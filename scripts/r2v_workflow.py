"""Experimental cloud-only Ref2VA workflow. No ML imports at module import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import struct
import time
import uuid
import zipfile

KIT = Path(__file__).resolve().parent.parent
# Isolated module instance: never mutate the published workflow's globals.
_spec = importlib.util.spec_from_file_location('h3_r2v_guard_helpers', KIT/'scripts/colab_workflow.py')
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)
guard.ROOT = Path('/content/h3-r2v-dev')
guard.SCRATCH = Path('/mnt/local-scratch/h3-r2v-dev') if Path('/mnt/local-scratch').is_dir() else Path('/content/h3-r2v-dev-cache')
guard.VENV = guard.ROOT/'venv'
guard.STATE = guard.ROOT/'state.json'
guard.KIT = KIT
WorkflowError = guard.WorkflowError
BASE_REV = '42ed227ee7df40d41602854ae760620d6eb651fe'
DIFFUSERS_REV = '85c9fa4caee33d1c5b473395784471d2d7957c1b'
MODELTC_REV = '02e26d591f7a04d5d1a074c9566d5dd4f22f6225'
LORA_REV = '3ec17a324ced54151364f24f8b5fb6bf7e26414f'
LORA_FILE = 'minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors'
LORA_SHA = '9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99'
PATTERNS = ['model_index.json', 'modular_model_index.json', 'LICENSE'] + [x+'/*' for x in
            ('transformer_ref','text_encoder','processor','tokenizer','vae','audio_vae','scheduler','audio_scheduler')]
PROFILES = {
    'near720_crop': {'megapixels':0.9,'native':[1280,736], 'delivery':[1280,720],
                     'filter':'crop=1280:720:0:8,setsar=1','transform':'8px top/bottom crop; no resize'},
    '544_upscale': {'megapixels':0.5,'native':[960,544], 'delivery':[1280,720],
                    'filter':'crop=960:540:0:2,scale=1280:720:flags=lanczos,setsar=1',
                    'transform':'2px top/bottom crop, then 4/3 upscale; no additional native detail'},
}


def budget_settings(settings):
    # Reuse ONLY the budget validator; the model task is built separately below.
    return {k:settings[k] for k in ('license_accepted','eligible_use_confirmed','prompts','count','seconds',
                                  'seed','cu_remaining','cu_rate','cu_budget')}


def validate(settings, *, with_reference=True):
    for k in ('gpu_trial_approved','source_execution_accepted','license_accepted','eligible_use_confirmed',
              'image_use_confirmed'):
        if settings.get(k) is not True:
            raise WorkflowError(f'{k}: 本人の確認が必要です。GPU未承認では実行しません。')
    if settings.get('count') != 1 or isinstance(settings.get('count'), bool) or settings.get('seconds') != 5:
        raise WorkflowError('このR2V試験は静止参照1枚、動画1本、指定5秒だけです。')
    if settings.get('profile') not in PROFILES:
        raise WorkflowError('未対応の解像度です。720は納品寸法であり生成寸法と区別します。')
    base=guard.validate_inputs(budget_settings(settings))
    elapsed=settings.get('gpu_minutes_before_input',0.0)
    if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not 0<=elapsed<float('inf'):
        raise WorkflowError('入力前のGPU接続経過分は0以上の有限な分数を入力してください。')
    if '<Picture 1>' not in base['prompts'][0] or re.search(r'<Picture\s+(?!1>)[^>]+>',base['prompts'][0]):
        raise WorkflowError('1枚の参照を <Picture 1> で明示してください。別の番号は使いません。')
    if settings['profile']=='near720_crop' and settings.get('higher_resolution_risk_accepted') is not True:
        raise WorkflowError('参照LoRA v0.1は544p学習です。1280×736の成功は1例だけで、品質・OOMリスクを確認してください。')
    if with_reference:
        if settings.get('image_content_confirmed') is not True:
            raise WorkflowError('正規化した参照画像の内容と向きを確認してください。')
        try:
            path=guard.verify_first_frame(settings.get('reference'))
        except WorkflowError as exc:
            raise WorkflowError(str(exc).replace('I2V','R2V').replace('始点画像','参照画像')) from exc
        ref=settings['reference']
        width,height=struct.unpack('>II',path.read_bytes()[16:24])
        if (ref.get('width'),ref.get('height'))!=(width,height):
            raise WorkflowError('参照画像の寸法記録が実際のPNGと異なります。')
        if max(width,height)>4*min(width,height):
            raise WorkflowError('参照画像の縦横比は1:4〜4:1です。')
    return dict(settings)


def jobs_document(settings):
    return {'examples':[{'task':'ref2va','prompt':settings['prompts'][0], 'duration':5,
                         'megapixels':PROFILES[settings['profile']]['megapixels'],'aspect_ratio':'16:9',
                         'references':[{'type':'image','path':settings['reference']['path']}]}]}


def verify_vendor():
    for entry in json.loads((KIT/'vendor/modeltc/manifest.json').read_text()):
        data=(KIT/'vendor/modeltc'/entry['path']).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise WorkflowError('固定したModelTCソースの検証に失敗しました。実行しません。')
        if MODELTC_REV not in entry['url']:
            raise WorkflowError('ModelTCのrevisionが固定版と異なります。')


def kit_digest():
    names=('scripts/colab_workflow.py','scripts/r2v_workflow.py','scripts/r2v_native.py',
           'requirements-r2v.in','requirements-r2v.lock','build-constraints.txt','vendor/modeltc/manifest.json')
    h=hashlib.sha256()
    for name in names:
        h.update(name.encode());h.update((KIT/name).read_bytes())
    return h.hexdigest()


def upload_reference(settings):
    guard.require_cloud()
    validate(settings,with_reference=False)
    # Generic bounded RGB normalization/upload helper. No I2V conditioning is used.
    return guard.upload_first_frame(image_use_confirmed=True)


def arm_budget(settings, previous_timer=None):
    guard.require_cloud()
    validate(settings,with_reference=False)
    initialize_budget(settings,allow_expired=True)
    return guard.schedule_budget_shutdown(budget_settings(settings),previous_timer=previous_timer)


def initialize_budget(settings, *, allow_expired=False):
    """Deduct observed GPU time before first input; never extend a deadline."""
    guard.require_cloud()
    settings=validate(settings,with_reference=False)
    state=guard.initialize_budget(budget_settings(settings),allow_expired=True)
    start=min(state['session_started_epoch'],time.time()-settings.get('gpu_minutes_before_input',0.0)*60)
    budget=float(settings['cu_budget']);rate=float(settings['cu_rate'])
    state.update(session_started_epoch=start,
                 deadline_epoch=min(state['deadline_epoch'],start+budget/rate*3600*.85),
                 shutdown_deadline_epoch=min(state['shutdown_deadline_epoch'],start+budget/rate*3600))
    guard.write_state(state)
    if not allow_expired:guard.deadline_remaining()
    return state


def environment(*, offline=False):
    env=dict(os.environ, UV_CACHE_DIR=str(guard.SCRATCH/'uv-cache'), HF_HOME=str(guard.SCRATCH/'hf-home'),
             HF_HUB_CACHE=str(guard.SCRATCH/'hf'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
             HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_OFFLINE='1' if offline else '0',
             TRANSFORMERS_OFFLINE='1' if offline else '0', TOKENIZERS_PARALLELISM='false')
    # Never propagate a token to public downloads or inference.
    for key in ('HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
        env.pop(key,None)
    return env


@guard.cloud_operation
def setup(settings):
    settings=validate(settings)
    initialize_budget(settings)
    verify_vendor()
    resources=guard.inspect_resources()
    guard.SCRATCH.mkdir(parents=True,exist_ok=True)
    state=guard.read_state()
    if state.get('setup_attempted'):
        if state.get('setup_complete') and state.get('kit_digest')==kit_digest():
            print('SETUP_ALREADY_COMPLETE');return
        raise WorkflowError('初回セットアップ失敗後の自動再試行はしません。ログを保存してGPUを終了してください。')
    state.update(setup_attempted=True, resources=resources,native_parent_pid=os.getpid(),cloud_boot_id=guard.boot_id())
    guard.write_state(state)
    env=environment()
    py=str(guard.VENV/'bin/python')
    try:
        guard.run_waiting(['python3','-m','pip','install','uv==0.13.0'],'install_uv',env=env)
        guard.run_waiting(['uv','venv','--python','3.12',str(guard.VENV)],'create_r2v_venv',env=env)
        lock=guard.ROOT/'requirements-r2v.lock'
        # The complete 67-dependency lock is the exact artifact used in the GPU trial.
        # Do not resolve new transitive versions in a paid runtime.
        lock.write_bytes((KIT/'requirements-r2v.lock').read_bytes())
        guard.run_waiting(['uv','pip','sync','--python',py,'--build-constraint',str(KIT/'build-constraints.txt'),
                           str(lock)],'sync_r2v_environment',env=env)
        # API/import compatibility and scheduler/dimensions checked BEFORE downloading 134GiB.
        guard.run_waiting([py,str(KIT/'scripts/r2v_native.py'),'preflight'],'r2v_api_preflight',env=env)
        download_code='''import hashlib,json,time
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
start=time.monotonic()
model=snapshot_download('MiniMaxAI/MiniMax-H3',revision=BASE_REV,allow_patterns=PATTERNS,cache_dir=CACHE,token=False,max_workers=4)
lora=hf_hub_download('lightx2v/Minimax-h3-Turbo',LORA_FILE,revision=LORA_REV,cache_dir=CACHE,token=False)
digest=hashlib.file_digest(open(lora,'rb'),'sha256').hexdigest()
assert digest==LORA_SHA, 'LoRA hash mismatch'
assert not (Path(model)/'transformer').exists(), 'inactive transformer downloaded'
for folder in ('transformer_ref','text_encoder','processor','tokenizer','vae','audio_vae','scheduler','audio_scheduler'):
    assert (Path(model)/folder).is_dir(), 'missing component '+folder
Path(REPORT).write_text(json.dumps({'model_path':model,'lora_path':lora,'lora_sha256':digest,'download_seconds':time.monotonic()-start},indent=2))
'''
        values={'BASE_REV':BASE_REV,'PATTERNS':PATTERNS,'CACHE':str(guard.SCRATCH/'hf'),
                'LORA_FILE':LORA_FILE,'LORA_REV':LORA_REV,'LORA_SHA':LORA_SHA,
                'REPORT':str(guard.ROOT/'download_report.json')}
        download=guard.ROOT/'download_ref.py'
        download.write_text('\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+download_code)
        guard.run_waiting([py,str(download)],'download_ref_weights',env=env)
        state=guard.read_state()
        state.update(setup_complete=True,download_complete=True,kit_digest=kit_digest(),requirements_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                     setup_profile=settings['profile'])
        guard.write_state(state)
        print('SETUP_OK: R2V生成はまだです。')
    except BaseException:
        state=guard.read_state();state.update(setup_failed=True);guard.write_state(state)
        print('SETUP_FAILED: 自動再試行せず、ログを保存して終了セルでGPUを終了してください。')
        raise


def probe(path):
    result=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],
                          capture_output=True,text=True,check=True,timeout=min(30,guard.deadline_remaining()))
    return json.loads(result.stdout)


def check_probe(info, width, height):
    videos=[s for s in info['streams'] if s['codec_type']=='video']
    audios=[s for s in info['streams'] if s['codec_type']=='audio']
    if len(videos)!=1 or len(audios)!=1:
        raise WorkflowError('動画1トラック・音声1トラックを検証できません。')
    v=videos[0]
    if (v['width'],v['height'])!=(width,height) or v.get('nb_frames')!='124' or v.get('avg_frame_rate')!='24/1':
        raise WorkflowError('生成寸法・フレーム数・FPSが期待値と異なります。')
    duration=float(info['format']['duration'])
    if not 5.1<=duration<=5.4:
        raise WorkflowError('5秒指定のフレーム整合後の長さと異なります。')
    return {'width':width,'height':height,'frames':124,'fps':24,'container_seconds':duration}


def fingerprint(settings):
    return hashlib.sha256(json.dumps({'prompt':settings['prompts'][0],'reference_sha256':settings['reference']['sha256'],
                                     'profile':settings['profile'],'seed':settings['seed'],'versions':[BASE_REV,LORA_REV,DIFFUSERS_REV,MODELTC_REV]},
                                     sort_keys=True).encode()).hexdigest()


@guard.cloud_operation
def generate(settings):
    settings=validate(settings)
    verify_vendor()
    initialize_budget(settings)
    state=guard.read_state();request=fingerprint(settings)
    if state.get('generation_attempted'):
        if state.get('generation_complete') and state.get('fingerprint')==request:
            return state['archive_path']
        raise WorkflowError('このVMでは試作1本の実行許可を消費済みです。再試行・設定変更後の再生成は別の明示承認と新しいVMが必要です。')
    if not state.get('setup_complete'):
        raise WorkflowError('SETUP_OKが必要です。')
    if state.get('kit_digest')!=kit_digest() or state.get('requirements_sha256')!=hashlib.sha256((guard.ROOT/'requirements-r2v.lock').read_bytes()).hexdigest():
        raise WorkflowError('セットアップ後にコード・依存が変更されています。生成を開始しません。')
    guard.inspect_resources()
    out=guard.ROOT/'outputs'/('run_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    state.update(generation_attempted=True,generation_complete=False,fingerprint=request,
                 output_dir=str(out),native_parent_pid=os.getpid(),cloud_boot_id=guard.boot_id())
    guard.write_state(state)
    run={'settings':settings,'profile':PROFILES[settings['profile']],'jobs':jobs_document(settings),
         'download':json.loads((guard.ROOT/'download_report.json').read_text()),'output_dir':str(out)}
    (guard.ROOT/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2))
    try:
        guard.run_waiting([str(guard.VENV/'bin/python'),str(KIT/'scripts/r2v_native.py'),'generate'],
                           'r2v_generation',env=environment(offline=True))
        native=out/'native.mp4';delivery=out/'delivery_720p.mp4';profile=PROFILES[settings['profile']]
        native_info=check_probe(probe(native),*profile['native'])
        guard.run_waiting(['ffmpeg','-v','error','-xerror','-i',str(native),'-map','0:v:0','-map','0:a:0',
                           '-f','null','-'],'decode_native',env=environment(offline=True))
        guard.run_waiting(['ffmpeg','-v','error','-xerror','-i',str(native),'-map','0:v:0','-map','0:a:0',
                           '-vf',profile['filter'],'-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p',
                           '-c:a','copy','-movflags','+faststart',str(delivery)],'export_exact720',env=environment(offline=True))
        delivered=check_probe(probe(delivery),1280,720)
        guard.run_waiting(['ffmpeg','-v','error','-xerror','-i',str(delivery),'-map','0:v:0','-map','0:a:0',
                           '-f','null','-'],'decode_delivery',env=environment(offline=True))
        metadata=json.loads((out/'metrics.json').read_text())
        metadata.update(native_validation=native_info,delivery_validation=delivered,delivery_transform=profile['transform'],
                        actual_cu_used=None,cu_start=state['cu_remaining_at_start'],
                        cu_note='Read start/end balance in Colab UI after shutdown; timing budget is not billing measurement.',
                        qualitative_review='pending: identity/reference adherence, background independence, motion/artifacts, audio listening',
                        ai_generated=True, source_revisions={'base':BASE_REV,'lora':LORA_REV,'diffusers':DIFFUSERS_REV,'modeltc':MODELTC_REV})
        (out/'metrics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        (out/'AI_GENERATED.txt').write_text('AI generated with MiniMax H3. Native and delivered resolutions are recorded separately. Reference image is not included.\n')
        archive=guard.ROOT/(out.name+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('native.mp4','delivery_720p.mp4','metrics.json','AI_GENERATED.txt'):
                z.write(out/name,name)
            z.write(guard.ROOT/'requirements-r2v.lock','requirements-r2v.lock')
        state=guard.read_state();state.update(generation_complete=True,archive_path=str(archive));guard.write_state(state)
        print('GENERATE_OK: 保存完了と画質の確認はまだです。');return str(archive)
    except BaseException:
        state=guard.read_state();state.update(generation_failed=True);guard.write_state(state)
        print('GENERATION_FAILED: 自動再生成・解像度変更はしません。ログを保存しGPUを終了してください。')
        raise


def shutdown(*, saved=False, stop_after_error=False):
    guard.require_cloud()
    if not saved and not stop_after_error:
        raise WorkflowError('正常時は実際の保存完了を確認してください。失敗・中止時はSTOP_AFTER_ERRORを確認して一時ファイルを破棄し終了できます。')
    return guard.shutdown(saved_confirmed=True)
