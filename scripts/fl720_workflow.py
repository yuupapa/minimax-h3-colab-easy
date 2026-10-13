"""Cloud-only FL2VA 720p with finite sequential queues; queue control offline-tested. No ML imports at module import."""
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
import serial_queue as serial

KIT = Path(__file__).resolve().parent.parent
# Isolated module instance: never mutate the published workflow's globals.
_spec = importlib.util.spec_from_file_location('h3_fl720_guard_helpers', KIT/'scripts/colab_workflow.py')
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)
guard.ROOT = Path('/content/h3-fl720-dev')
guard.SCRATCH = Path('/mnt/local-scratch/h3-fl720-dev') if Path('/mnt/local-scratch').is_dir() else Path('/content/h3-fl720-dev-cache')
guard.VENV = guard.ROOT/'venv'
guard.STATE = guard.ROOT/'state.json'
guard.KIT = KIT
serial.install_budget_policy(guard)
WorkflowError = guard.WorkflowError
BASE_REV = '42ed227ee7df40d41602854ae760620d6eb651fe'
DIFFUSERS_REV = '85c9fa4caee33d1c5b473395784471d2d7957c1b'
MODELTC_REV = '02e26d591f7a04d5d1a074c9566d5dd4f22f6225'
LORA_REV = '3ec17a324ced54151364f24f8b5fb6bf7e26414f'
LORA_FILE = 'minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors'
LORA_SHA = '1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c'
PATTERNS = ['model_index.json', 'modular_model_index.json', 'LICENSE'] + [x+'/*' for x in
            ('transformer','text_encoder','processor','tokenizer','vae','audio_vae','scheduler','audio_scheduler')]
PROFILE = {'megapixels':0.9,'native':[1280,736], 'delivery':[1280,720],
           'filter':'crop=1280:720:0:8,setsar=1',
           'transform':'8px top/bottom crop; no resize'}



def budget_settings(settings):
    # Reuse ONLY the budget validator; the model task is built separately below.
    result = {k:settings[k] for k in ('license_accepted','eligible_use_confirmed','prompts','count','seconds',
                                  'seed','cu_remaining','cu_rate','cu_budget')}
    return dict(result, count=1, prompts=[settings['prompts'][0]])


def validate(settings, *, with_image=True):
    if not isinstance(settings,dict):raise WorkflowError('現在のINPUT_OKが必要です。旧入力・旧出力へ戻りません。')
    for k in ('gpu_trial_approved','source_execution_accepted','license_accepted','eligible_use_confirmed',
              'higher_resolution_risk_accepted'):
        if settings.get(k) is not True:
            raise WorkflowError(f'{k}: 本人の確認が必要です。GPU未承認では実行しません。')
    approved=settings.get('approved_modes')
    if not isinstance(approved,list) or not approved or any(not isinstance(m,str) or m not in ('t2v','i2v') for m in approved) or len(approved)!=len(set(approved)):
        raise WorkflowError('今回承認したモードをapproved_modesへ明示してください。今回使用するモードを選択してください。')
    if settings.get('mode') not in approved:
        raise WorkflowError('選択モードのGPU試験は未承認です。')
    if settings.get('mode') not in ('t2v','i2v'):
        raise WorkflowError('FL720はT2V/I2V専用です。R2Vは別ノートブックです。')
    serial.validate_count(settings, WorkflowError)
    if settings.get('seconds') != 5:
        raise WorkflowError('720pは指定5秒です。1キューの始点画像は1枚です。')
    if settings.get('profile')!='native736_crop720':
        raise WorkflowError('生成1280×736→上下8px crop→1280×720だけです。544拡大は行いません。')
    guard.validate_inputs(budget_settings(settings))
    elapsed=settings.get('gpu_minutes_before_input',0.0)
    if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not 0<=elapsed<float('inf'):
        raise WorkflowError('入力前のGPU接続経過分は0以上の有限な分数を入力してください。')
    if settings.get('reference') or settings.get('references'):
        raise WorkflowError('意味的参照は受け取りません。R2Vは専用ノートブックで実行してください。')
    if settings['mode']=='t2v':
        if settings.get('first_frame'):
            raise WorkflowError('T2Vに始点画像は渡しません。I2Vを明示的に選んでください。')
    else:
        if settings.get('image_use_confirmed') is not True:
            raise WorkflowError('始点画像の権利・人物の同意・Colab送信を本人が確認してください。')
        if with_image:
            if settings.get('image_content_confirmed') is not True:
                raise WorkflowError('正規化した始点画像の内容と向きを確認してください。')
            path=guard.verify_first_frame(settings.get('first_frame'))
            width,height=struct.unpack('>II',path.read_bytes()[16:24])
            ref=settings['first_frame']
            if (ref.get('width'),ref.get('height'))!=(width,height):
                raise WorkflowError('始点画像の寸法記録が実際のPNGと異なります。')
            if max(width,height)>4*min(width,height):
                raise WorkflowError('始点画像の縦横比は1:4〜4:1です。初回は16:9を推奨します。')
    return dict(settings)


def jobs_document(settings):
    example={'task':'t2va' if settings['mode']=='t2v' else 'i2va',
             'prompt':settings['prompts'][0], 'duration':5,
             'megapixels':0.9,'aspect_ratio':'16:9'}
    if settings['mode']=='i2v':example['image']=settings['first_frame']['path']
    return {'examples':[example]}


def verify_vendor():
    for entry in json.loads((KIT/'vendor/modeltc/manifest.json').read_text()):
        data=(KIT/'vendor/modeltc'/entry['path']).read_bytes()
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:
            raise WorkflowError('固定したModelTCソースの検証に失敗しました。実行しません。')
        if MODELTC_REV not in entry['url']:
            raise WorkflowError('ModelTCのrevisionが固定版と異なります。')


def kit_digest():
    names=('scripts/colab_workflow.py','scripts/fl720_workflow.py','scripts/fl720_native.py','scripts/serial_queue.py',
           'requirements-r2v.lock','build-constraints.txt','vendor/modeltc/manifest.json')
    h=hashlib.sha256()
    for name in names:
        h.update(name.encode());h.update((KIT/name).read_bytes())
    return h.hexdigest()


def upload_first_frame(settings):
    guard.require_cloud()
    validate(settings,with_image=False)
    if settings['mode']!='i2v':raise WorkflowError('T2Vでは画像をアップロードしません。')
    state=require_session_scope(settings)
    if state.get('active_mode')!='i2v' or not state.get('mode_reviewed'):
        raise WorkflowError('I2Vへの切替と現在のモード確認が必要です。')
    # Bounded RGB first-keyframe normalization/upload helper.
    return guard.upload_first_frame(image_use_confirmed=True)


def arm_budget(settings, previous_timer=None):
    guard.require_cloud()
    validate(settings,with_image=False)
    initialize_budget(settings,allow_expired=True)
    return guard.schedule_budget_shutdown(budget_settings(settings),previous_timer=previous_timer)


def initialize_budget(settings, *, allow_expired=False):
    """Deduct observed GPU time before first input; never extend a deadline."""
    guard.require_cloud()
    settings=validate(settings,with_image=False)
    old=guard.read_state()
    modes=sorted(settings['approved_modes'])
    if old.get('generation_attempted') and not old.get('runs'):
        raise WorkflowError('旧候補の実行状態を新しい試験に流用しません。保存してGPUを終了してください。')
    state=guard.initialize_budget(budget_settings(settings),allow_expired=True)
    state.update(approved_modes=modes,session_schema=2)
    start=min(state['session_started_epoch'],time.time()-settings.get('gpu_minutes_before_input',0.0)*60)
    budget=float(settings['cu_budget']);rate=float(settings['cu_rate'])
    state.update(session_started_epoch=start,
                 deadline_epoch=min(state['deadline_epoch'],start+budget/rate*3600*.85),
                 shutdown_deadline_epoch=min(state['shutdown_deadline_epoch'],start+budget/rate*3600))
    guard.write_state(state)
    if not allow_expired:guard.deadline_remaining()
    return state


def inspect_resources():
    resources=guard.inspect_resources()
    # R2V used78.42GiB reserved. Require an almost-idle80GB card for this new trial.
    # This conservative entry check is not a bound on FL720's unmeasured peak.
    if resources['free_vram_mib']<80500:
        raise WorkflowError('FL720試験は空きVRAM80,500MiB以上のA10080GBを使います。他処理を止め、空きを確認してください。成功保証ではありません。')
    return resources


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
    require_active_request(settings)
    initialize_budget(settings)
    verify_vendor()
    resources=inspect_resources()
    guard.SCRATCH.mkdir(parents=True,exist_ok=True)
    state=guard.read_state()
    if state.get('setup_attempted'):
        lock=guard.ROOT/'requirements-r2v.lock'
        if state.get('setup_complete') and state.get('kit_digest')==kit_digest() and lock.is_file() and state.get('requirements_sha256')==hashlib.sha256(lock.read_bytes()).hexdigest():
            print('SETUP_ALREADY_COMPLETE');return
        raise WorkflowError('初回セットアップ失敗後の自動再試行はしません。ログを保存してGPUを終了してください。')
    state.update(setup_attempted=True, resources=resources,native_parent_pid=os.getpid(),cloud_boot_id=guard.boot_id())
    guard.write_state(state)
    env=environment()
    py=str(guard.VENV/'bin/python')
    try:
        guard.run_waiting(['python3','-m','pip','install','uv==0.13.0'],'install_uv',env=env)
        guard.run_waiting(['uv','venv','--python','3.12',str(guard.VENV)],'create_fl720_venv',env=env)
        lock=guard.ROOT/'requirements-r2v.lock'
        # The complete 67-dependency lock is the exact artifact used in the GPU trial.
        # Do not resolve new transitive versions in a paid runtime.
        lock.write_bytes((KIT/'requirements-r2v.lock').read_bytes())
        guard.run_waiting(['uv','pip','sync','--python',py,'--build-constraint',str(KIT/'build-constraints.txt'),
                           str(lock)],'sync_fl720_environment',env=env)
        # API/import compatibility and scheduler/dimensions checked BEFORE downloading 134GiB.
        guard.run_waiting([py,str(KIT/'scripts/fl720_native.py'),'preflight'],'fl720_api_preflight',env=env)
        download_code='''import hashlib,json,time
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
start=time.monotonic()
model=snapshot_download('MiniMaxAI/MiniMax-H3',revision=BASE_REV,allow_patterns=PATTERNS,cache_dir=CACHE,token=False,max_workers=4)
lora=hf_hub_download('lightx2v/Minimax-h3-Turbo',LORA_FILE,revision=LORA_REV,cache_dir=CACHE,token=False)
digest=hashlib.file_digest(open(lora,'rb'),'sha256').hexdigest()
assert digest==LORA_SHA, 'LoRA hash mismatch'
assert not (Path(model)/'transformer_ref').exists(), 'inactive transformer downloaded'
for folder in ('transformer','text_encoder','processor','tokenizer','vae','audio_vae','scheduler','audio_scheduler'):
    assert (Path(model)/folder).is_dir(), 'missing component '+folder
Path(REPORT).write_text(json.dumps({'model_path':model,'lora_path':lora,'lora_sha256':digest,'download_seconds':time.monotonic()-start},indent=2))
'''
        values={'BASE_REV':BASE_REV,'PATTERNS':PATTERNS,'CACHE':str(guard.SCRATCH/'hf'),
                'LORA_FILE':LORA_FILE,'LORA_REV':LORA_REV,'LORA_SHA':LORA_SHA,
                'REPORT':str(guard.ROOT/'download_report.json')}
        download=guard.ROOT/'download_fl720.py'
        download.write_text('\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+download_code)
        guard.run_waiting([py,str(download)],'download_fl720_weights',env=env)
        state=guard.read_state()
        state.update(setup_complete=True,download_complete=True,kit_digest=kit_digest(),requirements_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                     setup_profile=settings['profile'])
        guard.write_state(state)
        print('SETUP_OK: FL720生成はまだです。')
    except BaseException:
        state=guard.read_state();state.update(setup_failed=True,session_failed=True);guard.write_state(state)
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


def single_fingerprint(settings):
    value={'mode':settings['mode'],'prompt':settings['prompts'][0],
           'profile':settings['profile'],'seed':settings['seed'],
           'versions':[BASE_REV,LORA_REV,DIFFUSERS_REV,MODELTC_REV]}
    if settings['mode']=='i2v':value['first_frame_sha256']=settings['first_frame']['sha256']
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def fingerprint(settings):
    return serial.request_fingerprint(settings, single_fingerprint)


def require_session_scope(settings):
    state=guard.read_state()
    if state.get('session_schema')!=2 or state.get('approved_modes')!=sorted(settings['approved_modes']):
        raise WorkflowError('今回のVMの承認モードと一致しません。先に入力・共有予算を確定してください。')
    return state


@guard.cloud_operation
def activate_mode(settings):
    settings=validate(settings,with_image=False)
    guard.deadline_remaining()
    state=require_session_scope(settings)
    mode=settings['mode'];previous=state.get('active_mode')
    if mode==previous:return False
    queue=state.get('generation_queues',{}).get(state.get('active_queue_id'),{})
    if queue.get('status')=='running':
        raise WorkflowError('実行中です。先に実行を中断してください。')
    state.update(active_mode=mode,active_fingerprint=None,mode_reviewed=previous is None)
    state.pop('last_input_image',None)
    guard.write_state(state)
    return previous is not None


@guard.cloud_operation
def confirm_mode(settings):
    validate(settings,with_image=False)
    state=require_session_scope(settings)
    if state.get('active_mode')!=settings['mode']:
        raise WorkflowError('現在のモードが一致しません。入力セルで切替内容を確認してください。')
    state['mode_reviewed']=True;guard.write_state(state)


@guard.cloud_operation
def select_request(settings):
    settings=validate(settings)
    guard.deadline_remaining()
    state=require_session_scope(settings);mode=settings['mode'];request=fingerprint(settings)
    if state.get('active_mode')!=mode or not state.get('mode_reviewed'):
        raise WorkflowError('現在のモードを入力セルで確認してください。')
    if mode=='i2v' and state.get('last_input_image')!=settings['first_frame']:
        raise WorkflowError('現在のモードでアップロード・確認した新しい始点画像だけを使います。')
    return serial.select(guard, settings, fingerprint, mode=mode,
                         legacy_result=state.get('runs',{}).get(mode))


def require_active_request(settings):
    state=require_session_scope(settings)
    if state.get('active_mode')!=settings['mode'] or not state.get('mode_reviewed') or state.get('active_fingerprint')!=fingerprint(settings):
        raise WorkflowError('現在の入力とモードが一致しません。INPUT_OKからやり直してください。')
    if settings['mode']=='i2v' and state.get('last_input_image')!=settings['first_frame']:
        raise WorkflowError('現在の始点画像と入力が一致しません。旧画像へ戻りません。')
    return state


def current_output(settings):
    guard.require_cloud()
    settings=validate(settings)
    require_active_request(settings)
    return serial.current_output(guard,settings,fingerprint)


@guard.cloud_operation
def confirm_saved(settings, *, run_id, archive_path):
    entry=current_output(settings)
    if (run_id,archive_path)!=(entry['run_id'],entry['archive_path']):
        raise WorkflowError('保存確認の対象が現在のキュー/ZIPと異なります。')
    state=guard.read_state();state['generation_queues'][entry['queue_id']]['saved_confirmed']=True
    guard.write_state(state)
    return entry


@guard.cloud_operation
def generate(settings):
    settings=validate(settings)
    initialize_budget(settings)
    state=require_active_request(settings)
    if not state.get('active_queue_id'):
        settings=serial.select(guard,settings,fingerprint,mode=settings['mode'],legacy_result=state.get('runs',{}).get(settings['mode']))
    if serial.active(guard,settings,fingerprint)[1]['status']!='complete':
        if not state.get('setup_complete'):raise WorkflowError('SETUP_OKが必要です。')
        if state.get('kit_digest')!=kit_digest() or state.get('requirements_sha256')!=hashlib.sha256((guard.ROOT/'requirements-r2v.lock').read_bytes()).hexdigest():raise WorkflowError('セットアップ後にコード・依存が変更されています。')
    return serial.run(guard,settings,fingerprint,_queued_one)


def _queued_one(settings):
    state=guard.read_state();mode=settings['mode']
    previous=state.setdefault('runs',{}).get(mode)
    if previous:
        state.setdefault('run_history',[]).append(previous)
    state['runs'][mode]={}
    state['active_fingerprint']=fingerprint(settings)
    state.pop('session_failed',None)
    guard.write_state(state)
    _generate_one(settings)
    return dict(guard.read_state()['runs'][mode],mode=mode)


def _generate_one(settings):
    settings=validate(settings)
    verify_vendor()
    initialize_budget(settings)
    state=require_active_request(settings);request=fingerprint(settings);mode=settings['mode']
    if not state.get('setup_complete'):
        raise WorkflowError('SETUP_OKが必要です。')
    if state.get('kit_digest')!=kit_digest() or state.get('requirements_sha256')!=hashlib.sha256((guard.ROOT/'requirements-r2v.lock').read_bytes()).hexdigest():
        raise WorkflowError('セットアップ後にコード・依存が変更されています。生成を開始しません。')
    inspect_resources()
    run_id=mode+'_'+uuid.uuid4().hex[:8]
    out=guard.ROOT/'outputs'/('run_'+run_id);out.mkdir(parents=True)
    state.setdefault('runs',{})[mode]={'attempted':True,'complete':False,'failed':False,
                                     'saved_confirmed':False,'fingerprint':request,
                                     'run_id':run_id,'output_dir':str(out)}
    state.update(native_parent_pid=os.getpid(),cloud_boot_id=guard.boot_id())
    guard.write_state(state)
    run={'settings':settings,'profile':PROFILE,'jobs':jobs_document(settings),
         'download':json.loads((guard.ROOT/'download_report.json').read_text()),'output_dir':str(out)}
    (guard.ROOT/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2))
    try:
        guard.run_waiting([str(guard.VENV/'bin/python'),str(KIT/'scripts/fl720_native.py'),'generate'],
                           'fl720_generation',env=environment(offline=True))
        native=out/'native.mp4';delivery=out/'delivery_720p.mp4';profile=PROFILE
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
        metadata.update(mode=mode,run_id=run_id,request_fingerprint=request,native_validation=native_info,delivery_validation=delivered,delivery_transform=profile['transform'],
                        actual_cu_used=None,cu_start=state['cu_remaining_at_start'],
                        cu_note='Read start/end balance in Colab UI after shutdown; timing budget is not billing measurement.',
                        qualitative_review='pending: prompt adherence, I2V first-frame continuity, motion/artifacts, crop framing, audio listening',
                        ai_generated=True, source_revisions={'base':BASE_REV,'lora':LORA_REV,'diffusers':DIFFUSERS_REV,'modeltc':MODELTC_REV})
        (out/'metrics.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
        (out/'AI_GENERATED.txt').write_text('AI generated with MiniMax H3. Native and delivered resolutions are recorded separately. Input image is not included.\n')
        archive=guard.ROOT/(out.name+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in ('native.mp4','delivery_720p.mp4','metrics.json','AI_GENERATED.txt'):
                z.write(out/name,name)
            z.write(guard.ROOT/'requirements-r2v.lock','requirements-r2v.lock')
        state=guard.read_state();state['runs'][mode].update(complete=True,archive_path=str(archive));guard.write_state(state)
        print('GENERATE_OK:',mode,run_id,'保存完了と画質の確認はまだです。');return str(archive)
    except BaseException:
        state=guard.read_state();state['runs'][mode].update(failed=True);guard.write_state(state)
        print('GENERATION_FAILED: 自動再生成・解像度変更はしません。ログを保存しGPUを終了してください。')
        raise


def shutdown(*, saved=False, stop_after_error=False):
    guard.require_cloud()
    if not saved and not stop_after_error:
        raise WorkflowError('正常時は実際の保存完了を確認してください。失敗・中止時はSTOP_AFTER_ERRORを確認して一時ファイルを破棄し終了できます。')
    return guard.shutdown(saved_confirmed=True)
