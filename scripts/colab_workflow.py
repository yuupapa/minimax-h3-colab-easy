"""Colab-only workflow. No model downloads or setup on a user's Mac.

Original orchestration code; model and dependencies retain their own licenses.
"""
import argparse
import hashlib
import functools
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import struct
import sys
import time
import threading
import uuid
import warnings
import zipfile

BASE_REV = '42ed227ee7df40d41602854ae760620d6eb651fe'
SGLANG_REV = '448759b2358eb38e2a244fbe7d5a293a69dd314c'
LORA_REV = '797b284f774ff0bda805f7dd5917624670d5c4e0'
LORA_FILE = 'minimax_h3_fl2v_turbo_4step_v0.1.safetensors'
LORA_SHA = '5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29'
UV_VERSION = '0.13.0'
PYTHON_VERSION = '3.12'
ROOT = Path('/content/h3-easy')
SCRATCH = Path('/mnt/local-scratch/h3-easy') if Path('/mnt/local-scratch').is_dir() else Path('/content/h3-easy-cache')
VENV = ROOT / 'venv'
STATE = ROOT / 'state.json'
KIT = Path(__file__).resolve().parent.parent


class WorkflowError(RuntimeError):
    pass


class BudgetExceeded(WorkflowError):
    pass


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def require_cloud(*, verified_child=False):
    """Fail before creating directories, installing anything or downloading."""
    try:
        in_colab = importlib.util.find_spec('google.colab') is not None
    except (ModuleNotFoundError, ValueError):
        in_colab = False
    linux_content = platform.system() == 'Linux' and Path('/content').is_dir()
    if not in_colab and verified_child and linux_content:
        # The isolated model venv intentionally has no google.colab package.
        # Only accept a child of the guarded host process, on the same live VM.
        state=read_state()
        in_colab=state.get('native_parent_pid')==os.getppid() and state.get('cloud_boot_id')==boot_id()
    if not (in_colab and linux_content):
        raise WorkflowError('この処理はGoogle ColabのクラウドGPU専用です。MacやローカルPCでは実行しません。Colabノートブックを開いてください。')


def read_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def write_state(value):
    ROOT.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(STATE)


def validate_inputs(settings):
    if not settings.get('license_accepted') or not settings.get('eligible_use_confirmed'):
        raise WorkflowError('H3のライセンス・利用地域・利用条件を本人が確認し、2つの確認欄をオンにしてください。AIは代わりに同意できません。')
    count = settings.get('count')
    if isinstance(count, bool) or not isinstance(count, int) or count not in (1, 2):
        raise WorkflowError('初心者版の本数は1本または2本です。40本の一括生成は含みません。')
    if settings.get('seconds') not in (5, 6):
        raise WorkflowError('秒数は5または6を選んでください。実機検証済みは5秒です。')
    prompts = settings.get('prompts', [])
    if len(prompts) != count or any(not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in prompts):
        raise WorkflowError('本数に対応する文章を入力してください（各1〜4000文字）。')
    seed = settings.get('seed')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 2:
        raise WorkflowError('seedは0〜2147483646の整数です。')
    for key in ('cu_remaining', 'cu_rate', 'cu_budget'):
        try:
            value = float(settings[key])
        except (KeyError, ValueError, TypeError):
            raise WorkflowError('Colabのリソース画面の残CU・使用率・上限CUを入力してください。') from None
        if isinstance(settings[key], bool) or not 0 < value < float('inf'):
            raise WorkflowError('残CU・使用率・上限CUは0より大きい数を入力してください。')
    if float(settings['cu_budget']) > float(settings['cu_remaining']):
        raise WorkflowError('上限CUが残量を超えています。追加購入せず、上限を下げるか作業を終了してください。')
    if float(settings['cu_budget']) > 5:
        raise WorkflowError('初心者版の上限は5 CUです。既定3 CUから始めてください。')
    mode=settings.get('mode','t2v')
    if mode not in ('t2v','i2v'):
        raise WorkflowError('R2Vは別のRef2VA重み・対応検証が必要で、この候補ではまだ実行しません。T2VかI2Vを選んでください。')
    if mode=='i2v':
        if settings.get('image_use_confirmed') is not True:
            raise WorkflowError('画像の権利・写った人物の同意・Google Colabへの送信を本人が確認してください。')
        if settings.get('image_content_confirmed') is not True:
            raise WorkflowError('正規化した始点画像の内容と向きを本人が確認してください。')
        if count != 1:
            raise WorkflowError('未実機検証のI2Vは始点画像1枚・動画1本だけです。')
        verify_first_frame(settings.get('first_frame'))
    elif settings.get('first_frame'):
        raise WorkflowError('T2Vに画像は渡しません。I2Vを明示的に選んでください。')
    return dict(settings)


def verify_first_frame(frame):
    """Only accept a normalized, content-addressed image in our cloud input dir."""
    if not isinstance(frame,dict) or not isinstance(frame.get('path'),str):
        raise WorkflowError('I2Vの始点画像をアップロード・確認してください。')
    p=Path(frame['path']); base=ROOT/'inputs'
    digest=frame.get('sha256')
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise WorkflowError('始点画像のハッシュが不正です。')
    if p.parent!=base or p.name!=digest+'.png' or p.is_symlink() or not p.is_file():
        raise WorkflowError('このColabで確認した始点画像だけを使います。URL・Drive・任意パスは使いません。')
    if not 0<p.stat().st_size<=40*1024**2:
        raise WorkflowError('始点画像の容量が不正です。')
    data=p.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise WorkflowError('始点画像が変更・破損しています。再確認してください。')
    if len(data)<33 or data[:8]!=b'\x89PNG\r\n\x1a\n' or data[8:16]!=b'\x00\x00\x00\rIHDR':
        raise WorkflowError('確認済みの正規化PNGではありません。')
    w,h=struct.unpack('>II',data[16:24])
    if min(w,h)<64 or max(w,h)>8192 or w*h>16_000_000:
        raise WorkflowError('始点画像の画素数が不正です。')
    return p


def prepare_first_frame(source, *, image_use_confirmed=False):
    """Cloud-only, bounded decode; normalize orientation and remove metadata."""
    require_cloud()
    if image_use_confirmed is not True:
        raise WorkflowError('画像の権利とColabへの送信を本人が確認してください。')
    source=Path(source)
    if source.parent!=ROOT/'uploads' or source.is_symlink() or not source.is_file():
        raise WorkflowError('このColabの画像アップロード領域だけを読み取ります。')
    if not 0<source.stat().st_size<=20*1024**2:
        raise WorkflowError('画像は20MiB以内のPNG/JPEG/WebPを1枚選んでください。')
    try:
        from PIL import Image,ImageOps
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(source) as image:
                if image.format not in ('PNG','JPEG','WEBP') or getattr(image,'n_frames',1)!=1:
                    raise WorkflowError('静止画PNG/JPEG/WebPだけを使います。アニメ画像は使いません。')
                w,h=image.size
                if min(w,h)<64 or max(w,h)>8192 or w*h>16_000_000:
                    raise WorkflowError('画像は各辺64〜8192px、合計1600万画素以内です。')
                image.load()
                # A new RGB image intentionally drops EXIF, text and ICC metadata.
                oriented=ImageOps.exif_transpose(image).convert('RGB')
                clean=Image.new('RGB',oriented.size);clean.paste(oriented)
                import io
                stream=io.BytesIO();clean.save(stream,format='PNG');data=stream.getvalue()
    except WorkflowError:
        raise
    except Exception as exc:
        raise WorkflowError('画像をデコードできません。ColabのPillowと画像形式を確認してください。') from exc
    digest=hashlib.sha256(data).hexdigest();path=ROOT/'inputs'/(digest+'.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink(): raise WorkflowError('画像の保存先が不正です。')
    path.write_bytes(data)
    result={'path':str(path),'sha256':digest,'width':clean.width,'height':clean.height}
    verify_first_frame(result)
    return result


def upload_first_frame(*, image_use_confirmed=False):
    require_cloud()
    if image_use_confirmed is not True:
        raise WorkflowError('選んだ画像の権利・人物の同意・元画像のColab送信を本人が確認してください。')
    from google.colab import files
    source=ROOT/'uploads'/(uuid.uuid4().hex+'.upload')
    source.parent.mkdir(parents=True,exist_ok=True)
    state=read_state();state.pop('last_input_image',None);write_state(state)
    # Official single-file API: never use the uploaded filename as a path.
    print('本人が画像を1枚選択してください。元画像（EXIF等を含む）がGoogle Colabへ送信されます。')
    files.upload_file(str(source))
    result=prepare_first_frame(source,image_use_confirmed=True)
    state=read_state();state['last_input_image']=result;write_state(state)
    print('IMAGE_READY: 正規化した画像を確認してください。元画像はこのVMのuploads領域に残り、VM削除で消えます。')
    return result


def fingerprint(settings):
    keys = ('prompts', 'count', 'seconds', 'seed')
    value={k:settings[k] for k in keys}
    if settings.get('mode','t2v')=='i2v':
        verify_first_frame(settings['first_frame'])
        value.update(mode='i2v',first_frame_sha256=settings['first_frame']['sha256'])
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def dependency_digest():
    return hashlib.sha256(b'\0'.join((KIT/name).read_bytes() for name in ('requirements-colab.txt','build-constraints.txt'))).hexdigest()


def inspect_resources():
    require_cloud()
    if platform.machine() != 'x86_64':
        raise WorkflowError('この配布版はColab Linux x86_64用です。別環境へは導入しません。')
    if not shutil.which('nvidia-smi'):
        raise WorkflowError('GPUに接続されていません。ランタイムのタイプをA100・ハイメモリに変更してください。')
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise WorkflowError('この配布版はGPU 1枚用です。割当を確認してください。')
    name, total, free, driver = [x.strip() for x in rows[0].split(',')]
    memory = {a: int(b.split()[0]) for a, b in (line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines()) if b.strip() and b.split()[0].isdigit()}
    ram_gib = memory['MemTotal'] / 1024**2
    available_gib = memory['MemAvailable'] / 1024**2
    if 'A100' not in name or int(total) < 75000:
        raise WorkflowError(f'割当GPUは{name}（{int(total)/1024:.1f}GiB）です。この版はA100 80GB専用です。別GPUで進めず、割当できなければ終了してください。')
    if int(free) < 70000:
        raise WorkflowError('GPUメモリが他の処理で使用されています。別の生成を止め、空き容量を確認してください。')
    if ram_gib < 150 or available_gib < 140:
        raise WorkflowError(f'システムRAMは全体{ram_gib:.1f}GiB／空き{available_gib:.1f}GiBです。ハイメモリ（目安167GiB、空き140GiB以上）が必要です。')
    if int(driver.split('.')[0]) < 580:
        raise WorkflowError('固定したCUDA13系依存には新しいNVIDIAドライバが必要です。このランタイムでは進めません。')
    scratch_root = SCRATCH.parent
    if not scratch_root.is_dir():
        raise WorkflowError('モデル取得先のディスクがありません。このランタイムでは取得しません。')
    scratch_free = shutil.disk_usage(scratch_root).free / 1024**3
    content_free = shutil.disk_usage('/content').free / 1024**3
    shared_disk = scratch_root.stat().st_dev == Path('/content').stat().st_dev
    initial_required = 205 if shared_disk else 180
    if scratch_free < initial_required and not read_state().get('download_complete'):
        raise WorkflowError(f'モデル取得先の空きは{scratch_free:.1f}GiBです。初回は{initial_required}GiB以上必要です（重み・依存キャッシュ・共有時の環境用容量を含む）。')
    if content_free < 25:
        raise WorkflowError('/contentの空き容量が25GiB未満です。導入・生成を開始しません。')
    for tool in ('git', 'ffmpeg', 'ffprobe'):
        if not shutil.which(tool):
            raise WorkflowError(f'{tool}がありません。Colabの標準ランタイムに切り替えてください。')
    return {'gpu': name, 'vram_mib': int(total), 'free_vram_mib': int(free), 'driver': driver, 'ram_gib': round(ram_gib, 2), 'available_ram_gib': round(available_gib, 2), 'scratch_free_gib': round(scratch_free, 2), 'content_free_gib': round(content_free,2), 'cache_path': str(SCRATCH), 'shared_disk': shared_disk}


def deadline_remaining():
    state = read_state()
    if not state.get('deadline_epoch'):
        raise WorkflowError('先に入力とセットアップを完了してください。')
    seconds = state['deadline_epoch'] - time.time()
    if seconds <= 0:
        raise BudgetExceeded('CU上限から算出した時間枠に達しました。処理を停止し、ランタイムを切断してください。')
    return seconds


def kill_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_waiting(command, label, *, env=None):
    """One synchronous phase. A subprocess cannot outlive this function."""
    print(label, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    logfile = ROOT / (label.split(':', 1)[0] + '.log')
    process = None
    try:
        with logfile.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            last_progress=time.monotonic()
            while process.poll() is None:
                if time.monotonic()-last_progress>=30:
                    print(f'{label.split(":",1)[0]} 実行中（時間枠の残り約{deadline_remaining()/60:.1f}分）',flush=True)
                    last_progress=time.monotonic()
                if time.time() >= read_state()['deadline_epoch']:
                    raise BudgetExceeded('時間換算のCU上限に達したため子プロセスを停止しました。GPUも終了してください。')
                time.sleep(min(2, deadline_remaining()))
        if process.returncode:
            tail = logfile.read_text(errors='replace')[-2500:]
            raise WorkflowError(f'{label}に失敗しました。以降の生成は行いません。ログ末尾:\n{tail}')
        print(f'{label.split(":", 1)[0]} 完了', flush=True)
    finally:
        if process is not None:
            kill_group(process)


def cloud_operation(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        require_cloud()
        import fcntl
        ROOT.mkdir(parents=True, exist_ok=True)
        with (ROOT/'operation.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise WorkflowError('別のセットアップ・生成が実行中です。完了を待ってください。') from None
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return guarded


def initialize_budget(settings, *, allow_expired=False):
    require_cloud()
    settings=validate_inputs(settings)
    state=read_state()
    now=time.time()
    budget=float(settings['cu_budget']);rate=float(settings['cu_rate'])
    if state.get('deadline_epoch'):
        if budget!=state['cu_budget']:
            raise WorkflowError('同じセッション中にCU上限は変更できません。終了後、新しいランタイムで設定してください。')
        state['deadline_epoch']=min(state['deadline_epoch'],state['session_started_epoch']+budget/rate*3600*0.85)
        state['shutdown_deadline_epoch']=min(state['shutdown_deadline_epoch'],state['session_started_epoch']+budget/rate*3600)
        state['cu_rate']=rate
    else:
        state.update(session_started_epoch=now,deadline_epoch=now+budget/rate*3600*0.85,shutdown_deadline_epoch=now+budget/rate*3600,cu_budget=budget,cu_rate=rate,cu_remaining_at_start=float(settings['cu_remaining']))
    write_state(state)
    if not allow_expired:
        deadline_remaining()
    return state


def schedule_budget_shutdown(settings, *, previous_timer=None):
    """Call from the live notebook kernel. Timer does no model work.

    Processing stops at 85% of the time-derived budget. At 100%, request VM
    unassignment even if the user has not confirmed a download. Unsaved files
    may be lost. No exact billing/CU guarantee is possible with a typed rate.
    """
    # A processing deadline at 85% must not prevent arming cleanup at 100%.
    state=initialize_budget(settings,allow_expired=True)
    def finish_budget():
        print('時間換算のCU上限に達したためGPU終了を要求します。一時ファイルは消えます。',flush=True)
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            print('自動終了の確認が必要です。Colabで接続解除・削除、使用率0・稼働0件を確認してください。',flush=True)
    timer=threading.Timer(max(0,state['shutdown_deadline_epoch']-time.time()),finish_budget)
    timer.daemon=True
    timer.start()
    # Invalid inputs, changed budgets and Timer.start failures leave the old
    # cleanup timer in place. Cancel it only after its replacement is running.
    if previous_timer is not None:
        previous_timer.cancel()
    return timer


@cloud_operation
def setup(settings):
    require_cloud()
    settings = validate_inputs(settings)
    state = read_state(); state['setup_complete'] = False; write_state(state)
    resources = inspect_resources()
    state = initialize_budget(settings)
    state.update(resources=resources, license_accepted=True, eligible_use_confirmed=True)
    write_state(state)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SGLANG_BUILD_RUST_EXTS='none', UV_CACHE_DIR=str(SCRATCH/'uv-cache'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1')
    requirements = KIT/'requirements-colab.txt'
    digest = dependency_digest()
    if not (state.get('requirements_digest') == digest and (VENV/'bin/python').exists()):
        run_waiting([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', f'uv=={UV_VERSION}'], 'install_uv: uvを準備', env=env)
        if not (VENV/'bin/python').exists():
            run_waiting(['uv', 'venv', '--python', PYTHON_VERSION, str(VENV)], 'create_venv: Python3.12を準備', env=env)
        run_waiting(['uv', 'pip', 'install', '--python', str(VENV/'bin/python'), '--prerelease=allow', '--build-constraint', str(KIT/'build-constraints.txt'), '-r', str(requirements)], 'install_sglang: 固定依存を導入', env=env)
        run_waiting([str(VENV/'bin/python'), '-c', "import sys,torch,importlib.metadata as m; assert sys.version_info[:2]==(3,12); assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()==(8,0); assert m.version('sglang')=='0.5.6.post3.dev12129+g448759b23'; assert m.version('torch')=='2.14.1'; print('DEPENDENCIES_OK',torch.__version__,torch.version.cuda)"], 'check_dependencies: GPUと依存を確認', env=env)
        state = read_state(); state['requirements_digest'] = digest; write_state(state)
    download_code = f'''import json,time,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download
start=time.monotonic()
cache={str(SCRATCH/'hf')!r}
model=snapshot_download('MiniMaxAI/MiniMax-H3',revision={BASE_REV!r},cache_dir=cache,allow_patterns=['FL2VA/*','model_index.json','modular_model_index.json','LICENSE','.gitattributes'],ignore_patterns=['*.md','*.png','*.jpg','*.mp4'],max_workers=8,token=False)
lora=hf_hub_download('lightx2v/Minimax-h3-Turbo',{LORA_FILE!r},revision={LORA_REV!r},cache_dir=cache,token=False)
with open(lora,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest=={LORA_SHA!r},'LoRA SHA256 mismatch'
Path({str(ROOT/'download_report.json')!r}).write_text(json.dumps({{'model_path':model,'lora_path':lora,'lora_sha256':digest,'download_seconds':round(time.monotonic()-start,3)}},indent=2))
print('DOWNLOAD_OK',flush=True)
'''
    (ROOT/'download.py').write_text(download_code)
    run_waiting([str(VENV/'bin/python'), str(ROOT/'download.py')], 'download_model: クラウドへモデルを取得', env=env)
    state=read_state(); state.update(setup_complete=True, download_complete=True, model_revision=BASE_REV, sglang_revision=SGLANG_REV, lora_revision=LORA_REV); write_state(state)
    print('SETUP_OK: 設定とモデル取得が完了しました。次の生成セルへ進めます。', flush=True)
    return state


@cloud_operation
def generate(settings, new_run=False):
    require_cloud()
    settings=validate_inputs(settings)
    state=read_state()
    if not state.get('setup_complete'):
        raise WorkflowError('セットアップが未完了です。生成は開始しません。')
    key=fingerprint(settings)
    old=state.get('last_run')
    if old and not new_run:
        if old['fingerprint']==key and old.get('status')=='completed':
            p=Path(old['outdir']); report=json.loads((p/'sequence_report.json').read_text())
            if all((p/r['output']).is_file() for r in report['requests']):
                print('GENERATE_OK: この入力は完了済みです。追加生成せず既存ファイルを使います。'); return p
        raise WorkflowError('前回の生成記録があります。保存を確認してから「新しい生成」をオンにしてください。部分成功は自動再実行しません。')
    if state.get('requirements_digest') != dependency_digest() or any(state.get(k)!=v for k,v in [('model_revision',BASE_REV),('sglang_revision',SGLANG_REV),('lora_revision',LORA_REV)]):
        raise WorkflowError('固定依存・モデルのセットアップ記録が一致しません。先にセットアップを確認してください。')
    download=ROOT/'download_report.json'
    if not download.is_file() or json.loads(download.read_text()).get('lora_sha256')!=LORA_SHA:
        raise WorkflowError('検証済みTurbo取得記録がありません。先にセットアップを確認してください。')
    deadline_remaining()
    outdir=ROOT/'outputs'/('run_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]);outdir.mkdir(parents=True)
    inputs={'prompts':settings['prompts'],'duration_seconds':float(settings['seconds']),'seeds':[settings['seed']+i for i in range(settings['count'])]}
    if settings.get('mode','t2v')=='i2v':
        inputs.update(mode='i2v',first_frame=settings['first_frame'])
    (outdir/'inputs.json').write_text(json.dumps(inputs,indent=2))
    state['last_run']={'fingerprint':key,'outdir':str(outdir),'status':'running'}
    state.update(native_parent_pid=os.getpid(),cloud_boot_id=boot_id());write_state(state)
    env=dict(os.environ,PATH=str(VENV/'bin')+os.pathsep+os.environ['PATH'],HF_HUB_CACHE=str(SCRATCH/'hf'),HF_HUB_OFFLINE='0',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false',H3_OUTDIR=str(outdir),H3_DOWNLOAD_REPORT=str(ROOT/'download_report.json'))
    try:
        run_waiting([str(VENV/'bin/python'), str(KIT/'scripts/generate_native.py')], 'generation: モデル1回読込→連続生成', env=env)
        report=json.loads((outdir/'sequence_report.json').read_text())
        if report['status']!='completed' or report['model_load_count']!=1 or len(report['requests'])!=settings['count']:
            raise WorkflowError('生成レポートが成功条件を満たしていません。')
        for r in report['requests']:
            video=outdir/r['output']
            with (outdir/f"media_info_{r['index']:02d}.json").open('w') as media:
                subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],stdout=media,check=True,timeout=min(30,deadline_remaining()))
            subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,timeout=min(30,deadline_remaining()))
        state=read_state();state['last_run']['status']='completed';write_state(state)
        print('GENERATE_OK:',outdir,flush=True);return outdir
    except BaseException:
        state=read_state();state['last_run']['status']='failed_or_interrupted';write_state(state);raise
    finally:
        state=read_state();state.pop('native_parent_pid',None);state.pop('cloud_boot_id',None);write_state(state)


def package_outputs():
    require_cloud()
    state=read_state();last=state.get('last_run')
    if not last:
        raise WorkflowError('保存する生成記録がありません。')
    outdir=Path(last['outdir']);files=list(outdir.glob('*.mp4'))
    if not files:
        raise WorkflowError('完成したMP4がまだありません。')
    archive=ROOT/(outdir.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in outdir.iterdir():
            if p.is_file():z.write(p,p.name)
        if (ROOT/'generation.log').exists():z.write(ROOT/'generation.log','generation.log')
        z.writestr('AI-GENERATED.txt','Powered by MiniMax H3. These videos are AI-generated. Review before sharing.\n')
    print('SAVE_READY:',archive,flush=True);return archive


def shutdown(saved_confirmed=False):
    require_cloud()
    if not saved_confirmed:
        raise WorkflowError('ZIP/MP4を自分の端末へ保存したことを確認してください。確認前には一時ファイルを削除しません。')
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception as exc:
        raise WorkflowError('自動終了を確認できません。Colabの「ランタイム→ランタイムを接続解除して削除」を実行し、使用率0・稼働0件を確認してください。') from exc


def main():
    parser=argparse.ArgumentParser(description='ColabクラウドGPU専用。Macではdiagnose以外を実行しないでください。')
    parser.add_argument('command',choices=['diagnose','setup','generate','package','status','shutdown'])
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--new-run',action='store_true')
    parser.add_argument('--saved-confirmed',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='diagnose': print(json.dumps(inspect_resources(),ensure_ascii=False,indent=2))
        elif args.command=='status': require_cloud();print(json.dumps(read_state(),ensure_ascii=False,indent=2))
        elif args.command=='package':package_outputs()
        elif args.command=='shutdown':shutdown(args.saved_confirmed)
        else:
            if not args.settings:raise WorkflowError('--settingsで入力JSONを指定してください。')
            settings=json.loads(args.settings.read_text())
            if args.command=='setup':setup(settings)
            else:generate(settings,args.new_run)
    except (WorkflowError,subprocess.SubprocessError,OSError,json.JSONDecodeError) as exc:
        print('停止:',exc,file=sys.stderr)
        print('GPUはエラーでも割当が続くことがあります。Colabで切断・削除を確認してください。',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
