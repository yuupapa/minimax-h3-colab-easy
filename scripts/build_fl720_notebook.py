"""Build a local T2V/I2V-core-tested, UI-unverified 720p candidate using stdlib."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ['scripts/colab_workflow.py', 'scripts/fl720_workflow.py', 'scripts/fl720_native.py', 'scripts/serial_queue.py',
         'requirements-r2v.lock', 'build-constraints.txt', 'vendor/modeltc/manifest.json',
         'vendor/modeltc/LICENSE', 'vendor/modeltc/inference_minimax_h3.py',
         'vendor/modeltc/minimax_h3_ref2va_pipeline.py', 'vendor/modeltc/resolution_util.py']

INTRO = '''# MiniMax H3 T2V / I2V・1280×720候補

**T2V/I2Vは各1本・別クリーンVMのcore生成・端末保存・全デコード・GPU削除を確認済みです。既定UI・同VM切替・期限満了による終了はGPU未検証の試験版です。GPU使用は今回の目的/本数/CU上限の範囲です。連続制御の更新はGPUなしのオフライン検証です。** 既存Easy 544p・R2V720は維持しています。公開Easy/R2Vとは別のColab VM・venv・キャッシュ・状態を使い、1VMの共有予算で、承認した各モードを指定5秒・明示した本数だけ順に生成します。依存導入・モデル/LoRA取得は1回。入力を変えて次のキューへ進めます。無制限実行はせず、VIDEO_COUNTで指定した1〜100本だけ生成します。生成時のモデル読込は各モードで別々です。

T2Vは文章だけ、I2Vは始点画像1枚と動きの文章です。意味的参照を別場面へ持ち込むR2Vとは異なります。画像アップロード後は内容・向きの確認で必ず一度停止します。フォームの値が反映されない場合があるため、「この文章で生成します」を実際に読んでから生成します。

**1280×736で生成→上下各8pxを中心クロップ→1280×720を保存。** 544p拡大や1280×704の誤表記は使いません。約2.17%の縦画角と再圧縮分を失います。I2Vの入力は生成画角への調整と納品クロップがあり、重要部分を端に置かず、初回は16:9を推奨します。人物・商品・文字・足等の完全維持は保証しません。

FL2VA Turbo **4-step v1.0 768p BF16**（学習1344×768、video/audio shift6/3、alpha8、scale1、fuseなし）を固定。Ref用544p LoRAは使いません。実際のTransformer評価4回を検査します。生成画角は学習寸法そのものではないため、適合・速度・品質・VRAMを実機で検証する必要があります。

R2V実機使用の67依存lock、H3・Diffusers・ModelTC固定ソースを再利用し、依存を再解決しません。これはFL720での成功証拠ではありません。A10080GB（空きVRAM80,500MiB以上）、driver580以上、RAM約167GiB（空き140以上）、初期キャッシュ180GiB＋環境25GiB、共有ディスクなら205GiB以上を確認します。R2Vピーク76.35/78.42GiB（確保/予約）は別経路の値で、この経路の上限ではありません。重みはクラウドだけへ取得。FA3・Driveマウント・トンネル・トークン要求・追加購入は使いません。

**同VMで2モード合計5CUは未完了の計画値です。実際のT2V/I2V試験は別クリーンVMでした。** 新しい試験は目的/本数/上限を本人が判断し、実測費用や完了保証として扱いません。 残量と実際のCU/hを確認し、本人承認なしに接続しません。6.77CU/hなら合計5CUは約44.31分、処理期限は85%の約37.67分。モード切替・保存・待機もこの共通期限へ含め、モードごとに枠を作り直しません。接続から入力までの経過分を差し引きます。再入力で締切を延長せず、失敗後の自動再試行・解像度低下・別LoRAへの変更はしません。タイマーは請求保証ではなく、満了時の切断はGPU未検証で一時ファイルが消えます。

作業を終える前・CU期限に近づいた時は完成分を保存し、必ず **「ランタイム → ランタイムを接続解除して削除」** を実行し、ライブ画面でCU/h0・稼働0を確認。終了補助は追加せず、ブラウザを閉じる・子プロセス終了・成功表示をGPU停止の証拠にしません。

ModelTC仕様: https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md
LoRA: https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/3ec17a324ced54151364f24f8b5fb6bf7e26414f/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
LoRA SHA256: 1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c （T2V試験のColab取得時に実物照合済み。Macでは取得しません）

ModelTCとLoRAはApache-2.0、H3 Community License・AUPは継続。固定ソース実行、H3の条件、画像送信の判断を本人が行い、新規同意は初期オフです。

VIDEO_COUNTで有限本数、PROMPTS_JSONで文章配列（空ならPROMPTを本数分）、seedは1本ごとに+1。違う入力は新しいキュー。同じ入力の生成セル再実行は既存結果を返します。同じ入力で追加したい場合はNEW_RUN、停止/失敗した同じキューの未完成分だけを再開する場合はRESUME_FAILED_QUEUEをオンにし、入力セルだけ再実行します。完成分と過去の出力は消しません。CU上限は残量以内で指定し、入力やモード変更で期限を延長しません。GPU上での連続制御は未検証です。

T2V/I2V実測: conditioning/denoise/decode215.306/182.119秒、ロード9.679/10.003秒、native encode9.786/8.714秒、GPU確保70.89/76.52GiB・予約73.57/78.62GiB、RSS141.48/77.09GiB。1280×736からクロップした720p、124frames/24fps、5.175秒。猫の動きは見える一方、背景/毛並みににじみがあり、I2Vでは始点の猫/ポーチと歩行・停止・顔向けを確認しましたが、尾の上端切れや細部変化があります。音の聴感は未評価。既定フォームのままのGPU試験ではありません。詳細はdocs/FL720_CANDIDATE.md。

GPU接続前に入力値をソースへ設定して確認し、UIのfillだけに依存しません。切断中のファイルパネルは自動GPU接続を開始する場合があるため開かないでください。クラウド依存/モデルの導入はGPU/RAM/ディスク検査後のsetupが必要で、Macや未接続画面では完了しません。

<details><summary>本人が確認するH3 Community License全文・AUP</summary>

'''

INPUT = '''#@title 1. 入力・本数・CU上限（設定はGPU接続前に準備）
GPU_TRIAL_APPROVED = False #@param {type:"boolean"}
T2V_TRIAL_APPROVED = False #@param {type:"boolean"}
I2V_TRIAL_APPROVED = False #@param {type:"boolean"}
MODE_SWITCH_CONFIRMED = False #@param {type:"boolean"}
SOURCE_EXECUTION_ACCEPTED = False #@param {type:"boolean"}
LICENSE_ACCEPTED = False #@param {type:"boolean"}
ELIGIBLE_USE_CONFIRMED = False #@param {type:"boolean"}
HIGHER_RESOLUTION_RISK_ACCEPTED = False #@param {type:"boolean"}
GENERATION_MODE = "Text-to-Video" #@param ["Text-to-Video","Image-to-Video"]
IMAGE_USE_CONFIRMED = False #@param {type:"boolean"}
IMAGE_CONTENT_CONFIRMED = False #@param {type:"boolean"}
UPLOAD_FIRST_FRAME = False #@param {type:"boolean"}
PROMPT = "A small red sailboat slowly glides across a calm lake. A steady realistic shot, gentle ripples and soft wind. No people, speech, music, text or logos." #@param {type:"string"}
VIDEO_COUNT = 1 #@param {type:"integer"}
PROMPTS_JSON = "" #@param {type:"string"}
NEW_RUN = False #@param {type:"boolean"}
RESUME_FAILED_QUEUE = False #@param {type:"boolean"}
SEED = 42 #@param {type:"integer"}
CU_REMAINING = 0.0 #@param {type:"number"}
CU_PER_HOUR = 0.0 #@param {type:"number"}
CU_BUDGET = 5.0 #@param {type:"number"}
GPU_CONNECTED_MINUTES_AGO = 0.0 #@param {type:"number"}

settings=None
archive_path=None
import importlib.util, platform, sys
from pathlib import Path
try:
    cloud=importlib.util.find_spec('google.colab') is not None
except (ModuleNotFoundError,ValueError):cloud=False
if not (cloud and platform.system()=='Linux' and Path('/content').is_dir()):
    raise RuntimeError('Colabクラウド専用。Macへ導入・ダウンロード・生成しません。')
if not GPU_TRIAL_APPROVED or not SOURCE_EXECUTION_ACCEPTED:
    raise RuntimeError('FL720試験の目的・1本・CU上限と固定ソース実行の本人承認が必要です。')
if GENERATION_MODE not in ('Text-to-Video','Image-to-Video'):
    raise RuntimeError('R2Vは別ノートブックです。T2V/I2Vを選んでください。')
'''

BOOT = '''kit=Path('/content/h3-fl720-dev/kit')
for name,text in EMBEDDED_KIT.items():
    dest=kit/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
sys.path.insert(0,str(kit/'scripts'))
import fl720_workflow as fl720
import json
import importlib
fl720=importlib.reload(fl720)
candidate={'gpu_trial_approved':GPU_TRIAL_APPROVED,'source_execution_accepted':SOURCE_EXECUTION_ACCEPTED,
 'license_accepted':LICENSE_ACCEPTED,'eligible_use_confirmed':ELIGIBLE_USE_CONFIRMED,
 'higher_resolution_risk_accepted':HIGHER_RESOLUTION_RISK_ACCEPTED,
 'approved_modes':[mode for mode,ok in (('t2v',T2V_TRIAL_APPROVED),('i2v',I2V_TRIAL_APPROVED)) if ok],
 'mode':'t2v' if GENERATION_MODE=='Text-to-Video' else 'i2v',
 'image_use_confirmed':IMAGE_USE_CONFIRMED,'image_content_confirmed':IMAGE_CONTENT_CONFIRMED,
 'profile':'native736_crop720','prompts':json.loads(PROMPTS_JSON) if PROMPTS_JSON.strip() else [PROMPT],'count':VIDEO_COUNT,'seconds':5,'seed':SEED,
 'new_run':NEW_RUN,'resume_failed_queue':RESUME_FAILED_QUEUE,
 'cu_remaining':CU_REMAINING,'cu_rate':CU_PER_HOUR,'cu_budget':CU_BUDGET,
 'gpu_minutes_before_input':GPU_CONNECTED_MINUTES_AGO}
fl720.validate(candidate,with_image=False)
_fl720_timer=fl720.arm_budget(candidate,previous_timer=globals().get('_fl720_timer'))
_changed=fl720.activate_mode(candidate)
if _changed:
    MODE_SWITCH_CONFIRMED=False
    IMAGE_CONTENT_CONFIRMED=False
    UPLOAD_FIRST_FRAME=False
    _fl720_save_gate_target=None
    raise RuntimeError('モード切替で旧画像・出力選択を消去しました。生成セルに残るプレビューは前モードの旧出力です。表示モード/文章を確認してMODE_SWITCH_CONFIRMEDをオン、入力セルだけ再実行してください。')
if not fl720.guard.read_state().get('mode_reviewed'):
    if not MODE_SWITCH_CONFIRMED:raise RuntimeError('切替後のモード/文章を確認してください。')
    fl720.confirm_mode(candidate)
_upload=UPLOAD_FIRST_FRAME;UPLOAD_FIRST_FRAME=False
if candidate['mode']=='i2v':
    if _upload:
        IMAGE_CONTENT_CONFIRMED=False
        _first=fl720.upload_first_frame(candidate)
    else:
        _first=fl720.guard.read_state().get('last_input_image')
    if not _first:raise RuntimeError('UPLOAD_FIRST_FRAMEをオンにし、本人が始点画像1枚を選択してください。')
    fl720.guard.verify_first_frame(_first)
    from IPython.display import Image,display
    display(Image(filename=_first['path'],width=480))
    print('I2V始点画像です。意味的参照のR2Vとは異なります。内容・向きと、720p納品時のクロップを確認してください。')
    if _upload or not IMAGE_CONTENT_CONFIRMED:
        IMAGE_CONTENT_CONFIRMED=False
        raise RuntimeError('画像を確認し、IMAGE_CONTENT_CONFIRMEDをオン、UPLOAD_FIRST_FRAMEをオフにして入力セルだけ再実行。長く待つならGPU削除。')
    candidate['first_frame']=_first
elif _upload:
    raise RuntimeError('T2Vでは画像をアップロードしません。Image-to-Videoを明示的に選んでください。')
settings=fl720.select_request(candidate)
NEW_RUN=False
RESUME_FAILED_QUEUE=False
print('生成キュー:',settings['count'],'本、開始seed:',settings['seed'],'文章:',settings['prompts'])
print('INPUT_OK:',settings['mode'],'生成1280×736 / 納品1280×720。セットアップ・生成はまだです。')
'''

SETUP = '''#@title 2. 独立環境・FL API検査・固定重み取得
if settings is None:raise RuntimeError('INPUT_OKが必要です。')
fl720.setup(settings)
'''

GENERATE = '''#@title 3. 指定本数を順番に生成→720p・全デコード
if settings is None:raise RuntimeError('INPUT_OKが必要です。')
archive_path=fl720.generate(settings)
from IPython.display import Video,display
current=fl720.current_output(settings)
print('表示対象:',current['mode'],current['run_id'],current['completed_count'],'/',current['requested_count'],'本（最後の完成動画）')
display(Video(str(Path(current['output_dir'])/'delivery_720p.mp4'),embed=True))
print('文章・I2V始点画像の反映、クロップ、動き・破綻・音を本人が確認。保存はまだです。')
'''

SAVE = '''#@title 4. 現在のキューの完成分ZIP保存（停止・失敗時も使用可）
SAVED_CURRENT_RUN = False #@param {type:"boolean"}
if settings is None:raise RuntimeError('現在のINPUT_OKが必要です。旧出力は保存対象にしません。')
current=fl720.current_output(settings)
_target=(current['mode'],current['run_id'],current['archive_path'])
print('保存確認対象:',_target[0],_target[1])
if globals().get('_fl720_save_gate_target')!=_target:
    _fl720_save_gate_target=_target
    SAVED_CURRENT_RUN=False
    from google.colab import files
    files.download(current['archive_path'])
    print('端末でこのmode/run IDのZIP/MP4を開いて確認し、SAVED_CURRENT_RUNをオンにしてこのセルだけ再実行。以前の確認は使いません。')
elif not SAVED_CURRENT_RUN:
    print('SAVE_WAIT: 実際の保存が終わるまで次のモードを開始しません。')
else:
    fl720.confirm_saved(settings,run_id=_target[1],archive_path=_target[2])
    print('SAVED_CONFIRMED:',_target[0],_target[1],'保存確認済み。次の入力でもGPUは割当中で、共有CUを消費します。')
'''


STOP = '''#@title 5. 全試験保存後/失敗時: メニューでGPU削除・ライブ使用率0確認
print('承認済みモードがすべて保存できたら、または失敗・中断時は、ランタイム → ランタイムを接続解除して削除を実行してください。')
print('リソース画面でCU/h 0・稼働0件を確認。ブラウザを閉じる、子プロセス終了、成功文だけでは停止確認になりません。')
print('このセルは終了補助を実行しません。タイマー満了による切断もGPU未検証です。一時ファイルはVM削除で消えます。')
'''


def build():
    embedded = {name: (ROOT / name).read_text() for name in FILES}
    intro = INTRO + (ROOT / 'licenses/MINIMAX-H3-COMMUNITY.txt').read_text() + '\n</details>\n'
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': intro}]
    for code in (INPUT + '\nEMBEDDED_KIT=' + repr(embedded) + '\n' + BOOT, SETUP, GENERATE, SAVE, STOP):
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code})
    name = 'MiniMax_H3_T2V_I2V_720p_DEV_Colab.ipynb'
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'colab': {'name': name}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'cells': cells}
    out = ROOT / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n')
    return out


if __name__ == '__main__':
    print(build())
