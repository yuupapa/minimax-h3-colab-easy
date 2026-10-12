"""Build the isolated R2V notebook using standard library only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
FILES=['scripts/colab_workflow.py','scripts/r2v_workflow.py','scripts/r2v_native.py',
       'requirements-r2v.in','requirements-r2v.lock','build-constraints.txt','vendor/modeltc/manifest.json',
       'vendor/modeltc/LICENSE','vendor/modeltc/inference_minimax_h3.py',
       'vendor/modeltc/minimax_h3_ref2va_pipeline.py','vendor/modeltc/resolution_util.py']


def build():
    embedded={name:(ROOT/name).read_text() for name in FILES}
    intro='''# MiniMax H3 Reference-to-Video / exact720p

**R2V方式はA10080GBで1本実測。T2V/I2Vとは別のColab VM・Python環境を使用します。** この統合版の依存同期・経過時間補正はオフライン検査のみ。新しい終了補助セルとタイマー満了はGPU未検証です。保存後は必ず「ランタイム → ランタイムを接続解除して削除」を実行し、使用率0・稼働0を画面で確認してください。補助セルの成功表示を停止確認の代わりにしません。

参照画像1枚を人物・物体の意味的参照として渡すRef2VAです。始点画像I2Vとは別です。1本・5秒指定（124フレーム、24fps、約5.17秒）だけを試します。テンプレート文は自分の被写体に合わせて変更します。

`near720_crop`: 1280×736生成→上下8pxずつ中心クロップ→**1280×720納品**。縦の約2.17%を失い、再エンコードします。参照LoRA v0.1は544p用で、1280×736は非個人の静止画1枚で実機成功しましたが、別画像・別環境の成功は保証しません。

`544_upscale`: 960×544生成→960×540中心クロップ→1280×720拡大。こちらもR2V GPU未検証。拡大はネイティブの細部を増やしません。自動切替はしません。

事前にModelTC固定ソース、H3とLoRAの利用条件、画像の権利・人物の同意・Colab送信を本人が確認。外部コードの実行確認やColab警告は勝手に承認しません。試験承認フラグはGPU/CU使用の新しい明示承認があるときだけオンにします。

A10080GB、RAM約167GiB（空き140以上）、キャッシュ180GiB＋/content25GiB、共有ディスクなら205GiB、ドライバ580以上。モデル約134.13GiB＋参照LoRA約1.29GiBを**クラウドにだけ**取得。別VMと別venvでSGLang版を変更しません。FlashAttention3は使用しません。これらは必要条件の候補で、成功保証ではありません。

上限5CU、入力したCU/hから時間換算し85%で処理停止、100%でGPU終了要求。CUの請求保証ではなく、画面残量・使用率を定期確認してください。初期CUは0なので進みません。GPU接続から入力までの経過分をGPU_CONNECTED_MINUTES_AGOへ入れ、停止枠から差し引きます。本人確認とソース実行承認はGPU割当前に済ませてください。取得が終わらなくても上限を延ばさず停止します。自動終了は未保存ファイルを失います。

実測は参照処理・推論・復号合計180.00秒、読込9.76秒、原寸MP4書出し9.05秒、24fps・124フレーム・コンテナ5.175秒。茶トラの毛色・縞を引き継いで背景を変更できましたが、模様・体格・足の形に変化があり、完全一致は保証しません。音声は全デコード済み、聴感評価は未実施。実測と統合差分はdocs/R2V_VALIDATION.mdを参照。

67依存の実機使用lockを同梱し、固定版の同期→API検査→重み取得の順です。セットアップ・生成・保存・GPU終了は別々に確認します。Driveマウント、トンネル、認証トークン、追加購入、40本一括生成は使いません。

ModelTC: https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225 （Apache-2.0、原文LICENSEとsha256を同梱）
Diffusers: https://github.com/huggingface/diffusers/tree/85c9fa4caee33d1c5b473395784471d2d7957c1b （Apache-2.0）
参照LoRA: https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/3ec17a324ced54151364f24f8b5fb6bf7e26414f （配布メタデータApache-2.0、H3の条件も継続）

<details><summary>本人が確認するH3 Community License全文・AUP</summary>

'''+(ROOT/'licenses/MINIMAX-H3-COMMUNITY.txt').read_text()+'\n</details>\n'
    input_cell='''#@title 1. 追加試験の承認・参照・720p出力設定
GPU_TRIAL_APPROVED = False #@param {type:"boolean"}
SOURCE_EXECUTION_ACCEPTED = False #@param {type:"boolean"}
LICENSE_ACCEPTED = False #@param {type:"boolean"}
ELIGIBLE_USE_CONFIRMED = False #@param {type:"boolean"}
IMAGE_USE_CONFIRMED = False #@param {type:"boolean"}
IMAGE_CONTENT_CONFIRMED = False #@param {type:"boolean"}
HIGHER_RESOLUTION_RISK_ACCEPTED = False #@param {type:"boolean"}
PROFILE = "near720_crop" #@param ["near720_crop","544_upscale"]
PROMPT = "Use the object in <Picture 1> as the subject, keeping its shape and colors. Place it on a sunny sandy beach, a different environment from the reference photo. A gentle moving camera, realistic lighting, soft waves. No speech, music, text or logos." #@param {type:"string"}
SEED = 42 #@param {type:"integer"}
CU_REMAINING = 0.0 #@param {type:"number"}
CU_PER_HOUR = 0.0 #@param {type:"number"}
CU_BUDGET = 5.0 #@param {type:"number"}
GPU_CONNECTED_MINUTES_AGO = 0.0 #@param {type:"number"}
UPLOAD_REFERENCE = False #@param {type:"boolean"}

settings=None
import importlib.util,platform,sys
from pathlib import Path
try:
    cloud=importlib.util.find_spec('google.colab') is not None
except (ModuleNotFoundError,ValueError):cloud=False
if not (cloud and platform.system()=='Linux' and Path('/content').is_dir()):
    raise RuntimeError('Colabクラウド専用。Macへ導入・ダウンロード・生成しません。')
if not GPU_TRIAL_APPROVED or not SOURCE_EXECUTION_ACCEPTED:
    raise RuntimeError('先に新しいGPU試験/CU予算と固定ソース実行の本人承認が必要です。未承認では進めません。')
'''
    input_cell+='\nEMBEDDED_KIT='+repr(embedded)+'\n'
    input_cell+='''kit=Path('/content/h3-r2v-dev/kit')
for name,text in EMBEDDED_KIT.items():
    dest=kit/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
sys.path.insert(0,str(kit/'scripts'))
import r2v_workflow as r2v
import importlib
r2v=importlib.reload(r2v)
candidate={'gpu_trial_approved':GPU_TRIAL_APPROVED,'source_execution_accepted':SOURCE_EXECUTION_ACCEPTED,
 'license_accepted':LICENSE_ACCEPTED,'eligible_use_confirmed':ELIGIBLE_USE_CONFIRMED,
 'image_use_confirmed':IMAGE_USE_CONFIRMED,'image_content_confirmed':IMAGE_CONTENT_CONFIRMED,
 'higher_resolution_risk_accepted':HIGHER_RESOLUTION_RISK_ACCEPTED,'profile':PROFILE,
 'prompts':[PROMPT],'count':1,'seconds':5,'seed':SEED,
 'cu_remaining':CU_REMAINING,'cu_rate':CU_PER_HOUR,'cu_budget':CU_BUDGET,
 'gpu_minutes_before_input':GPU_CONNECTED_MINUTES_AGO}
r2v.validate(candidate,with_reference=False)
_r2v_timer=r2v.arm_budget(candidate,previous_timer=globals().get('_r2v_timer'))
_upload=UPLOAD_REFERENCE;UPLOAD_REFERENCE=False
if _upload:
    IMAGE_CONTENT_CONFIRMED=False
    _ref=r2v.upload_reference(candidate)
else:
    _ref=r2v.guard.read_state().get('last_input_image')
if not _ref:raise RuntimeError('UPLOAD_REFERENCEをオンにし、本人が静止画像1枚を選択してください。')
r2v.guard.verify_first_frame(_ref)
from IPython.display import Image,display
display(Image(filename=_ref['path'],width=480))
print('意味的参照です。始点画像として固定しません。内容と向きを確認してください。')
if _upload or not IMAGE_CONTENT_CONFIRMED:
    IMAGE_CONTENT_CONFIRMED=False
    raise RuntimeError('参照画像を確認し、IMAGE_CONTENT_CONFIRMEDをオン、UPLOAD_REFERENCEをオフにして入力セルだけ再実行。長く待つならGPU終了。')
candidate['reference']=_ref
settings=r2v.validate(candidate)
print('この文章で生成します:',settings['prompts'][0])
print('INPUT_OK: SETUP/GENERATIONは未実行。文章がフォームと違う場合は生成せず、入力を確定して再実行してください。')
'''
    setup='''#@title 2. 独立環境・API事前検査・参照重み取得
if settings is None:raise RuntimeError('INPUT_OKが必要です。')
r2v.setup(settings)
'''
    generate='''#@title 3. 試作1本→正確な720pへ変換・全体デコード
if settings is None:raise RuntimeError('INPUT_OKが必要です。')
archive_path=r2v.generate(settings)
from IPython.display import Video,display
display(Video(str(Path(r2v.guard.read_state()['output_dir'])/'delivery_720p.mp4'),embed=True))
print('被写体一致、別背景、動き、破綻と音を本人が確認。ZIPはまだ未保存。')
'''
    save='''#@title 4. ZIPを端末へ保存（生成映像2版＋測定、参照画像なし）
from google.colab import files
state=r2v.guard.read_state()
if not state.get('generation_complete'):raise RuntimeError('GENERATE_OKが必要です。')
files.download(state['archive_path'])
print('ダウンロードの開始だけでは保存確認になりません。端末でZIP/MP4を開いて確認。')
'''
    stop='''#@title 5. 必須: ColabメニューでGPU削除・使用率0確認
print('保存後は必ず「ランタイム → ランタイムを接続解除して削除」を実行し、使用率0・稼働0を画面で確認してください。')
print('以下の終了補助はGPU未検証です。既定は無効。補助が成功してもメニューと画面での終了確認は必須です。')
TRY_UNVERIFIED_SHUTDOWN_HELPER = False #@param {type:"boolean"}
SAVED_TO_MY_DEVICE = False #@param {type:"boolean"}
STOP_AFTER_ERROR = False #@param {type:"boolean"}
if not TRY_UNVERIFIED_SHUTDOWN_HELPER:
    print('終了補助は実行しません。Colabメニューから削除してください。ブラウザを閉じるだけでは停止確認になりません。')
elif 'r2v' in globals():
    _state=r2v.guard.read_state()
    _target=_state.get('archive_path')
    _first_successful_stop=_state.get('generation_complete') and globals().get('_r2v_stop_reviewed_target')!=_target
    if _first_successful_stop:
        _r2v_stop_reviewed_target=_target
        SAVED_TO_MY_DEVICE=False
        STOP_AFTER_ERROR=False
        TRY_UNVERIFIED_SHUTDOWN_HELPER=False
        print('初回は保存完了確認で待機。端末でZIP/MP4を開いてからSAVEDをオンにし、このセルだけ再実行。前回の確認は使いません。')
    elif not SAVED_TO_MY_DEVICE and not STOP_AFTER_ERROR:
        print('STOP_WAIT: 保存完了後にSAVED、失敗・中止で一時ファイル破棄に同意したらSTOP_AFTER_ERRORを確認してこのセルだけ再実行。')
    else:
        r2v.shutdown(saved=SAVED_TO_MY_DEVICE,stop_after_error=STOP_AFTER_ERROR)
        print('終了要求を送信。Colabリソース画面でCU/h 0・稼働GPU 0を確認。')
else:
    if not STOP_AFTER_ERROR:raise RuntimeError('セットアップ前の中止ではSTOP_AFTER_ERRORを確認してください。一時ファイルは消えます。')
    from google.colab import runtime
    runtime.unassign()
    print('終了要求を送信。Colabリソース画面でCU/h 0・稼働GPU 0を確認。')
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':intro}]
    for source in (input_cell,setup,generate,save,stop):
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source})
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'MiniMax_H3_R2V_720p_Colab.ipynb'},
        'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells}
    output=ROOT/'MiniMax_H3_R2V_720p_Colab.ipynb';output.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
    return output


if __name__=='__main__':print(build())
