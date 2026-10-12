"""Build a self-contained notebook without user data or execution outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def build():
    texts={name:(ROOT/name).read_text() for name in ['scripts/colab_workflow.py','scripts/generate_native.py','requirements-colab.txt','build-constraints.txt']}
    license_text=(ROOT/'licenses/MINIMAX-H3-COMMUNITY.txt').read_text()
    intro='''# MiniMax H3: AIと使うColabセットアップキット

**クラウドGPU専用。Macへモデルをインストールしません。** 公式H3 FL2VA + LightX2V Turbo v0.1 4-step。モデルを1回読み込み、1〜2本を順に生成します。公開H3 Baseを利用し、API専用フル2K機能は含みません。

**対応状況:** 開発版と配布版でText-to-VideoのGPU生成・保存・終了を実測。Image-to-Videoは始点画像1枚のアップロード・正規化・保存をCPUのColabで確認しました。始点画像1枚・5秒・動画1本のGPU生成・保存・GPU終了を実測しました。Reference-to-Videoは独立したMiniMax_H3_R2V_720p_Colab.ipynbで実行します。このT2V/I2Vノートブックでは実行しません。

I2VはフォームのGENERATION_MODEをImage-to-Videoへ変更し、権利・写った人物の同意・元画像（EXIF等を含む）のGoogle Colab送信を確認してIMAGE_USE_CONFIRMEDをオンにします。UPLOAD_FIRST_FRAMEをオンにして入力セルだけ実行し、画像を1枚選択。表示を確認してIMAGE_CONTENT_CONFIRMEDをオン、UPLOAD_FIRST_FRAMEをオフにして入力セルだけ再実行してください。元画像と正規化PNGはクラウドVMにだけ保管し、VM削除で消えます。画像は出力ZIPへ同梱せず、DriveやGitHubへ自動保存しません。PROMPTには動き・カメラ・音の指示を入れます。現在の出力は16:9固定なので横長16:9の画像から始めてください。新規アップロード直後は「すべてのセルを実行」でも画像確認で必ず停止します。

1. GoogleアカウントでColabにログインし、A100 **80GB**・ハイメモリを選択して接続。
2. Colabのリソース画面で残CU・1時間あたり使用率を確認し、次のフォームへ入力。ライセンスは本人が読み、同意・利用条件確認を行います。
3. 初回はフォームを入力してから「すべてのセルを実行」で順序実行できます。セットアップ・取得の完了を待って生成し、プレビューとZIP保存へ進みます。**保存した確認欄は初期のオフのまま**にし、保存完了後に終了セルだけを実行してください。
4. ZIPまたはMP4が自分の端末で開けることを確認してGPUを終了。エラー時もGPU終了を確認。

初回はモデル約134GiB＋依存を取得します。取得中・待機中もGPU接続にはCUが消費されます。既定1本・5秒・上限3CU。CU上限は入力した使用率から時間へ換算し15%の余裕を残す停止目安で、請求保証やリアルタイム計測ではありません。上限時間に達するとGPU終了を自動要求し、未保存ファイルが失われる場合があります。エージェントはColab画面で残量・使用率を定期確認してください。

この構成はA100 80GB・RAM約167GiB用です。GPU割当は保証されず、AI Proの200CCU特典にも条件があります。追加購入・自動課金の有効化は行いません。Driveマウント・トンネル・認証トークンは不要です。

H3 Community LicenseにはEU・英国・韓国・米国を除外する利用地域制限があります。利用地域、商用条件、Acceptable Use Policyを本人が確認してください。LICENSE_ACCEPTEDは本文への同意、ELIGIBLE_USE_CONFIRMEDは今回の利用条件を本人が確認したことを示します。ColabのVM所在地を確認・保証したという宣言ではありません。本キットは所在地や法的な利用資格を判定せず、本文の地域制限を変更しません。単なるチェックボックスは法的な利用資格を保証しません。[公式ライセンス](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)。

<details><summary>固定版H3ライセンス全文・Acceptable Use Policyを開く</summary>

'''+license_text+'\n</details>\n'
    bootstrap='''#@title 1. 文章・本数・秒数・CUと利用条件
# 利用者本人が規約・ライセンスを確認。AIは代わりにチェックしない。上限時間のGPU終了は未保存ファイルも削除します。
LICENSE_ACCEPTED = False #@param {type:"boolean"}
ELIGIBLE_USE_CONFIRMED = False #@param {type:"boolean"}
PROMPT = "A ginger cat slowly walks across a wooden porch in warm morning sunlight, then pauses and looks toward the camera. A steady realistic shot. Soft outdoor ambience. No speech, music, text or logos." #@param {type:"string"}
GENERATION_MODE = "Text-to-Video" #@param ["Text-to-Video","Image-to-Video","Reference-to-Video (別ノートブック)"]
IMAGE_USE_CONFIRMED = False #@param {type:"boolean"}
IMAGE_CONTENT_CONFIRMED = False #@param {type:"boolean"}
UPLOAD_FIRST_FRAME = False #@param {type:"boolean"}
FIRST_FRAME_PATH = "" #@param {type:"string"}
SECOND_PROMPT = "A small red sailboat slowly glides across a calm lake. A steady realistic shot, gentle ripples and soft wind. No people, speech, music, text or logos." #@param {type:"string"}
VIDEO_COUNT = 1 #@param [1,2] {type:"integer"}
DURATION_SECONDS = 5 #@param [5,6] {type:"integer"}
SEED = 42 #@param {type:"integer"}
CU_REMAINING = 0.0 #@param {type:"number"}
CU_PER_HOUR = 0.0 #@param {type:"number"}
CU_BUDGET = 3.0 #@param {type:"number"}
NEW_RUN = False #@param {type:"boolean"}

# このノートブックに埋め込まれたスクリプトをクラウドにだけ展開。
import importlib.util,platform,sys
from pathlib import Path
settings=None  # 入力・画像確認失敗時に以前のT2V設定で先へ進まない。
try:
    _is_colab=importlib.util.find_spec('google.colab') is not None
except (ModuleNotFoundError,ValueError):
    _is_colab=False
if not (_is_colab and platform.system()=='Linux' and Path('/content').is_dir()):
    raise RuntimeError('Google Colabクラウド専用です。MacやローカルPCではセットアップしません。')
if not LICENSE_ACCEPTED or not ELIGIBLE_USE_CONFIRMED:
    raise RuntimeError('H3ライセンス全文と利用地域・条件を本人が確認し、2つの欄をオンにしてください。')
if GENERATION_MODE not in ('Text-to-Video','Image-to-Video'):
    raise RuntimeError('Reference-to-Videoは別のMiniMax_H3_R2V_720p_Colab.ipynbを使用してください。このノートブックはT2V/I2V専用です。')
if GENERATION_MODE=='Image-to-Video' and (not IMAGE_USE_CONFIRMED or VIDEO_COUNT!=1):
    raise RuntimeError('I2V候補は画像1枚・動画1本です。画像の権利・人物の同意・Google Colabへの元画像送信（EXIF等を含む）を本人が確認してください。')
'''
    bootstrap+='\nEMBEDDED_KIT='+repr(texts)+'\n'
    bootstrap+='''kit=Path('/content/h3-easy/kit')
for relative,text in EMBEDDED_KIT.items():
    p=kit/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
sys.path.insert(0,str(kit/'scripts'))
import colab_workflow as h3
# フォーム変更後に再実行した場合も、更新した組込みコードを利用する。
import importlib
h3=importlib.reload(h3)
_base_settings=h3.validate_inputs({'license_accepted':LICENSE_ACCEPTED,'eligible_use_confirmed':ELIGIBLE_USE_CONFIRMED,'prompts':[PROMPT]+([SECOND_PROMPT] if VIDEO_COUNT==2 else []),'count':VIDEO_COUNT,'seconds':DURATION_SECONDS,'seed':SEED,'cu_remaining':CU_REMAINING,'cu_rate':CU_PER_HOUR,'cu_budget':CU_BUDGET})
_h3_budget_timer=h3.schedule_budget_shutdown(_base_settings,previous_timer=globals().get('_h3_budget_timer'))
h3.deadline_remaining()
if GENERATION_MODE=='Image-to-Video':
    # 明示したアップロード許可を1回消費。再クリックで再送信しない。
    _upload_requested=UPLOAD_FIRST_FRAME
    UPLOAD_FIRST_FRAME=False
    if _upload_requested:
        IMAGE_CONTENT_CONFIRMED=False
        FIRST_FRAME_PATH=''
        _first_frame=h3.upload_first_frame(image_use_confirmed=IMAGE_USE_CONFIRMED)
        FIRST_FRAME_PATH=_first_frame['path']
    else:
        # 以前確認したクラウド画像を使う。任意URL・Drive・別パスは拒否。
        import hashlib
        _cached_image=h3.read_state().get('last_input_image',{})
        _p=Path(FIRST_FRAME_PATH or _cached_image.get('path',''))
        if _p.parent!=h3.ROOT/'inputs' or _p.is_symlink() or not _p.is_file() or not 0<_p.stat().st_size<=40*1024**2:
            raise RuntimeError('最初はUPLOAD_FIRST_FRAMEをオンにして本人が画像を1枚選択してください。')
        _first_frame={'path':str(_p),'sha256':hashlib.sha256(_p.read_bytes()).hexdigest()}
        FIRST_FRAME_PATH=str(_p)
    from IPython.display import Image,display
    display(Image(filename=FIRST_FRAME_PATH,width=480))
    print('I2V始点画像: 内容と向きを確認してからセットアップへ。出力は16:9・544p固定のため別比率の画像はトリミングされ得ます。R2Vではありません。')
    if _upload_requested or not IMAGE_CONTENT_CONFIRMED:
        IMAGE_CONTENT_CONFIRMED=False
        raise RuntimeError('始点画像を確認してIMAGE_CONTENT_CONFIRMEDをオン、UPLOAD_FIRST_FRAMEをオフにし、入力セルだけ再実行してください。新規アップロード直後は必ずここで止まります。GPUの待機時間もCUを使います。長く待つならランタイムを終了してください。')
    _base_settings=dict(_base_settings,mode='i2v',first_frame=_first_frame,image_use_confirmed=IMAGE_USE_CONFIRMED,image_content_confirmed=True)
settings=h3.validate_inputs(_base_settings)
h3.deadline_remaining()
_h3_stop_gate_open=False
print('INPUT_OK: 次にセットアップを実行します。本数',VIDEO_COUNT,'秒数',DURATION_SECONDS)
print('入力使用率から算出した上限時間でGPU終了を要求します。未保存の一時ファイルは失われる場合があります。')
'''
    setup='''#@title 2. セットアップ（完了を待ってから生成）
if not globals().get('settings'): raise RuntimeError('先に1のフォーム・画像確認を完了してください。')
try:
    h3.setup(settings)
except BaseException:
    print('セットアップを停止しました。エラー時もGPUの割当は続くことがあります。終了セルまたはランタイムメニューから必ず切断してください。')
    raise
'''
    generate='''#@title 3. モデル1回読込 → 連続生成 → 検証・プレビュー
if not globals().get('settings'): raise RuntimeError('先に1と2を完了してください。')
_new_run_requested=NEW_RUN
NEW_RUN=False  # 追加生成の許可を1回だけ消費。失敗した場合も勝手に再試行しない。
try:
    outdir=h3.generate(settings,new_run=_new_run_requested)
except BaseException:
    print('生成を停止しました。完成したMP4があれば左側のファイル一覧から保存してください。GPUは終了セルまたはランタイムメニューから切断してください。')
    raise
import json
from IPython.display import Video,display
report=json.loads((outdir/'sequence_report.json').read_text())
print('モデル読込',report['model_load_seconds'],'秒／回数',report['model_load_count'])
for item in report['requests']:
    print('動画',item['index'],'生成',item['generation_seconds'],'秒',item['output'])
    display(Video(str(outdir/item['output']),embed=True,width=720))
print('プレビューだけで保存完了にはなりません。次のZIP保存へ進んでください。')
'''
    save='''#@title 4. ZIPを自分の端末へ保存
if 'h3' not in globals(): raise RuntimeError('先に入力と生成を実行してください。')
from google.colab import files
archive=h3.package_outputs()
files.download(str(archive))
print('ダウンロード完了をブラウザで確認し、ZIPまたはMP4が開くか確認してください。自動確認はできません。失敗したら左側のファイル一覧から保存します。')
'''
    stop='''#@title 5. 保存確認後にGPUを終了（このセルだけ再実行）
SAVED_TO_MY_DEVICE = False #@param {type:"boolean"}
STOP_AFTER_ERROR = False #@param {type:"boolean"}
# エラー後の終了は一時ファイルの喪失を承知して明示的に選択。
if not globals().get('_h3_stop_gate_open',False):
    _h3_stop_gate_open=True
    print('初回はここで待機します。ZIP/MP4の保存を確認して欄をオンにし、このセルだけ再実行してください。前回のオン状態では自動終了しません。エラー時はSTOP_AFTER_ERRORで終了します。')
elif not SAVED_TO_MY_DEVICE and not STOP_AFTER_ERROR:
    print('まだ終了しません。保存を確認してSAVED_TO_MY_DEVICEをオンにし、このセルだけ実行してください。エラーで終了する場合はSTOP_AFTER_ERRORをオンにします。')
else:
    print('クラウドGPUを終了します。Colabのリソース画面で使用率0・稼働0件を確認してください。')
    from google.colab import runtime
    runtime.unassign()
'''
    ending='''## 終了と再開

**ブラウザを閉じるだけでは終了確認になりません。** リソース画面で使用率0・稼働0件を確認してください。終了セルが動かない場合は「ランタイム → ランタイムを接続解除して削除」を使います。モデルと一時ファイルは削除されます。

同じVMで再開すると取得済みファイルは再利用されます。生成済みの同じ入力は追加生成せず既存出力を使います。新しい文章で生成するには保存後にNEW_RUNをオンにして入力セルを実行します。その許可は生成セルで1回だけ消費し、失敗時も消費済みです。再試行は診断・保存後に本人が許可し、入力セルを再実行して明示的に設定し直してください。フォーム再実行は許可を再設定する操作です。読込は1回の生成セルの間で共有し、そのセルが終わるとモデルプロセスを終了します。VM削除後は導入・取得からやり直します。

**検証範囲:** 配布版T2Vを2026-10-11、I2Vを2026-10-12に、A100 80GB・RAM167.05GB、各5秒・544p・4-step・動画1本で実測。T2Vは初期化199.132秒／生成92.688秒、I2Vは初期化205.243秒／生成106.352秒。どちらも960×544・24fps・124フレーム（約5.17秒）、ZIP保存・映像音声の全編デコード・GPU使用率0・稼働0件を確認しました。I2Vは画像表示後の内容確認と、保存確認後に終了セルを再実行するGPU終了も確認済みです。タイマー満了による自動終了、6秒、別GPU、高解像度は未実測。R2Vは別ノートブックです。このノートブックの解像度は544p・16:9固定でネイティブ1080p不可です。速度・CU・品質や人物・商品形状の完全維持は保証しません。
'''
    sources=[('markdown',intro),('code',bootstrap),('code',setup),('code',generate),('code',save),('code',stop),('markdown',ending)]
    cells=[]
    for kind,source in sources:
        cell={'cell_type':kind,'metadata':{},'source':source.splitlines(True)}
        if kind=='code':
            compile(source,'notebook-cell','exec')
            cell.update(execution_count=None,outputs=[])
            cell['metadata']={'cellView':'form'}
        cells.append(cell)
    nb={'nbformat':4,'nbformat_minor':0,'metadata':{'colab':{'name':'MiniMax_H3_Easy_Colab.ipynb','provenance':[],'private_outputs':True},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'},'cells':cells}
    target=ROOT/'MiniMax_H3_Easy_Colab.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    return target


if __name__=='__main__': print(build().name)
