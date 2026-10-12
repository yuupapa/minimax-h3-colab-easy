# T2V・I2V・R2Vの追加範囲と検証計画

現キットの解像度はshort_edge=544・16:9固定です。固定SGLangのコード上、720指定は1280×704、1080指定も1344×768となり、ネイティブ1920×1080は生成できません。高解像度の時間・CUは未計測で、画素比例の試算を保証にしません。[寸法の静的確認](VALIDATION.md#解像度の静的確認)

2026-10-12時点、配布版T2V・I2V各1本のGPU生成・保存・全編デコード・GPU終了を確認しました。I2Vは始点画像1枚を使います。R2Vは未実装で、この公開版の対象に含めません。以下のR2Vは静的調査・未実行の計画です。権限拒否や未承認のアクセスを別経路で迂回しません。

| 用途 | モデルと入力 | この候補の状態 |
|---|---|---|
| Text-to-Video | FL2VA、task=t2va、文章のみ | 開発版2本、配布版T2V1本の生成・保存・GPU終了を実測 |
| Image-to-Video | FL2VA、task=fl2va、始点画像1枚＋動きの文章 | 配布版で画像入力・Pillow正規化・GPU生成・保存・全デコード・GPU終了を確認 |
| Reference-to-Video | Ref2VA、task=ref2va、参照画像＋別場面の文章 | 仕様・固定コード・容量を調査済み。実行経路は未実装で、選択すると取得前に停止 |

I2Vは画像の始点構図から動かす機能です。R2Vは人物・商品の見た目を別場面へ持ち込む用途を想定しています。始点画像を渡すだけでR2V対応とは呼びません。見た目・文字・商品形状の完全な維持も保証しません。[公式H3仕様](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/README.md)

## I2Vで追加したこと

- モード選択、始点画像の1枚アップロード、権利と送信の本人確認、画像表示後の本人確認。アップロード直後は古い確認欄がオンでも必ず停止。失敗時に以前のT2V設定で先へ進みません。
- 公式Colabの単一ファイルアップロードAPIを使用。元画像はクラウドの専用uploads領域、正規化した画像はinputs領域に保存。画像選択は本人が行い、URL・Driveマウント・任意のファイルパスからの自動取得はしません。[Colab公式API](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py)
- 静止画PNG/JPEG/WebP、元ファイル20MiB以内、各辺64〜8192px、1600万画素以内。向きを適用して新しいRGB PNGへ変換し、生成に使うPNGからEXIF・テキスト・ICC情報を落とす設計。**アップロード前の元画像にはEXIF等が含まれ得ます。** 必要なら本人が送信前に端末側で取り除いてください。正規化前の元画像もこのVMに残ります。
- 正規化PNGのSHA256とモードを重複判定に追加。改変、URL、管理領域外のパス、symlinkを拒否。生成前に再確認。画像や元のファイル名を配布repoに加えず、出力ZIPへ画像を自動同梱しません。
- 現在は画像1枚・動画1本・5秒から検証。544p・16:9出力、既存FL2VA Turbo v0.1、4評価、shift12/3、alpha8、torch_sdpa、CPU offloadを保持。別比率の画像は出力時に再構図・トリミングされ得るため、初回は16:9を推奨。PROMPTは動き、カメラ、音の指示に使います。

固定SGLangのI2V条件は `type=image, role=keyframe, frame_index=0, uri=file://...`。既存T2Vのサンプリング値とモデル読込1回の経路は保持しています。[固定コードの入力規則](https://github.com/sgl-project/sglang/blob/448759b2358eb38e2a244fbe7d5a293a69dd314c/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/request_validation.py)

## R2Vで必要な追加と未解決点

Ref2VAはFL2VAと別のTransformer重みです。同じ固定H3 revisionのRef2VA側だけで約134.16GiB。HF公開サイズとLFS SHAの集計では、共通blobを再利用した両variantの重み合計は約195.85GiB、FL2VAに加える固有重みは約61.73GiB。これはダウンロード完了の実測ではなくメタデータからの見積もりです。別ディレクトリへコピーし共通blobを再利用しなければ約268GiBになるため、実際の取得方式と空きを確認します。[固定版ファイル一覧](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe)

現在固定のTurbo revision `797b284f774ff0bda805f7dd5917624670d5c4e0` にはFL2VA v0.1だけがあり、Ref2VAへ流用しません。追加候補はLightX2V revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f` の `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`（約1.289GiB、公開LFS SHA256 `9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`）。重みは取得しておらず、実ファイルのハッシュ一致も未確認です。[候補の固定ファイル](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/3ec17a324ced54151364f24f8b5fb6bf7e26414f/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors)

配布元はRef2VA v0.1を544p、4評価、shift12/3と説明し、推論コードのalpha既定値は8です。しかし参照画像の推奨は、出力の画素数に合わせて縮小し小さい画像は拡大しない `match`。固定SGLangは短辺を2048pxへ拡大する処理です。小さい画像を事前縮小してもSGLang側で再拡大されるため、この差は解決しません。SGLang公式もRef2VA LoRAを検証済みレシピとして固定していません。単に同じ4-step値を設定して高速R2V対応と宣言しません。[Turbo固定版仕様](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/README.md)、[固定SGLang画像処理](https://github.com/sgl-project/sglang/blob/448759b2358eb38e2a244fbe7d5a293a69dd314c/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py)、[SGLang LoRA案内](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#5-lora-recipes)

FastH3 8-stepは固定SGLangでT2VA専用と検証され、FL2VA/Ref2VAを拒否します。I2V/R2V用の高速モデルとして置き換えません。[固定FastH3の検証](https://github.com/sgl-project/sglang/blob/448759b2358eb38e2a244fbe7d5a293a69dd314c/python/sglang/multimodal_gen/configs/sample/minimax_h3.py)

R2Vの最初の候補環境はA100 80GB・RAM約167GiB、参照静止画1枚だけ。現在の150GiB以上／空き140GiB、空きVRAM約70GiBというチェックを出発点にしますが、R2Vの必要量を保証する数値ではありません。2048px画像のVAE/Qwen処理や条件tokenが増えるため、実測ピークで判断します。公式32GB GPU＋60GB RAMの480p例はFL2VAの試験で、R2Vへそのまま適用しません。Hopper専用FlashAttention3をA100へ適用しません。[SGLangハードウェア例](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3#9-hardware-tuning-and-benchmarks)

ディスクはRef2VAだけなら既存と同じ初回180GiB＋環境25GiBを仮の下限にし、重複と一時ファイルを実測。両variantを同時にキャッシュする場合、約195.85GiB＋LoRA約2.58GiB＋環境25GiB＋余裕を見て**単一ディスク260GiB以上の空き**を計画値とします。これは未実測です。共通blob再利用が無い場合は320GiB以上へ見直します。小さいルートディスクに無理に取得せず、大容量scratchがなければ別VMでRef2VAだけ取得する案を検討します。Macへは取得しません。

実装順は、別variant・LoRA・依存digest・状態を分ける→参照画像1枚と `<Picture 1>` に対応した文章を作る→画像サイズ方針を解決する→VRAM/RAM/ディスク診断と保存・終了を通す。解決できない場合は、公式Baseの50評価を候補にするか、配布元のmatch対応経路を別途監査・固定する判断が必要です。50評価の速度・CUを4-stepの実測から保証しません。公開されていないContext-IRと2K再生成APIは含めず、新規API契約へ進みません。

## 本人確認・ライセンス

H3-BaseのRef2VAもH3 Community Licenseの対象で、利用地域・商用規模・AUP・下流表示の確認を引き継ぎます。確認欄はColabのVM所在地の確認・保証を求めるものではなく、本文の地域制限を変更しません。Ref Turbo候補の公開ライセンス表記はApache-2.0で、H3の条件が消えるわけではありません。別の配布元コードを採用するなら、そのソースとライセンス・実行承認も追加で確認します。

画像については利用権、人物の同意、非公開商品・個人情報がGoogle Colabへ送信される範囲を本人が確認します。人物・商品の画像を勝手に探して送信しません。画像をGitHub・公開ノートブック・Libraryへ自動で出しません。Library保存する動画やレポートに元画像の見た目や文章が含まれる場合も、保存先と共有範囲を確認します。認証情報や画像内の個人情報を抽出する機能はありません。

## GPU試験と追加範囲

T2V・I2Vは、A10080GB・RAM167.05GB、各5秒・動画1本・544pで配布版を実測しました。モデル取得・初期化・生成、ZIP保存、全映像音声デコード、入力画像反映の目視、GPU使用率0・稼働0件まで確認。処理時間・CU表示差と未検証範囲は[VALIDATION](VALIDATION.md)に記載しています。Macではモデルを実行せず、33項目の標準ライブラリテストと保存媒体のデコードを行いました。

新たなGPU試験は、目的・モード・本数・上限CUを本人が許可した場合だけ実施します。R2Vは互換性の方針決定・実装後に、参照画像1枚と別場面の5秒動画1本から検証する案です。目安5CUは未実測・未承認の予算案で、実際の残量・レート・取得方式から見直します。Base50評価なら完了に足りると保証せず、上限到達で停止します。複数画像・動画・音声参照や40本連続生成は追加範囲にしません。

R2Vは参照画像と異なる背景で主体の見た目を保つ用途です。I2Vの始点再現をR2V試験の代わりにせず、T2V・I2Vの成功を3モードすべての成功と表現しません。

## 追加調査の結果

公式更新と配布元経路を調べ、R2V専用の別ノートブック・別VMで、ModelTC固定版のmatch対応＋H3対応Diffusers固定commitを使う案を優先しました。現行Diffusers0.37.0にはH3 APIが無く、同じvenvへの上書きは避けます。SGLang原形式同士の61.73GiB追加と異なり、SGLang FL2VA＋Diffusers Ref2VAの併用では固有追加約71.99GiB、総重み約206.12GiBになるため、同一VMで併用するなら空き280GiB以上を計画します。詳細な比較と未検証事項は[R2V対応案](R2V_OPTIONS.md)を参照してください。


## 画像入力の追加診断

2026-10-11、GPUを起動せず、CPU・標準メモリで配布版と同じupload_first_frame関数を実行。静止画PNG1枚のアップロード、Pillow正規化、960×544 PNG保存、last_input_imageの一致、レポートの端末保存を確認しました。2026-10-12の配布版GPU試験でも画像入力・内容確認からI2V生成へ進む経路が成功しました。ファイル選択の操作ツールによるタイムアウトは[TROUBLESHOOTING](TROUBLESHOOTING.md)を参照してください。

## 2026-10-12のGPU試験結果

配布版のI2Vを始点画像1枚・5秒・動画1本で生成し、ZIP保存・全映像音声デコード・GPU使用率0・稼働0件を確認しました。詳細と品質の観察は[VALIDATION](VALIDATION.md)。T2VとI2Vの公開前試験は完了しましたが、R2Vは未実装のままです。以前の画像選択中止と未承認予算案は履歴であり、現在の検証状態はこの節とVALIDATIONを参照してください。
