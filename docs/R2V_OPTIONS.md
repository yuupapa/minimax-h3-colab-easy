# R2Vに採用した独立構成

ModelTC固定版の `match` 対応Ref2VAと、H3対応Diffusers固定版を採用しました。GPU試験1例の結果は[R2V_VALIDATION](R2V_VALIDATION.md)。統合版への変更はGPUを接続せず検証しています。

- H3 `42ed227ee7df40d41602854ae760620d6eb651fe` のDiffusers形式 `transformer_ref` と共通コンポーネントだけ取得。
- ModelTC `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` の3ソースは改変せず同梱し、サイズ・SHA256を検証。Apache-2.0を保持。
- Diffusers `85c9fa4caee33d1c5b473395784471d2d7957c1b`。全67依存の `requirements-r2v.lock` をGPU試験時の実物から固定し、新規解決せずsync。モデル取得前にAPI・scheduler・寸法をpreflight。
- Ref Turbo v0.1 BF16 `3ec17a324ced54151364f24f8b5fb6bf7e26414f`、LoRA SHA256 `9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99` を取得後に照合。
- Ref2VAの参照画像1枚を `references` と `<Picture 1>` で渡す。`match` は大きい画像を出力の画素予算まで縮小し、小さい画像を強制拡大しない。I2Vの始点画像条件へ置き換えない。
- sigma gridは5点、実際のTransformer評価4回をhookで検証。alpha8、scale1、fuse=False、標準attention、CPU offload。A100へHopper専用FA3を指定しない。

## 環境と余裕

別Colab VM・別venv・別状態・別キャッシュを使用します。既存のSGLangと異なるモデル形式を同じVMへ追加すると重複が増えます。Refのみモデル約134.13GiB＋LoRA約1.29GiB。初期空きはキャッシュ180GiB、環境25GiB以上、同じファイルシステムなら205GiB以上。Macへ重みを取得しません。

実測はA100-SXM4-80GB、RAM167.05GiB、driver580.82.07。診断でA100 80GB、RAM150GiB以上・空き140GiB以上、driver580以上、ディスクを確認します。実測CUDA最大確保76.35GiB・予約78.42GiBで余裕は小さく、別画像の成功を保証しません。32GB GPUの別レシピをこのRef構成へそのまま適用しません。

## 解像度と再現性の限界

`near720_crop` は1280×736生成→上下8pxずつクロップ→1280×720納品。ネイティブ高さ720は32の倍数ではないため直接指定しません。`544_upscale` は960×544→960×540クロップ→1280×720拡大で、GPU未検証です。拡大は生成時の細部を増やしません。

Ref v0.1は544p学習。FL用768p LoRAやFastH3へ勝手に変更しません。H3-Baseの公開機能とAPI専用フル2Kを区別し、新規API契約を作りません。

依存版・ソースcommit・LoRAハッシュは固定しますが、全wheelのハッシュ、Pythonパッチ版、Colab OS・GPU driverまでは固定していません。ビット単位の再現を保証しません。

## 入力・予算・終了

生成セルへ進む前に実際に表示されたプロンプトを読み、フォームで意図した内容と違えば停止します。GPU接続から最初の入力までの経過分を `GPU_CONNECTED_MINUTES_AGO` に入力して予算の開始を遡ります。再入力で締切を延長しません。中断・再開を含む合計の承認上限CUをリセットしません。

保存後はColabメニューから接続解除・削除し、使用率0・稼働0件を確認します。GPU未検証のR2V終了補助は既定オフ。タイマー満了による切断もGPU未検証で、終了メッセージだけを停止証拠にしません。1画像・1本・指定5秒の経路で、失敗後の自動再試行・自動ダウングレードはありません。
