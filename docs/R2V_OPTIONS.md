# R2Vを最小構成で追加する案

2026-10-11の追加静的調査。GPU接続・パッケージ導入・モデル取得・画像送信・公開は行っていません。T2V/I2Vの現行環境とコードはこの調査では変更していません。

**優先案は、R2V専用Colabを別セッションで用意し、Turbo配布元の固定版コード＋H3対応Diffusers固定commitで、Ref2VAだけを取得することです。** 公式SGLangの処理を独自に書き換える案は採用しません。まだ依存導入・実import・GPU生成は検証しておらず、この案を実証済みの対応とは扱いません。

## 調査結果

SGLang公式main `46a97e55400d871ba05537f21ed603321801da2d` を調べました。H3の参照画像処理とSamplingParamsは現行固定commitとバイト単位で同一。ServerArgsとSamplingParamsに `reference_resize_mode` / `ref_image_size` の公開設定は見当たりません。最新公式版への更新だけで短辺2048px処理をmatchに変更できる根拠は得られませんでした。[最新公式処理](https://github.com/sgl-project/sglang/blob/46a97e55400d871ba05537f21ed603321801da2d/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py)

Turbo配布元ModelTCの固定commit `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` には、Ref2VA専用の画像サイズ選択を備えたパイプラインと単一GPU CPU offloadの推論経路があります。設定でmatch、4評価、video shift12、audio shift3、alpha8、scale1を選べます。matchは参照比率を保ち、生成画面の画素数に合わせて縮小し、小さい画像は拡大しません。これは配布元が提供する実装であり、現行SGLangをこちらで改変したものではありません。[配布元の単一GPU手順](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/DIFFUSERS_SETUP_AND_INFERENCE.md)、[matchパイプライン](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/minimax_h3_ref2va_pipeline.py)

**現在のDiffusers0.37.0にはMiniMax H3 APIがありません。** 公式tag `d791c5c024014784df4a8dac2601e19fb4d300fc` のソース一覧でもH3ファイルが存在しません。SGLangによる既存生成はそのDiffusers H3 APIを使わないため、これまでの成功と矛盾しません。R2V用に同じvenvのDiffusersだけを上書きせず、別環境を用意します。[0.37.0公式ソース](https://github.com/huggingface/diffusers/tree/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/modular_pipelines)

Diffusers公式main `85c9fa4caee33d1c5b473395784471d2d7957c1b` には、配布元コードが参照するH3クラス・ヘルパー・workflow選択が存在します。Ref2VAを先頭に持つbefore-encode分岐などを静的に照合しました。ただしソース上のAPI存在確認であり、組合せの実importや実行成功を保証しません。[H3の公式ブロック](https://github.com/huggingface/diffusers/blob/85c9fa4caee33d1c5b473395784471d2d7957c1b/src/diffusers/modular_pipelines/minimax_h3/modular_blocks_minimax_h3.py)

4-stepの数え方にも注意が必要です。SGLangは要求値4を使いますが、ModelTCのCLIは4評価に対して終端0を含む5点のsigmaをDiffusersへ渡します。直接パイプラインへ4を渡すと評価回数が変わります。配布元のスケジュール処理を保持し、4評価をレポートで確認します。[配布元コード](https://github.com/ModelTC/Minimax-H3-Turbo/blob/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/inference_minimax_h3.py)、[公式scheduler](https://github.com/huggingface/diffusers/blob/85c9fa4caee33d1c5b473395784471d2d7957c1b/src/diffusers/schedulers/scheduling_minimax_h3.py)

## 案の比較

| 案 | match・高速化 | ディスク・環境 | 判断 |
|---|---|---|---|
| SGLangの公式更新または設定だけ | 調べた最新mainにmatch設定を確認できず | 現行環境を保てるが問題は残る | 更新だけでは採用しない |
| 同じVMでSGLang T2V/I2V＋ModelTC R2Vを別venv | 配布元のmatch＋Ref専用4-step候補 | 両形式の固有blob合計約206.12GiB、LoRA・2環境・余裕込み空き280GiB以上を計画 | 大容量scratchでのみ検討。現時点では最小案ではない |
| **別ノートブック・別VMでR2Vだけ取得** | **配布元のmatch＋Ref専用4-step候補** | **R2V約134.13GiB＋LoRA約1.29GiB。現行環境を変更しない** | **優先案。再取得のCUは必要だが切替と保存が明確** |
| 同じVMでFL2VAキャッシュを消してR2V取得 | match経路は別venvが必要 | ピークを抑えられるが、削除対象・共通blob・再取得を誤りやすい | 自動削除切替は採用しない。別VMを優先 |
| 公式ComfyUI＋配布元ワークフロー | ref_image_size=matchが存在。Ref4-stepの配布元例あり | 別形式のモデル・ワークフロー・導入・保存経路の検証が増える | 二番目の候補。現行キットへ未検証のまま追加しない |
| SGLang Ref2VA Baseの標準50評価 | 公式Baseは2048px処理。Turboのmatch差を避けられる | Ref2VA原形式だけ取得可能。生成時間は増える | 正しい参照機能の基準候補。高速版とは呼ばない |

ComfyUI公式固定commit `7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217` のRef2VA nodeにはmatch処理があります。ただしnative SGLangのrequestへ同じ設定を足すだけの経路ではなく、別の画像エンコード・conditioning経路です。[公式ComfyUI node](https://github.com/comfyanonymous/ComfyUI/blob/7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217/comfy_extras/nodes_minimax_h3.py)

前の約61.73GiB追加・両方195.85GiBは、SGLang原形式のFL2VA/Ref2VAを同じHF cacheに保持する見積もりです。**SGLang FL2VAとDiffusers Ref2VAを併用すると、VAE等の形式差により固有追加は約71.99GiB、合計約206.12GiB**になります。HF固定revisionのサイズとLFS SHAからの集計で、ダウンロード実測ではありません。両形式のキャッシュを同時保持するなら、以前の260GiB案ではなく2環境分も含む280GiB以上を計画し、実際の空きを確認します。共通blobを再利用しないコピーならさらに増えます。

## 優先案の準備・実行順

1. 今回承認待ちのT2V/I2Vは現行ノートブックと現行venvで実機確認し、動画とレポートを端末等の承認先へ保存・全デコード。使用率0・稼働0件を確認してVMを終了。R2V試験を勝手に加えない。
2. R2V専用の自己完結ノートブックを別途用意する。現行T2V/I2Vの依存ファイル・コード・キャッシュ状態を変更しない。R2Vのbudget/state/outputも分離する。ModelTCの固定版3コード（inference、Ref pipeline、resolution util）とApache-2.0本文を監査・ハッシュ固定し、変更しない画像処理と4評価スケジュールを採用する。オーケストレーションのcloud-only、本人確認、1本、CU期限、保存・GPU終了を再利用する。
3. R2V環境はPython3.12、H3対応Diffusersを上の公式commitで固定し、PEFTとAccelerateを追加する。現行環境に両パッケージは無い。確認時の公式PyPIはPEFT0.21.2・Accelerate1.15.0で、候補として依存解決を行い、実際に導入できた全依存を別lockへ固定する。既存Torch/Transformers等との組合せは未実行。Colabクラウド内で重み取得前にAPI importとdry-runを確認し、失敗時は重みを取らない。Macへ導入しない。
4. H3は現行固定revision `42ed227ee7df40d41602854ae760620d6eb651fe` のDiffusers形式だけを取得する。対象はrootのmodel_index/modular_model_index、LICENSE、transformer_ref、text_encoder、processor、tokenizer、vae、audio_vae、scheduler、audio_scheduler。**FL2VA、Ref2VA原形式、transformer側は取得しない。** Ref専用LoRAは別固定revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f` のv0.1 bf16ファイルだけを取得し、公開SHA256と実ファイルを照合する。
5. モデルindexのcomponent sourceにはリモートHub IDとrevision=nullが書かれている。完全なローカルsnapshotを使い、配布元が実装するpretrained_model_name_or_pathの明示上書きを保持する。workflow=ref2vaでtransformer_refだけをloadし、missing componentsを確認する。HFの暗黙token利用・telemetryを無効化し、取得後はofflineで生成。欠けたcomponentをmainから自動取得せず停止する。
6. A10080GB・ハイメモリ約167GiBを候補にする。単一GPU、BF16、配布元の自動CPU offloadと既定12GBのVRAM余裕から始める。全コンポーネントGPU常駐、FSDP2、torch compile、Hopper専用FlashAttention3の指定を行わない。標準SDPA経路と必要VRAM/RAM、CPU offloadの動作を診断し、Ref1枚でも不足すれば停止する。ディスクは既存と同じ単一205GiB以上、別ディスクならcache180GiB＋環境25GiBを仮下限にし、依存解決後の実容量で確認する。80GBでの成功・速度・最大メモリはまだ保証できない。
7. 個人情報を含まないテスト用の商品画像1枚を参照にし、文章で別背景・別場面を指定する。初回は5秒1本・960×544・124frames、matchと4評価。元画像の利用権と送信先を確認。参照の見た目を維持した別背景になったかをI2Vの始点再現と分けて評価する。計画する実機試験は目安5CUだが未承認で、事前診断と開始時の実残量・CU/hourで再評価する。
8. 取得、初期化、4評価、生成、VRAM/RAMピーク、実消費CU、全動画デコード、保存先を記録。動画を保存した後にGPUを終了し、使用率0・稼働0件を確認する。自動タイマーと例外終了も新経路で検証する。試験に成功してから対応範囲と検証記録を更新し、公開へ進む。

## 残る検証と条件

現時点で確認したのは公式コード・公開メタデータ・API名と構造のみです。新venvの依存解決・実import、LoRA tensorの形状、実ハッシュ、モデル初期化、VRAM/RAM、速度、品質、保存、期限停止は未検証。ModelTCコードはLoRAの対象・rank・形状を照合して不一致を停止する実装ですが、実ファイルをまだ読んでいません。

ModelTCとDiffusers/PEFT/AccelerateはApache-2.0系の配布で、H3 Community Licenseと画像の権利・人物同意・クラウド送信確認は別に引き継ぎます。追加コードを同梱する場合はソース、ライセンス・NOTICEと固定ハッシュをレビューし、実行時の確認を解決します。API契約、Driveマウント、外部トンネル、公開サービスを追加する必要はありません。

新しいGPU実行とモデル取得は未承認です。現在質問中のT2V/I2V計2本・目安4CUの回答を待ち、R2Vの試験は別途確認します。公開は引き続きテスト後で、ソース調査や設定完了を生成成功として扱いません。
