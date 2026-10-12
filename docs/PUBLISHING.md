# 公開・更新時の確認手順

対象：yuupapa/minimax-h3-colab-easy。2026-10-12に配布版のT2V・I2V各1本のGPU試験、端末保存・全編デコード・GPU停止を確認しました。T2V/I2Vは544p・16:9固定。別R2V経路で1280×736生成から1280×720納品の1例を実測しました。統合差分とR2V終了補助はオフライン検証のみで、ネイティブ1080pは未対応です。公開・更新は、本人が具体的な対象と操作を許可した場合だけ実施します。キットの利用はGitHubへの書き込み許可ではありません。

1. README、AGENTS、ライセンス本文・NOTICE、実測と未検証範囲を確認。H3の地域・商用・AUP条件は保持し、所在地や法的資格を保証しない。
2. build_notebook、build_r2v_notebook、validate_integration、全unittestを実行し、埋込み一致と出力・個人メタデータなしを確認。
3. 配布はレビュー済みのソース・ライセンス・2つの出力なしノートブックだけ。親フォルダ、試験MP4・画像、モデル重み、依存バイナリ、ログ、個人URL、利用者のsettings/state、Library ID、秘密を加えない。
4. GPU試験が必要な変更は、目的・本数・CU上限を確認してから実施。生成成功、媒体の全デコード・保存、GPU使用率0・稼働0件まで記録する。
5. 初回公開/更新後、確定commit SHAと公開URLを記録。未認証でもREADMEが読めること、Open in Colabが正しい自己完結ノートブックを開くこと、全セル出力が空であることを確認する。表示の確認でGPUを再割当しない。

[Open in Colab](https://colab.research.google.com/github/yuupapa/minimax-h3-colab-easy/blob/main/MiniMax_H3_Easy_Colab.ipynb)。GPU実測と未検証の終了枝は[VALIDATION](VALIDATION.md)に記録しています。

## R2V統合の公開前レビュー

T2V/I2Vの実測済みコアと依存を保持し、R2Vを別ノートブックにします。2本のビルド・統合検証・標準ライブラリテスト・配布ZIPのプライバシー監査を完了してから、変更内容をレビューします。GPU試験済みのコアと統合後のオフライン差分を区別し、R2V終了補助・タイマー切断のGPU未検証状態を明記。同意は初期オフ。元画像・動画・個人の入力・Library ID・アカウント残量・私的Colab URLは公開しません。レビュー中に追加GPUを使わず、pushは公開指示後に行います。
