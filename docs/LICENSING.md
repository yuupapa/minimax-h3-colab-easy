# ライセンス確認と帰属

確認日：2026-10-11。独自の導入・連続生成用の制御コードと文書はMITで配布する案です。APIを呼ぶ制御コードを公開することは、モデル重みや第三者バイナリをMITで再配布することではありません。モデル重み・LoRA重み・外部依存バイナリ・試験動画は同梱しません。

| 対象 | 確認した条件・扱い |
|---|---|
| 独自スクリプト・README・AGENTS | ルートLICENSEのMIT。第三者のライセンスを変更しない |
| MiniMax H3 | 固定revisionのCommunity License本文を `licenses/MINIMAX-H3-COMMUNITY.txt` に原文で収録。NOTICEへ帰属を記載。利用者は全文・Exhibit Aを読み、自身の利用条件を確認 |
| LightX2V Turbo LoRA | [固定revisionの配布元](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/797b284f774ff0bda805f7dd5917624670d5c4e0)のメタデータはApache-2.0。H3由来モデルに適用するためH3の条件も残る。重み非同梱 |
| SGLang | [固定commitのLICENSE](https://github.com/sgl-project/sglang/blob/448759b2358eb38e2a244fbe7d5a293a69dd314c/LICENSE)はApache-2.0。原文を `licenses/APACHE-2.0.txt` に収録。パッケージは公式ソースから利用者VMに導入 |
| uv | [公式PyPI](https://pypi.org/project/uv/0.13.0/)はMIT OR Apache-2.0。公式レジストリから利用者VMに導入 |
| Qwen3-VL | H3ライセンスのAdditional NoteがApache-2.0と明記。H3の構成部品として取得、重み非同梱 |
| その他234依存 | 各上流のライセンス・NOTICEに従う。バージョン固定は配布ライセンスを変更しない。バイナリをこのrepoへコピーしない |

## H3の条件は利用者と公開者の双方が確認する

[固定版H3ライセンス](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE)は、EU・英国・韓国・米国をExcluded Territoriesとします。利用地域・予定用途・商用条件・Acceptable Use Policyを本人が確認し、条件の適用が曖昧な場合はMiniMaxの許諾窓口へ確認してください。LICENSE_ACCEPTEDは本文への同意、ELIGIBLE_USE_CONFIRMEDは今回の利用条件を本人が確認したことを示します。ColabのVM所在地を確認・保証したという宣言ではありません。本文はVM所在地の調査・証明を明示的に求めていませんが、これはクラウド利用を一律に許可する解釈ではありません。本キットは所在地や法的な利用資格を判定せず、本文の地域制限を変更しません。

年間売上2000万米ドル超の商用製品・サービスには事前の個別許諾条件があります。商用UIのMiniMax H3表記、他AIモデルの改善への利用制限、公開生成物のAI生成表示、下流利用者への制限通知・拘束、安全対策と違反報告対応などの条件もあります。これらの要約で本文を置き換えることはできません。

本キットは本人のColabで少数の動画を作る補助であり、共有の公開生成サービス/APIではありません。チェックボックス・少数制限・クラウド限定ガードを設けていますが、それだけでライセンス上の安全対策・法的条件を満たすと保証しません。公開前に配布者が適用範囲と下流利用者への条件の扱いを確認してください。

違反・不適切な利用が見つかった場合の窓口は、公開後のGitHubリポジトリのIssuesを使用する予定です。公開前に管理者の対応方針を決めてください。秘密・個人データをIssuesへ投稿せず、必要なら適切な非公開窓口を設けてください。

## 固定識別子

- MiniMaxAI/MiniMax-H3: `42ed227ee7df40d41602854ae760620d6eb651fe`
- sgl-project/sglang: `448759b2358eb38e2a244fbe7d5a293a69dd314c`
- lightx2v/Minimax-h3-Turbo: `797b284f774ff0bda805f7dd5917624670d5c4e0`
- Turbo: `minimax_h3_fl2v_turbo_4step_v0.1.safetensors`
- Turbo SHA256: `5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29`

H3本文SHA256：`59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`。同梱本文には改変を加えていません。


## R2Vの追加配布物

ModelTC固定commit `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` の3ソースとLICENSEを `vendor/modeltc` に原文で保持。Apache-2.0、manifestに不変URL・SHA256を収録。実行には当該ソースの承認を必要とする。Diffusers固定commit `85c9fa4caee33d1c5b473395784471d2d7957c1b`、PEFT0.21.2、Accelerate1.15.0もApache-2.0。Ref LoRAはLightX2V固定revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`、配布メタデータApache-2.0。H3由来の利用条件は継続し、H3ライセンス・AUP・利用資格の本人判断を省略しない。重み・外部バイナリは同梱しない。実機で使った67依存lockの固定は各上流のライセンスを変更しない。
