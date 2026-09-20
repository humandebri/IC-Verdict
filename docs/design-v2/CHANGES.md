# v1 → v2 設計変更

2026-09-19（JST）。これは会話中に示された設計方針に対する変更記録であり、前版ファイルの機械的なdiffではない。

| 前版/直前の議論 | v2で固定 |
|---|---|
| NLIを基礎にしたengine | Laya式option-marker backend |
| DeBERTa-small/ModernBERT-NLI比較 | Laya typed-decisionsを第一参照に固定。資源/品質未達なら同じprimitiveを維持して適応/蒸留 |
| Choice/Scoreを後続へ | 三primitiveをcoreへ。ADR-014を置換 |
| NLI neutralをuncertaintyへ寄せる説明 | NLIラベルで代用しない。外側のabstentionと分布を分離 |
| 1質問/callだけの説明 | 1評価1質問、1workflow最大3質問の逐次実行を明確化 |
| Scoreを平均にまとめる | rubric ID、期待段階、正規化平均、分布、上側確率を維持 |
| 128をlocal attention由来の自然境界とする説明 | 製品の資源profileとする。attention maskの同値化はしない |
| act headを無条件で削る | 参照parityでは保持。実行権限に使わず、logit不変確認後に配備計算を省略 |
| 短いmax_lenだけ設定 | prefix≤64をコンパイル時検査。無言切詰め禁止 |
| temperatureは付属設定で十分という印象 | 上書き優先順位を確認し、task/profileごとに有効値を一意に固定 |
| 型付きdecisionをTxへ接続 | private認可型・委任・nonce・業務重複・不明予約を維持/強化 |

新設ADR-015はScoreの数値契約、ADR-016はCompact128/少数逐次workflow。既存の認可・ledger回復・監査・upgradeの境界は撤回しない。
