# v2 自己レビューと修正記録

2026-09-19（JST）。対象は会話中の設計説明と今回の最終仕様・ADR。公開一次資料との突合、型/数値契約、状態遷移、受入条件を自己レビューした。第三者監査ではない。

## 修正・明文化した事項

| ID | 指摘 | v2への反映 |
|---|---|---|
| V2-R01 | Scoreを後回し/削除すると要件を満たさない | core primitiveへ昇格。ADR-014置換、ADR-015新設。 |
| V2-R02 | root Layaとtyped-decisions specialistを同じ汎用性で扱う危険 | 独立repoを第一参照に固定し、対象外業務の品質を別Gateにする。 |
| V2-R03 | Scoreの元値は0〜1ではなく0〜K−1 | 期待段階と正規化平均の両方を単位付きで定義。 |
| V2-R04 | 平均が同じなら同じriskと扱ってしまう | 分布と上側確率を正本にする。中央集中/両端分裂の例を試験。 |
| V2-R05 | ordinalの等間隔化を実世界の事故率と誤解する | 表示上の規約と明記。rubric、方向、utilityを分離。 |
| V2-R06 | 1質問/callと3signalのTx合成が矛盾し得る | engineは1問、workflow最大3問の逐次ジョブ。必須slotが揃うまで許可しない。 |
| V2-R07 | 逐次質問で別snapshot/modelの結果を混ぜられる | immutable bindingと全slot一致、stale/late callback排除を規定。 |
| V2-R08 | 3質問へ分割すると総計算も1質問分になるという印象 | 最大60Bは評価分だけの目標と明記し、cycles/latencyを別集計。 |
| V2-R09 | max_lenだけ128にするとheadがstateを圧迫しうる | prefix≤64をコンパイル時検査。rubric/option/stateの無言切詰めを禁止。 |
| V2-R10 | local attention 128なら128token入力の全attentionが同じになるという解釈 | 実際のlocal maskを保持。128は製品profileとして選択。 |
| V2-R11 | headのactivationをすべてGELUとして移植する誤り | PyTorch default ReLU、scorer GELU、backbone GeGLUを分離して照合。 |
| V2-R12 | 付属temperatureをそのまま使えば校正済みという過信 | priorityを解決したeffective設定とtask/profile別再校正を必須化。 |
| V2-R13 | entropy集中度を正答確率として扱う | diagnosticsとして返し、calibrationとabstention ruleを別途評価。 |
| V2-R14 | act headを削ると大幅高速化できるという過大評価 | logits不変の確認後に省略。主要最適化とは見積もらない。 |
| V2-R15 | 型付き出力なら直接Txへ渡して安全という混同 | 外部DTOと内部AuthorizedTransferを分離。runtime権限/予算検査を維持。 |
| V2-R16 | transport再送で同じ評価を何度も計算 | evaluation IDによるlive重複排除、expiry、quotaを規定。 |
| V2-R17 | unknown送金を後続TooOld等だけで失敗扱い | 不明予約とoperation予約を保持。ledger固有の確実な照合を要求。 |
| V2-R18 | INT8化すれば必ず速いという仮定 | kernel込みで実測。F32全展開だけの「最適化」を採用しない。 |
| V2-R19 | 型名が同じなら違うrubricでも使える | schema/version/hashとcandidate順をSDK/runtimeで検査。 |
| V2-R20 | 品質未達時にScoreを無効化して受入完了にする | 三primitive別Gateと最終policy Gateをすべて必須化。 |

## 第2パス

本文、ADR、profileの上限（1/3質問、128tokens、2〜5Choice、3〜7Score、標準5段階）、Scoreの単位、同じsnapshot、ReportOnly、unknown予約保持を読み合わせる。文書構造と数値fixtureを`validate.py`で確認する。

validate.pyは実モデル、Rustの型実装、認可コード、ledger接続を試験するものではない。文書の整合と設計上の数値例を検査する補助である。

## 残る受入作業

固定したrevision/hash、実tokenizerのschema token数、raw logitsの一致、ICPのinstructions/heap、INT8の実利、Scoreの過小評価と校正、最終workflowの誤受理率、特定ledgerの回復、独立レビューは未実施。

設計方針は固定したが、これらの未実測値を推測で合格にしない。未合格のまま実資金Txを有効化しない。
