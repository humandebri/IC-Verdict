# IC-Laya — 最終仕様・計画・ADR v2.0

**英語 / Rust / ICP / Choice・Noul・Score / 少数質問 / 型付き関数連携 / 制約付きTx**  
2026-09-19（JST）

## 結論

Laya型backendを使い、1評価1質問・1workflow最大3質問を逐次実行する。Scoreを第一級とし、rubric、期待段階、平均、分布、上側確率を保持する。推論engineと権限/資金を持つexecutorを分離する。

初期profileはCompact128、Choiceは2〜5候補、Noulは2候補、Scoreは3〜7段階（標準5）。入力超過を無言で切らない。F32参照から実測によってINT8等へ最適化する。実送金は初期状態で無効。

## 文書

| ファイル | 内容 |
|---|---|
| [FINAL_SPEC.md](FINAL_SPEC.md) | 最終仕様の正本 |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | 実装フェーズと受入Gate |
| [ADR.md](ADR.md) | 更新済み16件のADR |
| [adr/README.md](adr/README.md) | 分割したADRの索引 |
| [specs/SCORE_AND_WIRE.md](specs/SCORE_AND_WIRE.md) | Scoreの数値、wire DTO、型境界 |
| [specs/WORKFLOW_AND_TX.md](specs/WORKFLOW_AND_TX.md) | 逐次評価、認可、送金回復 |
| [profiles/compact128.json](profiles/compact128.json) | 機械可読の設計profile。runtime用の完成設定ではない |
| [CHANGES.md](CHANGES.md) | 会話中のv1設計からの変更 |
| [review/REVIEW.md](review/REVIEW.md) | 自己レビューの20件の指摘と反映 |
| [review/CHECKS.txt](review/CHECKS.txt) | 文書整合性・数値例の検査結果 |
| [SOURCES.md](SOURCES.md) | 一次資料と確認範囲 |

## 検証範囲

設計の作成・公開資料確認・自己レビュー・文書と数値例の検査を実施した。実モデル推論、Rust/Wasm build、ICP deploy、ベンチマーク、実資金送金、第三者監査は今回行っていない。

旧版ファイルの原本は同梱していない。CHANGES.mdは会話で確認できる設計の変更記録であり、元ファイルに対するdiffではない。
