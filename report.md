# 計算効率・状態管理の改善記録

## Summary

- 対象: 承認済み計画の実装と再検証。比較元は `4219f136507dad819d0c1bb3e74f3572c238755c`。
- 更新日: 2026-09-22。実装・再検証完了。1,000件の一致率は99.7%だが、既存の危険な棄権反転1件により品質gateはFAIL。
- 確定結果: 同一bundleの再warm-upとバッチ回答の不具合を修正。重み複製・全状態clone・全件比較を削減。
- 状態数: RETEST PASS 6 / PASS 1 / FAIL 1。指摘数: P1 3 / P2 5（修正済みを含む）。
- 主な計測: Wasmメモリ最大到達量47.7%減。比較できた7入力はlogits完全一致。
- 本番配備・費用モデル更新は行わない。既存の危険な棄権反転を別途解決する必要がある。

## Status Legend

`RETEST PASS`: 修正して再検証済み。`PASS`: 期待する動作を確認済み。
`TESTING`: 実行中。`FAIL`: 基準未達。`deferred`: 今回の実装対象外。

## Allowed Status Transitions

`TODO → TESTING → PASS / FAIL / BLOCKED`、`FAIL → FIXED → RETEST PASS / FAIL`。
以下は実装・再検証後の状態を記録する。

## Feature Inventory / User Stories

| ID | 利用者の操作・期待 | 実装根拠 | 検証と状態 |
|---|---|---|---|
| US-01 | 同一モデルを再warm-upしても登録契約と評価IDを継続利用できる | `canisters/verdict-engine/src/lib.rs`: `activate_bundle`, `finish_warmup` | snapshot復元unit、旧版→新版upgrade、再warm-up、失敗upgradeの復旧: RETEST PASS |
| US-02 | 複数質問が同じ選択肢IDと棄権を使え、各質問の確率和が1になる | 同上: `batch_layout`, `batch_results` | ID重複の範囲、独立softmaxのunit、ローカル2質問: RETEST PASS |
| US-03 | 同じモデルを少ないメモリでロードし、既存と同じ推論結果を得る | `crates/verdict-candle/src/lib.rs`: `assemble` | 同じpackのWasm旧新版比較: RETEST PASS |
| US-04 | 各forwardで同じ局所attention maskを再生成しない | `crates/modernbert-candle/src/lib.rs`: `attention_masks` | 境界mask、cached/単独forward一致、実モデル比較: RETEST PASS |
| US-05 | executorの正常・異常・非同期遷移が保存される | `crates/ic-laya-core/src/workflow.rs`: `Changes`; `canisters/executor/src/stable.rs`: `sync` | stale/ReportOnly等のErr、未知結果と遅延成功、登録/取消/メタ情報、旧新版upgrade: RETEST PASS |
| US-06 | verdict-engineのheap更新がupgrade時に復元される | `canisters/verdict-engine/src/lib.rs`: `mutate`, `pre_upgrade` | 新旧upgradeと意図的な失敗upgrade: RETEST PASS |
| US-07 | 新カーネル候補を計測し、悪化する候補を採用しない | `crates/verdict-simd/src/lib.rs`; owner限定benchmark API | block32 84点、attention 14点、Wasm端数tile: PASS、全候補不採用 |
| US-08 | 著者記録1,000件以上・一致率99.5%以上・欠落0・危険反転0を要求する | `crates/verdict-candle/src/bin/verdict_infer.rs`: `quality_pass`; `tests/golden.rs` | 判定unit PASS。実モデル997/1000一致・欠落0・危険反転1: FAIL |

## Findings / Fix Log

| ID | 重要度 | 問題と影響 | 対応・状態 |
|---|---|---|---|
| F-01 | P1 | warm-upが毎回EngineStateを初期化し、登録スキーマ・校正・キャッシュを消去する | 同一bundleは保持。別bundleはモデル依存情報だけをリセットしcaller利用枠を保持。RETEST PASS |
| F-02 | P1 | batchの全質問でIDを一意に要求し、全選択肢のsoftmaxを分割して返す | 質問内のID検証と質問単位softmaxへ変更。質問IDは空白・重複を拒否。RETEST PASS |
| F-03 | P2 | モデル構築時に量子化重み全体を複製する | QuantMapからremoveで所有権移動。メモリ実測で確認。RETEST PASS |
| F-04 | P2 | 推論予算判定のためのPersistent cloneとlayerごとのmask生成 | 必要なscalarだけを読む。maskはforward内で共有し全0なら省略。RETEST PASS |
| F-05 | P2 | executorの各更新で全状態をcloneし、全mapを比較する | 一時的な変更キー集合で保存。Errでも保存し、保存失敗はtrapする。stable形式・MemoryIdは維持。RETEST PASS |
| F-06 | P2 | awaitのないverdict-engineで更新ごとにsnapshot全体を直列化する | init/pre_upgradeだけでsnapshotを保存。ローカルupgradeで確認。RETEST PASS |
| F-07 | P1 | 既存INT8モデルが棄権から具体クラスへ反転する | `test_oos_00033`はF32で棄権、INT8重みだけの独立参照でも具体クラスへ反転。量子化形式の修正は範囲外。deferred |
| F-08 | P2 | 過去のper-row INT8性能値が現行block-32の説明に混在する | 現行実測と履歴を区別し、query上限の未再測定を明記。RETEST PASS |

## 計測結果

同一モデル・tokenizer・著者データのSHA256は [baseline.json](artifacts/efficiency/baseline.json)、
比較に使用した全入力が変更されていないことも[確認済み](artifacts/efficiency/unchanged-inputs.json)。
測定用・最終Wasmとnative binaryのハッシュは [provenance.json](artifacts/efficiency/provenance.json) に記録した。
管理されたタスク専用ローカルreplicaのport 8012を使用した。ユーザーのproject設定・port 8011は変更していない。
ローカル検証完了後に専用networkを停止し、生成した一時PEMを削除した。canisterの状態は保持している。

| 指標 | 旧版 | 改善版・候補 | 判定 |
|---|---:|---:|---|
| warm-up後のWasmメモリ最大到達量 | 355,729,408 B | 185,925,632 B | 47.7%減 |
| T=120推論命令数 | 39,782,583,123 | 39,769,124,903 | 約0.034%減 |
| 7入力のlogits | 比較元 | 全て完全一致 | PASS |
| T=128 | ガード拒否、profile経路でもIC0522 | 同左 | 上限変更なし |
| block32 2×4 | 標準kernel | 命令数182〜184%増 | 不採用 |
| block32 4×4 | 標準kernel | 命令数13.2〜13.7%増 | 不採用 |
| block32 8×4 | 標準kernel | 命令数73.5〜74.4%増 | 不採用 |
| 局所attention QK/softmax/AV | dense参照 | 命令数4.33〜10.04倍 | 不採用 |

[7入力の比較](artifacts/efficiency/comparison.json)、[最終WasmのT=120/98再計測](artifacts/efficiency/final-comparison.json)、[84点のkernel計測](artifacts/efficiency/kernels.json)、
[14点のattention計測](artifacts/efficiency/attention.json)、[Wasm端数tileの検証](artifacts/efficiency/wasm-tail-check.json)を参照。
メモリ値はWasm linear memoryの最大到達量であり、解放後のlive heapではない。
初回7入力比較時の候補heapは187,695,104 Bだった。上表は最終Wasmへのupgrade後に再warm-upした値を示す。
同一条件でのwarm-up時間は未計測。executorは履歴1/64/512件で変更キー1件を検証したが、
実canister上の永続化命令数比は未計測である。変更キー数kに応じて保存するため、全map比較は発生しない。
候補採用基準は中核処理5%以上改善・end-to-end悪化2%以内・品質gate合格であり、今回は満たす候補がなかった。

## Retest Log

| 対象 | コマンド・方法 | 結果・記録 |
|---|---|---|
| Rust workspace | `cargo test --workspace` | PASS、[log](artifacts/efficiency/rust-tests.log)。外部モデルgoldenは既定でignored、下記の直接実行で検査 |
| 差分永続化追加検査 | `cargo test -p executor --lib` | PASS、[log](artifacts/efficiency/executor-persistence-tests.log) |
| lint | `cargo clippy --workspace --all-targets -- -D warnings` | PASS、[log](artifacts/efficiency/clippy.log)。追加executor検査も[PASS](artifacts/efficiency/executor-clippy.log) |
| Python | `python3 -m unittest discover -s tests` | 34件PASS、[log](artifacts/efficiency/python-tests.log) |
| 4 canister | `bash tools/build_one.sh {decision-engine,verdict-engine,executor,mock-ledger}`を個別実行 | 全てPASS。最終verdict再buildの[log](artifacts/efficiency/verdict-build.log)。既存SIMD関数のunused_unsafe警告4件あり |
| mock workflow | `python3 tools/local_integration.py --project-root /tmp/ic-laya-efficiency-local --port 8012 --keep` | PASS。返信消失・重複排除・予約保持・権限拒否。qrun成功時に詳細一時logが削除されたため、永続logはない |
| executor旧版→新版 | 同上に `--build-dir /tmp/ic-laya-efficiency-baseline --upgrade-executor-wasm build/executor.wasm` | PASS。旧版は別source/target-dirから構築し、異なるWasm hashを確認。未確定送金中のupgrade拒否、park後の予約保持を検証 |
| verdict旧版→新版 | `tools/verdict_regression.py --project /tmp/ic-laya-efficiency-local --identity ic-laya-efficiency-owner --controller ic-laya-local-test --canister upgrade-fixture` | PASS、[log](artifacts/efficiency/verdict-regression.log)。小型fixtureを新規canisterへ投入して実行 |
| full品質gate | `target/release/verdict-infer check --pack models/verdict-pack --tokenizer models/verdict-151m/tokenizer.json --cases models/verdict-parity/cases.jsonl --predictions models/verdict-parity/predictions.jsonl --quiet` | FAIL（危険反転1件）、[全件log](artifacts/efficiency/quality-full.log)、[集計](artifacts/efficiency/quality-summary.json) |
| 最終CLIの品質判定 | 件数不足・1件診断・既知の危険反転 | 期待したexitを確認、[log](artifacts/efficiency/quality-cli-smoke.log) |
| 不一致の切り分け | `python3 tools/diagnose_verdict.py --out artifacts/efficiency/diagnosis.json` | 3例をF32 / INT8重みのみ / 重み+activation量子化 / nativeで比較。[結果](artifacts/efficiency/diagnosis.json) |

## E2E Candidates / Untested / Deferred Scope

- 計画範囲のcanister操作はCLIで検証済み。UIの変更はない。
- F-07の量子化方式変更や再校正は別作業。NumPy参照は独立したF32 BLAS累積であり、本番kernelとのbit一致を主張しない。
- 全件検証は数値演算が同じ検証中binaryで完走した。最終binaryでは判定unitと拒否動作を再確認した。
- 全1,000件をWasmで実行したわけではない。Wasm旧新版比較は7入力で、全件gateはnativeで行う。
- T=121〜127、block-32 queryの境界、最大登録件数での実canister命令数は未計測。
- cost model更新・本番配備・commit/pushは未実施。
- テストcanisterのreinstallは自動承認レビューで拒否されたため実行せず、新規の空canisterでupgradeを検証した。削除・初期化による回避はしていない。

## Open Questions

実装を妨げる未確定事項はない。品質gateが通るまでは本番投入を再開しない。
