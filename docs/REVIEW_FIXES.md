# レビュー指摘の処置台帳

対象レビュー: 2026-09-21、HEAD `e701ad2` + 作業ツリー。重大度は P0（実資金/本番破壊）、
P1（通常経路の重大欠陥・検証の空振り）、P2（条件付き不具合・文書と証拠の不一致）。

**並行作業について**: レビュー中に別セッションが同じ修正を並行実装していた
（`canisters/verdict-engine/src/lib.rs`、`tools/verdict-upload/src/main.rs`、
`docs/VERDICT_ENGINE.md`）。競合を避けるため、こちらは**相手が触っていないファイル**だけを
修正し、`verdict-engine` のコードは相手の実装（予算ガードの受理上限120）を採用した。
そのため #2・#3・#7・#8・#9・#14・#16・#17 は未処置のまま残っている。

## 処置一覧

| # | 重大度 | 指摘 | 処置 | 検証 |
|---|---|---|---|---|
| 1 | P0 | `tools/verdict_canister.py` に実網ガードが無く、`--env` 次第で 100 ICP 送金と `-m reinstall` が可能。根拠にしていた `icp.yaml` のマッピングは存在しない | **修正** `tools/icp_guard.py` を新設し、`local_integration.py`/`measure_inference.py` の重複定義を統合。`Icp.run` の choke point（state-changing コマンドのみ）+ 両 main の事前チェック + `--replica` の loopback 検査（`--env local --replica https://ic0.app` の迂回を閉鎖） | `tests/test_icp_guard.py` 10件PASS。ガードを無効化した状態で transfer が実行されることを実測（修正前FAIL相当） |
| 2 | P1 | 予算ガードの既定値が文書の「T=120が成功」と矛盾（旧既定は実効上限117） | **相手セッションの実装を採用**（`COST_FIXED=254_400_000`/`COST_PER_TOKEN=327_600_000`/margin 1005 → 上限120）。`max_accepted_tokens` と `ProfileReply.would_accept` は未実装 | `projected(120)=39.76e9 ≤ 40e9`、`projected(121)=40.09e9 > 40e9` を算術で確認 |
| 3 | P1 | 118/119/126/128 の測定行が `artifacts/verdict_sweep.json` に存在しない（記録は1点のみ、sweep ログは失敗run） | **未処置**（replica + 605 MiB の再測定が必要） | — |
| 4 | P1 | `tools/verify.py` は要求したチェックが NOT_RUN でも exit 0 | **修正** 要求フラグごとに `requested` を定義し、NOT_RUN を失敗として exit 1。フラグ無しは exit 0 のまま `NOTHING VERIFIED` を明示 | `tests/test_verify_contract.py` 4件PASS（ツールチェーン不在で exit 1、フラグ無しで exit 0） |
| 5 | P1 | CI が verdict 系を一切ビルドせず、`generate-lockfile` → `--locked` で lock 検査が無効 | **修正** `verdict-engine` の wasm ビルド追加、`generate-lockfile` 削除、`verify.py --rust --require-rust` を集約ゲートとして追加、native job に wasm target | `ci.yml` 差分。未追跡 member の commit は承認が必要（未実施） |
| 6 | P2 | `PERFORMANCE_MEASUREMENTS.md:381` が同 doc `:170` で撤回した 1.43 instr/MAC を再利用。下限も 0.5 と 0.25 の二重定義（10倍/20倍） | **修正** 1.43 を encoder 単独 4.13/4.12 に、下限を 0.5 に統一、mlp_up を「下限の約2倍」、全体を「約8〜10倍」に | #22 の再計算値と一致 |
| 7 | P2 | `VERDICT_ENGINE.md:17-20` と `:175-180` で同一1000件の分布が異なる | **未処置**（同ファイルは相手セッションが編集中） | — |
| 8 | P2 | 検証件数の stale（README:7 の62件、STATUS:11,15,19 の45件/3 canister、VERDICT_ENGINE:83 の8件） | **未処置**（README/STATUS/VERDICT_ENGINE は競合リスク） | 実測は `cargo test --workspace` = 75 passed / 1 ignored、Python 50件、canister 4 |
| 9 | P2 | `README.md:16-24` の「今回の実行検証」表が v0.1 のまま（Rust未実行/Wasm未ビルド/ICP境界未実行） | **未処置**（同上） | artifacts と README:7 が反証 |
| 10 | P2 | `tools/local_integration.py` の `check(True, ...)` 10箇所は失敗し得ない（`icp canister call` は Candid `Err` でも exit 0） | **修正** `Icp.call(expect_ok=True)` が `(variant { Err ... })` を検出して例外化（anchored regex、Err を期待する呼出しは `expect_ok=False`） | 正例3/誤検出0/Err 2/2 検出を実測。`local_integration.py` と `measure_inference.py` の両方 |
| 11 | P2 | `measure_inference.py`/`measure_phases.py` は全 tier 失敗でも exit 0 | **修正** `harness_error` を検出したら artifact は書いた上で exit 1 | py_compile + 差分 |
| 12 | P2 | owner 秘密鍵（canister owner 唯一の鍵）を既定パーミッションで作成 | **修正** 新規は `os.open(..., 0o600)`、既存は `chmod 0600`、`--home` は 0700 | 新規=0600 / 既存の再締め=0600 を実測 |
| 13 | P2 | `MANIFEST.sha256` は生成器も検証者も無く、89件中27件が不一致・57件が未収載 | **修正** `tools/manifest.py --write/--check` を追加し `verify.py --manifest` から検査。追跡ファイル146件で再生成、0 stale / 0 missing | `--check` PASS |
| 14 | P2 | `verdict-upload` の `begin_upload` 応答を decode せず成功表示 | **未処置**（相手セッションが同ファイルを編集中） | — |
| 15 | P2 | `Calibration.test_only` がどこからも読まれず「fixture calibration は実資金を有効化できない」が未強制 | **受容（未到達）** `LimitedLive` は `set_mode` が拒否し、live dispatch 経路が存在しないため現状は到達不能。live 有効化時に engine 側で強制する必要がある（`docs/REVIEW_FIXES.md` の「live 前の必須項目」） | 全文grepで参照が定義と demo 生成のみ |
| 16 | P2 | `decide` が label 数と class slot 数の一致を検査せず、`<<LABEL>>` 注入で panic/誤選択 | **未処置**（`verdict-engine` 占有） | — |
| 17 | P2 | `verdict-engine` に per-caller quota が無く、allowlist 済み1主体が 40B 命令/call を無制限に消費 | **未処置**（同上） | — |
| 18 | P2 | `pack_verdict.py` が tensor 長を shape から検算しない | **修正** `length != 4*numel(shape)` を tensor 名付きで拒否 | `tests/test_pack_verdict.py`。チェック無効時は exit 0 で不整合 pack を書くことを実測 |
| 19 | P2 | `--random --source-revision <40hex>`（`--test` なし）で `test_only=false` の pack が作れる | **修正** `--random` は `--test` 必須 | 同上（4テスト） |
| 20 | P2 | `projector_hidden_act` の無言 Relu 化、`normalize_features=true` での `logit_scale` 無言除外、`layer_types` 無視 | **修正** いずれも明示エラー化 | 同上（activation/normalize/layer_types 各1テスト） |
| 21 | P2 | tier fixture の seed が `hash(name)` でプロセス毎に変わる | **修正** `zlib.crc32(name)` | `PYTHONHASHSEED` を変えても同一値を実測 |
| 22 | P2 | `measure_phases.encoder_macs` が window を全層に課し、鍵数に `local_attention` を使用 | **修正** `i % global_every != 0` の層のみ window、鍵数は `distance=local_attention//2` の包括範囲（端のqueryも考慮） | `tests/test_measure_phases.py` 7件PASS。分母は measure-m で 2,214,592,512→2,202,206,208（-0.56%）、instr/MAC は 4.94→4.97 |
| 23 | P2 | `ReservationOracle.invariant()` が bare `assert`、625トレース列挙に unittest 断言が無い | **修正** `raise ValueError` 化 + トレース数の断言 + 強制テスト | `python -O` でも同一結果、`assert False` が `-O` で消えることを実測 |
| 24 | P2 | `laya_port_bridge.fetch` が Range 応答の status/長さを検査しない | **修正** `read(limit)` で上限を切り、206 と長さを検査 | 差分。803 MiB 取得は未実施（メモリ膨張は仮説のまま） |
| 25 | P1 | `crates/laya-candle/tests/parity.rs` の許容差 1e-3 がクラス間 spread（2.3e-5〜3.0e-5）より大きく、軸誤りを検出できない | **修正** 実測偏差 1.118e-8 に基づき `5e-7 + 1e-5|b|`。`許容差 < spread/10` をテスト内で断言 | `cargo test -p laya-candle --test parity` 5件PASS |
| 26 | P2 | Laya decision head の activation が producer 間で矛盾（bridge=Gelu / fixture=Relu / FINAL_SPEC・V2-R11=ReLU） | **修正** bridge は PyTorch 既定の ReLU を採用（upstream が明示した場合のみ上書き）し、`port_report.json` の未解決事項に記録 | `translate_config` の出力が Relu になることを実測 |
| 27 | P2 | `GLICLASS_FORWARD_SPEC.md:322` の「`T = 1.0` as shipped」が同 doc `:32`/`:313`・ship済み calibrator・`verdict_infer.rs:145` と矛盾 | **修正** 著者 artifact（1.0）と ship済み calibrator（1.4265148639678955）を区別して明記 | 差分 |
| 28 | P2 | `golden.rs` が `skipped_no_truth` を拘束せず、predictions 1行でも gate がPASS | **修正** `skipped_no_truth==0` と `cases >= 1000` を要求 | 差分（605 MiB pack が必要なため実行は未実施） |

## 検証コマンドと結果

```
CARGO_HOME=$PWD/.cargohome cargo test --workspace --offline   # 75 passed / 0 failed / 1 ignored
cargo test -p laya-candle --test parity                        # 5 passed（締めた許容差で）
python3 -m unittest tests.test_icp_guard tests.test_verify_contract \
       tests.test_pack_verdict tests.test_measure_phases tests.test_reference
                                                              # 失敗は torch 未導入の1件のみ（環境要因）
python3 tools/manifest.py --check                             # 146 tracked files, 0 stale, 0 missing
```

## 実資金を動かす前に必要な残項目（優先順）

1. **#3 の再測定**: replica を起動し 605 MiB pack を投入して `--sweep 118,119,120,121,126` を実行し、
   `artifacts/verdict_sweep.json` を更新。文書の上限表はこの artifact の値だけを引用する。
2. **#16・#17（verdict-engine）**: `decide` の class slot 数検査と per-caller quota。
   どちらも `verdict-engine` の API/state 変更を伴うため、次の1回の reinstall に同梱する。
3. **#2 の残り**: `EngineInfo.max_accepted_tokens` と `ProfileReply.would_accept`（測定値が serving 側で
   拒否される長さかを artifact に残す）。
4. **#15**: live dispatch を有効化する前に `test_only` calibration を engine 側で拒否する。
5. **Laya 実pack の primitive→qtype 行順**: 参照と Rust が同じ不透明 index を使うため現状は反証不能。
   実checkpointで live を開く前の人手確認事項（`README.md` 6節、`docs/MODEL_PORT_FINDINGS.md:5` が
   未検証と明記済み）。
6. **#8・#9・#7**: README / IMPLEMENTATION_STATUS / VERDICT_ENGINE の数値と状態表の更新
   （並行セッションの編集が止まってから）。
