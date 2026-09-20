# 実装状態 v0.2

v0.1からの差分: Rust toolchainのある環境でビルドとテストを実行し、判明した不具合を修正した。**「ソース納品・未コンパイル」ではなくなった。** ただし実Laya checkpoint、ICP実測、実送金は引き続き未検証である。

## この環境で実際に実行した検証

実行環境: macOS 26.5.2 (arm64), rustc/cargo 1.97.1, `wasm32-unknown-unknown`, Python 3.14.7 + torch 2.14.0。

| 検証 | 結果 |
|---|---|
| `cargo test --workspace` | **PASS 62件** (ic-laya-core 57 / laya-candle parity 4 / compile_fail doctest 1) |
| `cargo check -p decision-engine --features candle` (native) | **PASS** |
| `cargo build -p decision-engine --features candle --target wasm32-unknown-unknown` | **PASS** |
| `bash tools/build_one.sh` × decision-engine / executor / mock-ledger | **PASS**。Candid 3件と Wasm 3件を生成 |
| `IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine` | **PASS** (Wasm 4.8 MiB, Candle込み) |
| `cargo run -p ic-laya-core --example mock_workflow` | **PASS**。三primitive逐次評価 → mock送金 → `Succeeded`, reserved=0/spent=110 |
| `cargo run -p laya-candle --bin laya-infer -- fixtures/tiny-prenorm ...` | **PASS**。合成weightでlogitsを出力 |
| `tools/local_integration.py`（local replicaで3 canister実行） | **PASS**。下記のworkflow全体を実機で確認 |
| Python unittest | **PASS 45件** |
| `tools/verify.py --rust --require-rust` | 下表参照 |

### local replica統合試験で確認したこと

`icp` CLI の管理networkに3 canisterをinstallし、実際のmessage boundary越しに動作を確認した。

1. engine fixture mode、executorのcaller登録、schema 3件のcompile、calibration 3件、plan、operation、grant、mock ledger handshakeがすべて成立。
2. 最大3質問の逐次評価が通り、3件目のScoreまで揃って`ReadyToDispatch`に到達。
3. mock ledgerの**`CommitThenCallbackTrap`**（commit後にmessage boundaryを跨いでreplyを失う故障）を撃ち、executorは`Submitted`のままでも成功扱いでもなく**`OutcomeUnknown`**へ遷移。予約は保持。
4. 同じfrozen payloadでretryし、ledgerが重複として検出。`committed_transfers`は**1のまま**（二重送金なし）で、requestは`Succeeded`に確定。

これは「結果不明を失敗扱いして新しいtimestampで送り直さない」という設計主張が、実replica上で成立することを示す。`cargo test`では到達できない領域である。

**注意:** 使用したledgerはテストダブルであり、実assetは動いていない。`LimitedLive`は依然コードで拒否される。

`AuthorizedTransfer`が生成Candidに一切現れないことを確認した。`dispatch`はidのみを受け取り内部で再認可するため、型による誤接続防止は実際のAPI境界でも成立している。

## v0.1の記述を訂正した点

v0.1は「Rust未コンパイル、未確認」としていた。実際にビルドした結果、**3件の実バグ**が出たので修正した。

1. **`tools/build_one.sh`がbash 3.2で全滅** — `features=()` を `"${features[@]}"` で展開すると、macOS同梱のbash 3.2は`set -u`下で空配列をunboundとして扱う。`features[@]: unbound variable`でCandle以外の全ビルドが即死していた。`run()`経由の条件分岐に変更。
2. **`build_one.sh`が`CARGO_TARGET_DIR`を無視** — artifactを`target/...`固定で探すため、target dirを上書きした環境では`cp`が失敗。`cargo metadata`から実際の`target_directory`を解決するよう変更。
3. **`--features candle`のWasmビルドが不可能だった** — `candle-core`→`rand`→`rand_core`、および`tokenizers`→`ahash`が`getrandom 0.3`を非optionalに引き込み、`getrandom 0.3`は`wasm32-unknown-unknown`でbackendを持たないためE0425で失敗していた。`.cargo/config.toml`で`--cfg getrandom_backend="custom"`をwasm32に限定して指定し、`canisters/decision-engine/src/getrandom_ic.rs`で`__getrandom_v03_custom`を実装した。**暗号学的強度の限界はそのファイルのdocコメントに明記**した（`raw_rand`の代用ではない。用途はcandleのtensor初期化とahashのseedのみ）。

また、**toolchain依存**が判明したため`rust-toolchain.toml`で1.97.1に固定した。`candle-core 0.11.0`はaarch64で`fp16` target featureが有効だと`stdarch_neon_f16`を使うが、これは1.93.0では未安定のためE0658で失敗し、1.97.1では通る。wasm32は`cpu/neon.rs`をコンパイルしないためcanister buildは影響を受けない。

## 未完了・未確認

| 項目 | 状態 / 次の作業 |
|---|---|
| 実Laya weights | 未同梱・未ロード。**構造の突き合わせは完了**（[MODEL_PORT_FINDINGS.md](MODEL_PORT_FINDINGS.md)）。803 MiBの取得とexportは未実施 |
| upstream tensor/tokenizer/qtype一致 | **名前とshapeは全206 tensorで一致**。ただしQKV順・RoPE・prompt形式など数値parityの前提は未確認 |
| 実checkpoint inference品質 | 未測定。`fixtures/`はランダムweightで言語理解を証明しない |
| ICP heap/instructions/cycles | **未測定**。20B instructions / 2.5GiB heapの受入目標は未検証。F32 pack 1.57 GiBは実測済み |
| local replica / PocketIC | **PASS**。`tools/local_integration.py`が`icp` CLIの管理networkで3 canisterを実行 |
| 本番ledger / live transfer | `LimitedLive`はコードで拒否。実asset接続機能は未有効化 |
| Human review承認再開 | NeedsReviewで停止する。承認endpointは未実装 |
| temperature fitとholdout校正 | 受入機構のみ。**上流はprimitive別・候補数別のtemperatureを持つ**が現行型はスカラー。仕様判断が未解決 |
| stable table / compaction | bounded snapshotのみ。削除なし、上限で停止 |
| INT8 / SIMD専用kernel / 蒸留 | 未実装 |

## 性能について

ビルドとテストが通ったことは、性能目標を満たしたことを意味しない。**1質問20B instructions以下、warm heap 2.5GiB以下、cold peak 3.0GiB以下の実測は依然として無い。** Candle F32の421MモデルがICPのinstruction上限に収まる保証はどこにも無く、次に測るべき対象である。
