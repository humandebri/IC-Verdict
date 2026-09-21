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
| 実checkpoint inference品質 | Layaは未測定（`fixtures/`はランダムweight）。**openJev 151Mは著者記録の1000件でargmax 1000/1000一致**（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 3.1節） |
| ICP heap/instructions/cycles | **instructionsは実測済み**（Laya合成pack、openJev実checkpoint）。heap（warm 2.5GiB / cold peak 3.0GiB）はcanisterのheapを読む口がなく**未測定** |
| openJev 151Mの実測上限 | **T=120トークンが成功、T=126が40B上限で拒否**（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 5.1.1節）。実benchmarkの入力長は**中央値95トークン**で、**1000件中975件（97.5%）が予算内**。超過は121〜150の25件のみ |
| local replica / PocketIC | **PASS**。`tools/local_integration.py`が`icp` CLIの管理networkで3 canisterを実行 |
| 本番ledger / live transfer | `LimitedLive`はコードで拒否。実asset接続機能は未有効化 |
| Human review承認再開 | NeedsReviewで停止する。承認endpointは未実装 |
| temperature fitとholdout校正 | 受入機構のみ。**上流はprimitive別・候補数別のtemperatureを持つ**が現行型はスカラー。仕様判断が未解決 |
| stable table / compaction | bounded snapshotのみ。削除なし、上限で停止 |
| INT8 / SIMD専用kernel | **実装中**。ADR-017で方針確定（SIMDはICPで実行可能と実測済み）。蒸留は後段 |

## 性能について

**測定した。** `tools/measure_inference.py` がlocal replica上で合成packの `measured_instructions` を実測した（詳細は[PERFORMANCE_MEASUREMENTS.md](PERFORMANCE_MEASUREMENTS.md)）。

| tier | hidden | 層 | 質問あたり instructions |
|---|---|---|---|
| measure-s | 128 | 2 | 193M〜228M |
| measure-m | 512 | 4 | 4.73B〜5.43B |

これを実checkpoint（28層・hidden 1024・128 tokens）へ外挿すると **494B〜567B instructions**。**設計目標20Bの25〜28倍、ICPのupdate上限40Bの12〜14倍**である。

つまり **F32のままでは実checkpointは載らない**。必要な削減は最低12倍で、INT8化で見込める4倍では足りない。INT8と蒸留の併用、または層数・hiddenの再検討が必要になる。ただし「Scoreを削る」「尺度説明を短縮する」「入力を切る」といった意味を削る最適化は、この結果を理由にしても認められない。

### openJev 151Mは外挿ではなく実測で上限を押さえた

`tools/measure_verdict.py` が**実checkpoint（605 MiB pack、151M params）**を投入し、
実benchmarkの1件（自然長118トークン）を118/119/120/126/128トークンで測った:

| T | instructions | 判定 |
|---|---|---|
| 118 | 38,998,851,854 | 予算内 |
| 119 | 39,293,820,829 | 予算内 |
| 120 | **39,568,299,856** | **予算内（成功した最長）** |
| 126 / 128 | — | **replicaが40B上限で拒否（IC0522）** |

**上限はT=120と126の間**である。`VERDICT_ENGINE.md` 5.1節の4点から出した外挿 `T ≈ 123` は
この範囲に入っており、外挿としては妥当だったが、**上限値は外挿ではなくこの実測で押さえる**
（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 5.1.1節、`artifacts/verdict_sweep.json`）。
151Mは量子化なしでも**実benchmarkの97.5%（中央値95トークン）が1 callに収まる**。
当初「平均383トークンで3.2倍超過」と書いていたが、383はJevBench生入力の素朴な計数であり、
この実装が食わせるprompt（選択肢の説明文を含む）の長さではない。**測り直した値が上の分布である。**

### ボトルネックを位相別に特定した（openJev実checkpoint、T=120）

`infer_profiled`（owner専用、`instruction_counter`を位相境界で読む）で内訳を測った。
**同一入力で2回測って同じ配分**である（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 5.2節）:

| 位相 | 割合 | instr/MAC |
|---|---|---|
| `attn.pre`（qkv射影＋RoPE＋head整形） | **33.0%** | 2.79 |
| `layer.mlp_up`（Wi＋GELU） | **31.9%** | 2.70 |
| `layer.mlp_down`（Wo） | 15.8% | 2.68 |
| `layer.attn`（out射影） | 10.5% | 2.66 |
| `attn.core`（scores＋softmax＋AV） | 6.8% | 5.52 |
| norm・活性化・embedding・projector・decode | 2.1% | — |

**密なmatmulが92%**で、norm・softmax・gather・decodeは2%未満。しかも**どのmatmulも
2.66〜2.79 instr/MACとほぼ一定**で、f32x4の下限0.5の約5.5倍で回っている。

**ただし「5.5倍の伸びしろ」ではない。** 差の大半は計算ではなく**演算ごとの周辺コスト**
（tensor確保、`narrow`/`transpose`/`contiguous`のコピー、カーネル起動）である。
理論MACがほぼ同じ `attn.pre`(2.79) と `layer.attn`(2.66) の差がその証拠で、
必要なのは「速いgemm」ではなく「**演算の融合**」であり、見込みは**2〜3倍**である。
最長150トークンは実測の傾き（2.85e8/token）から**約42.7B instructions（40Bの1.07倍）**で、
裾は約7%の超過にとどまる。それでも**裾を消すにはカーネル改善ではなくINT8・蒸留・入力契約の設計が要る**。

**内訳を実測した**（`artifacts/phase_measurements.json`）: encoderが**83%**、decisionが16%、
softmax/norm/活性化/gather/decodeは合計1%未満。ADR-010の「hot linearへ適用」は裏付けられた。

**ただし効率はすでに最適に近い**: encoder **1.43** instructions/MAC、decision **0.92**。
スカラーなら3〜4、SIMDなら1前後が下限なので、**現状は下限から1.4倍以内**である。
したがって INT8/SIMD の現実的な伸びは**最大3〜4倍**で、×4を当てても421Mは127B（40Bの**3.2倍超過**）。

**サブフェーズまで特定した**（128-token profile、6層h768）: **MLP 51%**、attention 31%、
decision head 16%、その他 約2%。

**効率について当初の記載は誤っていた。** 測定ツールが理論MACを128 tokens固定で計算する一方、
実際の入力は27〜38 tokensだったため **instructions/MAC を約4倍過小評価**していた。
修正して128-token入力を実際に測ると **約4.8〜5.2 instr/MAC**（encoder）で、
**f32x4の理論下限0.25の約20倍**。つまり**カーネル改善の余地は大きい**（当初「最適に近い」と
書いたのは誤り）。

**壁時計が本質的な制約**（公式: update 40B、query 5B、目標2B instructions/秒）:

| 応答時間 | 必要なinstructions |
|---|---|
| 3秒 | 6B以下 |
| 10秒 | 20B以下 |

6層h768は128-token profileで**79.8B（約40秒）**。単独では上限40Bも超える。
**カーネル改善（×5）とINT8（×3）の併用**で数秒台に入る見込み。

[ADR-017](design-v2/adr/ADR-017.md)に記録した。

**未測定**: heap使用量（`warm 2.5GiB` / `cold peak 3.0GiB`）はcanisterのheapを読む口がなく未測定。1.57 GiB packの投入も未実施（CLI経由のアップロードはargv長制約で256 KiB chunkが上限、約6400回の呼び出しになり非現実的。これはcanister側ではなくクライアント側の制約）。
