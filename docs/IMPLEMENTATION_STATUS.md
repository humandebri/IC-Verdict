# 実装状態 v0.2

v0.1からの差分: Rust/Wasm/local replicaを実行検証し、openJev実checkpoint parityとICP実測を完了した。実送金は引き続き無効である。

## 未配備: INT8 pack全面移行

`ic-verdict-int8-pack-v1`を実装し、全2次元重み（embeddingを含む）をblock-32 INT8、Norm・bias・scaleだけをF32補助値として保存する。生成packは **170,408,640 bytes**（旧F32 605,512,704 bytesの28.1%）で、旧F32 manifestと2次元F32 entryはloaderが拒否する。Wasm buildと通常テストは通過した。

著者記録1000件とのargmax一致は **997/1000**（per-row max-absは996/1000）で、許容された99.5%基準は通過する。しかし3反転中1件が`__insufficient_evidence__`から`lost_or_stolen_phone`への変化であり、「危険な反転ゼロ」の安全性ゲートには不合格である。したがって本番配備、INT8専用temperatureの再校正、実canisterでのcost model更新は停止中であり、以下の1000/1000記録は従来F32基準の記録である。

## この環境で実際に実行した検証

実行環境: macOS (arm64), rustc/cargo 1.97.1, `wasm32-unknown-unknown`, Python 3.12.14（torchは未導入。依存していた検査はLaya削除時に撤去済み）。

| 検証 | 結果 |
|---|---|
| `cargo test --workspace` | **PASS 93件**（Layaバックエンド削除後。decision-engine は fixture モード。query経路のguardテスト5件を含む） |
| `bash tools/build_one.sh` × decision-engine / executor / mock-ledger / verdict-engine | **PASS**。Candid 4件と Wasm 4件を生成 |
| `bash tools/build_one.sh verdict-engine` | **PASS**（block-32 INT8専用pack＋Wasm SIMD kernel） |
| `cargo run -p ic-laya-core --example mock_workflow` | **PASS**。三primitive逐次評価 → mock送金 → `Succeeded`, reserved=0/spent=110 |
| `tools/local_integration.py`（local replicaで3 canister実行） | **PASS**。下記のworkflow全体を実機で確認 |
| Python unittest | **PASS 33件**（query経路の契約テスト10件を含む） |
| openJev実checkpoint parity（`verdict-infer check --limit 1000`） | **PASS 1000/1000**（最大偏差 5e-6） |
| query経路（`infer_tokens_query` / `query_limits`、ローカルreplica実測） | **PASS**。上限は既定F32で**14トークン**（T=14: 4.33e9、T=15は`Capacity`）、int8ビルドでは**40トークン**。queryとupdateのlogits一致、query前後で`info`不変も確認 |
| `tools/verify.py --rust --require-rust --manifest --verdict --local-integration` | **PASS 14/17行**。残る`NOT_RUN`は`openjev_canister_instructions`と`openjev_query_canister`（どちらも温まったreplicaが要る）、`real_ledger_transfer`（実送金）のみ |

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
3. **（歴史）`--features candle`のWasmビルドが不可能だった** — `candle-core`→`rand`→`rand_core`、および`tokenizers`→`ahash`が`getrandom 0.3`を非optionalに引き込み、`getrandom 0.3`は`wasm32-unknown-unknown`でbackendを持たないためE0425で失敗していた。`.cargo/config.toml`で`--cfg getrandom_backend="custom"`をwasm32に限定して指定し、`canisters/decision-engine/src/getrandom_ic.rs`で`__getrandom_v03_custom`を実装した。**暗号学的強度の限界はそのファイルのdocコメントに明記**した（`raw_rand`の代用ではない。用途はcandleのtensor初期化とahashのseedのみ）。**このfeatureはLayaバックエンドとともに削除済み**（本項は当時の記録）。

また、**toolchain依存**が判明したため`rust-toolchain.toml`で1.97.1に固定した。`candle-core 0.11.0`はaarch64で`fp16` target featureが有効だと`stdarch_neon_f16`を使うが、これは1.93.0では未安定のためE0658で失敗し、1.97.1では通る。wasm32は`cpu/neon.rs`をコンパイルしないためcanister buildは影響を受けない。

## 今回直した不具合: replica停止状態から起動できない

`icp_guard`をfail-closedにした際、3つのツールが `require_local_network`（`managed: true` の確認）を
`network start` より**前**に呼ぶようになり、replicaが停止している状態では必ず
「cannot confirm that environment 'local' is a locally launched replica」で拒否されていた。
`network start` はguard対象外（cyclesもstateも動かさない）なので、**起動を先に行い、確認を
その直後・cycles mint / install / call の前**に移した。

* 対象: `tools/local_integration.py`、`tools/verdict_canister.py`、`tools/measure_verdict.py`。
* `local_integration.py` はさらに、**到達可能だがmanagedでない**ネットワーク（`--env ic` など）では
  起動を試みずに従来どおり拒否する（`any_network_status` で区別）。
* 検証: `tools/verify.py --local-integration` が停止状態から起動して **PASS**（`artifacts/local_integration.log`
  の先頭が "starting local network ..."）。既存の`tests/test_icp_guard.py` 23件もPASS。
* 副次: `measure_verdict.py --out` にリポジトリ外のパスを渡すと、書き込み成功後に
  `relative_to` でtracebackしていたため、絶対パスを表示するよう修正。

## 未完了・未確認

| 項目 | 状態 / 次の作業 |
|---|---|
| Layaバックエンド | **削除済み**（`laya-candle` crate・decision-engine の `candle` feature・Laya計測ツール）。weightは未取得のままで、合成weightからの外挿が40B上限の12〜14倍だった。記録は[archive/](archive/)に退避 |
| 実checkpoint inference品質 | Layaは未測定（`fixtures/`はランダムweight）。**openJev 151Mは著者記録の1000件でargmax 1000/1000一致**（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 3.1節） |
| ICP heap/instructions/cycles | **instructionsは実測済み**（Laya合成pack、openJev実checkpoint）。heapは`heap_bytes`で実checkpoint warm時 **1.02 GiB**（1,099,694,080 B）を記録済み（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 5.1.2節。今回の再測は未実施）。cyclesは未計測 |
| openJev 151Mの実測上限 | **T=120トークンが成功、T=126が40B上限で拒否**（[VERDICT_ENGINE.md](VERDICT_ENGINE.md) 5.1.1節）。実benchmarkの入力長は**中央値95トークン**で、**1000件中975件（97.5%）が予算内**。超過は121〜150の25件のみ |
| local replica / PocketIC | **PASS**。`tools/local_integration.py`が`icp` CLIの管理networkで3 canisterを実行 |
| 本番ledger / live transfer | `LimitedLive`はコードで拒否。実asset接続機能は未有効化 |
| Human review承認再開 | NeedsReviewで停止する。承認endpointは未実装 |
| temperature fitとholdout校正 | 受入機構のみ。**上流はprimitive別・候補数別のtemperatureを持つ**が現行型はスカラー。仕様判断が未解決 |
| stable table / compaction | bounded snapshotのみ。削除なし、上限で停止 |
| INT8 / SIMD専用kernel | **実装済み、安全性ゲート未通過**。block-32でargmax 997/1000（99.5%基準は通過）だが危険な反転1件。過去per-row kernelはT=120で14.57e9 instructions、block-32は未測定 |

## 性能について

**測定した。** `tools/measure_inference.py` がlocal replica上で合成packの `measured_instructions` を実測した（詳細は[PERFORMANCE_MEASUREMENTS.md](archive/PERFORMANCE_MEASUREMENTS.md)）。

| tier | hidden | 層 | 質問あたり instructions |
|---|---|---|---|
| measure-s | 128 | 2 | 193M〜228M |
| measure-m | 512 | 4 | 4.73B〜5.43B |

これを実checkpoint（28層・hidden 1024・128 tokens）へ外挿すると **494B〜567B instructions**。**設計目標20Bの25〜28倍、ICPのupdate上限40Bの12〜14倍**である。

> **履歴**: この外挿を根拠にLayaバックエンドは削除した。ただし外挿は当時の効率（gemm f32）が前提で、現在のカーネル最適化（int8 8行カーネル＋softmax融合、実測 −60.8%）を当てると約194〜222B（128 tokens時）まで縮む。

つまり **F32のままでは実checkpointは載らない**。必要な削減は最低12倍で、INT8化で見込める4倍では足りない。INT8と蒸留の併用、または層数・hiddenの再検討が必要になる。ただし「Scoreを削る」「尺度説明を短縮する」「入力を切る」といった意味を削る最適化は、この結果を理由にしても認められない。

### openJev 151Mは外挿ではなく実測で上限を押さえた

`tools/measure_verdict.py` が**実checkpoint（577.5 MiB pack、151,378,176 params）**を投入し、
実benchmarkの1件（自然長118トークン）を118/119/120/126/128トークンで測った:

| T | instructions | 判定 |
|---|---|---|
| 118 | 38,998,851,854 | 予算内（当時のカーネル） |
| 119 | 39,293,820,829 | 予算内（当時） |
| 120 | **36,976,071,434** | **予算内（現行カーネル。成功した最長。当時の実測は39,568,299,856）** |
| 126 / 128 | — | **replicaが40B上限で拒否（IC0522）** |

T=118/119の行は int8カーネル整理（commit `b38d606`/`52f47e9`）より前の測定である。現行カーネルでの
再測は T=120（36,976,071,434、308,133,929/token）で、位相別内訳も `artifacts/verdict_sweep.json` に更新済み。

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
最長150トークンは最小二乗式（切片込み、2.8465e8/token）から**約48.1e9 instructions（40Bの1.20倍）**で、
裾は約7%の超過にとどまる。それでも**裾を消すにはカーネル改善ではなくINT8・蒸留・入力契約の設計が要る**。

**内訳を実測した**（`artifacts/phase_measurements.json`）: encoderが**83%**、decisionが16%、
softmax/norm/活性化/gather/decodeは合計1%未満。ADR-010の「hot linearへ適用」は裏付けられた。

**効率（現行の実測）**: gemm **2.501**、int8カーネル **0.780** instructions/MAC。以前の「encoder 1.43」は分母の取り違えで撤回済み（`docs/archive/PERFORMANCE_MEASUREMENTS.md`）。
スカラーなら3〜4、SIMDなら1前後が下限なので、**現状は下限から1.4倍以内**である。
INT8は実装済みで、T=120の実測は **14.57e9**（現行F32 36.98e9 比 −60.6%）。T=126以上は予算ガードで拒否される。
**ただしガードの費用モデルは最適化前の傾き（3.276e8/token）のまま**で、現行の実測傾きは 3.081e8/token である
（ガードは約7%保守的。T=126の実測見積りは約38.8e9で予算内だが、ガードは41.74e9と見て拒否する）。

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

ADR-017（当時の設計記録。本文書とともに削除済みで、git履歴に残る）に記録した。

**heap**: wasm32の線形メモリ上限は4 GiB。`canisters/verdict-engine` に `heap_bytes` query を追加したので測れる（実checkpoint warmで**1.02 GiB**）。Layaの1.57 GiB packは未投入のまま削除した。
