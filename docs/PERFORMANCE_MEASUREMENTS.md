# ICP性能の実測

対象: `tools/measure_inference.py`（local replica上の実測）
測定日時の記録: `artifacts/inference_measurements.json`

**この文書は測定値と、明示的に区別した外挿値から成る。** 外挿を測定値として扱わない。

## 何を測ったか

`tools/generate_fixtures.py --tier` が生成する合成packを、`IC_LAYA_CANDLE=1` でビルドした decision-engine canister に
`begin_upload` → `upload_chunk` → `start_warmup` → `warmup_next` で投入し、
`register_schema` / `register_calibration` を通してから `evaluate` を呼び、
canister 自身が返す `measured_instructions` を読んだ。

合成weightは**同じ `laya-candle` カーネル**を通るため、層あたり・tokenあたりの計算量は意味を持つ。
一方で**判断品質や上流parityの証拠にはならない**。

評価は設計の返金workflowと同じ三primitive（Noul → Choice → Score）で、入力は Compact128 profile に収まる長さ。

## 測定値

| tier | hidden | encoder層 | params | pack | 質問あたり instructions | 実測 wall |
|---|---|---|---|---|---|---|
| `measure-s` | 128 | 2 | 約0.2M | 3.1 MiB | **193M / 207M / 228M** | 0.30 s |
| `measure-m` | 512 | 4 | 約13M | 81.1 MiB | **4.73B / 5.02B / 5.43B** | 0.93–1.02 s |

（3値は Noul / Choice / Score の順。候補数が増えるほど高い。）

正規化すると **encoder層 × token あたり**:

- `measure-s`: 3,270,454 instructions
- `measure-m`: 39,581,128 instructions

hidden 128 → 512（4倍）で **12.1倍**。指数は `log(12.1)/log(4) = 1.80` で、二次（2.0）よりやや緩やか。
これはattentionの `O(hidden²)` に加えて、層数に比例しない定数項（embedding lookup、scorer、Candid decode）が小さいtierでは相対的に重いためである。

## 外挿（測定値ではない）

実checkpointは **encoder 28層 / hidden 1024 / vocab 50368**。`measure-m` から hidden をさらに2倍に上げ、
28層・128 tokens に当てはめると:

| 前提 | 421Mモデルの1質問あたり | 20B目標に対して | 40B上限に対して |
|---|---|---|---|
| hiddenに対し二次（保守的・上限側） | **567B instructions** | 28倍超過 | **14倍超過** |
| 実測指数 1.80 を延長 | **494B instructions** | 25倍超過 | **12倍超過** |

**結論: 現状のF32実装は、ICPの1 updateあたり40B instructionsという公開上限に対して12〜14倍超過する。**
設計の受入目標20Bに対しては25〜28倍である。これは性能の「不足」ではなく、**アーキテクチャ選択の棄却**に近い。

外挿の仮定と限界:

- 実checkpointの `vocab_size` は 50368 で `measure-m` の 2048 より大きい。embedding lookup と最終normは層数に比例しないため外挿に含まれず、**実測はこれより悪化する方向**に働く。
- 実checkpointは `local_attention=128` のsliding windowを持つ。`measure-m` も `local_attention=128` に設定してあるが、`global_every` の層配置は層数に依存するため厳密ではない。
- decision head（2層）の分は含めていない。encoder 28層に比べて小さいが、無視できる大きさではない。
- したがって上表の値は**下限寄りの推定**であり、実際はこれより大きい可能性が高い。

## ICPのwasmはSIMDを実行できる（実測）

v2.0は「SIMD専用kernel」を将来の最適化として挙げていたが、**使えるかどうか自体が未確認**だった。
wasm-feature probe canisterを作り、local replicaへinstall・実行して確認した。

probeは `std::arch::wasm32` のintrinsicを使い、wasmがSIMD必須であることを検証したうえで
（`wasm-tools validate --features mvp` は失敗、`--features simd` は成功）実際に呼び出した:

| 命令 | 結果 | 意味 |
|---|---|---|
| `i32x4.mul` | 実行成功（50） | SIMD算術が動く |
| `i32x4.dot_i16x8` | 実行成功（36） | **4レーン整数ドットが動く** |
| `i16x8.extend_low/high_i8x16` | 実行成功（72） | i8→i16拡張が動く |
| `i16x8_dot_i8x16_i7x16_s` | **Rustでは未公開** | 直接のi8ドットはコンパイルできない |

したがって**INT8カーネルは「i8→i16拡張 + `i32x4.dot_i16x8`」の経路で書ける**（4 MAC/命令）。
これはADR-010の方針が実行可能であることの確認であり、同時にADR-017として記録した。

## フェーズ別の内訳（実測）

`measure_phases`（測定専用endpoint。cacheを迂回し、資金移動経路に触れない）で内訳を測った。

| tier | embedding | **encoder** | final_norm | decision | scorer | decode |
|---|---|---|---|---|---|---|
| measure-s | 0.0% | **70.3%** | 0.0% | 27.7% | 1.4% | 0.0% |
| measure-m | 0.0% | **83.3%** | 0.0% | 15.7% | 0.9% | 0.0% |

**encoderが支配的**で、softmax・norm・活性化・gather・decodeは合計1%未満である。
ADR-010の「hot linearへINT8/SIMDを適用する」という判断は、**この内訳で裏付けられた**。

## 効率: すでにほぼ最適である（重要な訂正）

当初この文書は「約31命令/MAC」と記載していた。**これは誤りだった。**
全フェーズの命令数をencoderのMACで割っていたためで、正しくは**フェーズごとに**割る必要がある。

`measure-m` の実測値から:

| フェーズ | 理論MAC | 実測instructions | instructions/MAC |
|---|---|---|---|
| encoder | 2,751,463,424 | 3,942,825,715 | **1.43** |
| decision | 805,306,368 | 742,253,057 | **0.92** |

**どちらもほぼ最適である。** スカラー実装なら3〜4命令/MAC、SIMDを使えば1命令/MAC前後が下限なので、
現状は下限から1.4倍以内に収まっている。つまり**カーネルは非効率ではなく、削る余地は小さい。**

この事実は最適化の期待値を大きく変える。middle tensorの除去やper-opオーバーヘッドの削減に期待できる余地は、
「31命令/MAC → 数命令/MAC」ではなく、**「1.43 → 1.0」程度**しかない。

## 示唆

- **この構成のままでは実checkpointを載せられない。** F32での正しさを優先した設計判断（ADRの順序: F32で一致確認 → buffer再利用 → INT8/SIMD → 蒸留）は妥当だったが、その最初の実測で予算に届かないことが確定した。
- 必要な削減は**最低でも12倍**（40B上限に収めるだけでも）。INT8化で見込める4倍では届かない。
  残差は**3.1倍**（123B → 40B）で、これはADR-017のとおり融合カーネルで中間tensor生成を削ることで
  埋める見込みである。20B目標に対しては6.2倍。
- **INT8 + SIMDだけでは届かない。** 実測の効率はすでに 1.43 instructions/MAC で、
  INT8 SIMDの理論上限（4 MAC/命令）を当てても**最大で3〜4倍**が現実的な天井である。それを当てると:

| INT8/SIMDの伸び | 421M投影 | 40B比 | 20B比 |
|---|---|---|---|
| ×2 | 249B | 6.2倍 | 12.5倍 |
| ×4（理論上限近く） | 127B | **3.2倍** | 6.4倍 |
| ×6（非現実的） | 86B | 2.2倍 | 4.3倍 |

  **×4でも40Bの3.2倍超過**である。したがって:

  - INT8/SIMDは必要だが**十分ではない**。単独では本番採用基準を満たせない。
  - **蒸留が必須**である。8層/hidden 768 まで落とせば 84B（40Bの2.1倍）、
    6層/hidden 640 で 45B（1.1倍）。INT8/SIMD（×3〜4）と組み合わせれば
    **6〜8層で20B以内**に入る。
  - つまり正しい順序は「**蒸留で規模を落とし、その上でINT8/SIMD**」である。
    INT8だけを先に完成させても本番基準には届かない。
- 「Scoreを削る」「尺度説明を短縮する」「入力を切る」といった意味を削る最適化は、この測定結果を理由にしても**依然として認められない**。削るなら計算側（量子化・蒸留・カーネル）である。

## 未測定（重要）

1. **heap使用量**。canisterのheapを直接読む口がないため、`warm heap 2.5GiB以下` / `cold peak 3.0GiB以下` は**測っていない**。
   実checkpointのF32 packは1.57 GiBで、warm-upは1 tensorずつstableからheapへ読む。`Builder::push` が `Tensor::from_raw_buffer` でstableの読み出し元bufferをそのまま使うため、恒久的な二重化は避けられる設計だが、**実測していない**。
2. **1.57 GiB packの投入そのもの**。今回の最大は81 MiBである。
   CLI経由の `upload_chunk` は `icp canister call` が引数をテキストで受け取りCandidが各バイトを `\xx` に展開するため、argv長で1 MiB chunkが使えない（実測: `OSError: Argument list too long`）。256 KiB chunkで1.57 GiBは約6400回の呼び出しになり、この経路では非現実的である。
   これは**canister側の制約ではなく、CLIというクライアントの制約**である。実運用のアップロード経路（binaryを渡せるクライアント）を別途設計する必要がある。
3. **wall-clockの本番相当値**。local replicaは cycle 課金されないため、この 0.3〜1.0 秒を mainnet の見積もりに使ってはならない。判定に使ったのは `instructions` のみである。

## 再現手順

```bash
python tools/generate_fixtures.py --tier measure-s --tier measure-m
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
python tools/measure_inference.py --tier measure-s --tier measure-m --chunk-kib 256 --timeout 900
```

`--chunk-kib` はCLIの引数長制約に対する回避策であり、canisterの上限（`upload_chunk` は1 MiBまで）とは別物である。
