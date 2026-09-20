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

## 示唆

- **この構成のままでは実checkpointを載せられない。** F32での正しさを優先した設計判断（ADRの順序: F32で一致確認 → buffer再利用 → INT8/SIMD → 蒸留）は妥当だったが、その最初の実測で予算に届かないことが確定した。
- 必要な削減は**最低でも12倍**（40B上限に収めるだけでも）。INT8化で見込める4倍では足りない。**INT8 + 蒸留、または層数・hiddenの再検討**が必要になる。
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
