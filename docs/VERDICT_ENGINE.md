# openJev (GLiClass / ModernBERT-151M) を canister で動かす

対象checkpoint: `heman10x/rlcd-modernbert-151m` (revision `70fa1982…`)、Apache-2.0、151,378,177 parameters。
移植仕様の一次資料は [GLICLASS_FORWARD_SPEC.md](GLICLASS_FORWARD_SPEC.md)（gliclass 0.1.20 / transformers 5.17.0 /
ONNX先頭16 MiBの実測に基づく）。

## 1. なぜこのモデルか

Jev系は「デコードしないone-pass」なので、1 decisionのコストは `α_eff × 2P × T`
（P=パラメータ、T=state+選択肢+markerのトークン数）。40B命令/updateに対し、**Pが唯一の支配的レバー**である。
実測（5.1.1節）では F32 でこのモデルは **T=120が成功、T=126が40B上限で拒否**される。これは
Laya（421M、F32、128トークンで494〜567B）と同じ結論だが、151Mは同じ命令予算で扱える入力が長い。
量子化カーネルを書いた場合の射程は 2.2節の表を参照。

**着手時に置いていた「平均383トークンで13倍超過」という見積りは、このリポジトリのprompt contractには
当てはまらない。** 383はJevBenchの生入力を素朴に数えた値で、本実装が測る入力
（選択肢の説明文を含む実際のprompt）ではない。実benchmark 1000件をこの実装の契約で数えると
**中央値94・平均96.5・p90 111・最長150トークン**（`models/verdict-parity/cases.jsonl`、
`verdict-infer tokens`。先頭120件のサブサンプル）で、**120/120件中118件（98%）が予算内**である。
1000件全体では中央値95・平均95.8・p90 108・予算内975/1000（97.5%）である。5.1.1節の結論は
「実入力はほぼ収まるが、長い側の裾は収まらない」であって「3.2倍超過」ではない。

## 2. 実装したもの

| 追加物 | 中身 |
|---|---|
| `crates/verdict-candle` | GLiClass uni-encoder forward。encoder は `modernbert-candle::encoder::ModernBert` を共有し、head だけが違う |
| `canisters/verdict-engine` | `begin_upload` / `upload_chunk` / `start_warmup` / `warmup_next` / `infer_tokens` / **`infer_profiled`**（owner専用の位相別命令数） / `decide` / `info` / `allow_caller` / `set_max_input_tokens` |
| `crates/modernbert-candle`（変更） | encoder の公開面を追加（型と `forward` の可視性、`new`、`pub mod encoder`）。既存57テストは green |
| `tools/pack_verdict.py` | safetensors → canonical F32 pack（142テンソル、605,512,704 B、SHA-256検証つき） |
| `tools/make_verdict_fixture.py` | canisterスモーク用の32次元2層pack（同一カーネル、重みは乱数） |
| `tools/verdict-upload` | agent経由でpack投入＋warm-up＋`infer_tokens`（`icp canister call` のargv制限を回避） |
| `tools/verdict_canister.py` | ローカルreplicaでの作成→install→投入→推論の実行 |
| `tools/verify.py`（変更） | fixture pack生成・`verdict-candle` テスト・wasmビルドを追加、`--verdict` で実checkpoint parity |

### 2.1 移植仕様で外せない点

* `model.logit_scale` は **適用されない**（`normalize_features=false`）。packから意図的に除外し、
  packの総バイト数はsafetensorsのデータ部よりちょうど4バイト小さい。
* layer 0 には `attn_norm` が **無い**（HFは `nn.Identity()`）。`first_layer_attention_norm=false` を
  tensor inventory から導出する。
* class特徴は `<<LABEL>>`(50368) の**その位置**の隠れ状態（`embed_class_token=true`）、
  text特徴は `[CLS]`（位置0）、scorerは内積、projectorは bias つき2層GELU(erf)。
* sliding window は `local_attention/2 = 64`（`|i-j| <= 64`、129キー）。

### 2.2 射程（実測と推定）

| 実装 | instr/token | 40Bで入るT | 根拠 |
|---|---|---|---|
| F32（本次実装） | 3.28e8 | **120〜126の間** | 実benchmark入力の直接測定（5.1.1節）。120が成功、126が拒否 |
| F32 + 毎callの重み転置（修正前） | 3.21e8 + 固定1.77e10 | 69 | ローカルreplicaで実測（5.1節、固定費が支配） |
| INT4/ternary packed | 約0.16e9（推定） | 約240（推定） | 限界費用が既に1.06 instr/MACなので伸びは最大2倍 |
| Laya-large 421M F32 | — | 約10 | 本リポジトリの実測（128トークンで494B＝12.3倍超過） |

限界費用はすでに **1.06 instructions/MAC** で、論文の `α_eff ≈ 1.0`（ternary床）と同じ水準にある。
したがって量子化で得られるのは（重み容量とメモリ帯域を除けば）**約2倍**であり、
「量子化すれば5倍」という想定は本実装では成り立たない。

## 3. 正解性の証拠

### 3.1 実checkpointとの一致（外部証拠）

packの`--out`には存在しないパスを指定します。既存pack・空ディレクトリ・ファイル・symlinkは上書きせず拒否します。再生成時は別の出力先を使ってください。入力検証後に出力先を排他的に作成し、通常の書込みエラーでは今回作成したファイルを削除します。強制終了で残った未完成ディレクトリも次回は拒否するため、内容を確認してから別名で再実行してください。

`reports/v2/predictions_v2.jsonl`（著者自身が記録した argmax と校正済み確率）を、
`data/real_banking_test.jsonl` の入力で再実行して突き合わせる。

```bash
python3 tools/pack_verdict.py \
  --safetensors models/verdict-151m/model.safetensors \
  --config models/verdict-151m/config.json \
  --tokenizer models/verdict-151m/tokenizer.json \
  --out models/verdict-pack --source-revision 70fa19828074e1199e4a793c4af4dba3bfd1d222
cargo build --release -p verdict-candle
./target/release/verdict-infer check --pack models/verdict-pack \
  --tokenizer models/verdict-151m/tokenizer.json \
  --cases models/verdict-parity/cases.jsonl \
  --predictions models/verdict-parity/predictions.jsonl
```

結果は `docs/VERDICT_ENGINE.md` の 5節（測定値）に記録する。`cargo test --release -p verdict-candle
--test golden -- --ignored` が同じ比較を gate として実行する。

### 3.2 カーネル単位のテスト

`cargo test -p verdict-candle` が11件（+1 ignored: 577.5 MiB（605,512,704 B） packが必要なgolden）。pack往復（`Builder`→`VerdictModel`）が直接構築と一致すること、
独立に再計算したhead（projector＋内積）と `logits()` が一致すること、class位置の順序、
`max_classes` 超過の拒否、tensor set不一致の拒否を含む。

## 4. canisterでの実行

課金対応版では、以下のupdateを含むコマンドに`--proxy PRINCIPAL --max-cycles N`を追加します。
再install後の初期単価は`--execution-pricing BASE,NUMERATOR,DENOMINATOR`で設定してください。
使用するowner PEMのidentityをproxyのcontrollerとして認可する必要があります。
無料の`--query`のみの実行には支払い引数は不要です。詳細は[サイクル課金仕様](CYCLES_BILLING.md)を参照してください。

```bash
bash tools/build_one.sh verdict-engine
cargo build -p verdict-upload
python3 tools/make_verdict_fixture.py
python3 tools/verdict_canister.py --pack fixtures/verdict-tiny \
  --tokenizer fixtures/verdict-tiny/tokenizer.json --ids 1,3,11,3,12,2 --keep --decide
```

実pack（577.5 MiB（605,512,704 B））の場合:

```bash
python3 tools/verdict_canister.py --pack models/verdict-pack \
  --tokenizer models/verdict-151m/tokenizer.json \
  --ids 50281,50368,2000,50368,3000,50282 --keep
```

`--ids` は**fixture用の既定値（`1,3,11,3,12,2`）を実packに使い回せない**。実packの語彙には
`<<LABEL>>`(50368) が含まれないid列を渡すと `Invalid("no class tokens")` で拒否される（正しい拒否）。
実benchmark入力をそのまま測る場合は、prompt contractをRust側で組み立てる次を使う:

```bash
python3 tools/measure_verdict.py --sweep 118,119,120,126 --top-up 200t
# 温まったreplicaを使い回す場合（577.5 MiB（605,512,704 B）の再投入を避ける）
python3 tools/measure_verdict.py --skip-upload --sweep 120 --keep
```

* icp-cli の状態は `.icphome/`、Cargoのregistryは `.cargohome/` に置き、リポジトリ外へ書かない。
* ローカルreplicaでは **anonymous identity にICPがpre-fundされている**ため、
  テスト用identityへtransferしてから `icp cycles mint` する（`tools/verdict_canister.py::ensure_cycles`）。
* canister owner は `.icphome/verdict-owner.pem`（Ed25519, PKCS#8 v1）。
  icp-cli の identity は secp256k1 PKCS#8 v2 で、ic-agent の `BasicIdentity` が読めないため使い分けている。

### 4.1 `icp canister call` でpackを運べない理由

`upload_chunk(offset: nat64, bytes: vec nat8)` に 1 MiB を渡すには Candid 引数を argv にエスケープする必要があり、
`docs/archive/PERFORMANCE_MEASUREMENTS.md` が記録した argv 長の壁に当たる。icp-cli 1.0.2 の
`--args-format hex` / `bin` は `--args-file` と組み合わせても引数をデコードせず
（canister側で Candid パースエラーになることを確認）、この経路は使えない。
そこで `tools/verdict-upload` が ic-agent で ingress を直接話し、1 MiBを1 update callで送る。

## 5. 測定値

### 5.1 実checkpoint（F32、151,378,176 params（packは`logit_scale`を除く142テンソル。checkpoint全体は151,378,177）、ローカルreplica実測）

> **測定の来歴**: ここから5.1.13までの数値は、このリポジトリのローカルreplicaで
> `bench_matmul` / `bench_simd` / `bench_int8` / `infer_tokens` / `infer_profiled` /
> `tools/measure_verdict.py --sweep` を実行して得たもので、生ログの一部は `artifacts/` に
> 残っていない。**`bench_simd`・`--simd`・f32 `f32x4` カーネル一族はその後の整理
> （commit `b38d606`）で削除した**ため、それらに依存する値は履歴であり再実行できない。
> 現在も再現できるのは `bench_int8`（0.780 instructions/MAC）、`infer_tokens`、
> `infer_profiled`、`--sweep`、および `verdict-infer check` である。
>
> **現行カーネル（HEAD）の実測**: T=120 の実checkpointは **36,976,071,434 instructions**
> （308,133,929/token、`artifacts/verdict_sweep.json`）。これは最適化前の基準値
> 39,568,154,800（`overflow-checks` 有効時）／37,311,793,353（無効化直後）より低い。
> 以下の 5.1.3〜5.1.13 の対比表は**その当時のF32基準値（37,311,793,353）を1.00とした履歴**で、
> 現行値に置き換えたものではない。

`infer_tokens` の `measured_instructions`（forwardのみ、引数のdecodeとreplyのencodeは含まない）。
入力は `[CLS] <<LABEL>> 2000 <<LABEL>> 3000 [SEP]` に filler を足したもの。

| T（トークン） | 2 | 6 | 33 | 70 |
|---|---|---|---|---|
| 重み転置の修正前 | 18,383,286,061 | 19,598,547,957 | 28,337,233,046 | — |
| **修正後** | **909,587,511** | **2,124,706,257** | **10,863,606,034** | **22,821,340,023** |

修正前の最小二乗は `I(T) ≈ 1.774e10 + 3.21e8 × T`。**Tに依存しない固定費 1.77e10 instructions/call** が
支配的だった。原因は `Linear::forward` が毎回 `weight.t()?.contiguous()?` で151Mパラメータぶんの重みを
転置・実体化していたこと（T=2でも満額払う）。ロード時に一度だけ転置する形へ変更した結果:

**`I(T) ≈ 2.66e8 + 3.22e8 × T`**（固定費は66分の1、T=2で20倍、T=6で9.2倍の改善）

* 限界費用 **3.22e8 instructions/token ≒ 1.06 instructions/MAC**（`2P = 3.03e8`）。
  wasm SIMDの4レーンで回っており、論文の `α_eff ≈ 1.0`（ternary床）と同じ水準。
* **40B上限で T ≈ 123トークン**（修正前は69）。この値は T=2/6/33/70 の4点から出した
  **一次式による外挿**である。実入力長での直接測定は下の5.1.1節で行った。

### 5.1.1 実benchmark入力での直接測定（`tools/measure_verdict.py`）

5.1節の T=2〜70 は合成id列である。実benchmarkの1件（`test_in_00665`、
`models/verdict-parity/cases.jsonl` の先頭）を
**canisterと同じRustのprompt contract**（`verdict-infer tokens`）でtokenizeし、
その自然長118トークンから上へpaddingして測った:

| T（トークン） | instructions | 判定 |
|---|---|---|
| 118 | 38,998,851,854 | 予算内（最適化前のカーネル） |
| 119 | 39,293,820,829 | 予算内（同） |
| 120 | **36,976,071,434** | 予算内（現行カーネル。成功した最長。最適化前は 39,568,154,800） |
| 126 | — | **replicaが40B上限で拒否（IC0522）** |
| 128 | — | 同上 |

* **現行カーネルでの再測は T=120 の1点**（36,976,071,434、308,133,929/token）。118/119/126/128 の行は
  int8カーネル整理（commit `b38d606`/`52f47e9`）より前の測定で、そのままでは現行値と混ぜられない。
* **上限は T=120（成功）と T=126（拒否）の間**。5.1節の外挿 `T ≈ 123` はこの範囲に入っており、
  外挿としては妥当だった。ただし**上限値そのものは外挿ではなくこの3点で押さえる**。
* この3点の厳密な最小二乗は `I(T) ≈ 5.41e9 + 2.8465e8 × T`（切片を落とすと過小評価になる）。5.1節の式（`2.66e8 + 3.22e8 × T`）と
  固定費が3.6e7、傾きが2%ずれる。**実benchmark入力でも限界費用はほぼ同じ**。
* T=120 は 2B instructions/秒で**約19.8秒**。応答時間の観点では依然として重い。

**実入力の長さ分布（この契約で1000件すべてを数えた）:**

| 指標 | トークン数 |
|---|---|
| 最小 / 中央値 | 66 / **95** |
| 平均 / p90 / p99 | 95.8 / 108 / 132 |
| 最長 | **150** |
| T=120を超える件数 | **25/1000 = 2.5%**（超過分は121〜150） |

**したがって「平均383トークンで3.2倍超過」という当初の見積りは誤りである。**
383はJevBenchの生入力の素朴な計数で、本実装が実際に食わせるprompt（選択肢の説明文を含む）ではない。
正しい結論は「**中央値95トークンは予算内、上側2.5%の裾だけが予算外**」である。

1件の内訳を数えると、**選択肢の説明文が53%、state＋questionが50%**（`test_in_00665`、118トークン）。
つまり入力の約半分は選択肢ラベルの記述であり、選ぶ余地があるのはそこである。
ただしそれは**checkpointが学習時に見た文字列を変える**ことを意味し、parityと校正の再取得が要る
（5.3節の1000件一致はこの契約に対する証拠である）。性能を理由にstateやScore尺度を削るのは
従来どおり認められない。

ログ: `artifacts/verdict_sweep.json`。再現は
`python3 tools/measure_verdict.py --sweep 118,119,120,126`（replicaを起動し577.5 MiB（605,512,704 B） packを投入する）。

実用面の解釈:

* 実benchmarkの中央値（94トークン）は**量子化なしでも1 callに収まる**。収まらないのは裾だけである。
* 1000トークン級のstateは、量子化カーネルか入力設計の変更なしには載らない。
* 限界費用がすでに 1.06 instructions/MAC なので、INT4/ternaryで得られるのは**約2倍**（`α≈0.5`）であって
  5倍ではない。2.2節の表はこの実測を踏まえて読み直す必要がある。


### 5.1.2 Phase 1 の実測（heap、実重みdecide）

| 項目 | 値 |
|---|---|
| warm時のwasm linear memory | **1,099,694,080 B ≒ 1.02 GiB**（`heap_bytes` = `memory_size(0)×64KiB`、実checkpoint warm時。現行wasmでの再測でも同値: `info.heap_bytes = 1_099_694_080`） |
| `decide`（state+question+2選択肢+abstention、実重み） | 52トークン、**16,831,252,741 instructions**、`card_lost` を 0.9731 で選択（targetと一致）、abstention 0.0245 |
| 事前予算ガード | `I(T)=2.544e8+3.276e8·T`（実測2点 T=2/T=120 にフィット）＋余裕0.5%。**以下はコードの実値**（以前の39.10e9/41.04e9は傾きの取り違え）。`estimated_cost(120)=39,764,232,000`（39.76e9）≤ 40e9 で**許可**、`estimated_cost(126)=41,739,660,000`（41.74e9）で**拒否** |
| ガードの実地確認 | T=120: 現行カーネルで `36,976,071,434` が **Ok**（最適化前の同条件は `39,567,477,066`、ガードはそこから更新していない）。T=126: 40B超過を**trap ではなく `Capacity`** で返す（以前は replica が IC0522 で拒否） |

**ガードの費用モデルは最適化前の実測に固定されたままである。** 傾き 3.276e8 instructions/token は
今も `estimated_cost(120)=39.76e9` / `estimated_cost(126)=41.74e9` を返すが、現行カーネルの実測傾きは
**3.081e8**（36,976,071,434 − 固定費2.544e8 を120で割った値）で、T=126 の実測見積りは約38.8e9と
40B以内に入る。つまり**ガードは約7%保守的**で、予算内の長さを `Capacity` で拒否しうる。
費用モデルの再フィット（`set_cost_model`）はコード変更なので、ここでは数値のずれとして記録するにとどめる。

`decide` が実重みで通ったことで、**typed decision 経路（tokenizer→prompt契約→forward→temperature→softmax）が canister 上で完走する**ことが確認できた。

### 5.1.3 カーネル診断（`bench_matmul` / 削除済み`bench_simd`、T=120形状。SIMD側の値は履歴）

| カーネル | instr/MAC | 備考 |
|---|---|---|
| candle gemm（n=768/1152/2304 の3形状） | **2.648〜2.653** | 形状に依存しない。gemmの `simd128` カーネルがartifact内で有効（`f32_4x3x4` にSIMD命令241個） |
| 自前 f32x4（`crates/verdict-simd`） | 12.505 | scalar参照は55.026（4.4倍改善だがgemmより4.7倍遅い） |

**原因**: `#[target_feature(enable="simd128")]` はLLVM IRには `target-features="+simd128"` として入るが、
wasmのコード生成はモジュール単位で決まるため、artifact内の自前カーネルは**スカラー化**される
（`verdict_simd7gemm_n4` のSIMD命令は6個のみ）。gemmのカーネルがSIMDになっているのは、
candle が gemm を `wasm-simd128-enable` 付きでビルドしているため。
つまり **モジュール全体で `-C target-feature=+simd128` を有効にする以外に手はなく、その唯一の障害は
candle-core 0.11.0 の `CurrentCpuF16` 欠落**（`src/cpu/mod.rs` の f16 ゲートが `simd128` を含むのに
`src/cpu/simd128.rs` が `CurrentCpu` しか定義していない）。
2.65 instr/MAC は「4×3ブロッキング + mul/add（wasmにFMAが無い）」の設計値（31命令/12 MAC = 2.58）と一致し、
f32x4の素直な4×4ブロッキングなら 0.8〜1.0 instr/MAC、つまり**2.6〜3.3倍の余地**が残っている。

**simd128 をモジュール全体で有効化する試行（実測して巻き戻し）**: `RUSTFLAGS='--cfg getrandom_backend="custom"
-C target-feature=+simd128'` でビルドすると `candle-core 0.11.0` が `CurrentCpuF16`/`CurrentCpuBF16` 未定義で
落ちる（`src/cpu/mod.rs` の f16/bf16 演算が `any(neon, avx2, simd128)` でゲートされ、同時に
`not(any(avx2, neon))` のスカラー版も定義されるため二重定義にもなる）。修正には **12箇所以上のゲート変更**
（f16/bf16 から simd128 を外し、スカラー版の `not(...)` に simd128 を足す）が必要で、
candle を vendor して当てる必要がある。実際に vendor パッチ（f16/bf16 の8ゲートから simd128 を外す）を当ててビルドを通し、実測した:

| 項目 | simd128 無効 | simd128 有効（モジュール全体） |
|---|---|---|
| end-to-end T=120 | 39,567,477,066 | **38,736,773,563（−2.1%）** |
| `bench_matmul`（3形状） | 2.648〜2.653 instr/MAC | **2.648〜2.653（変化なし）** |
| native と canister の logits | — | 最大 1e-5 で一致 |

**dense matmul は動かなかった。** 理由は gemm の wasm SIMD マイクロカーネル
（`gemm-f32-0.19.0/src/microkernel.rs:504` が `core::arch::wasm32` を使う）が
**フラグ無しでも既に選択されていた**こと。つまり 2.65 instr/MAC は「SIMDが無効なスカラー」ではなく、
gemm の 4×3 ブロッキング（31命令/12 MAC = 2.58）という**設計値そのもの**である。
一方、自前カーネルは simd128 をモジュール全体で有効にしても artifact 内でスカラー化したまま
（12.505 instr/MAC、SIMD命令6個）。gemm のやり方（`gemm-f32-0.19.0/src/microkernel.rs:500` の
`#[cfg(target_arch="wasm32")] pub mod simd128` から intrinsic を直接呼ぶだけ。`#[target_feature]` は
付けていない。candle は gemm を `features=["wasm-simd128-enable"]` で引く）を3通り試した:

| 自前カーネルの書き方 | 実測 instr/MAC |
|---|---|
| `#[target_feature(enable="simd128")]`（現状） | 12.505 |
| `#[target_feature]` 無し（gemm と同じ書き方） | **30.512（悪化）** |
| `#[inline(never)]` を追加 | 12.507（変化なし） |

追加で2通り試した（いずれも改善せず）:

| 試行 | 結果 |
|---|---|
| intrinsic を `#[inline(always)]` の薄いラッパで包む（gemm と同じ書き方） | 12.505（変化なし） |
| `tools/simd-probe`（依存ゼロの独立クレート）で同じカーネルをビルドし、artifact を検査 | probe 関数内の v128 は **0**（SIMD は `core::arch::wasm32::simd128::*` のアウトオブライン関数側にある） |

つまり **この toolchain では `core::arch` の intrinsic が呼び出し側へインライン展開されず**、
per-MAC の呼び出し境界が演算より高くつく。gemm の `simd128` カーネルは artifact 内で
インラインな v128（241命令）になっており、この差を我々のクレートで再現する方法は
5通りの試行では見つからなかった。**int8/ternary カーネルを書く場合の障害はここ**である。

**gemm の効率は形状・トークン数に依存しない**: m=120 で 2.649、m=512 で 2.635 instr/MAC。
つまり 2.65 は「小さな m のパッキング負荷」ではなく **gemm のカーネル設計値そのもの**である。
int8/ternary カーネルを書く場合の最初の障害はここ（同一 toolchain で v128 を出させる方法）にある。

得られた改善は 2.1% に対して vendor 1.9 MiB と第三者パッチの保守コストが見合わないため**巻き戻した**
（作業ツリーに `vendor/` は無く、`.cargo/config.toml` のフラグも戻してある。判断根拠は同ファイルのコメントに残した）。

**結論（Phase 2 の判定）**: dense 91% を 2.65 → 0.8〜1.0 に下げるには gemm を置き換える必要があり、
その機構解明が前提。T=383 を通すには **0.83 instr/MAC 以下**が必要なので、モジュールフラグだけでは届かない。
残るレバーは (a) int8/ternary カーネル（論文の `α_eff≈0.53` なら 2.65→約1.3、T≈240）＋
(b) 蒸留（パラメータ半減で T≈480）の併用、または (c) 入力設計の変更。

### 5.1.4 量子化カーネル（candle 同梱）の実測と、カーネル路線の結論

`candle-core 0.11.0` は wasm SIMD の量子化カーネルを同梱している
（`src/quantized/simd128.rs`、`k_quants.rs:288-2295` が `#[cfg(target_feature="simd128")]` で
Q4_0/Q8_0/Q2K〜Q8K をそこへディスパッチする）。5.1.3 でビルドを通したモジュール全体の
`simd128` フラグを戻したうえで、`verdict-engine` に `bench_qmatmul`（`QTensor::quantize` →
`QMatMul::forward`）を追加して実測した:

| 方式 | mlp_down (120×768×1152) | qkv (120×2304×768) | f32 との最大差 |
|---|---|---|---|
| gemm f32（現行） | **2.648** | **2.653** | — |
| candle Q8_0（SIMD） | 5.458 | 5.391 | 0.004 |
| candle Q4_0（SIMD） | 5.739 | — | 0.046 |

**量子化は f32 gemm の約2倍遅い。** 理由は (a) 呼び出しごとに活性を q8_0 へ量子化する費用、
(b) `vec_dot` ベースで gemm のようなレジスタブロッキングが無いこと。したがって
**candle 内に dense matmul を安くする道は無い**（量子化の重み容量・メモリ帯域の利点は別途ある）。

### 5.1.5 カーネル路線の結論（6通り試行、すべて実測）

| 試行 | 結果 |
|---|---|
| simd128 モジュール全体（vendor パッチ） | end-to-end −2.1%、dense は 2.65 のまま |
| 自前 f32x4 `#[target_feature]` | 12.505 instr/MAC |
| 同・属性なし（gemm と同型） | 30.512 |
| 同・`#[inline(never)]` | 12.507 |
| 同・intrinsic を `#[inline(always)]` ラップ | 12.505 |
| candle 同梱の量子化 SIMD カーネル（Q8_0/Q4_0） | 5.4〜5.7 |

**結論**: この環境で dense matmul を 2.65 から下げる手段は見つからなかった。gemm の 2.65 は
「4×3 ブロッキング＋mul/add」の設計値であり、それを超えるには同一 toolchain で
インラインな v128 を出す方法（gemm は出せている）が必要。上限は **T=120 トークン**
（T=126 は 40B 超過で拒否）で、**T=383 は約3.2倍超過**という判定は変わらない。

T=383 を通す残りの道はカーネルではなくモデル側: **蒸留**（パラメータ半減で T≈240、
1/4 で T≈480）または入力設計の短縮。蒸留は torch/GPU が要るため別環境での作業になる。

### 5.1.6 コスト削減レバーの実測（T=383は目標にしない）

1決定あたりの命令数を実際に下げられたもの／下げられる見込みのものを、すべてローカルreplicaで測った。

| レバー | 変更 | 実測効果 |
|---|---|---|
| **overflow-checks** | release プロファイルを `false` に | gemm 2.648→**2.501** instr/MAC、T=120 が 39,567,439,408→**37,311,793,353（−5.7%）**。`panic_const_*_overflow` の呼び出しがホットループから消える |
| **T削減（プロンプト設計）** | 選択肢説明を「3語」または「id語」に短縮 | 同一ケース・同一5選択肢で T=118→83/82、命令数 **36,770,062,509 → 25,537,809,056 / 25,272,479,222（−31%）**。T にほぼ完全に線形 |
| **P削減（層削減＝蒸留の代理）** | 22層→11層の pack を作成（hidden等は同じ） | ネイティブのforward時間 **53.6→27.0 ms/call（1.99倍）**。パラメータ半減がそのままコスト半減 |
| **固定費** | ロード時の重み転置を事前計算に | 1.77e10→2.5e8 instructions/call（既出、T=2で20倍） |
| 自前カーネル（進行中） | カーネルを canister クレート内でコンパイル＋4行×4列ブロッキング | 12.505→**3.628** instr/MAC（gemm は 2.501。**まだ勝てていない**のでモデルには未配線） |

**canister 実測（同一ケース・同一5選択肢、命令数は `infer_tokens` の実測値）**:

| 構成 | T | instructions | 対 現行 |
|---|---|---|---|
| 22層 + 元のプロンプト | 118 | 36,770,062,509 | 1.00 |
| 22層 + 短縮プロンプト | 82 | 25,272,479,222 | 0.69 |
| 11層 + 元のプロンプト | 118 | **18,407,087,909** | **0.50** |
| **11層 + 短縮プロンプト** | 82 | **12,657,757,164** | **0.34（2.9倍安い）** |

P は層数に、T はトークン数に、どちらもほぼ完全に線形で、**併用すると積で効く**。
12.66e9 は 40B 上限に対して大きな余裕があり、11層なら T=118 でも 18.4e9 で収まる。

#### 質問バッチ（state共有）の実測

IC-Laya の実ワークフローは「1つの state に3質問」なので、state を1回だけ送る
1パス（8ラベル、Simple Jev の shared-prefix と同じ形）と、3回に分ける形を比較した。

| 構成 | T（合計） | instructions | 対 分離 |
|---|---|---|---|
| 分離3回（state を3回） | 70+72+64 = 206 | 21,514,021,453 + 22,034,864,595 + 19,559,641,027 = **63,108,527,075** | 1.00 |
| **バッチ1回（state を1回）** | **110** | **34,194,754,666** | **0.54（−45.8%）** |

**精度への影響（要対応）**: 同一state・同一選択肢で logits を突き合わせると、
**argmax は3問すべて一致**したが **logit は最大 1.7 ずれた**（Q1 max|Δ|=1.57, Q2 0.91, Q3 1.70）。
GLiClass は全ラベルを1つの `[CLS]` 表現に対して採点するため、質問文を連結しラベルを並べると分布が動く。
したがってバッチ構成は**別schemaとして校正を再フィット**する必要がある
（`Calibration` は (model, schema, tokenizer) 単位なので構造的には対応済み）。決定のみを採るなら影響は小さいが、
Score の分布・ppm を使う設計では必須。

**ラベル短縮との併用**: バッチ構成でラベルを id 語に短縮すると T=110→87、
instructions は **34,194,953,142 → 26,809,909,631**。分離＋長いラベルの 63,108,527,075 に対して
**−57.5%（2.35倍安い）**。さらに層を半減（P/2）すれば、2本の実測線形則から約13.4e9（−79%、4.7倍）と見込まれる。

#### 推奨構成（実測に基づく）

IC-Laya のような「1 state × 複数質問」では、次の順で効く:

1. **質問を1パスにまとめる**（state共有）: −45.8%
2. **選択肢説明を短く**（id語または3語）: −31%（分離時）/ −22%（バッチ併用時）
3. **層数を絞る**（蒸留の代理、P に線形）: 半減で −49.9%
4. `overflow-checks = false`: −5.7%
5. 固定費（転置の除去）: 済み

1+2+4 だけで 63.11e9 → 26.81e9。ここに 3（P/2）を足すと約13.4e9。
**ただし 1 は分布を動かす**（argmax は3/3一致、logit は最大1.7ずれ）ので、
バッチ構成は独立した schema として再校正が必要。

**実装（`decide_batch`）**: `verdict-engine` に「1 state × 複数質問（最大8問）」の
`decide_batch` を追加した。全質問のラベルを1つのラベルブロックに並べ、質問文は `q1 | q2 | …` に連結し、
state は1回だけ送る。返り値は質問ごとの ids/logits/probabilities/selected/confidence。

canister 実測（同じ3質問、state は約15トークン）:

| 構成 | 合計T | instructions | 対 分離 |
|---|---|---|---|
| `decide` を3回（state を3回） | 47+46+42 = 135 | 41,459,382,180 | 1.00 |
| **`decide_batch` 1回** | **89** | **27,512,022,620** | **0.66（−33.6%）** |

state が長いほど節約が大きい（先の実測では state を含む206→110トークンで −45.8%）。
3問の選択は一致したが、**信頼度は大きく動いた**（例: 0.9346 → 0.0198）。
バッチ構成は独立した schema として校正を再フィットする前提で使うこと。

#### 自前カーネルが canister 内でベクトル化しなかった原因（判明）

同一ソースが **依存ゼロの `cdylib` クレートでは v128 になる**（`tools/simd-probe`、57〜77 SIMD命令）のに、
`verdict-engine` から **rlib 依存**として使うと artifact 内でスカラー化していた（該当関数のSIMD命令は0〜6）。
`lto = false`、`#[inline(never)]`、`#[inline(always)]` ラッパ、`simd128` フラグのいずれでも変わらない。
**カーネルを `#[path]` で canister クレート内に取り込むと v128 が入る**（`bench_simd` に14命令、artifact全体で190関数）。
この二重取り込みは整理（commit `b38d606`）で撤去した。残したint8経路は rlib 依存でも同一の 0.780 instructions/MAC を
再現したため、canister内の複製は不要だった。
つまり **クレート境界が原因**で、回避策は「カーネルを canister クレート内でコンパイルする」こと。

残る性能差の原因も artifact から確認できた: 4×4 カーネルの内側ループで
**アキュムレータがメモリにスピルし、毎回 `v128.store` でゼロ初期化されている**（LLVM の wasm レジスタ割当）。

スピルを避ける狙いで **2×4 版**（アキュムレータ2個でレジスタ圧を下げる）も実測したが
**4.504 instr/MAC と悪化**（重みの再利用が半分になる方が効いた）。

生成コードの内訳を測ると、内側ループ1回あたり **v128 13命令に対し `local.get/set`＋`i32` の
アドレス計算が31命令**あった（wasm にはレジスタファイルが無く、オペランドは必ず local 経由）。
k方向を4段アンロールして帳簿を64 MACに償却した結果 **3.628 → 2.988 instr/MAC**（誤差0）。
gemm の 2.501 まで残り19%。さらに8段アンロールも試したが **5.357 instr/MAC と悪化し、値も 0.031 ずれた**
（レジスタ圧の増加と実装誤り）ため破棄した。**f32での手書きカーネルはここが限界**で、
gemm を下回るには int8（`i32x4.dot_i16x8`、8 MAC/命令）へ進む必要がある。
gemm 2.501 を下回るには、スピルを消す（アキュムレータをローカルに保つ構造へ変える）か、
`i32x4.dot_i16x8` を使う int8 版（8 MAC/命令、必要なlive値も少ない）へ進む必要がある。

#### int8 カーネル（gemm を初めて下回った）

設計の要点は **重みを `[n,k]`（k連続＝checkpointの元レイアウト）で持つ**こと。
`i32x4.dot_i16x8_s` は8ペアの積和を返すので「k方向の内積」と相性が良く、
k連続なら重みベクトルをそのままロードできる（f32経路は n方向のベクトルロードのために
`[k,n]` へ事前転置していた）。量子化は行単位（出力要素ごと・トークンごとに1スケール）、
アキュムレータは i32（127×127×1152 = 1.9e7 で余裕）。

| カーネル | instr/MAC | 対 gemm |
|---|---|---|
| gemm f32（candle、現行） | 2.501 | 1.00 |
| 自前 f32x4（4×4、k4アンロール） | 2.988 | 1.19 |
| **自前 int8（`i32x4.dot_i16x8_s`、4列出力）** | **1.605 / 1.626** | **0.64（1.56倍速）** |
| candle f16（`to_dtype(F16)` + gemm-f16） | **12.048 / 11.910** | 4.82（約5倍遅い） |

付随コストと精度（実測、canister内）:
- 活性の量子化: **24.2e6 instructions/call**（120×768）＝1決定 37e9 の 0.07%。無視できる。
- 重みの量子化: 768×1152 で 143.8e6＝**162 instr/param**。モデル全体では約24.5e9（1回）。
  → 本番では**packに量子化済みで入れる**（オフライン量子化）ことでこの費用は消せる。
- f32 との差: max_abs 0.0066 / 0.0042（logit値はO(5)なので約0.1%）。
  **argmax が実データで保たれるかは未検証**（golden 1000件での確認には、pack/モデルへの配線が必要）。

**モデルへの配線と end-to-end 実測**: `verdict-candle` の dense 重み（encoder 4種＋projector）を
int8 化し（`modernbert-candle::Linear` に量子化重みを持たせ、`verdict_simd::matmul_i8` を呼ぶ）、
canister で実測した。

| 構成 | T=120 の instructions | 対 f32 |
|---|---|---|
| f32（gemm、既定） | 37,311,793,353 | 1.00 |
| **int8（dense 91%、改善量子化）** | **27,104,461,469** | **0.73（−27.4%）** |

予測（−34%）より小さいのは、dense 以外の 9% と活性量子化（24e6/call）が減らないため。

**精度の代償（実測）**: 著者記録との argmax 一致は **1000件で 994/1000（99.40%）**（f32 は 1000/1000）。
再現は int8 ビルドでの `verdict-infer check --limit 1000`（この測定の生ログは`artifacts/`に未保存）。
確率偏差は max 0.0835 / 平均 0.0121。**1%の判断が変わる**ため、
既定ビルドでは f32 のまま（`cargo test` の parity gate は 100% を維持）とし、
canister は `IC_VERDICT_INT8=1 bash tools/build_one.sh verdict-engine` で明示的に opt-in する
（`verdict-candle` の `int8` フィーチャ）。1%の変化を許容できない用途では、次の改善
（活性を32要素ブロックごとに量子化＝Q8_0と同じ方式）で誤差を1〜2桁下げるのが本筋。

なお重みの量子化は現状ロード時（`from_tensors`）に行っており、モデル全体で約18〜24e9 instructions が
warm-up の最終呼び出しに集中する（40B以内なので動作はする）。本番では pack に量子化済みで入れて消すべき。

**f16 の評価（項目(d)、実測で棄却）**: 重みを f16 にすると容量は半分になるが、
wasm には f16 演算が無いため candle の f16 matmul は要素ごとに f32 変換が入り、
**12.05 instr/MAC** と f32 gemm（2.501）の約5倍遅い。f32 との差は max_abs 0.0016
（精度は int8 より良いが、命令数では最悪）。**命令数を下げる目的では採用しない**。

### 5.1.7 int8 カーネルのチューニング（実測）

| 版 | instr/MAC | 対 gemm |
|---|---|---|
| 1行×4列、k非展開 | 1.605 | 0.64 |
| **1行×4列、k32アンロール＋ポインタ前進** | **1.250** | **0.50（gemmの2.0倍速）** |
| 上記＋重みを i16 へ事前拡張（i8→i16拡張命令を削除） | 1.274 | 0.51（**悪化**） |

32要素アンロールでは、内側ループ1回（32 k × 4列 = 128 MAC）が
`v128.load` 4（活性）＋ 16（重み）＋ `i32x4.dot_i16x8` 16 ＋ `i32x4.add` 16 = 52命令に、
ポインタ前進5本とループ制御が加わる。**定数オフセットをロード命令に畳ませる**のが要点で、
アドレス計算をロードごとではなく4ロードごとに1回へ減らせる。
重みの i16 事前拡張は**効かなかった**（`i16x8.extend_low_i8x16` は実質無料で、
メモリが倍になるだけ）ので i8 のままにした。

**end-to-end への効果**（T=120、実checkpoint）:

| 構成 | instructions | 対 f32 |
|---|---|---|
| f32（gemm） | 37,311,793,353 | 1.00 |
| int8（1.605） | 27,104,461,469 | 0.73 |
| **int8（1.250）** | **22,434,775,026** | **0.60（−39.9%）** |

位相内訳（int8最新）: matmul4種 = 85.9%、**attn.core = 10.7%（3番目に大きい）**、
mlp_act 2.2%、attn_norm 0.8%。`attn.core` は f32 のままなので、次に効くのはここ。

### 5.1.8 カーネルと量子化のさらなるチューニング（実測）

| 版 | kernel instr/MAC | T=120 instructions | 対 f32 |
|---|---|---|---|
| f32 gemm | 2.501 | 37,311,793,353 | 1.00 |
| int8 1行×4列 | 1.605 | 27,104,461,469 | 0.73 |
| int8 + k32アンロール | 1.250 | 22,434,775,026 | 0.60 |
| **int8 + k の const 特殊化（K=768/1152）＋活性量子化のSIMD化** | **1.212** | **20,738,374,647** | **0.556（−44.4%）** |

- **const 特殊化**: `K` を const generic にすると重み4列が `0, K, 2K, 3K` の定数オフセットになり、
  ポインタが1本に減って各ロードのオフセットが即値に畳まれる（1.250 → 1.212）。
- **活性量子化のSIMD化**: 従来は要素ごとに除算＋`round()`＋`clamp()`で **167命令/要素**。
  逆数乗算＋`f32x4.nearest`＋`i32x4.trunc_sat_f32x4`＋`i16x8.narrow_i32x4` で8要素/回にし、
  `quant_a` が **23,085,592 → 5,504,344**（120×1152）。これだけで end-to-end が −7.6%。
- **重みの i16 事前拡張は不要**と判明: artifact を見ると `i16x8.extend_low_i8x16` は
  すでに `v128.load8x8_s` に融合されており、i16化はメモリを倍にするだけだった（実測でも悪化）。
- 残る差: カーネルの理論床は約0.4〜0.8 instr/MAC（現状1.212）で、行・列方向のオーバーヘッドは
  実測上ゼロ（m=1でも同じ、n=64でも同じ）。次の効き所は `attn.core`（10.7%、f32のまま）。

### 5.1.9 attn.core の内訳と、RoPE テーブルの再利用

attention を細分化して測った結果（T=120、int8ビルド、22層×44回）:

| 位相 | instructions | 割合 |
|---|---|---|
| attn.scores（q·kᵀ＋スケール） | 844,565,207 | 4.1% |
| attn.softmax | 836,572,397 | 4.1% |
| attn.core（重み·v＋転置） | 650,659,256 | 3.2% |
| attn.mask（スライディング窓） | 66,823,547 | 0.3% |
| RoPE 適用 | 約310,000,000 | 1.5% |

**attn.core は合計 2.4e9（11.6%）で、4つの位相に薄く分散**しており、単独の大きな的はない。
一方で matmul は全体の約76%（qkv 5.66e9、wi 5.66e9、wo 2.6e9、out 1.8e9）で、依然そこが本命。

**マーカー帰属の落とし穴（記録）**: 最初の計測で `rope.table` が **6.14e9（29.6%）** と出たが、
これは qkv 射影と RoPE テーブル生成の間にマーカーが無く、**qkv の行列積（5.66e9）が
rope.table に計上されていた**ためだった。位相計測では「重い処理の直前にマーカーを置く」こと。

**それでも直した無駄**: RoPE の cos/sin テーブルは `(tokens, head_dim, theta)` だけで決まるのに、
層ごと・q/k ごとに計 **44回** 作り直していた（`powf`/`cos`/`sin` は libm 呼び出し）。
エンコーダごとに theta 種別（global/local）で **2回** にまとめた（値は同一なので parity 不変、
`tests/parity.rs` の PyTorch 一致テストも通過）。実測 **20,738,374,647 → 20,565,692,991（−0.8%）**。

### 5.1.10 int8 カーネルの2行化（実測）

重みロードを2行で共有する版を追加した（8個のi32x4アキュムレータを保持）。

| 版 | instr/MAC | 対 gemm | T=120 全体 | 対 f32 |
|---|---|---|---|---|
| f32 gemm | 2.501 | 1.00 | 37,139,559,055 | 1.00 |
| int8 1行 const-K | 1.212 | 0.48 | 20,565,692,991 | 0.554 |
| **int8 2行×4列** | **1.022 / 1.036** | **0.41（2.45倍速）** | **17,994,755,333** | **0.485（−51.5%）** |

- 実装は生成コード（4列×4チャンクを明示展開）。**最初の版は誤差0.95で誤り**だった：
  重みチャンクのオフセットを `o`（要素番号）とすべきところ `o*8`（バイト）にしていなかった。
  修正後は max_abs 0.0033（1行版と同一値）で、値は1行版と一致する。
- 残る全体の内訳: matmul 約73%、attn.scores 4.7%、attn.softmax 4.6%、attn.core 3.6%、mlp_act 2.8%。
- これ以上のカーネル改善は4行化（アキュムレータ16個でスピル）か、LLVMが畳まない
  ロードの定数オフセット（artifact に `offset=` が無い）を消す構造変更が必要。

### 5.1.11 int8 カーネルの4行化（実測）

重みロードを4行で共有する版を追加（`m % 4` の端数は2行版へ委譲）。

| 版 | instr/MAC | 対 gemm | T=120 全体 | 対 f32 |
|---|---|---|---|---|
| f32 gemm | 2.501 | 1.00 | 37,139,559,055 | 1.00 |
| int8 1行 | 1.212 | 0.48 | 20,565,692,991 | 0.554 |
| int8 2行 | 1.022 | 0.41 | 17,994,755,333 | 0.485 |
| **int8 4行×4列** | **0.856 / 0.867** | **0.34（2.92倍速）** | **15,766,321,467** | **0.424（−57.6%）** |

精度は不変（f32参照との max_abs は 0.0033 / 0.0023 で1行版と同一値）。
行数を増やすほど重みロードの共有が効く（1→2行で −16%、2→4行で −16%）。
これ以上は8行（アキュムレータ32個でスピル確実）で、伸びは鈍る見込み。

**計測手法の失敗記録**: 「ICが1つのSIMD命令に何単位課金するか」を小さなループで測る
`bench_ops` を試したが、**LLVMがループ不変な演算を巻き上げる**ため演算の実数を制御できず、
「dot = 25単位」という結果はカーネル実測（1.022/MAC＝256MACあたり262単位）と矛盾した。
**演算単価の校正はマイクロベンチでは不可能**で、カーネル全体の `instr/MAC` を測るのが正しい。
この endpoint は誤解を招くので削除した。

### 5.1.12 int8 カーネルの8行化と、カーネル路線の限界

2026-09-23追記: 以下は8行×4列版を測った当時の記録。
その後16/8/4行×8列と端数処理の改善で、48行の主形状は約0.681 instructions/MACまで下がった。
「実用上の床」という当時の見立ては更新した。追加最適化では約0.591 instructions/MACまで下がった。
現行の53-token query受入は[QUERY_OPTIMIZATION_V3.md](QUERY_OPTIMIZATION_V3.md)を参照。

| 版 | instr/MAC | 対 gemm | T=120 全体 | 対 f32 |
|---|---|---|---|---|
| f32 gemm | 2.501 | 1.00 | 37,139,559,055 | 1.00 |
| int8 1行 | 1.212 | 0.48 | 20,565,692,991 | 0.554 |
| int8 2行 | 1.022 | 0.41 | 17,994,755,333 | 0.485 |
| int8 4行 | 0.856 | 0.34 | 15,766,055,138 | 0.424 |
| **int8 8行×4列** | **0.780 / 0.789** | **0.31（3.21倍速）** | **14,736,937,737** | **0.397（−60.3%）** |

行数ごとの改善は 1→2行 −16%、2→4行 −16%、4→8行 −9% と鈍り、精度は全版で同一
（f32参照との max_abs 0.0033 / 0.0023）。**ここが実用上の床**である理由:

- 内側ループの命令数は `(R+C) + 2RC` オペレータ / `8RC` MAC（R行×C列×8k）。
  `2RC` の項（`i32x4.dot` と `i32x4.add` の対）は**8 MACあたり2オペレータ**で、wasm には
  積和融合命令が無いため削れない。R=C=8 で `16/512 + 0.25 = 0.28` オペレータ/MAC、
  R=16 でも `0.27` にしかならず、行を増やす意味がなくなる。
- 実測 0.780 は「1オペレータ≒2単位」と整合する（ベクトル304オペレータ×2＋アドレス約100 ≈ 708/1024）。

**残る全体の内訳（8行版、T=120）**: matmul 系が約7割、
`attn.softmax` 5.7%、`attn.scores` 5.7%、`mlp_act`（GELU）3.4%、`attn_norm` 1.3%。
非matmulは単独では小さく、次に狙うなら softmax の融合（candleの複数パス＋一時テンソルをやめる）が最大。

### 5.1.13 attention softmax の融合とSIMD化（実測）

candle の `softmax_last_dim` は `max_keepdim` → `broadcast_sub` → `exp` → `sum_keepdim` →
`broadcast_div` の5段で、各段が中間テンソルを確保して要素ごとにディスパッチする。
これを1バッファ内の処理に置き換え（`f32::exp` はそのままなので**値は不変**）、
さらに max・exp後の総和・除算をSIMD化した。

| 版 | attn.softmax | T=120 全体 |
|---|---|---|
| candle の5段チェーン | 836,589,434 | 14,736,937,737 |
| 融合（1パス、スカラー） | 729,515,588 | 14,630,692,487 |
| **融合＋SIMD** | **671,255,933** | **14,572,786,121** |

- `exp` は libm 呼び出しのままで、3.8M要素 × 約40単位 ≈ 152M が下限の大半を占める
  （残りはロード/ストアとスカラーexpループ）。**多項式近似に置き換えれば更に削れるが
  数値が変わり parity gate に影響するため実施していない**（要判断）。
- 精度: modernbert-candle の PyTorch 一致テスト、verdict-candle の profiled/plain 一致テストは通過。

### 5.2 位相別の内訳とボトルネック（`infer_profiled`、T=120、実checkpoint）

canisterにowner専用の `infer_profiled(input_ids, detailed)` を追加し、`logits` の位相境界で
`instruction_counter` を読んだ。**同一入力で2回測って同じ配分**（`rope.apply` 32.9%）である。
位相名は現行の計装（`--profile-detailed`）のもので、`artifacts/verdict_sweep.json` の値をそのまま載せる。

| 位相 | instructions | 割合 | 理論MAC | instr/MAC |
|---|---|---|---|---|
| `rope.apply`（qkv射影＋RoPE＋head整形） | 12,147,651,930 | **32.9%** | 4,671,406,080 | 2.60 |
| `layer.mlp_up`（Wi＋GELU） | 11,950,467,823 | **32.3%** | 4,671,406,080 | 2.56 |
| `layer.mlp_down`（Wo） | 5,908,475,625 | **16.0%** | 2,335,703,040 | 2.53 |
| `layer.attn`（out射影） | 3,911,952,785 | 10.6% | 1,557,135,360 | 2.51 |
| `attn.scores`＋`attn.mask`＋`attn.softmax`＋`attn.core`（AV） | 2,234,092,695 | 6.0% | 486,604,800 | 4.59 |
| `layer.attn_norm`＋`mlp_act`＋`attn_resid` | 733,751,878 | 2.0% | — | — |
| embedding / rope.table / attn.pre / encoder / projector / decode | 89,982,739 | 0.2% | — | — |
| **合計** | **36,976,375,475** | 100% | 13,722,255,360 | **2.69** |

（最適化前の同じ計装は合計 39,568,154,800・2.88 instr/MAC で、位相名も `attn.pre`／`attn.core` だった。
差分の大半は softmax の融合とカーネル整理である。）

**読み方:**

1. **密なmatmulが92%を占める**（`rope.apply`＋`mlp_up`＋`mlp_down`＋`layer.attn` ＝ 91.8%）。
   norm・活性化・softmax・gather・decodeは合計2%未満で、最適化対象ではない。
2. **どのmatmulも instr/MAC が 2.51〜2.60 でほぼ一定**。つまり特定の1カ所が遅いのではなく、
   **カーネル全体がf32x4の下限0.5の約5倍**で回っている。`attn.scores`〜`attn.core` の4.59は
   `t×t` の要素演算（mask加算・softmax）を含むためで、これも別種のカーネル問題である。
3. **形状依存がある**（`verdict-infer gemm`、native、同一カーネル）:

   | 形状 | gmac/s | 備考 |
   |---|---|---|
   | m=120 n=2304 k=768 | **112.7** | qkv射影・mlp_up と同じ形 |
   | m=120 n=768 k=1152 | 169.3 | mlp_down と同じ形 |
   | m=120 n=768 k=768 | 159.9 | attention out射影と同じ形 |

   同じMAC数でも **n=2304 は n=768 より1.4〜1.5倍遅い**。canister実測の
   `rope.apply`(2.60) 対 `layer.attn`(2.51)・`mlp_down`(2.53) の比（1.04）とは桁が違うので、
   位相間の差の主因は形状ではなく**位相ごとの周辺コスト**である。

**ボトルネックの結論:** 費用は「特定の遅い演算」ではなく**密行列積の総量**にある。
モデルは1トークンあたり 1.14e8 MAC（22層・hidden 768）を必要とし、それが2.69 instructions/MACで
実行されている。したがって改善のレバーは次の2つだけで、優先順位は明確である:

| レバー | 効果の見積り | 根拠 |
|---|---|---|
| **カーネル効率**（f32x4下限0.5へ） | 2.7〜5.4倍 | 実測2.69 instr/MAC ÷ 下限0.5。**ただしFMAが無いwasmでは現実的な下限は1.0前後**なので、過度な期待は禁物 |
| **MAC総量**（INT8・蒸留・入力長） | 削減率そのもの | instr/MACが既に下限近辺なら、これが唯一の大きなレバー |

**注意（誤読しやすい点）:** 2.88 instr/MAC を「5.8倍の伸びしろ」と読んではいけない。
差の大部分は**計算そのものではなく演算ごとの周辺コスト**（tensor確保、contiguityコピー、
カーネル起動）である。`attn.pre` には `narrow`3回＋`transpose`＋`contiguous`が含まれ、
理論MACがほぼ同じ `layer.attn` との差はそこから出ている。つまり必要なのは
「より速いgemm」ではなく「**演算を融合して周辺コストを消す**」ことで、それは
5.8倍ではなく**2〜3倍**の見込みである。最長150トークンでは最小二乗式から約48.1e9（40Bの1.20倍）で、
**裾の超過はカーネル改善では消えない**。

### 5.2.1 fixture pack（hidden 32 / 2層、同一カーネル）

| 呼び出し | 入力 | instructions |
|---|---|---|
| `infer_tokens` | 6トークン | 5,636,424 |
| `decide`（state+question+2選択肢+abstention） | 43トークン | 15,231,847 |

（この2つは重み転置の修正**前**のビルドで測定した。修正後は同じ入力でより少なくなる。）

fixtureはモデルではない（重みは乱数）。ここで測っているのは
「pack投入 → warm-up → forward → 命令数の報告」という経路が canister 上で成立することである。

### 5.3 ネイティブF32での一致（過去の外部証拠）

以下はF32経路で `data/real_banking_test.jsonl` の入力を、著者が記録した `predictions_v2.jsonl` と突き合わせた過去の結果であり、現行INT8の値ではない。

| 項目 | 値 |
|---|---|
| 比較件数 | **1000（全件、`skipped_no_truth=0`）** |
| argmax一致 | **1000/1000 (100.00%)** |
| 校正済み top 確率の最大偏差 | 5e-6 |
| 平均偏差 | 0.000000 |
| 最長入力 | 150トークン |
| 重み転置の修正後（50件で再確認） | 50/50 (100.00%)、最大偏差 3e-6（in-session実行。生ログは`artifacts/`未保存） |

tokenizer・prompt contract・projector・内積scorer・temperature 1.4265148639678955 まで含めて
一致している。ログは `artifacts/verdict_parity.log`。`cargo test --release -p verdict-candle --test golden -- --ignored` が同じ比較を
gateとして実行する。現行INT8の基準は1,000件以上・argmax一致99.5%以上・欠落0・危険な棄権反転0である。
現行の検証結果は5.4節と `report.md` を参照。


### 5.3 query 経路（5B上限）と、そこでの実測上限

`infer_tokens_query` / `decide_query` は同じ forward を **query call** として実行する。resource limits
（[canister resource limits](https://oa7fk-maaaa-aaaam-abgka-cai.icp0.io/docs/building-apps/canister-management/resource-limits)）では
update 40B に対して **query は 5B**、応答サイズは update 2MiB / query 3MiB、canister あたりの query 実行スレッドは 2、
replicated query の stable アクセスは 1GiB である。query は cycles を消費せず、合意も要らない。
代償は**上限の低さ**と、**応答が certified でない**こと（呼び出し側は「誰が何を実行したか」を検証できない）。

上限は update と同じ費用モデルから導出する:
`T_query = floor(((QUERY_BUDGET×1000/1005) − COST_FIXED) / COST_PER_TOKEN)`。canister は `query_limits()` で
`budget` / `margin_permille` / `max_tokens` / `max_input_tokens` を返すので、呼び出し側がこの値を
ハードコードする必要はない。超過は replica が切る前に **ガードが `Capacity` で拒否**する（`Error` に変種は足していない:
variant 一覧は公開 Candid surface であり、呼び出した method で区別できる）。

既定（F32、保守的な費用モデル）での実測。canister `verdict-engine` に実checkpointを投入し warm した状態で、
canonical な短い id 列（`cls,<<LABEL>>,2000,<<LABEL>>,3000,sep`）を neutral filler `[PAD]` で pad して測った
（`artifacts/verdict_query_sweep.json`）:

| T | instructions | 判定 |
|---|---|---|
| 6 | 2,000,352,930 | 予算内 |
| 10 | 3,159,259,650 | 予算内 |
| 12 | 3,652,108,732 | 予算内 |
| 14 | **4,325,353,975** | **予算内（成功した最長）** |
| 15 | — | **ガードが `Capacity` で拒否（実測見積りは約4.6e9で5B以内）** |

* `query_limits()` は `max_tokens=14` を返す。限界費用は実測 308.1e6/token で update と同一だが、
  **ガードは最適化前の傾き（3.276e8）のままなので1トークン保守的**である（update 側の T=120/126 と同じ性質。
  owner が実測傾斜を `set_cost_model` で入れると 15 になる）。
* **JevBench の実benchmark入力（自然長118）は query には絶対に入らない。** `decide` の実測 52トークン
  （16.83e9）も同様で、`decide_query` は現実的な入力では `Capacity` を返す。これは仕様どおりの拒否である。
* 同一 id 列で query と update の logits は一致した（1.950261 / 1.879874）。query は状態を変えない
  （複数 query の前後で `info` の `warmed` / `active_model` / `callers` / `tensors` / `heap_bytes` が不変）。
* **int8 ビルドでは上限が大きく上がる。** 実測2点（T=6: 774,918,466 / T=120: 14,572,176,608）から
  フィットした `fixed=48,746,986, per_token=121,028,580` を `set_cost_model` で入れると
  `query_limits().max_tokens` は **40** になり、実測も T=40 まで成功した
  （`artifacts/verdict_query_sweep_int8.json`）。F32 の 14 に対して **約2.9倍**である。
  int8 の実測は T に対して単調でない（T=38: 4.317e9、T=39: 4.889e9、T=40: 4.447e9）— int8 カーネルに
  データ依存の分岐（範囲クランプ）があるためで、フィットは平均として扱う。
* **資金移動の根拠に query を使わない。** 応答は certified ではなく、`executor` は verdict-engine を
  呼んでいない。この経路は対話的な短入力の採点と計測のためのもので、資金を動かす判断は update 経路のままである。
* 実測の再現: 温まった replica に対し `python3 tools/measure_verdict.py --query --skip-upload --keep`
  （opt-in の gate は `python3 tools/verify.py --verdict-query`）。int8 の手順は上記のフィット→`set_cost_model`。


### 5.4 block-32の再計測と状態管理の改善（2026-09-22）

現行packの比較記録は [`report.md`](../report.md) に集約した。
同じpackで比較した7入力（合成T=8/16/32/64/96/120、実例T=98）のlogitsは旧版と完全一致した。
T=120の推論命令数は39,782,583,123から39,769,124,903へ約0.034%減少した。
warm-up後のWasmメモリ最大到達量は355,729,408から185,925,632 bytesへ約47.7%減少した。
これは量子化済み重みを複製せず移動する効果であり、解放後のlive heapやwarm-up時間を測った値ではない。

block-32用の2×4・4×4・8×4タイルは、標準カーネルより命令数が増えたため採用しなかった。
局所attentionのQK/softmax/AV候補も増加したため採用せず、attention maskのforward内共有だけを適用した。
測定用のowner限定 `bench_int8_block32` と `bench_local_attention` は通常推論から呼ばれない。
既定の費用モデルと予算は変更していない。T=128は両版とも通常経路のガードで拒否され、
ガードを持たない計測経路でも40B命令上限に達した。

同一bundleの再warm-upはスキーマ・校正・キャッシュ・利用枠を維持する。
別bundleではモデル依存の登録情報を消去し、callerの利用枠を保持する。
質問バッチは質問ごとに選択肢IDを検証し、質問ごとにsoftmaxを計算する。
質問をまたぐ同じ選択肢IDや棄権選択肢を許可するが、質問IDの重複は拒否する。
executorは変更キーだけをstable memoryへ保存し、Errを返す状態遷移も保存する。
verdict-engineはawaitを含まないupdateのheapコミットを利用し、snapshotはinit/pre_upgrade時に作成する。

## 現行方式への訂正

速度優先の指定により、現行packは`i8_row_symmetric`のper-row INT8へ復帰した。
block-32とその品質ゲートに関する以下の記載は旧版の履歴である。
現在の仕様・測定結果は[PER_ROW_INT8.md](PER_ROW_INT8.md)を参照。

## 6. 旧block-32版の制約と未検証

* コード経路は `ic-verdict-int8-pack-v1` 専用で、旧F32 packを拒否する。全2次元重みはblock-32 INT8、
  Norm・bias・scaleだけがF32補助値である。packは170,408,640 bytes。著者記録とのargmaxは
  997/1000で許容基準99.5%を通過するが、`__insufficient_evidence__`から具体クラスへの危険な反転が1件あるため、
  本番配備とcost model更新は停止中。過去のper-row INT8実測は
  0.780 instructions/MAC・T=120で14.57e9だったが、現行block-32のT=120は39.77e9である（5.4節）。
  F32やper-row INT8の過去の実測値は、現行packの費用推定には利用できない。
* 校正は同梱artifactの **5候補限定** temperature（1.4265148639678955）をそのまま使う経路のみ。
  候補数・qtype別の再校正は未実施。
* `crates/verdict-simd` は **int8 経路でモデルに配線されている**（`Linear::forward` が量子化重みを持つとき
  `matmul_i8_blocked` を呼び、softmaxは `softmax_rows_inplace` を使う）。attentionのF32行列積は
  candle gemmを使う。
* `verdict-engine` はraw推論に加えてexecutor互換の`register_schema`・`register_calibration`・`evaluate`を持ち、
  実checkpointのlogitsを型付き`Receipt`へ変換できる。実資金dispatchは引き続き無効である。
* **query 経路（`infer_tokens_query` / `decide_query` / `query_limits`）の上限は5B**である。
  5.3節のper-row INT8で40トークンという記録は、現行block-32には適用できない。
  今回のupdate計測ではT=16で約5.13Bだったが、現行queryの最長入力は再測定していない。
  応答は非certifiedなので、資金を動かす判断には使わない。
* 現行block-32のcanister実行は今回T=8〜120で確認し、成功した最長はT=120である。
  T=128は40Bを超過した。T=121〜127の境界は再測定しておらず、上限拡張は行っていない。
  383トークンの実入力は1 callに収まらない。
* `tools/measure_verdict.py` の全実行はモデル投入とローカルreplicaを必要とするため、既定の
  `verify.py` gateには含まれない。`--verdict-canister` / `--verdict-query` は温まったreplicaに対して
  実行するopt-in検査である。今回の旧版との比較は `tools/compare_verdict.py` で再現できる。
* `tools/verify.py` の `PASS` の意味は `artifacts/verification.json` の各行が示すとおりで、
  実行していない検査は `NOT_RUN` として残る。`python_reference_and_export_tests` は
  `python3 -m unittest discover -s tests` の結果である（torch依存の検査はLaya削除時に撤去済み）。
* 著者記録との一致検証は実施したが、JevBench自体は再評価していない。JevBenchの独立値は Intelligence 59.0 / hard 38.2%（公開GLiClass重み、
  著者エンジン）で、Laya-large は 63.2 / 34.1%。このリポジトリで再測定したものではない。
