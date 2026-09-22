# IC-Verdict — Rust implementation v0.2

**Choice・Noul・Score / 英語 / ICP canister / 型付き判断 / 制約付きmock Tx**

設計だけではなく、Rust workspace、推論演算、canister adapter、テスト、checkpoint変換ツールを実装したソースパッケージです。

> **検証状態 (v0.2):** Rust toolchainのある環境で実際にビルド・テストしました。`cargo test --workspace`は**97件PASS**、`python3 -m unittest discover -s tests`は**33件PASS**、**4 canister分のWasmとCandidを生成済み**です。`tools/verify.py --rust --require-rust --manifest --verdict --local-integration`は**17行中14件PASS**（`artifacts/verification.json`。`NOT_RUN`は温まったreplicaを要する`openjev_canister_instructions`と`openjev_query_canister`、および実送金の`real_ledger_transfer`のみ）。実checkpoint parityはargmax 1000/1000、ローカルreplica統合試験は**replica停止状態からの起動**でPASSしました。
>
> **Layaバックエンドは削除しました。** 実Laya checkpointは一度もロードしておらず（weightは未取得）、合成weightからの外挿では40B上限の12〜14倍だったためです（記録は[docs/archive/PERFORMANCE_MEASUREMENTS.md](docs/archive/PERFORMANCE_MEASUREMENTS.md)）。判断バックエンドはopenJev 151M（GLiClass）に置き換え、実checkpointで実測しています: 著者記録の1000件とargmax 1000/1000一致、120トークンで1決定は既定F32が36.98e9、int8が14.57e9 instructions（[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md)）。実ledger送金は引き続き未検証（heapは実checkpoint warmで1.02 GiBを`heap_bytes`で実測済み）で、`fixtures/`はランダムweightです。
>
> v0.1の「cargoが無くRust未確認」という記述は誤りでした。実際にビルドした結果、`tools/build_one.sh`のbash 3.2非互換、`CARGO_TARGET_DIR`無視、Candle Wasmの`getrandom`欠落という3件の実バグが出たため修正しています。詳細は[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)。

## 1. 入っているもの

| 部分 | ソース実装 | 今回の実行検証 |
|---|---|---|
| 三primitive、整数ppm分布、Scoreの平均・期待段階・上側確率 | Rust core / typed SDK | Rust/Python数値参照とも検証済み |
| schema固定、prefix上限64・全体128tokens、特殊token拒否 | Rust core / HF tokenizer adapter | Rustテスト済み |
| workflow最大3質問、snapshot照合、失効、予算、nonce、業務重複 | Rust core | unit/local replicaとも検証済み |
| unknown時の予約保持、同一payload再送、遅延成功・upgrade扱い | Rust core / canister adapter | message境界を含むlocal replica試験済み |
| openJev GLiClass（ModernBERT-151M） | Candle F32演算＋int8カーネルを実装 | **実checkpointで著者記録とargmax 1000/1000一致**（int8は994/1000） |
| モデルpack・分割アップロード・段階warm-up | Rust loader / engine canister / Python exporter | Python exporterを合成weightで検証 |
| engine / executor / mock ledger | Rustの4 canister | Wasm/Candidをビルド済み。ローカルreplicaで統合試験PASS |
| native例、Candid生成、ローカルmock bootstrap、CI | スクリプト・設定を同梱 | native/Wasm/local統合を実行済み |

**実資金の送金は無効です。** `LimitedLive`を指定しても`LiveDisabled`を返します。実装したoutcallは、専用のmock識別APIを確認したledgerだけに向けます。mock ledgerは残高を模擬するテストダブルで、実在の資産を扱いません。

## 2. ディレクトリ

```text
crates/
  ic-laya-core/       # 型、数学、schema、engine契約、policy、状態機械
  modernbert-candle/  # 共有ModernBERTエンコーダ（attention/RoPE/Linear）
  verdict-candle/     # openJev GLiClassバックエンド（int8対応）
  verdict-simd/       # wasm SIMDカーネル（f32x4 / i32x4.dot_i16x8）
  hf-tokenizer/      # tokenizersのRust adapter
  canister-common/   # stable snapshot、ICRC-1 argument adapter
canisters/
  decision-engine/   # schema/calibration/engine（fixtureモード）
  verdict-engine/    # openJev推論canister（pack upload + warm-up + decide）
  executor/          # 委任・workflow・予算・mock dispatch
  mock-ledger/       # 転送、重複、結果不明のテスト用
fixtures/            # 小さいランダムweight（verdict-tiny、トークナイザ）
tools/              # 数値参照、export、ビルド、ローカル起動
tests/              # Python参照・exportのテスト
artifacts/           # 実行ログと検証状態
.github/workflows/   # Rust native / Wasm CI。未実行
docs/               # 最終設計、16 ADR、実装状況、レビュー
```

## 3. 最初の確認

### Pythonで、この納品時と同じ参照テストを実行

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python tools/verify.py
```

このテストはRustを呼びません。抽象Txモデルは実装の独立した参照で、Rust/ICPの挙動を証明するものではありません。`artifacts/verification.json`が各検査を`PASS / FAIL / NOT_RUN`に分けます。

### Rust toolchainのある環境で

```bash
cargo generate-lockfile
cargo test --workspace
cargo run -p ic-laya-core --example mock_workflow
```

`mock_workflow`は三primitiveを固定のスコアで返し、Unknown→Duplicate成功までの状態遷移を試す例です。**英語を理解するモデルではありません。**
openJevバックエンドの実測・再現手順は[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md)にあります。

実行できた最初の環境で`Cargo.lock`と`rustc --version`等を記録し、lockをレビュー・commitしてください。この環境では依存解決すら実行できないため、架空のlockfileを作成していません。CIも最初にlockを生成する構成です。

## 4. 型付き結果

```rust,ignore
let risk: Score<PaymentRisk> =
    Score::try_from_receipt(&registered_schema, &receipt, &expected_stamp)?;

let mean = risk.mean_ppm();
let severe_tail = risk.tail_ppm(3)?;
```

`PaymentRisk`は`DecisionSchema`の実装で、schema ID/version/primitive/option順を宣言します。`expected_stamp`は呼出し側が登録情報と要求から作る値であり、受け取ったreceiptからコピーして検査を省くものではありません。

Scoreは分布を保持します。期待段階は`Σi·p[i]`、平均はそれを`K−1`で割った整数ppm、tailは指定bin以上の質量です。`top1`・`margin`は分布の特徴であって正答確率ではありません。モデル出力からTx権限は発生しません。

## 5. canisterをローカルで試す

前提はRustとDFINITY SDKのインストール済み環境です。以下はローカル統合試験で実行済みです。defaultのengineには学習済みモデルを含めません。

```bash
rustup target add wasm32-unknown-unknown
bash tools/build_one.sh decision-engine
bash tools/build_one.sh executor
bash tools/build_one.sh mock-ledger

# 別プロセスでローカルreplicaを起動してから実行
# dfx start --background
python tools/local_demo.py
```

`local_demo.py`は`--network local`固定で、`install`のみ使用し、既存stateを消す`reinstall`をしません。ローカルでも同名canisterが既に導入済みなら停止します。既定identityをowner/delegateとして使い、テスト専用calibration、trusted operation、mock grantを登録します。自動的なmainnet配備・入金・実資金送信はしません。

CandidはRustの`export_candid!()`から生成して`build/*.did`へ置きます。未コンパイルの段階で手書きのDIDを正本として同梱していません。

verdict-engineのコード経路は速度を優先したper-row INT8専用です:

```bash
bash tools/build_one.sh verdict-engine
```

packは全2次元重みをper-row INT8（`i8_row_symmetric`）で保持し、旧F32重みとblock-32 packは受理しません。Norm・bias・scaleだけがF32補助値です。精度差・棄権からの反転は測定結果として報告し、自動停止条件にはしません。必要な場合のみ`--min-argmax-ratio`で比較閾値を明示できます。詳細は[per-row復帰の検証記録](docs/PER_ROW_INT8.md)を参照してください。

## 6. 判断バックエンド

判断バックエンドは**openJev 151M（GLiClass uni-encoder）**です。canonical tensor名・prompt形式・headの意味は
[docs/GLICLASS_FORWARD_SPEC.md](docs/GLICLASS_FORWARD_SPEC.md)、実測（instructions・parity・コスト削減）は
[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md)に記録しています。

かつて計画していたLaya checkpointの接続は**削除しました**（weightは未取得で、合成weightからの外挿が40B上限の
12〜14倍だったため）。経緯と外挿の根拠は[docs/archive/](docs/archive/)に残しています。

## 7. 実送金を閉じている理由と残作業

未完了なのは用途別calibration、実モデルworkflow接続のlocal replica受入、cycles実測、対象本番ledgerの確認です。checkpoint parity、Rust/Wasmビルド、instruction/heap、mock非同期障害試験は実施済みです。

現状は安全側の有限容量PoCです。512 workflow、64 registryの上限は残り、compactionは未実装です。executorはstable tableへ移行し、旧bounded snapshotをupgrade時に自動変換します。人間reviewの承認再開endpoint、蒸留、実ledger adapterの有効化は未実装です。

次の着手順は、**INT8品質・instructions gate → 実モデルworkflow接続のlocal replica常時検査 → 用途別calibration → 実ledger adapter**です。[実装状況](docs/IMPLEMENTATION_STATUS.md)と[検証記録](artifacts/verification.json)を先に確認してください。

## 8. openJev（GLiClass / ModernBERT-151M）バックエンド

Layaの代わりに **openJev-verdict 系の151Mモデル**（`heman10x/rlcd-modernbert-151m`、Apache-2.0）を
同じcanister実行モデルで動かす実装を追加しました。設計判断の根拠（40B命令予算に対してパラメータ数が
唯一のレバーであること、151Mなら短い入力は量子化なしでも予算内に入ること）と移植仕様は
[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md) と [docs/GLICLASS_FORWARD_SPEC.md](docs/GLICLASS_FORWARD_SPEC.md)
にあります。

| 追加物 | 役割 |
|---|---|
| `crates/verdict-candle` | GLiClass uni-encoder の forward（encoder は `modernbert-candle` と共有） |
| `canisters/verdict-engine` | pack投入・warm-up・`infer_tokens`・`decide` と instructions 実測。**query経路**（`infer_tokens_query`・`decide_query`・`query_limits`）も実装 |
| `tools/pack_verdict.py` / `tools/verdict-pack` | HF checkpoint → canonical per-row INT8 pack（142テンソル、約145.2 MiB） |
| `tools/make_verdict_fixture.py` | canisterスモーク用の小型pack（同一カーネル） |
| `tools/verdict-upload` | 全entry hashを事前検証してagent経由でpack転送し、queryも呼び出す |
| `tools/verdict_canister.py` | ローカルreplicaで作成→install→投入→推論までを実行（`--query`でqueryスモーク） |
| `tools/measure_verdict.py` | 実benchmark入力をtokenizeし、長さを振って40B予算の上限を実測（`--query`で5B経路） |

検証は次の2つに分かれています。**実checkpointの一致**は
`cargo test --release -p verdict-candle --test golden -- --ignored`（INT8 packが必要）、
**canister上での実行とinstructions実測**は `python3 tools/verdict_canister.py` です。

**上限は外挿ではなく実測で押さえました。** `tools/measure_verdict.py` が実benchmarkの1件
（自然長118トークン）を測り、**T=120（現行カーネルで36,976,071,434 instructions。`overflow-checks`有効時は39.57e9、無効化直後は37.31e9、int8は14.57e9）が成功**
するところまで確認しています。T=126は予算ガードの計算値41.74e9が40Bを超えるため拒否される設計です（記録された実測点はT=120のみ）（[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md) 5.1.1節、
`artifacts/verdict_sweep.json`）。151Mは量子化なしでも短い入力は1 callに収まりますが、
JevBench生入力は中央値95・平均95.8トークンで、予算外は上側2.5%のみです（「平均383トークンで3.2倍超過」は誤りと判明済み）。量子化なし、校正は同梱artifactの5候補用temperatureのみ、
という制約はそのまま残っています。

**query呼び出し（5B上限）も実装・実測しました。** `infer_tokens_query`・`decide_query`・`query_limits` を
`verdict-engine` に追加し、同じ費用モデルから上限を導出して事前に`Capacity`で拒否します。実測した上限は
**既定F32で14トークン**（`artifacts/verdict_query_sweep.json`）、**int8ビルドでは40トークン**
（`artifacts/verdict_query_sweep_int8.json`）。queryはcyclesを消費せず合意も不要ですが、応答は
**certifiedではない**ため、資金を動かす判断には使いません（[docs/VERDICT_ENGINE.md](docs/VERDICT_ENGINE.md) 5.3節）。
