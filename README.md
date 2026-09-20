# IC-Laya — Rust implementation v0.2

**Choice・Noul・Score / 英語 / ICP canister / 型付き判断 / 制約付きmock Tx**

設計だけではなく、Rust workspace、推論演算、canister adapter、テスト、checkpoint変換ツールを実装したソースパッケージです。

> **検証状態 (v0.2):** Rust toolchainのある環境で実際にビルド・テストしました。`cargo test --workspace`は**62件PASS**、`cargo check -p decision-engine --features candle`はPASS、**3 canister分のWasmとCandidを生成済み**（Candle込みで4.8 MiB）、Python参照テスト45件もPASSです。`tools/verify.py --rust --require-rust`は実行可能な8項目すべてPASSです。
>
> ただし**実Laya checkpointのparity、ICP上のinstructions/heap実測、実ledger送金は未検証**です。`fixtures/`はランダムweightで言語理解を証明しません。ビルド成功を性能・品質の証拠とは扱っていません。
>
> v0.1の「cargoが無くRust未確認」という記述は誤りでした。実際にビルドした結果、`tools/build_one.sh`のbash 3.2非互換、`CARGO_TARGET_DIR`無視、Candle Wasmの`getrandom`欠落という3件の実バグが出たため修正しています。詳細は[docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)。

## 1. 入っているもの

| 部分 | ソース実装 | 今回の実行検証 |
|---|---|---|
| 三primitive、整数ppm分布、Scoreの平均・期待段階・上側確率 | Rust core / typed SDK | 独立Python数値参照を検証。Rustは未実行 |
| schema固定、prefix上限64・全体128tokens、特殊token拒否 | Rust core / HF tokenizer adapter | Rustテストを用意。未実行 |
| workflow最大3質問、snapshot照合、失効、予算、nonce、業務重複 | Rust core | Python抽象状態機械は検証。Rustは未実行 |
| unknown時の予約保持、同一payload再送、遅延成功・upgrade扱い | Rust core / canister adapter | 同上。ICPのmessage境界は未実行 |
| ModernBERT + decision Transformer + marker scorer | Candle F32演算を実装 | 合成weightのPyTorch/NumPy一致のみ。Candle・実Layaは未確認 |
| モデルpack・分割アップロード・段階warm-up | Rust loader / engine canister / Python exporter | Python exporterを合成weightで検証 |
| engine / executor / mock ledger | Rustの3 canister | ソースのみ。Wasm未ビルド |
| native例、Candid生成、ローカルmock bootstrap、CI | スクリプト・設定を同梱 | CIとdfx手順は未実行 |

**実資金の送金は無効です。** `LimitedLive`を指定しても`LiveDisabled`を返します。実装したoutcallは、専用のmock識別APIを確認したledgerだけに向けます。mock ledgerは残高を模擬するテストダブルで、実在の資産を扱いません。

## 2. ディレクトリ

```text
crates/
  ic-laya-core/       # 型、数学、schema、engine契約、policy、状態機械
  laya-candle/        # batch=1 F32推論、canonical model pack
  hf-tokenizer/      # tokenizersのRust adapter
  canister-common/   # stable snapshot、ICRC-1 argument adapter
canisters/
  decision-engine/   # 推論。defaultはモデル未ロード
  executor/          # 委任・workflow・予算・mock dispatch
  mock-ledger/       # 転送、重複、結果不明のテスト用
fixtures/            # 小さいランダムweight。実Layaではない
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
python tools/generate_fixtures.py
python -m unittest discover -s tests -v
python tools/verify.py
```

このテストはRustを呼びません。抽象Txモデルは実装の独立した参照で、Rust/ICPの挙動を証明するものではありません。`artifacts/verification.json`が各検査を`PASS / FAIL / NOT_RUN`に分けます。

### Rust toolchainのある環境で

```bash
cargo generate-lockfile
cargo test --workspace
cargo run -p ic-laya-core --example mock_workflow
cargo run -p laya-candle --bin laya-infer -- \
  fixtures/tiny-prenorm fixtures/tiny-prenorm/input.json
cargo check -p decision-engine --features candle
```

`mock_workflow`は三primitiveを固定のスコアで返し、Unknown→Duplicate成功までの状態遷移を試す例です。**英語を理解するモデルではありません。** `laya-infer`の同梱fixtureも、ランダムweightのニューラル演算テストです。

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

前提はRustとDFINITY SDKのインストール済み環境です。以下は**未実行の手順**です。defaultのengineには学習済みモデルを含めません。

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

実推論featureのビルド候補:

```bash
IC_LAYA_CANDLE=1 bash tools/build_one.sh decision-engine
```

Candle/tokenizersのWasm依存経路は未検証です。ブラウザWasm対応をICP互換性の証拠にしていません。CIではこのビルドも必須にして、不適合を隠さない設定にしています。

## 6. 実Layaを接続する場所

F32推論演算は書いてありますが、**公開checkpointを読み込んだ実測はありません**。本パッケージのcanonical tensor名を、元checkpointの実際の名前だと見なさないでください。元実装とconfigから、QKV順、RoPE、norm、GeGLU、decision head、scorer、qtype順、入力token列を確認して対応表を作ります。

```bash
python tools/pack_checkpoint.py inspect /path/to/checkpoint
python tools/pack_checkpoint.py export \
  --source /path/to/checkpoint \
  --config reviewed-runtime-config.json \
  --mapping reviewed-tensor-map.json \
  --tokenizer /path/to/tokenizer.json \
  --repo convaiinnovations/laya-typed-decisions \
  --revision ACTUAL_40_CHARACTER_COMMIT \
  --qtypes 0 1 2 \
  --out checkpoints/laya-f32
```

上記qtype順は**形式例であり未確認の値**です。元コードで確認した順に置き換えてください。`--revision`も実在するimmutable revisionを入力します。exporterはこの値の形式は検査しますが、HFへ問い合わせて真正性を確認しません。

対応表はcanonical name→source nameのJSONで、明示的transposeまたはaxis-0結合も指定できます。自動推測・任意Python式・pickle・remote code実行はしません。詳細は[MODEL_PORT.md](docs/MODEL_PORT.md)に記載しました。

## 7. 実送金を閉じている理由と残作業

未完了なのは実モデルparity、用途別calibration、Rust/Wasmビルド、ICP instruction/heap実測、非同期障害試験、対象本番ledgerの確認です。実装済みコードがあることは、これらを通過したことを意味しません。

現状は安全側の有限容量PoCです。512 workflow、64 registry、engine cache1024件で、削除・compactionは実装していません。上限で停止し、古いnonceや不明Txを消して処理を続けることはしません。stable storageはbounded snapshotで、全量再保存のコストがあり、本番の高頻度処理にはstable table化が必要です。人間reviewの承認再開endpoint、INT8、蒸留、実ledger adapterの有効化も未実装です。

次の着手順は、**Rustのビルド修正 → 合成fixtureでCandle照合 → 実Layaのtensor/tokenizer/logit照合 → ICP性能 → fault injection**です。[実装状況](docs/IMPLEMENTATION_STATUS.md)と[自己レビュー](docs/IMPLEMENTATION_REVIEW.md)を先に確認してください。
