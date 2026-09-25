# 推論updateのサイクル払い

`infer_tokens`、`decide`、`decide_batch`は、必要なサイクルを添付したcanisterから利用できます。owner・allowlistにも無料枠はありません。`evaluate`にも同じ課金を適用しますが、既存のworkflow認可・quota・キャッシュを維持します。管理操作はowner限定です。

通常のqueryは無料です。汎用推論queryのallowlistと、公開Tetris queryの有効・無効設定を維持します。5つの推論queryをupdate経路で実行すると、推論前に`Denied`を返し、添付サイクルは受領しません。

## 料金と添付額

料金は **3 ×（実行固定費 + ceil(実測命令数 × 命令単価の分子 / 分母)）** です。命令数はICの`instruction_counter()`で、引数デコードから応答のCandidエンコードまでを計測します。最後の課金・応答コピー処理、継続的なメモリ保管料、呼び出し側の通信費は含めません。canister残高の減少額そのものを測る方式ではありません。

1. `cycles_pricing()`をqueryし、`required_attachment`を取得します。
2. 呼び出し元canisterがその額以上を`.with_cycles(...)`で添付します。
3. 推論前に40B命令上限分の資金を確認します。実行後に実測額だけ受領し、未受領分はICが返却します。

料金設定・資金が不足する呼び出しは推論前に拒否し、受領額は0です。資金確認後の通常の`Result::Err`は、検証に使った実測分も課金します。Wasm trapでは受領もロールバックされるため、その実行費はサービス側の負担です。`evaluate`のキャッシュ再取得は再推論せず、その呼び出しの検証・キャッシュ読出し・応答生成分を課金します。

## 設定と移行

初回install・旧版からのupgrade直後は料金未設定で、推論updateを停止します。ownerが`set_execution_pricing(opt record { ... })`で**対象サブネットの実際の単価**を設定して有効にします。`null`で停止できます。設定はupgradeを越えて保持されます。モデル用の`set_cost_model`は命令数の推定・入力制限用であり、サイクル単価とは別です。

[ICの料金表](https://docs.internetcomputer.org/references/cycle-costs/)（2026-09-23確認）では13ノードの例は`base_cycles=5_000_000`、`instruction_cycles_numerator=1`、`instruction_cycles_denominator=1`です。この設定で上限添付額は120,015,000,000 cycles、実測1B命令なら受領額は3,015,000,000 cyclesです。他のサブネットや料金改定後にこの値をそのまま使わないでください。

ブラウザ・通常のCLI ingressからは直接サイクルを添付できません。推論updateには、呼び出し元canisterまたは認可されたproxyが必要です。従来のowner直呼び出しも支払いが必要になります。query利用のTetris UIには変更不要です。

## executorと計測ツールの設定

executorのownerは、engineの`cycles_pricing().required_attachment`を確認して、executorの`set_engine_cycles(額:nat)`へ設定します。`engine_cycles()`で読み戻せます。評価と最大2回の送信（初回＋再送）は、それぞれこの額をexecutorの残高から添付し、未受領分はexecutorへ戻ります。executor自体の実行費・通信費も賄える残高を用意してください。添付額だけでも残高を超える場合は、評価状態や再送回数を変更せず`Budget`を返します。

設定は専用のstable memory ID 6に保存し、既存のworkflowの保存形式を変更しません。旧版からのupgrade時は0で、無料の`decision-engine`との接続を維持します。有料engineでは、評価を開始する前に料金設定・executorのallowlist登録・添付額設定が必要です。engineの料金を変更した場合も添付額を更新してください。不足額やengine側の停止を見逃して評価を送ると、既存のエラー処理どおり`NeedsReview`になります。

`verdict-upload`、`profile_query.py`、`verdict_regression.py`、`compare_verdict.py`、`measure_verdict.py`、`verdict_canister.py`の有料updateには、`--proxy PRINCIPAL --max-cycles N`を指定します。`N`は1回ごとの添付上限（10進数）です。各呼び出しの直前に料金をqueryし、上限以内の必要額だけを添付します。`verify.py --verdict-canister`にも同じ引数を渡せます。`compare_verdict.py --unpaid-baseline`は、課金導入前のbaselineだけを明示的に直接呼び出す指定です。無料queryや管理操作はproxyへ転送しません。料金未設定・上限超過は推論送信前に失敗します。上限は実行全体の総額制限ではありません。

proxyは[icp-cliのproxyインターフェース](https://github.com/dfinity/icp-cli/blob/main/crates/icp-canister-interfaces/src/proxy.rs)を使用します。実行するCLI identity、または`verdict-upload --pem`のidentityをproxyのcontrollerとして事前に認可し、proxyに資金を用意してください。`evaluate`の認可対象は転送後のcallerであるproxyです。`verdict_regression.py`は指定proxyをengineのallowlistへ登録します。ツールはproxyの作成・controller追加・資金補充を自動実行しません。

必要な場合のみ、ownerとして`--execution-pricing BASE,NUMERATOR,DENOMINATOR`を指定して単価を設定できます。既存設定を使う場合は省略します。次は既に温めたローカルfixtureでの例です（proxyとowner PEMは事前準備）。

```sh
target/release/verdict-upload --url http://localhost:8011 --canister "$ENGINE" \
  --pem "$OWNER_PEM" --no-upload --proxy "$PROXY" --max-cycles 120015000000 \
  --execution-pricing 5000000,1,1 --infer 1,3,11,3,12,2
python3 tools/profile_query.py --canister "$ENGINE" --identity "$OWNER_IDENTITY" \
  --mode update --lengths 6 --repeats 1 --out /tmp/paid-profile.json \
  --proxy "$PROXY" --max-cycles 120015000000
```

後者のトークンIDは実モデル用です。fixtureの計測には前者を使います。`verdict_canister.py`はローカル再installを行うため、有料スモーク時には`--execution-pricing`も指定して初期設定してください。無料スモークだけなら`--query`で支払い引数を省略できます。

## ローカル検証

`bash tools/build_one.sh verdict-engine`でWasmを生成してから、ローカルPocketICバイナリを指定して実行します。

```sh
POCKET_IC_BIN=/path/to/pocket-ic cargo test -p verdict-engine --test cycles_billing -- --ignored --nocapture
```

事前に`bash tools/build_one.sh executor`も実行します。同じテストが実executorの課金付き評価、owner認可、添付資金不足時の状態保持、upgrade後の設定保持を確認します。executor旧版からの移行には`EXECUTOR_OLD_WASM=/path/to/old/executor.wasm`を指定します。

PocketIC serverは15.xを使います。`decision-engine`と`mock-ledger`のWasmもビルドした上で、`cargo test -p verdict-engine --test executor_reconcile -- --ignored --nocapture`を実行すると、無料engineとの互換性、および応答を失ったmock送金の保留・upgrade・期限切れ・grant失効後の公開APIによる回復を検査します。未検出・不一致・通信失敗では予約を保持し、成功の再照合では二重計上しないことも確認します。

Tetris対応・課金導入前のengineからの移行も確認する場合は、さらに`VERDICT_OLD_WASM=/path/to/old/verdict-engine.wasm`を指定します。テストは使い捨てのローカルcanisterだけを作り、既存環境や本番には接続しません。
