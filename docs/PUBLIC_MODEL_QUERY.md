# Public model query guide / 汎用モデルqueryの使い方

## English

The production canister is `qojfj-6qaaa-aaaam-qjkaq-cai`. Its Candid interface is public on the [ICP Dashboard](https://dashboard.internetcomputer.org/canister/qojfj-6qaaa-aaaam-qjkaq-cai). This API **selects among short candidate descriptions** given some context. It is not a chat or free-text generation API. Other applications can use the same `decide_query` method as the Tetris demo.

### Choose the method

| Method | Purpose | Caller |
| --- | --- | --- |
| `decide_query` | Choose an option and return scores for all options | Public query, including anonymous callers |
| `info` | Check `warmed` and the 32-byte model digest | Public query |
| `query_limits` | Read the current token limit and instruction budget | Public query |
| `infer_tokens_query` | Run inference on token IDs directly | Query for the owner or allowlisted callers |

Use `decide_query` when your application has text and candidate answers. The canister no longer has Tetris-specific methods. `decide_query` must be called as a query. If a canister update needs a consensus-executed decision, see the paid `decide` update and the [cycles billing guide](CYCLES_BILLING.md).

### Request fields and limits

`decide_query` takes one `DecideRequest` record.

| Field | Candid type | Meaning |
| --- | --- | --- |
| `state` | `text` | Context for the decision; must not be empty |
| `question` | `text` | The question; may be empty |
| `options` | `vec record { id : text; "text" : text }` | 1–24 options with nonempty, unique IDs and nonempty descriptions |
| `abstention` | `bool` | Append an “insufficient evidence” option when `true` |
| `temperature` | `float64` | Softmax temperature, greater than `0` and at most `100`; use `1.0` for the raw distribution |

With `abstention=true`, the reply also contains the option ID `__insufficient_evidence__`. Do not use that ID for one of your own options. Reserved model delimiter tokens are also rejected in the question, state, and option descriptions.

The public query enforces UTF-8 byte limits before tokenization: 16 KiB for the combined state, question, option IDs, and descriptions; 4 KiB for the question; 1 KiB per option description; and 128 bytes per option ID. A separate limit applies to the complete tokenized model input. Option labels and delimiters consume tokens too, so the byte counts alone do not determine the token count.

As checked on 2026-09-24, `query_limits().max_tokens` is **52** and `budget` is **5,000,000,000 instructions**. These values can change; read `query_limits` instead of hardcoding them. After an upgrade, `info().warmed` is `false` until the model is ready again.

### Call it with the `icp` CLI

Fetch the exact public Candid from metadata, then check the model and its limits:

```sh
CANISTER=qojfj-6qaaa-aaaam-qjkaq-cai
icp canister metadata "$CANISTER" candid:service -n ic --identity anonymous --json \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["value"])' \
  > verdict-engine.did

icp canister call "$CANISTER" info '()' --query -n ic --identity anonymous --candid verdict-engine.did
icp canister call "$CANISTER" query_limits '()' --query -n ic --identity anonymous --candid verdict-engine.did
```

This decision request was run successfully as an anonymous query on production. The choice can change with the input or model state.

```sh
icp canister call "$CANISTER" decide_query \
  '(record {
    state = "Need a brief greeting";
    question = "Which reply fits?";
    options = vec {
      record { id = "short"; "text" = "Hello!" };
      record { id = "long"; "text" = "Hello, how are you today?" }
    };
    abstention = false;
    temperature = 1.0
  })' \
  --query -n ic --identity anonymous --candid verdict-engine.did
```

A successful call returns `variant { Ok = record { ... } }`; a rejected request or inference failure returns `Err`. This example returned `ids = ["short", "long"]`, `selected = "short"`, and `input_tokens = 29`. One example does not establish the model's general accuracy.

### Call it from TypeScript

Create an actor from the public Candid's generated types and IDL. The working IDL and reply checks in this repository are in the [Tetris query client](../web/tetris/src/placement-query-api.ts). Replace the candidate descriptions with those for your application.

```ts
const info = await actor.info();
const limits = await actor.query_limits();
if (!info.warmed || limits.max_tokens < 1) throw new Error("Model unavailable");

const result = await actor.decide_query({
  state: "Need a brief greeting",
  question: "Which reply fits?",
  options: [
    { id: "short", text: "Hello!" },
    { id: "long", text: "Hello, how are you today?" },
  ],
  abstention: false,
  temperature: 1.0,
});
if ("Err" in result) throw new Error(`Model query failed: ${JSON.stringify(result.Err)}`);
// Check result.Ok.ids and result.Ok.model before using result.Ok.selected.
```

This snippet assumes you have already created `actor`. In TypeScript actor bindings, Candid `nat64` is represented as `bigint`; compare `model` as a 32-byte array.

### Read the reply and handle failures

| `Ok` field | Meaning |
| --- | --- |
| `ids` | Order of the result arrays; matches the submitted option order, with abstention appended when enabled |
| `selected` | Chosen option ID, possibly the abstention ID |
| `logits` | Raw scores before temperature scaling |
| `probabilities` | Softmax values after temperature scaling, in `ids` order |
| `confidence` | Softmax value of `selected`, not a calibrated probability of being correct |
| `model` | 32-byte model digest; compare it with `info().model` |
| `input_tokens` | Actual token count sent to the model |
| `measured_instructions` | Measured model forward-pass instructions, not total query cost |

Before using the decision, check the count and order of `ids`, that `selected` belongs to it, that result arrays have the expected lengths and finite values, and that `model` matches the digest you checked. Treat an `Err` or transport failure as “no decision”; retry or report it according to your application's needs.

Common errors include `TooLong` for byte, option-count, or token limits; `Capacity` for the 5B-instruction preflight guard; `Invalid` for duplicate IDs or reserved tokens; `Numeric` for an invalid temperature; and `ModelUnavailable` while the model is not warmed. Calling a query method through replicated update execution returns `Denied`.

A query response is not a consensus-certified decision record. Do not use it as the authority for moving funds or for a result that multiple canisters must agree on.

## 日本語

本番canisterは `qojfj-6qaaa-aaaam-qjkaq-cai` です。[ICP Dashboard](https://dashboard.internetcomputer.org/canister/qojfj-6qaaa-aaaam-qjkaq-cai)で公開Candidを確認できます。このAPIは、短い背景文と候補文からモデルに**候補を選ばせる**ものです。自由文を生成するチャットAPIではありません。Tetris以外のアプリからも同じ `decide_query` を利用できます。

### どのメソッドを呼ぶか

| メソッド | 用途 | 呼び出し元 |
| --- | --- | --- |
| `decide_query` | 文章と選択肢を渡し、選択IDと各候補のスコアを受け取る | 匿名を含む公開query |
| `info` | モデルの `warmed` 状態と32バイトのモデルdigestを確認する | 公開query |
| `query_limits` | その時点のトークン上限と命令予算を確認する | 公開query |
| `infer_tokens_query` | トークンID列を直接推論する | ownerまたは許可済みcallerのquery |

外部アプリで文章と候補を扱うなら `decide_query` を選びます。現在のcanisterにTetris専用メソッドはありません。`decide_query` はquery専用です。canisterのupdate処理から合意対象の判断が必要な場合は、有料updateの `decide` と[サイクル課金仕様](CYCLES_BILLING.md)を確認してください。

### `decide_query` の入力

引数は1つの `DecideRequest` レコードです。

| フィールド | Candid型 | 意味 |
| --- | --- | --- |
| `state` | `text` | 判断の背景。空文字は不可 |
| `question` | `text` | 質問。不要なら空文字にできる |
| `options` | `vec record { id : text; "text" : text }` | 1〜24件の候補。IDと候補文は空にできず、IDは重複不可 |
| `abstention` | `bool` | `true` なら「情報不足」の候補を末尾に追加する |
| `temperature` | `float64` | スコア分布の温度。`0`より大きく`100`以下。元の分布を見る例では`1.0` |

`abstention=true` の場合、返る候補IDに `__insufficient_evidence__` が追加されます。このIDを自分の候補に使うと重複エラーになります。候補文・背景文・質問文に、モデルの区切りに使う予約トークンを含めることもできません。

公開queryでは、文字列のUTF-8バイト数を追加で制限します。`state`・`question`・すべての候補IDと候補文を足して16 KiB、`question`単体で4 KiB、候補文は1件1 KiB、候補IDは1件128バイトまでです。これとは別に、モデル入力全体のトークン上限があります。候補ラベルや区切りもトークンに数えるため、送信文字列だけから正確なトークン数は決まりません。

`query_limits().max_tokens` は現在 **52トークン**、`budget` は **5,000,000,000命令**です（2026-09-24確認）。上限は設定変更で動くため、クライアントで固定せず `query_limits` を読んでください。モデルがupgrade直後などで再準備中なら、`info().warmed` が `false` になります。

### CLIから試す

`icp` CLIを使う例です。公開metadataをJSONから取り出すと、公開した `.did` をそのまま保存できます。

```sh
CANISTER=qojfj-6qaaa-aaaam-qjkaq-cai
icp canister metadata "$CANISTER" candid:service -n ic --identity anonymous --json \
  | python3 -c 'import json,sys; sys.stdout.write(json.load(sys.stdin)["value"])' \
  > verdict-engine.did

icp canister call "$CANISTER" info '()' --query -n ic --identity anonymous --candid verdict-engine.did
icp canister call "$CANISTER" query_limits '()' --query -n ic --identity anonymous --candid verdict-engine.did
```

次の判定例は本番の匿名queryで実行済みです。モデルの選択結果は、入力やモデルの状態に応じて変わり得ます。

```sh
icp canister call "$CANISTER" decide_query \
  '(record {
    state = "Need a brief greeting";
    question = "Which reply fits?";
    options = vec {
      record { id = "short"; "text" = "Hello!" };
      record { id = "long"; "text" = "Hello, how are you today?" }
    };
    abstention = false;
    temperature = 1.0
  })' \
  --query -n ic --identity anonymous --candid verdict-engine.did
```

成功時は `variant { Ok = record { ... } }`、入力や推論に問題があれば `Err` を返します。上の例では `ids = ["short", "long"]`、`selected = "short"`、`input_tokens = 29` という応答を確認しました。この1件はモデルの一般的な正答率を示すものではありません。

### ブラウザやTypeScriptから呼ぶ場合

公開Candidから生成した型とIDLでactorを作り、`info`、`query_limits`、`decide_query` の順に利用します。実際に動いているIDL定義と応答検証は[Tetrisのqueryクライアント](../web/tetris/src/placement-query-api.ts)を参照してください。別の用途では、候補文だけをそのアプリの内容に置き換えます。

```ts
const info = await actor.info();
const limits = await actor.query_limits();
if (!info.warmed || limits.max_tokens < 1) throw new Error("Model unavailable");

const result = await actor.decide_query({
  state: "Need a brief greeting",
  question: "Which reply fits?",
  options: [
    { id: "short", text: "Hello!" },
    { id: "long", text: "Hello, how are you today?" },
  ],
  abstention: false,
  temperature: 1.0,
});
if ("Err" in result) throw new Error(`Model query failed: ${JSON.stringify(result.Err)}`);
// result.Ok.idsとmodelを照合してからresult.Ok.selectedを使用する。
```

これは既に生成したactorを使う呼び出し部分の例です。Candidの `nat64` はTypeScriptのactorでは `bigint` として扱い、`model` は32バイトの配列として比較します。

### 応答の読み方と失敗時の扱い

`Ok` の主なフィールドは次のとおりです。

| フィールド | 意味 |
| --- | --- |
| `ids` | 結果配列の順序。通常は送った候補IDの順序と同じ |
| `selected` | モデルが選んだ候補ID。棄権が有効なら追加IDになる場合もある |
| `logits` | 温度適用前の各候補の生スコア |
| `probabilities` | 温度適用後のsoftmax値。`ids`と同じ順序 |
| `confidence` | 選択候補のsoftmax値。正答率の保証や校正済み確率ではない |
| `model` | 32バイトのモデルdigest。`info().model`との照合に使える |
| `input_tokens` | 実際にモデルへ渡したトークン数 |
| `measured_instructions` | モデルforward passの計測命令数。query全体の費用ではない |

アプリ側では `ids` の個数・順序、`selected` がその中にあること、配列長と数値の有限性、`model` の一致を確認してから選択結果を使ってください。`Err` や通信失敗の場合は「判定できなかった」と扱い、用途に応じて再試行または利用者へ通知します。

よくある `Err` は、文字列・候補数・実トークン数の上限を超えた `TooLong`、5B命令予算の事前ガードにかかった `Capacity`、予約トークンや重複IDなどの `Invalid`、不正な温度値の `Numeric`、モデルが温まっていない `ModelUnavailable` です。queryをupdate経由で実行しようとすると `Denied` になります。

queryの結果は合意済み・認証済みの判断記録ではありません。資金移動など、複数canisterで同じ結果を確定する必要がある処理の根拠には使わないでください。
