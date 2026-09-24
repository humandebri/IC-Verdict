# 汎用モデルqueryの使い方

本番canisterは `qojfj-6qaaa-aaaam-qjkaq-cai` です。[ICP Dashboard](https://dashboard.internetcomputer.org/canister/qojfj-6qaaa-aaaam-qjkaq-cai)で公開Candidを確認できます。このAPIは、短い背景文と候補文からモデルに**候補を選ばせる**ものです。自由文を生成するチャットAPIではありません。Tetris以外のアプリからも同じ `decide_query` を利用できます。

## どのメソッドを呼ぶか

| メソッド | 用途 | 呼び出し元 |
| --- | --- | --- |
| `decide_query` | 文章と選択肢を渡し、選択IDと各候補のスコアを受け取る | 匿名を含む公開query |
| `info` | モデルの `warmed` 状態と32バイトのモデルdigestを確認する | 公開query |
| `query_limits` | その時点のトークン上限と命令予算を確認する | 公開query |
| `infer_tokens_query` | トークンID列を直接推論する | ownerまたは許可済みcallerのquery |

外部アプリで文章と候補を扱うなら `decide_query` を選びます。現在のcanisterにTetris専用メソッドはありません。`decide_query` はquery専用です。canisterのupdate処理から合意対象の判断が必要な場合は、有料updateの `decide` と[サイクル課金仕様](CYCLES_BILLING.md)を確認してください。

## `decide_query` の入力

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

## CLIから試す

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

## ブラウザやTypeScriptから呼ぶ場合

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

## 応答の読み方と失敗時の扱い

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
