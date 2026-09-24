# Terrarium・Last Exit・Hundred のローカル適性検証

2026-09-23。現在の151Mモデルでは、今回試した入力形式のいずれも、小さな
プロトタイプに進む目安（全体90%以上、各分類80%以上）を満たさなかった。
作品をそのまま移植することは勧めない。別の入力形式でも必ず失敗するという証明ではない。

## 対象と範囲

- [Terrarium](https://github.com/TheGali/terrarium): 動物と周囲の物の説明から、
  食べる・逃げる・調べる・無視するを選ぶ部分。原作の多項目判定や移動は再現しない。
- [Last Exit](https://github.com/0x963D/last-exit): 検問での主張と証拠について、
  整合・矛盾・情報不足を区別する部分。会話全体、検査行動や駆け引きは未検証。
- [Hundred](https://github.com/jammaru/jev-lab): 人物の状況から、食事・休息・仕事・
  他者への援助を選ぶ部分。100人の同時稼働、関係性・記憶・街のシミュレーションは未検証。

これは原作コードのベンチマークではなく、それぞれの基本判断を短い英文にした
能力スクリーニング。原作のTypeSafe Jevの実績を、このモデルの実績とは扱わない。

## 方法

各作品24局面、各局面に2種類の言い換え、候補を通常順・逆順で各1回提示し、
各作品96回答を取得した。元の局面は24件であり、96件の独立した問題ではない。
候補ごとの正解数は均等。12組の局面ペアでは、重要な条件を変えて正解も変えた。
Terrariumの行動正解は、この簡略化したゲームの意図として人手で設定したもの。
「調べる／無視する」には主観的な余地があるため、食用でない物を食べる等の具体例も確認した。

最初は短い候補名を使用。結果を確認した後、同じ局面で候補を説明文にした追加実験を1回行った。
追加実験は探索的な比較であり、独立した未知問題による評価ではない。
両方とも全入力・正解・候補順を推論前に保存し、モデルには正解や局面IDを送らなかった。

| 作品 | 短い候補 | 説明付き候補 |
| --- | --- | --- |
| Terrarium | eat / flee / investigate / ignore | eat the food / escape danger / inspect a novel object / leave it alone |
| Last Exit | consistent / contradictory / insufficient information | evidence supports the claim / evidence contradicts the claim / evidence cannot verify the claim |
| Hundred | eat / rest / work / help | eat a meal / sleep to recover / start the job / assist someone |

既存の `target/release/tetris_probe` を使い、ネイティブCPU推論のみ実行した。
Canisterや外部モデルへの呼び出し、重み変更、学習、デプロイは一切ない。
入力上限は既存probeの52トークンを維持。全文英語で、日本語の自由入力は未検証。
モデル・tokenizer・実行ファイルのハッシュは `protocol.json` に記録した。

## 結果

| 作品 | 短い候補 | 説明付き候補 | 説明付きでも残る問題 |
| --- | ---: | ---: | --- |
| Terrarium | 63/96（65.6%） | 72/96（75.0%） | 食べる24/24、逃げる24/24だが、調べる15/24、無視9/24 |
| Last Exit | 43/96（44.8%） | 46/96（47.9%） | 情報不足が0/32。断定を保留できない |
| Hundred | 54/96（56.3%） | 76/96（79.2%） | 食事21/24、休息16/24、仕事22/24、援助17/24 |

説明付き形式でも、候補の順番だけを変えた際の回答一致は、Terrarium 42/48、
Last Exit 44/48、Hundred 41/48だった。
条件を変えた2局面×言い換え2通り×候補順2通りをすべて正解したペアは、
順に4/12、0/12、2/12。簡単な問題での正解が、条件変更に安定して追従するわけではない。

説明付き形式の実際の誤答例（日本語は説明用の訳）:

- 空腹のウサギがプラスチックの人参を見つける → 食べる。
- 空腹の猫が魚の写真を見つける → 食べる。
- 「薬を運んでいる」という主張、貨物は未検査 → 矛盾。
- 一晩寝ていないが食事は済んだ人物、ベッドが利用可能 → 食べる。
- 食事・休息が足りた人物の近くで友人が溺れている → 休息。

## 判断

Terrariumは本物／模造品の違いを扱うことがゲームの面白さになるが、そこで失敗した。
「食べる」の正答率が高くても、食べてはいけない物まで選ぶため信頼できない。
Last Exitは情報不足と矛盾の区別ができず、会話を長くする前の基本段階で不合格。
Hundredが今回もっとも高いが、他の欲求を満たしていると明記した簡単な局面でも誤るため、
複雑な住民生活を任せられる根拠はない。

この結果から画面や原作規模のゲームの実装には進まない。
残る可能性を調べるなら、プロンプト変更と独立した評価問題を伴う別実験として扱う。
特定の例だけ動くものを、一般的な意味理解が成立したデモとはしない。

## 証跡と再実行

```sh
python3 -m unittest discover -s tools -p test_semantic_game_study.py
python3 tools/semantic_game_study.py
python3 tools/semantic_game_study.py --descriptive
```

`artifacts/semantic-game-study/` および `descriptive/` 配下に、推論前の
`fixtures.json`、`protocol.json`、各回答の `responses.json`、実際の入力と確率を含む
`inferences.json`、集計の `result.json` を保存。
計576回のネイティブ推論が完了し、入力上限による除外はなかった。
4件のテストが通過。全576回答の候補対応・最大確率の選択・確率値・入力上限・
fixtureハッシュ・集計値を再確認した（`audit.json`）。UI変更はない。
