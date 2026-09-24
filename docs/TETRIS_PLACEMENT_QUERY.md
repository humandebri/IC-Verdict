# 4候補・1 queryのTetris

候補の多様化・相対コスト表現を使う改善版を本番へ反映。比較・実測・再現手順は [TETRIS_PROMPT_SEARCH.md](TETRIS_PROMPT_SEARCH.md) を参照。

現行の公開UIは、開始をブラウザ内で処理し、1ターンにつき `tetris_placement_query` を1回だけ呼ぶ。自動リトライなどのHTTP再送を除くアプリ側の呼び出し数。updateのゲーム作成・着手APIは呼ばない。

入力は盤面200セル、seed、turn、消去行数、モデル/評価関数モード。クライアントは候補・評価値・選択IDを送らない。
Canisterが到達可能な合法配置を生成し、同じ `(lines, holes, height)` をまとめ、特徴量の距離で最大4種類の候補を選ぶ。総合評価値では選別しない。
消去行は候補内の最多消去行との差 `missed` に変換する。各特徴を独立に `min/mid/max/equal` と表現し、151Mは `min holes,mid height,max missed` のような最大4ラベルから選ぶ。指示は `Min holes`、最大41トークン。評価関数は同じ候補について `10*lines - 8*holes - height` を比較用に計算する。モデルなしモードも1 queryで盤面計算を返す。
推論はqueryの5B命令ガードを通る。エラー時に評価関数へフォールバックしない。12候補の段階選考は採用しない。

候補生成・特徴量・推論・次盤面の計算はCanister内で行う。返された盤面の保持と着手アニメーションはブラウザで行う。queryはCanister状態を変更しないので、判定ログは永続・合意済み記録ではない。UIから保存するJSONも入出力ログであり、証明書ではない。
旧update版のstable snapshotと過去の証明付き記録は互換性のため保持する。旧update版のモデル予算は公開移行時に0へ設定し、旧モデルupdateを無効化する。新query版には50手の共有予算や32ゲーム枠は適用されない。1ゲーム128ターンの区切りは維持する。

## 検証

```sh
QUERY_CANISTER=<local-id> node --import ./web/tetris/node_modules/tsx/dist/loader.mjs web/tetris/scripts/placement-query-smoke.ts
npm --prefix web/tetris test
npm --prefix web/tetris run build
```

ローカル検証ではHTTP `/call` を検出すると即失敗するfetchラッパーを使用。3モデル手・3比較手を実行して各手の候補・盤面を再計算し、全推論が5B命令未満、ゲーム枠とupdate推論予算が不変であることを確認する。
