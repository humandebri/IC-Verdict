# 汎用Queryの48トークン対応（2026-09-23）

後続の[追加最適化](QUERY_OPTIMIZATION_V2.md)で既定上限は52へ拡張した。以下は48対応時点の計測記録。

per-row INT8の行列積を最適化し、既定の汎用query上限を40から48トークンへ拡張した。
実モデルと5B命令上限のローカルreplicaで検証した。入力長は本文だけではなく、
質問・選択肢・特殊トークンを含むモデル入力全体を指す。本番への配備はこの作業では行っていない。

[前回の詳細計測](QUERY_PROFILE.md)では費用の約82%がINT8射影だった。
今回、K=768/1152の専用カーネルを16/8/4行×8列へ置き換え、入力・重みのロードを共有した。
残り1行も既存SIMDへ渡す。15・7・3行の端数は、行列積内部でゼロ行を1行だけ補い、
16・8・4行の一括処理後に実際の行だけ取り出す。
Attentionへ渡すtoken列や長さは変えず、重み・量子化・整数積和・scaleの適用順も変えていない。

単に加算式をまとめる案は同一のWasm命令数になり、改善しなかった。
16行の本体だけを広げる案では48が成功しても47が超過したため、端数側まで最適化した。
旧4行・8行の重複した実装は、同じ8列カーネルの定数パラメータへ統合して削除した。
入力shapeの積はポインタ演算前にoverflowを検査する。

以下は同じtoken列・同じモデルのupdate forwardを比較した値。単位は10億命令。
queryのガードやtokenizer費用とは区別して、カーネル変更による効果を見るために使う。

| 全体tokens | 変更前 | 最適化後 | 削減率 |
|---:|---:|---:|---:|
| 39 | 4.886 | 3.988 | 18.4% |
| 40 | 4.444 | 3.995 | 10.1% |
| 41 | 5.032 | 4.178 | 17.0% |
| 47 | 5.816 | 4.803 | 17.4% |
| 48 | 5.372 | 4.797 | 10.7% |
| 120 | 14.559 | 13.158 | 9.6% |
| 128 | 15.670 | 14.144 | 9.7% |

48行のQKV/Wi形状（N=2304、K=768）の行列積単体は、
67,040,870から57,853,762命令、約13.7%減った。
nativeの時間から換算した値ではなく、canisterのinstruction counterで測っている。
比較値は[comparison.json](../artifacts/query-optimization/comparison.json)、
最終Wasmの実測は[final-logits.json](../artifacts/query-optimization/final-logits.json)に保存した。

最終受入ではqueryの事前ガードを緩和せず、次を検証した。成功78件、事前拒否5件だった。

- 2クラスの固定入力を6〜48トークンまで1トークン刻みで呼び出す。
- 25クラス・通常token IDを変えた入力を27〜48まで1トークン刻みで呼び出す。
- 1クラスの3〜5トークン入力も呼び出す。
- 未パディング英文の`decide_query`を48まで呼び出す。
- 49トークンは実行を始める前に`Capacity`で拒否する。

記録は[final-query.json](../artifacts/query-optimization/final-query.json)、
[final-maxclasses.json](../artifacts/query-optimization/final-maxclasses.json)、
[final-short.json](../artifacts/query-optimization/final-short.json)、
[final-real.json](../artifacts/query-optimization/final-real.json)。
同一queryのcacheによる繰り返しを速度サンプル数として数えていない。

最終受入の最大は25クラス・47トークンで4,821,506,417命令、5Bに対してforwardの残予算は約3.6%。
48トークンの通常英文は約4.788Bで成功した。49は2クラスで成功する場合があったが、
25クラスや通常英文では超過したため、既定上限には採用していない。
64トークンは約6.52Bで、依然としてqueryには入らない。

新しい既定cost modelは`fixed=170,000,000`、`per_token=100,000,000`。
既存の0.5%余裕を含めると、48は4,994,850,000、49は5,095,350,000となる。
query予算5B、update予算40B、モデル入力のpolicy上限128は据え置き。
この式はquery範囲に合わせた推定で、長い入力全般を上から抑える式ではない。
Attentionの二次項により、120/128の費用は式より大きいが、実測でupdate予算内に収まる。
この式だけを根拠に入力policy上限128を引き上げない。

既存canisterの保存済みcost modelはupgradeで保持する。ローカルでも最適化Wasmへの
upgrade直後は40のままであることを確認し、その後、所有者の`set_cost_model`で48へ更新した。
既存配備へ適用するときも、最適化Wasm・対象per-row pack・再warm-upを確認してから
明示的に設定を更新する。block-32 packへこの費用推定を流用しない。
ゲーム用adapterの固定上限36/40は変更していない。

Wasm専用の検査例を追加し、Nodeの実Wasmエンジンで3,480ケースを独立したi64内積と
float32のbit単位で比較した。K=8/40/768/1152、行・列タイルの端数、正負の最大振幅を含む。
nativeテストだけでは実行されないWasm SIMDを検査するためのものだ。
canisterの`bench_int8`も、計時区間の外で量子化スカラー参照との完全一致を確認する。

実モデルでは38〜48の11入力と120/128の2入力で、変更前のWasm出力とlogitが完全一致した。
通常英文の完走ケースでも前回の出力と一致した。モデルを再量子化したり精度を落としたりはしていない。
これは有限の入力での検証であり、未測定の入力・負荷に対する性能保証や本番latencyの測定ではない。
Wasm hashとモデルhashは[provenance.json](../artifacts/query-optimization/provenance.json)、
検証の一覧は[verification.json](../artifacts/query-optimization/verification.json)に保存した。

```sh
cargo build -p verdict-simd --example wasm-check --release --target wasm32-unknown-unknown
node tools/check_verdict_simd_wasm.mjs
```

専用ローカルcanisterは`4xhad-gd777-77775-aaacq-cai`、endpointは`http://localhost:8011/`。
最終状態は最適化Wasm、warm済み実モデル、48トークンのガード。
ガードを緩和せず境界を再確認するコマンドは次のとおり。

```sh
python3 tools/profile_query.py \
  --canister 4xhad-gd777-77775-aaacq-cai \
  --identity ic-verdict-acceptance-h1vkKw \
  --controller-proxy 4ld2s-rd777-77775-aaaaq-cai \
  --respect-guard --mode query --lengths 47,48,49 \
  --out /tmp/query-48-check.json
```
