# 推論時の中間領域削減と53トークンquery（2026-09-23）

現行のper-row INT8モデルで、射影入力の余分なf32コピー、RoPEの中間Tensor、
正規化の中間Tensorを減らした。専用ローカルreplicaで52トークンの詳細計測は
4,763,145,490から4,632,915,381命令（2.73%減）、128トークンは
12,878,489,681から12,559,662,128命令（2.48%減）になった。
本番canisterには配備していない。トークン数は特殊tokenと選択肢を含む入力全体を指す。

## 採用した変更

- 連続したCPU Tensorの射影入力はCandleのstorageを借用して量子化し、`to_vec1`へのコピーを省く。非連続入力は従来の経路で処理する。
- RoPEは同じ乗算・加減算順で一つの出力領域へ書く。非連続入力などは従来のTensor演算に戻す。
- biasのない2次元正規化は行ごとに平均・分散を求めて一つの領域へ書く。それ以外の形状・biasありでは従来の経路を使う。

試したMLPの`gelu_erf`とgate積の手書き融合は、52トークンで`layer.mlp_act`を
約3,418万命令増やしたため採用しなかった。モデルpack、重み、Attentionの入力長、
量子化方式と行列積カーネルは変更していない。

52トークンの詳細計測では、RoPEが約5,481万命令、attention/MLP正規化が合計約5,772万命令減った。
残りは射影入力コピーなどによる削減である。計測用マーカーを含むため、
詳細計測と通常queryの命令数はわずかに異なる。
生値は[baseline-phases.json](../artifacts/query-optimization-v3/baseline-phases.json)と
[candidate-phases.json](../artifacts/query-optimization-v3/candidate-phases.json)。

## query境界と費用モデル

5B命令上限の実queryで、53トークンは2クラス4,821,015,902命令、
最大25クラスで4,845,929,864命令、未パディング英文で4,814,138,316命令だった。
入力IDを変えた54トークンは4,929,581,209命令で、今回の採用基準4.9Bを超えた。
57・58トークンはreplicaの命令上限で失敗した。
最大25クラスの27〜53トークン、2クラスの6〜53トークン、
1クラスの3〜5トークン、未パディング英文15件はいずれも成功した。
生値は[query-sweep.json](../artifacts/query-optimization-v3/query-sweep.json)、
[maxclasses-query.json](../artifacts/query-optimization-v3/maxclasses-query.json)、
[short-query.json](../artifacts/query-optimization-v3/short-query.json)、
[real-query.json](../artifacts/query-optimization-v3/real-query.json)、
[boundary-query.json](../artifacts/query-optimization-v3/boundary-query.json)。
最終Wasmに新設定を適用した実queryでは、25クラスの53トークンが
4,834,820,093命令で成功し、54トークンは`Capacity`で事前拒否された
（[final-guard.json](../artifacts/query-optimization-v3/final-guard.json)）。
未パディング英文15件も新ガードのままで成功した
（[final-real.json](../artifacts/query-optimization-v3/final-real.json)）。
53トークンを異なる通常token IDで6回測ったCLI往復時間は中央値0.832秒、
範囲0.754〜1.758秒だった（[final-latency.json](../artifacts/query-optimization-v3/final-latency.json)）。
ローカルホスト負荷を固定していないため、本番応答時間の保証や旧Wasmとの速度差には使わない。

新規canisterの既定値は`cost_fixed=170,000,000`、`cost_per_token=90,000,000`。
既存の0.5%マージンを含む推定は53で4,964,700,000、54で5,055,150,000命令となり、
`query_limits().max_tokens`は53を返す。既存canisterの保存済み費用設定はupgrade後も保持されるため、
専用ローカルcanisterでは`set_cost_model`で明示的に適用する。
query予算5B、update予算40B、入力policy上限128、Tetris専用上限は変えない。
この線形費用モデルはquery域の校正値であり、長い入力全般の厳密な上界ではない。

## 正確性と適用条件

旧Wasmの保存済み出力と同じtoken列で比べ、2クラス47件、25クラス26件、
短入力3件、未パディング英文14件のlogitがすべてbit単位で一致した。最大絶対差は0で、
選択ラベルの変更もない。連続領域・オフセット付き領域・非連続領域のRustテストを追加した。
比較件数とsource hashは[verification.json](../artifacts/query-optimization-v3/verification.json)に保存した。
3ライブラリのRustテスト、native/Wasmのclippy（警告をエラー扱い）、
1000件のcheckpoint parityテストが通った。
各環境への適用では、最終Wasm hashとモデルhash、warm-up完了を確認し、
新しい費用モデルを設定してから、53成功・54事前拒否を確認する。
最終ローカルWasmのSHA-256は
`e64f53a14bf5ddc75f9209c092f86c1bba3bd41874fd439af8ff590260859624`、
model pack binaryのSHA-256は
`c70f28d115880a30ba02416ba318d67f293b00451a9f73e25993c13ac9787815`。
ローカルcanisterのinstalled module hashは最終Wasmと一致した。
