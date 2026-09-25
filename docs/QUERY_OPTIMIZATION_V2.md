# INT8タイルと内積ループの追加最適化（2026-09-23）

16/8/4行×16列のタイルと64要素ずつの内積処理を採用し、既定の汎用query上限を
48から52トークンへ拡張した。48トークンでは、行列積単体の命令数を約13.2%、
実モデルのquery forwardを約9.9%削減した。本番への配備は行っていない。
トークン数は質問・選択肢・特殊トークンを含む入力全体である。

比較基準は[前回の48トークン対応](QUERY_OPTIMIZATION.md)の16/8/4行×8列、
内積32要素展開版。計測は同じper-row INT8 packと専用ローカルreplicaで実施した。
命令数はcanisterのinstruction counterによる実測で、ホストの処理時間から換算していない。

## タイルとループ展開の比較

QKV/Wiに相当するM=48、N=2304、K=768の行列積。表の命令数は1回分で、
計時前に量子化スカラー参照とのbit単位の一致を検査し、その後3回実行している。

| 最大行タイル×列タイル | 内積の展開幅 | 命令数 | 基準からの削減 |
|---|---:|---:|---:|
| 16×8（基準） | 32 | 57,853,762 | — |
| 16×16 | 32 | 55,068,653 | 4.8% |
| 16×32 | 32 | 54,051,515 | 6.6% |
| 32×16 | 32 | 54,392,378 | 6.0% |
| 32×32 | 32 | 52,837,667 | 8.7% |
| 32×32 | 64 | 49,468,490 | 14.5% |
| 32×32 | 128 | 47,985,002 | 17.1% |
| 16×32 | 64 | 49,783,760 | 13.9% |
| **16×16（最終版）** | **64** | **50,237,792** | **13.2%** |

列を広げると同じ入力ベクトルを多くの出力列に使い回せる。内積の展開幅を32から64へ
増やすと、ループ制御とアドレス計算をまとめられる。量子化方式・重み・整数内積・
scaleの適用順は変更していない。元の15/7/3行の端数補完も維持している。

最大タイルを広げ続ける案は採用しなかった。32×32・128要素版は命令数が最少だったが、
48トークンqueryの応答時間6件の中央値は約2.76秒だった。基準は約0.82秒、
16×32・64要素版は約0.53秒、16×16・64要素版は約0.49秒だった。
16×32と16×16の差は全体命令数で約0.7%なので、保持する積和が少ない16×16を選んだ。

ただし、最終版の再測定では中央値が約1.39秒になった。ホスト負荷を固定しておらず、
応答時間の改善は確証がない。上記を速度保証や有意な速度差とは扱わない。
各回で通常token IDを変え、同一queryのcache hitを独立サンプルとして数えないようにした。
ビルド終了後に比較したが、ローカル環境全体の負荷変動は残っている。

候補は40/48/64行、N×Kが2304×768・768×768・768×1152の3形状で計測した。
全数値は[kernel-comparison.json](../artifacts/query-optimization-v2/kernel-comparison.json)、
応答時間の生値は[latency-comparison.json](../artifacts/query-optimization-v2/latency-comparison.json)。
候補patchとWasm hashは同じartifactディレクトリに保存した。

## 実モデルの効果と残った費用

同一入力のquery forwardを比較した。単位は10億命令。

| tokens | 基準 | 最終版 | 削減 |
|---:|---:|---:|---:|
| 39 | 3.985 | 3.608 | 9.4% |
| 40 | 3.992 | 3.616 | 9.4% |
| 41 | 4.175 | 3.799 | 9.0% |
| 47 | 4.800 | 4.325 | 9.9% |
| 48 | 4.794 | 4.319 | 9.9% |

[query-comparison.json](../artifacts/query-optimization-v2/query-comparison.json)に比較値を保存した。
64トークンのupdate forwardは約5.891Bで、依然として5Bのquery予算には入らない。
120/128トークンは約11.990B/12.879Bで、40Bのupdate予算内に収まった。

52トークンの詳細計測では、総費用4.763Bの内訳は次のようになった。
計測マーカーの費用を含むため、通常queryの値とはわずかに異なる。

| 工程 | 命令数（B） | 総費用に占める割合 |
|---|---:|---:|
| QKV射影 | 1.276 | 26.8% |
| MLP up射影 | 1.276 | 26.8% |
| MLP down射影 | 0.666 | 14.0% |
| Attention出力射影 | 0.450 | 9.5% |
| Attentionのscores/mask/softmax/AV | 0.462 | 9.7% |
| MLP活性化 | 0.228 | 4.8% |
| RoPE | 0.150 | 3.1% |
| その他 | 0.255 | 5.4% |

4つのINT8射影が合計約77.0%を占め、引き続き最大のボトルネックである。
この割合は射影内の量子化・付随処理を含み、純粋な積和だけの割合ではない。
40/48/52/64の工程別集計は[final-phases.json](../artifacts/summaries/query-optimization-v2/final-phases.json)。

## 52トークンの受入と設定

事前ガードを緩めずに、次の90件が成功し、53トークンの3件は事前拒否された。
命令上限超過はなかった。

- 2クラスの6〜52トークンを1刻みで検査。
- 最大25クラス・通常token IDを変えた27〜52トークンを1刻みで検査。
- 1クラスの最短3〜5トークンを検査。
- パディングしていない英文14件を`decide_query`で検査。

最大は25クラス・52トークンの4,778,768,127命令で、forwardの残予算は約4.4%。
52トークンの通常英文は4,754,135,627命令だった。53以上を一律に不可能とはしていないが、
今回の既定値では余裕を確保して52までに制限する。

新しい既定値は`cost_fixed=170,000,000`、`cost_per_token=92,000,000`。
既存の0.5%余裕を含む推定は52で4,978,770,000、53で5,071,230,000になる。
query予算5B、update予算40B、入力policy上限128は変更していない。
この推定はquery範囲の校正であり、長い入力全般を上から抑える式ではない。
Tetris adapterの個別上限も変更していない。

既存canisterの保存済み設定はupgradeで保持される。最終Wasmへのupgrade後にも
旧`per_token=100,000,000`、48トークンの設定が残ることを確認し、実モデルのwarm-up後、
専用ローカルcanisterだけを所有者の`set_cost_model`で52へ更新した。
別環境への適用には対象Wasm・per-row pack・warm-up・費用設定を一緒に確認する。
block-32 packにはこの校正を流用しない。

## 正確性と再現

Wasm専用検査は独立したi64内積に対して9,840ケースがfloat32のbit単位で一致した。
K=8/40/768/1152、列16の境界、8列への切替、行タイルの端数、52付近の全長、
正負の最大振幅を含む。16の倍数以外の列数では8列タイルを残し、
小さい行列が大きなスカラー端数へ落ちることを避けている。

実モデル出力は、変更前に成功していた78件、長いupdate入力3件、応答時間比較用の6件、
計87件でlogitが完全一致した。Rustのモデル・カーネル・canisterテスト、
Wasmライブラリとnative SIMDのclippy（警告をエラー扱い）、Python query契約テスト11件が通った。
Wasmのclippy対象はライブラリに限定した。全targetを選ぶと、元々native専用の
`verdict-infer`がWasmでは利用できない`load_directory`を参照するため失敗する。

[verification.json](../artifacts/query-optimization-v2/verification.json)に検証件数、
[provenance.json](../artifacts/query-optimization-v2/provenance.json)にWasm/model hashを保存した。
測定レポートの`module_hash`が実際にインストールされたWasmを識別する。
作業ツリーには別作業の変更もあるため、候補作成中のsource snapshotと混同しない。

```sh
cargo build -p verdict-simd --example wasm-check --release --target wasm32-unknown-unknown
node tools/check_verdict_simd_wasm.mjs
python3 tools/profile_query.py \
  --canister 4xhad-gd777-77775-aaacq-cai \
  --identity ic-verdict-acceptance-h1vkKw \
  --controller-proxy 4ld2s-rd777-77775-aaaaq-cai \
  --mode query --lengths 51,52,53 --respect-guard \
  --out /tmp/query-52-check.json
```

専用ローカルcanisterは最終Wasm、warm済みモデル、52トークンのガードで残している。
多数のupgradeで検証用cyclesが不足したため、ローカルproxyから3Tを実行残高へ補充した。
本番のcanisterや残高は変更していない。
