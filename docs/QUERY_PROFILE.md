# Queryの40トークン境界の実測（2026-09-22）

40トークン超は可能だった。現行カーネルでも42・44トークンのqueryは成功するが、
41・43トークンは命令上限に達する。奇数の最後の1行がスカラー処理になるためである。
この1行を既存SIMDカーネルへ渡す試作では41・43も成功し、44トークンの通常の文章分類も成功した。
45以上は依然として超過する。44は余裕が小さく、安全な一般上限としての採用判断はしていない。

測定対象は22層の実openJevモデル、per-row INT8 pack。
入力長は本文・質問・選択肢・特殊トークンを含むモデル入力全体を指す。
専用のmanaged local canister `4xhad-gd777-77775-aaacq-cai`、`http://localhost:8011/`で測った。
本番canister、公開UI、モデルの重みは変更していない。

| 識別対象 | SHA-256 |
|---|---|
| 本番と同じ既存Wasm（最初のquery sweep） | `b4da1963c401f72a4d34913018edd644a29d3250b856917a7befda44b3e7e123` |
| モデルmanifest | `0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0` |

計測点追加版・試作版のWasm hashは[provenance.json](../artifacts/query-profile/provenance.json)、
各呼び出し時に取得したinstalled module hashは各JSONに記録した。
`source_hashes`はその時点の作業ツリーのスナップショットであり、installed Wasmと同一であるとは限らない。

アプリの事前ガードと実行上限を区別するため、専用canisterだけで`set_cost_model(1,1,40B)`を
一時設定した。replicaのquery上限は変更していない。すべてのquery sweepは`finally`で
元の`fixed=48,746,986 / per_token=121,028,580`に戻し、`query_limits`の一致を確認した。
したがって、以下の「超過」は40トークンの事前拒否ではなく、実queryの`IC0522`である。

| 全体tokens | 現行query命令数 | 末尾SIMD試作のquery命令数 |
|---:|---:|---:|
| 38 | 4,312,145,204 | 4,307,122,277 |
| 39 | 4,884,158,567 | 4,473,735,543 |
| 40 | 4,441,576,452 | 4,436,715,616 |
| 41 | 超過 | 4,619,481,076 |
| 42 | 4,739,768,572 | 4,734,918,823 |
| 43 | 超過 | 4,902,389,726 |
| 44 | 4,938,978,154 | 4,934,098,478 |
| 45–48 | 各長さで超過 | 各長さで超過 |

固定の2選択肢token列をPADで延長した測定。
[現行の生データ](../artifacts/summaries/query-profile/baseline-query.json)と
[試作の生データ](../artifacts/query-profile/candidate-query.json)を保存した。
同じqueryを2回呼んでいるが、replicaのquery cacheに当たる可能性があるため、
これを独立した2回の速度測定とは扱わない。CLI経由のwall timeも純粋な推論時間ではない。

既存の詳細プロファイラは、QKV射影を最初の`rope.apply`へ含めていた。
QKV、head整形、RoPE、出力射影、MLP正規化、残差加算の境界を分離した。
以下は追加した計測点を使った40トークン・22層の集計である。

| 演算 | 命令数 | 全体の割合 |
|---|---:|---:|
| Attention QKV射影 | 1,278,289,674 | 28.77% |
| MLP Wi射影 | 1,277,882,926 | 28.76% |
| MLP Wo射影 | 659,170,596 | 14.83% |
| Attention出力射影 | 445,078,756 | 10.02% |
| MLP GELUとgate積 | 176,103,511 | 3.96% |
| Attention QKスコア積 | 127,148,996 | 2.86% |
| RoPE適用 | 115,686,742 | 2.60% |
| Attention重みとVの積・head結合 | 82,790,720 | 1.86% |
| Attention softmax | 78,542,413 | 1.77% |
| その他の正規化・残差・embedding・head等 | 203,069,447 | 4.57% |
| 合計 | 4,443,763,781 | 100% |

4種類のINT8射影で82.37%を占める。各射影にはactivation量子化とTensor変換も含む。
一方、Attentionの2つの積は合計4.72%、RoPEテーブル生成は1,150,319命令、約0.026%だった。
40トークンではlocal attentionの距離制限にかかる要素がなく、mask適用もほぼ費用を持たない。
この長さではAttentionの疎化やRoPEテーブルの再利用だけで大きく上限を延ばすのは難しい。

層別では最小199,038,357、最大202,123,752命令。
特定の1層に集中せず、22層を通して同じ射影費用を払っている。
各phase集計と生データのハッシュを[phases.json](../artifacts/summaries/query-profile/phases.json)へ保存した。
各測定についてphase合計と総命令数の一致をスクリプトで検査している。
40トークンの詳細計測値は既存queryのforward計測値より約0.05%大きい。
計測callbackや実行状態の差を含むため、詳細phaseの値をそのままqueryの実行費用とは扱わない。

`verdict-simd`のK=768/1152専用カーネルは、8行→4行→2行と処理し、残り1行を
スカラー内積で計算する。QKV/Wiと同じN=2304・K=768の単体測定は次のとおりだった。

| 入力行数 | 現行の命令数/回 | 命令数/MAC | 試作の命令数/回 |
|---:|---:|---:|---:|
| 1 | 8,760,173 | 4.951 | 2,253,783 |
| 2 | 3,667,722 | 1.036 | 3,667,718 |
| 39 | 63,261,583 | 0.917 | 56,755,187 |
| 40 | 55,867,428 | 0.789 | 55,867,428 |
| 41 | 64,627,415 | 0.891 | 58,121,025 |
| 42 | 59,534,964 | 0.801 | 59,534,960 |

単体ベンチは3回のカーネル実行を計時して平均する。重み量子化・activation量子化・
F32参照計算は計時区間の外。40行のactivation量子化は1,296,614命令で、
同じshapeの行列積55,867,428命令の約2.3%にとどまる。
他にN=768・K=768とN=768・K=1152も測定した。
[現行kernel測定](../artifacts/query-profile/kernels.json)、
[試作kernel測定](../artifacts/query-profile/candidate-kernels.json)を参照。

試作は新しい近似計算を追加せず、残り1行を既存の`matmul_i8_simd`に渡す変更である。
差分を[tail-candidate.patch](../artifacts/query-profile/tail-candidate.patch)へ保存し、
測定後に作業ツリーのカーネルとローカルcanisterを元へ戻した。
計測点追加版同士の比較では奇数長で約4.10億命令削減し、偶数長では約400万命令削減した。
偶数入力でもtext projectorが1行処理になるため、小さな削減が残る。

同一モデル・同一token列の38〜48トークンの11ケースで、変更前後のWasmのfloat32 logitが
すべて完全一致した。計測点追加そのものも、比較可能な38・39・40・42・44トークンの
5ケースで既存Wasmとlogitが一致した。
[comparison.json](../artifacts/query-profile/comparison.json)に記録している。
これは限定した入力の確認であり、全入力・全shapeに対する品質保証ではない。

PADだけへの依存を避けるため、通常token IDを変えた8選択肢の合成入力でも39〜44のquery成功、
45・46の超過を確認した。さらに実tokenizerを使う`decide_query`でも、
次の未パディング英文を44トークンとして分類できた。

> I lost my wallet yesterday and need to stop my debit card immediately. Please help.

質問は`What is the request?`、候補は`lost card`・`dispute payment`・`insufficient evidence`。
試作で4,927,732,825命令。45・46トークンになる英文の続きでは超過した。
[入力ケース](../artifacts/query-profile/real-requests.json)、
[試作の文章分類](../artifacts/query-profile/candidate-real.json)、
[現行の文章分類](../artifacts/query-profile/baseline-real.json)を保存した。
44トークンでもforwardだけで予算の約98.6%を使っており、十分な安全余裕とはいえない。
同じ英文の41トークンは現行版で超過した。両版で完走した32・37・38・42・44トークンの
5ケースでは、通常の文章分類でもlogitが完全一致した。

長い入力の費用は、同じforwardを実行するupdateの詳細計測経路で取得した。
各長さで2回実行した平均を使う。queryで成功したという意味ではない。

| 全体tokens | 現行forwardの命令数 | 5Bまでに必要な全体削減率 |
|---:|---:|---:|
| 48 | 5,371,634,936 | 6.92% |
| 56 | 6,322,666,667 | 20.92% |
| 64 | 7,289,418,478 | 31.41% |
| 80 | 9,297,915,406 | 46.22% |
| 96 | 11,352,116,526 | 55.96% |

この削減率はtokenizer・応答処理・安全余裕を含まない楽観的な下限である。
48トークンを狙う次の対象は、8行SIMDカーネル本体の命令数と重みload・拡張の再利用。
射影以外の費用が同じなら、48には主射影部分をさらに約8.5%、64には約39.2%削減する必要がある。
64以上を末尾処理だけで実現することはできない。現状からの試算であり、これらの追加最適化は未検証。

計測器は[tools/profile_query.py](../tools/profile_query.py)。例えば次のように再測定できる。
既存のローカル計測canisterはモデルwarm済み、元の40トークンガードに戻してある。

```sh
python3 tools/profile_query.py \
  --canister 4xhad-gd777-77775-aaacq-cai \
  --identity ic-verdict-acceptance-h1vkKw \
  --controller-proxy 4ld2s-rd777-77775-aaaaq-cai \
  --mode profile --lengths 39,40,41,42,43,44,48,64 \
  --out /tmp/query-phases.json
```

`--mode query`は実query、`kernels`は単体計測、`update`はlogit比較用。
`--mode decide --requests artifacts/query-profile/real-requests.json`は通常の文章分類を測る。
query/decideモードは専用canisterの見積りガードを一時変更するため、専用のローカル試験先で使う。
最終的に残したソース変更は計測境界の分離とその回帰テスト、計測スクリプトのみ。
Rustのモデルテスト、Wasmビルド、Python構文検査、差分の空白検査を実施した。
検証と復元確認の一覧は[verification.json](../artifacts/query-profile/verification.json)に保存した。
