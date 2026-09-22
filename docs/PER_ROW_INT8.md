# per-row INT8への復帰（2026-09-22）

速度を優先する指定に従い、block-32からper-row INT8へ戻した。
全2次元重み（embeddingを含む）はオフラインでINT8化し、各行にF32 scaleを1個保持する。
Norm・bias・scale以外にF32重みはアップロードしない。

## 変更

- packのmatrix encodingは`i8_row_symmetric`。重みbytes数は`rows * cols + rows * 4`。
- loaderはper-row表現を直接読み、dense演算は既存のper-row SIMDカーネルを使う。
- F32 matrixおよび旧`i8_block32_symmetric`を拒否する。既存block-32 packをそのまま流用しない。
- 実モデルは152,245,512 bytes（約145.2 MiB）。`models/verdict-pack`を再生成済み。
  以前のpackは`models/verdict-pack-block32-backup`へ移動して保持した。
- 新規canisterの既定cost modelは`fixed=48,746,986, per_token=121,028,580`で、query上限40。
  upgradeは既存設定を保持するため、既存canisterでは`set_cost_model`を明示して更新する。
- 精度差と棄権から具体クラスへの反転は報告値とし、既定の失敗条件から外した。
  入力欠落・空の評価集合等は引き続き失敗する。必要な比較閾値は`--min-argmax-ratio`で明示する。

## ローカル実モデル測定

Endpoint: `http://localhost:8011/`。新規テストcanister: `4fbx2-kt777-77775-aaabq-cai`。
次のquery/update測定は以前のquery sweepと同じ短いtoken列へpaddingを追加したもの。

| モード | tokens | instructions |
|---|---:|---:|
| query | 6 | 773,816,963 |
| query | 14 | 1,630,950,264 |
| query | 20 | 2,223,958,355 |
| query | 30 | 3,394,926,890 |
| query | 39 | 4,884,089,567 |
| query | 40 | 4,441,624,625 |
| update | 120 | 14,559,052,908 |
| update | 128 | 15,670,269,425 |

query 40は5B以内で成功し、41はガードが`Capacity`を返した。
これは測定入力での結果で、全候補数・全入力内容で同じコストになるという保証ではない。

財布紛失・カード停止の英文（52 tokens）は`card_lost`、softmaxスコア0.970080、
5,879,227,976 instructions。block-32版の同一文章16,762,943,658 instructionsから約65%減った。
別canister（ローカルCLI付属proxy）からの`decide`呼び出しも成功した。
同一Wasmへのupgrade後、再アップロードせずwarm-upでき、文章分類のlogitとquery上限40が保持された。

Rust workspaceテスト、Python 34件、fixture検証、Wasmビルドが成功。
nativeで20件の比較を行い参照argmaxと19/20一致、最大確率差0.016831だった。
これは小規模な動作確認であり、1000件の精度値として外挿しない。
精度差を理由に推論方式を再変更しない。

証跡: `artifacts/verdict_per_row_upload.log`、`artifacts/verdict_per_row_local.log`、
`artifacts/verdict_per_row_local.json`、`artifacts/verdict_per_row_sample.log`。
JSONには使用したpack/Wasmのハッシュを含む。メインネット操作は行っていない。
