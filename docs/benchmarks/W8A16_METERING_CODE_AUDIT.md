# 行列積を展開した Wasm 表現への課金

対象は実測と SHA-256 が一致する Wasm (`3bbb0359cd243985ccff6879ba003563cc7f7646f52b9d035deaac971c03799f`)。GitHub Actions run `36112530260` の `candid-and-wasm` から取得し、wabt 1.0.39 で flat WAT に逆アセンブルした。IC の参照ソースは `d26cd031176beec51b39fbb9e39e80a3a46a748e`。

## コードが行っていること

1. `verdict-simd` は行列積を 128-bit SIMD の dot、加算、ロード、ローカル変数操作、ループ制御へ展開する。
2. IC はその Wasm の各命令を `instruction_to_cost` に渡し、基本ブロック内の料金を加算する。行列積全体の形状・演算量を認識した特別料金はない。dot 1 個は 1、加算 1 個も 1、`local.get/set/tee` も各 1。
3. 基本ブロックにまとめてカウンタを減算する命令を挿入する。1 命令ごとにカウンタを更新しているわけではないが、金額は元の細粒度の命令コストの合計。
4. この挿入後に Wasmtime でネイティブコードにコンパイルする。生成された機械語の命令数や CPU の同時実行能力から金額を再計算しない。なお IC は Cranelift の最適化レベルを `None` にしており、「強い JIT 最適化がすべてを消す」との説明は正しくない。

## 実測の約 1.264 億命令の内訳

120×2304×768 は 212,336,640 MAC。16×16 タイルの内側ループは 1 周で 16,384 MAC を処理し、コストは 9,325。その内訳には `local.get` 4,386 個、`local.tee` 290 個、`local.set` 257 個が含まれる。16 行タイルを 7 個と、8 行の残りタイルを処理する経路で、以下の金額になる。

| 内訳 | 計上コスト | 実測カーネル全体に対する割合 |
| --- | ---: | ---: |
| 主要ループの `local.get/set/tee` | 63,993,024 | 50.63% |
| 主要ループの SIMD dot と加算 | 53,084,160 | 42.00% |
| 主要ループのベクトルロード | 3,428,352 | 2.71% |
| 主要ループのアドレス計算・定数・分岐等 | 1,156,032 | 0.91% |
| 上記以外の処理（差分） | 4,740,567 | 3.75% |
| 合計 | 126,402,135 | 100% |

主要ループの静的計算は実測全体の 96.25% を説明する。ループ外の処理は独立に分類せず差分として残している。カウンタ取得等の端点の微小な費用もその差分に含まれる。

SIMD の 8 個分の積を 8 回分に分解して課金しているわけではなく、`i32x4.dot_i16x8_s` は 1 と数えている。それでも、そのオペランドを渡すローカル変数操作などは別に加算される。実測の約半分はそのローカル変数操作の金額であり、行列計算を表す Wasm の形式が料金に大きく影響している。

この結果は「細粒度の Wasm 表現への加算が、行列積の料金を決めている」という指摘を支持する。ただし `local.*` がすべて実 CPU 上で無料とはいえない。このカーネルは多数のベクトル累積変数を持ち、レジスタからメモリへの退避も起こり得る。上記の 50.63% をそのまま過大課金率とすることはできない。評価すべき実装上の論点は、Wasm 変数操作の固定単価と、行列演算全体を単位とする費用モデルの違いにある。

再現スクリプト: `tools/audit_w8a16_metering.py --wat verdict-engine.wat --wasm verdict-engine.wasm --out audit.json`。生の集計は `w8a16-metering/code-audit.json`。

参照コード:

- [IC の命令単価](https://github.com/dfinity/ic/blob/d26cd031176beec51b39fbb9e39e80a3a46a748e/rs/embedders/src/wasm_utils/instrumentation.rs#L171)
- [基本ブロックのコスト合算](https://github.com/dfinity/ic/blob/d26cd031176beec51b39fbb9e39e80a3a46a748e/rs/embedders/src/wasm_utils/instrumentation.rs#L1221)
- [計上コード挿入後にコンパイル](https://github.com/dfinity/ic/blob/d26cd031176beec51b39fbb9e39e80a3a46a748e/rs/embedders/src/wasm_utils.rs#L241)
- [行列積カーネル](../../crates/verdict-simd/src/lib.rs) はリポジトリルートの `crates/verdict-simd/src/lib.rs:442`。
