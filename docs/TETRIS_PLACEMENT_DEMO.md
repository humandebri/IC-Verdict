# Canister内で配置を選ぶTetris

> この文書は旧update版。現在の公開版は [4候補・1 query版](TETRIS_PLACEMENT_QUERY.md)。

151Mの推論から着手まで、ICP canister内で動くテトリス。
ゲームの強さではなく、合法候補の生成・モデルの選択・確定した盤面と記録の境界を見せるデモ。

## 1ターンの処理

署名したブラウザセッションが `tetris_demo_start(seed, mode, nonce)` でゲームを作り、`tetris_demo_step(game, expected_turn)` を呼ぶ。クライアントは盤面・特徴量・選択結果を指定できない。
Canisterは中心上端から移動・右回転・下移動で到達できる着地を列挙する。正規化した回転形、壁蹴りなし、固定seedの7-bag。
回転・x・y順に整列し、12件を超えた場合は先頭から末尾までの等間隔の添字を採用する。評価値による候補選別はしない。

各候補について消去行数、消去後の穴数、最大高さをコードで計算する。
モデルに渡すラベルは `linesN holesN heightN`、文脈は `Tetris. Clear lines, avoid holes and height.`。
12候補の分布を実モデルで計算し、argmaxの候補を採用する。評価関数の推奨は `10*lines - 8*holes - height`、同点は候補順。
モデルには推奨順位・評価値を渡さない。比較モードは同じ候補から評価関数で選び、モデルを呼ばない。
失敗時の評価関数フォールバックはない。

生成・推論・盤面更新・記録はawaitなしの同一update内で行う。直前ターンの再送には同じ記録を返し、二重着手・二重予算消費を防ぐ。
ブラウザは応答後に選択先をゴースト表示し、合法経路を再生して確定盤面に切り替える。リアルタイム自然落下ではない。
実測応答時間は通信・IC合意を含む。Canister内の時刻は同一メッセージ中に進まないので、純粋な推論秒数とは表示しない。推論の実命令数は別表示する。

## 記録検証

`tetris_demo_record(id)` は記録のUTF-8 JSON bytes、保持中の全記録の `id:sha256` 索引、IC証明書を返す。
`certified_data = SHA256(index bytes)`。UIの「証明を検証・保存」はSDKで証明書の署名・時刻・Canister IDを検証し、certified_dataとの索引一致、記録のハッシュ・ID一致を確認して証拠JSONを保存する。
localhostのみ開発root keyを取得し、結果をローカルネットワークと明示する。本番では既知のIC root keyを使用する。
証明書は記録の完全性を証明する。モデルの正しさ・実行コードの同一性・controllerによる将来の更新不能性を保証しない。
記録には盤面前後、候補経路・特徴量、選択・推奨、分布、生logits、入力文字列、モデルID、トークン数、命令数、前の保持記録のハッシュが入る。

## ローカル起動と制限

`tools/build_one.sh verdict-engine` でWasm/Candidを作る。既存モデル入りのローカルCanisterはupgradeし、所有者がwarmup、`set_tetris_enabled(true)`、`tetris_demo_budget(50)` を行う。本番反映は別作業。

```sh
VITE_IC_HOST=http://localhost:8011 VITE_CANISTER_ID=<local-id> npm --prefix web/tetris run dev -- --port 5185
```

予算はデフォルト0、所有者のみ最大4096手を設定可能。推論費用はデモ運営側が負担する。公開署名セッションは予算を消費できる。
32ゲーム、各128ターン、直近512記録。ゲーム枠の自動回収・管理削除は未実装で、32ゲーム到達後は新規開始不可。一般公開向けの無制限サービスではない。
ゲーム・予算・記録・有効化フラグ・課金設定をstable snapshotに保存し、upgrade後に認証ルートを再構築する。モデルは既存仕様どおり再warmupが必要。
旧版Tetris query APIは互換用に残る。

## 今回の検証

- 実151Mモデル、localhost:8011、検証用Canister `4fbx2-kt777-77775-aaabq-cai`。
- 同じseed 184でモデル4手・評価関数4手。全8件で証明書・記録ハッシュ検証、独立したTypeScriptによる盤面・特徴量再計算、同一ターン再送を確認。予算消費はモデル4手のみ。
- モデル入力78〜99トークン。先頭手は評価関数「回転なし・左端」、モデル「右回転・左から2列」。応答約1.5秒はこのローカル環境での値であり、本番の速度を保証しない。
- PocketIC 15で旧版→新版→新版upgrade、所有者以外の着手・予算設定拒否、有効化スイッチ、記録・予算保持を確認。
- UIをPlaywright Chromiumで確認。証明書検証・JSON保存が成功。実測と全応答は `artifacts/tetris-placement/`。
- 強さの改善を示す評価ではない。本番Canisterは未変更。

再現コマンド（ローカルモデルのwarmupと予算設定後）：

```sh
PLACEMENT_CANISTER=<local-id> node --import ./web/tetris/node_modules/tsx/dist/loader.mjs web/tetris/scripts/placement-smoke.ts
POCKET_IC_BIN=/path/to/pocket-ic-15 VERDICT_OLD_WASM=/absolute/path/to/old.wasm cargo test -p verdict-engine --test tetris_placement -- --ignored --nocapture
cargo test -p verdict-engine placement_real_tokenizer_budget --lib -- --ignored --nocapture
npm --prefix web/tetris test
npm --prefix web/tetris run build
```
