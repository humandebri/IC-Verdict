# INT8実モデルのローカル受入結果（2026-09-22）

対象commit: `47754d1bff713ad4914a56fe4dc8b997e67a7d3d`。最新ソースからWasmを再ビルドし、
このプロジェクトのmanaged local network `http://localhost:8011/` に新規canister
`4caro-hl777-77775-aaaba-cai`を作成して検証した。既存canisterのreinstallは行っていない。

## 結果

実モデルのアップロード、warm-up、文章分類、canister間evaluate、同一Wasmへのupgradeと復帰は成功した。
nativeとWasmのlogitは完全一致していない。以下は動作確認であり、全入力の品質・性能保証ではない。

- Wasm SHA256: `524bc25be73baec85e0f03bf8bed9afefa867087da36b21412ada9256645b40c`
- pack manifest SHA256: `2db054927c36f12f14b96b4e924ff8e85a316f8f1e684f39c569f8e9a8c2ad50`
- 重み: block-32 INT8、170,408,640 bytes。tokenizer込み173,992,236 bytesを約191秒で転送。
- 142テンソルのwarm-up完了。復帰後のWasm heapは186,646,528 bytes。
- 財布紛失・カード停止の英文を`card_lost`へ分類。52 tokens、softmaxスコア0.972573、16,762,943,658 instructions。
- ローカルCLI付属proxy canisterから`evaluate`を呼び、実checkpointのReceiptを取得。
  schema/model/stateのハッシュを返すことを確認。executorの業務workflow全体を使った試験ではない。
- 未許可callerの推論を`Unauthorized`で拒否。
- upgrade後はupload済みpackを保持し、heap modelをリセット。再warm-up前の推論は`ModelUnavailable`。
- 重みの再転送なしでwarm-up成功。caller・schema・キャッシュ済みReceiptを保持し、再推論logitもupgrade前と一致。

## 入力長と命令数

以下は固定の合成token列による測定。長さは本文だけでなく候補・質問等を含むモデル入力全体。

| 入力tokens | update instructions |
|---|---:|
| 8 | 2,590,912,904 |
| 32 | 10,255,615,208 |
| 64 | 20,736,809,682 |
| 120 | 39,769,110,363 |

既定cost modelでは121 tokensを`Capacity`で事前拒否した。
queryは14 tokensで4,480,512,904 instructions、15 tokensは`Capacity`。
120-token測定は40Bに約0.58%しか余裕がなく、すべての入力に対する安全な最大長を示すものではない。

## native/Wasm差と未確認事項

- 同一pack・同一8-token入力でlogit最大絶対差0.0116877。argmaxは一致。
- 同一52-token実文章では最大絶対差0.0056236。argmaxは両方`card_lost`。
- 試験スクリプトの`native_wasm_parity_pass=false` / `passed=false`は診断用の絶対差1e-4条件を超えたという意味。
  `lifecycle_passed=true`が配備・呼び出し・復帰の結果。1e-4は新たな本番採否基準として設定していない。
- 数値差の原因、Wasm上での1000件品質評価、用途別calibration、executorから実送金までのworkflowは未検証。
- 今回は同一版へのupgrade試験であり、旧FP32版からの移行試験ではない。

証跡: `artifacts/verdict_local_upload.log`、`artifacts/verdict_local_acceptance.log`、
`artifacts/verdict_local_acceptance.json`、`artifacts/verdict_local_real_parity.json`。
検証canisterはwarm済みでローカルネットワーク上に残した。メインネットには操作していない。
