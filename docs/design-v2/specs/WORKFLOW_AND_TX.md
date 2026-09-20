# Workflow・Tx状態機械

v2.0 / 2026-09-19（JST）

## 1. 固定する記録

`WorkflowRecord`はcaller/nonce、operation ID、typed proposal、元evidence、snapshot hash、workflow plan、schema群、model bundle、calibration群、policy/delegation revision、slots、現在状態、送信args、予約、receiptを持つ。

`Snapshot`はapp側の真実性が検証されたstateと、単なる主張としての非信頼テキストを区別する。nonceやhashがあるだけで文中の事実を真実と認定しない。

## 2. 最大3slotの逐次評価

planは登録時に固定する。slotは`Pending → InFlight(evaluation_id) → Complete(result)`または`Abstained/Error`。順番を変えたり、望ましくない結果だけを無視したりしない。

各評価は独立したengine update。1slotに重複リクエストが来ても同じevaluation IDへ収束する。engineはlive IDについて同じpayloadのresultを返し、計算を再発行しない。キャッシュはexpiryとcaller別上限で管理し、expired IDには新規評価しない。期限・回数・cyclesの予算を予約して資源攻撃を防ぐ。

次slotを開始する条件はplanに明記。deny/reviewへのshort-circuitは可能だが、approveのために必須signalを省くことは不可。必要signal同士の独立性は仮定しない。

## 3. Workflow状態

| 状態 | 次の状態・処理 |
|---|---|
| Received | 入口検査・nonce消費済み。Evaluatingへ、またはRejected/Cancelled。 |
| Evaluating | slotごとに結果を保存。必要結果が揃えばReadyToDispatch、問題があればNeedsReview/Rejected/Stale。未送信なら取消可能。 |
| NeedsReview | 承認者が同じsnapshot/hashに対して判断。hard capは迂回しない。 |
| Stale | 元結果では許可しない。明示的な新attemptか取消。 |
| ReadyToDispatch | 永続的な実行許可ではない。現在の権限/予算を再検証してSubmittedへ。 |
| Submitted | ledger callを発行・結果待ち。Succeeded / FailedDefinitive / OutcomeUnknown。 |
| OutcomeUnknown | 予約維持。照合で成功/確実な未実行へ、または不明のまま停止。 |
| Succeeded | terminal。遅延エラーで失敗へ戻さない。 |
| FailedDefinitive | 全attemptが未実行と確定した場合のみ。予約解除は1回。 |
| Rejected / Cancelled | 送信前のterminal。一般retryで未完了へ戻さない。 |

`NeedsReview`はactorの権限不足や上限超過を解除するための裏口ではない。ownerが上限変更を行う場合はversion変更として再検査する。modelが`Review`候補を選ぶことと、runtimeの`NeedsReview`は別の由来を持つが、どちらでも自動送金は止まる。

## 4. 認可・資金・二重処理

初期はexecutorの専用treasuryのみ。ownerがdelegate、allowlist、task、期限、一回/累積/期間別予算、fee上限を登録する。modelの提案は委任を作らない。[S10]

callerごとのmonotonic nonceを永続化する。同一nonce/同一payloadは既存workflowへ収束。違うpayloadは拒否。compact後もwatermarkを保持し、低いnonceを再実行しない。

別nonceによる同じ業務を防ぐため、appで検証されたoperation状態を`Available → Reserved(workflow_id) → Consumed(workflow_id)`とする。任意の新IDをcallerが与えただけでは新規業務と認めない。同時proposalがあっても予約は一つ。

初回dispatch直前に、全signalのmodel/profile/schema/calibration/snapshot、policy/delegation revision、expiry、revocation、現在予算を確認する。金額+feeをchecked arithmeticで予約し、送信argsを固定、Submittedを記録して送信する。この間に他の外部awaitを挟まない。[S09]

## 5. 結果不明

ICRC-1 transfer argsはledger、caller identity、from_subaccount、to、amount、fee、memo、created_at_timeを固定する。unknown状態で新timestamp、memo、feeにして再送しない。[S08][S10]

初回のみで確実な未実行エラーならFailedDefinitiveにできる。先行unknownがあるとき、後のTooOld/BadFeeはそのattemptの不受理を示しても、過去の成功を否定しない。予算とoperationの予約を保持する。

確認したledger profileのdedup期間・条件を満たす同一payloadだけ、回数を限定して再送を検討する。ICRC-1のdedupはSHOULDであり、任意ledgerのexactly-onceを保証しない。履歴に見つからないだけで未実行としない。先行callが後から実行され得ないことも確認する。[S10][S08]

失効やpause後は受動照合を既定とする。不明だったcallを新しく成立させ得るretryは別途明示承認が必要。それでもdedup・同一args条件を省けない。

## 6. 会計と更新

不明予約を日次resetや期限で解放しない。元のbudget epochに紐づけ、累積委任上限とoutstanding reservation上限を併用する。Succeededは予約→spentを1回、FailedDefinitiveは予約解除を1回だけ行う。同一payload retryで二重に予約しない。

model/schemaを更新するときは新workflowから新bundleを使う。旧workflowが終わるまで旧bundleを保てない場合、drainまたは明示的にstaleへ移す。旧結果と新結果を合成しない。

executorのcode/model rollbackと会計state rollbackを混同しない。nonce、送金成功、消費済みoperation、不明予約は保全する。不明Txが残る通常upgradeは既定で停止。災害復旧の古いsnapshotからは送金を凍結してledgerと照合する。

## 7. 不変条件

I01: modelの最大スコアでも委任・allowlist・上限を越えられない。

I02: 異なるsnapshot/schema/model/calibrationのsignalを合成して許可できない。

I03: 全required slotなしのapprove、失効後の新規dispatchはできない。

I04: 同一caller nonce・同一operationは多重実行できない。

I05: unknownの予約は失効・日付変更・一般retryで消えない。

I06: ledger成功はcallback trapやローカルrollbackで取り消せると扱わない。

I07: Scoreの平均だけへ情報を縮退させ、上側risk ruleを無効化しない。

I08: prefix/入力超過を無言で切り、異なる意味の質問として処理しない。

I09: quality/calibration未承認で本番送金を有効化しない。

I10: engineは送金を発行できず、externally deserialized valueはAuthorizedTransferにならない。
