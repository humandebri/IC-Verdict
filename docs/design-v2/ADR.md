# Architecture Decision Records — v2.0

2026-09-19（JST）

本版は会話中のv1設計を置き換える。ここでの確定は設計・実装方針であり、未実施の品質/性能Gateや本番運用承認を完了扱いしない。


---

# ADR-001: 三つのtyped decision primitiveを製品の核にする

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

## 背景

従来のNLI-only設計では、実行時候補のChoiceとordinal Scoreを不自然に代用することになる。

## 決定

Choice<T>・Noul<P>・Score<S>をv2の必須primitiveとして固定する。Jevの内部構造や完全互換は主張しない。

## トレードオフと不採用案

NLIの候補別確率を後から正規化する方式を既定backendにしない。モデル交換後もprimitive契約を維持する。

## 検証条件

三primitiveの入出力と失敗型を別々に試験する。[S01][S04]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-002: Laya typed-decisionsを第一参照checkpointに固定する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

## 背景

小型NLIではなく、動的候補をmarkerで読むarchitectureが目的に合う。ただしspecialist checkpointの汎化は未確認。

## 決定

独立repo convaiinnovations/laya-typed-decisionsをrevision固定して移植する。四つの合成workflow以外への品質は独自評価する。

## トレードオフと不採用案

root Layaへのsilent routingや、モデルカードの数値だけによる本番採用はしない。必要なら同型のtask別fine-tuningまたはstudentへ移行する。

## 検証条件

品質・資源Gate前は配備適合と認定しない。将来のweights交換もschema/profile/calibrationを再固定する。[S02]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-003: Candle backboneとLaya固有headを分離する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

## 背景

CandleにModernBERTはあるが、Layaの完成済みcanisterではない。標準sequence classifierとも異なる。

## 決定

backbone hidden出力にqtype embedding、2層decision transformer、marker scorerを移植する薄いadapterを作る。

## トレードオフと不採用案

Candleだけで無修正動作するとは仮定しない。PyTorch既定のReLU、scorer GELU、backbone GeGLU、mask、normを混同しない。

## 検証条件

Python/native/Wasm/ICPのfixture一致とimport/数値検査を通す。[S03][S05][S06]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-004: 分布・診断値・校正・abstentionを区別する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

## 背景

entropyが低くても正しいとは限らず、Scoreの大きさは確信度ではない。付属temperatureには優先順位の注意点がある。

## 決定

全primitiveで分布を保存し、calibration artifactをmodel/dtype/schema/profile/taskに結び付ける。effective temperatureを一意に解決する。

## トレードオフと不採用案

普遍的confidence閾値や、Scoreの平均だけを承認根拠にする処理は採用しない。learned act headは参照で保持し、policyから独立させる。

## 検証条件

元出力parity後に再校正し、最終policyのrisk–coverageも測る。[S02][S03][S04]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-005: engineとexecutorの二canister境界を維持する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

## 背景

重い推論と資金権限を同じ変更範囲に入れない。

## 決定

engineは推論のみ、executorは認証・workflow・委任・予算・ledger adapterを担当する。

## トレードオフと不採用案

callが増え、非同期状態管理が必要になる。GPU的な一括低latencyは約束しない。

## 検証条件

engineから送金できず、クライアントの持込みscoreをexecutorが信用しないことを試験する。[S09]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-006: 委任と内部capabilityを送金の必須条件にする

版: v2.0 / 状態: 設計採用 / 前版から: 維持

## 背景

正しい型の候補を選ぶことと、資金所有者の許可は別。

## 決定

executor所有の専用treasuryへ明示委任を設定。AuthorizedTransferはprivateな内部型で、外部DTO/Deserialize/Clone/Copyにしない。

## トレードオフと不採用案

Rustの型だけでone-time実行や意味的正しさを証明しない。nonce、operation、現stateを併用する。

## 検証条件

偽造型、権限外caller、金額/宛先差替え、失効でdispatchできないことを確認する。[S10][S12]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-007: 複数signalを同一snapshotに束縛する

版: v2.0 / 状態: 設計採用 / 前版から: 強化

## 背景

逐次評価中にstateやbundleが変わり、都合のよい結果だけ混ぜると誤った認可になる。

## 決定

全slotのworkflow/model/schema/profile/calibration/snapshotを照合。必須slot完了後に現stateと委任/予算を再検査する。

## トレードオフと不採用案

違うslotの確率を独立として掛けない。古い結果を新attemptへ使わず、失効/stale時はreviewへ。

## 検証条件

partial set、stale callback、model変更、budget競合、早期approveを拒否する。[S09]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-008: 結果不明を保持する永続Tx状態機械を維持する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

## 背景

外部ledgerの処理はcallbackの失敗で巻き戻らない。retryで二重送金が起き得る。

## 決定

FrozenTransfer、Submitted、OutcomeUnknown、nonce/operation消費状態を永続化する。不明予約は保持し、同一argsかつ確認済みdedup条件でのみ回復する。

## トレードオフと不採用案

unknown後のTooOld/BadFeeで過去の未実行を断定しない。新timestampによるblind retry、一般的なexactly-once保証は不採用。

## 検証条件

late success、callback trap、日次reset、dedup expiry、revocation後retryを故障注入する。[S08][S10]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-009: モデルは一つだけ常駐しロードpeakを管理する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

## 背景

保存ファイルが入っても、raw bytesとtensorの二重保持でheapが不足し得る。

## 決定

stable memoryにchunk uploadしhash/shapeを検査。管理されたwarm-upとactive bundle切替を行う。

## トレードオフと不採用案

無停止切替の二重常駐は初期要件にしない。init/upgradeで無制限の一括ロードをしない。

## 検証条件

warm/cold heap、ロード命令数、破損chunk、upgrade回復を測る。[S07]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-010: F32基準から実測でINT8へ進む

版: v2.0 / 状態: 設計採用 / 前版から: 維持

## 背景

INT8はサイズを減らしても、kernel次第では速くならず出力も変わる。

## 決定

F32でcorrectnessを確立し、prefix/scratch再利用後、hot linearへINT8/SIMDを適用する。

## トレードオフと不採用案

いきなりINT4化、全weightsの毎回F32展開、未検証fast mathによる速度主張は採用しない。

## 検証条件

三primitiveのparity/校正とScoreの上側誤りを再評価。速度とheapが改善した方式だけ採用する。[S05]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-011: 入口制限とevaluationの重複排除を必須にする

版: v2.0 / 状態: 設計採用 / 前版から: 強化

## 背景

少数質問でも巨大入力・無限retry・大量未完了ジョブで資源を消費する。

## 決定

認証、16KiB byte cap、token cap、slot cap、quota、expiry、live evaluation IDの重複排除を設ける。

## トレードオフと不採用案

無制限query推論、無言切詰め、expired IDを新規計算する処理は不採用。

## 検証条件

最大入力、予約特殊token、同じID別payload、transport再送、pending枯渇を試験する。

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-012: schema・model・calibrationの版を一体管理する

版: v2.0 / 状態: 設計採用 / 前版から: 強化

## 背景

同じRisk型でもrubricが変われば数値の意味は変わる。

## 決定

weights/config/tokenizer/runtime/renderer/profile/schema/calibration/ruleをhashで識別する。原文とeffective inputのhashも記録する。

## トレードオフと不採用案

main追従、無言モデル変更、ログへの無制限原文、会計snapshotの巻戻しは禁止。

## 検証条件

旧新schema混在、in-flight更新、nonce維持、原文保持方針、controllerの更新手続を確認する。

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-013: 全primitiveと合成policyを別Gateで受け入れる

版: v2.0 / 状態: 設計採用 / 前版から: 強化

## 背景

Choiceだけ良くてもScoreの重大な過小評価で危険な処理を許可し得る。

## 決定

Choice/Noul/Score、最終workflow、実装parity、資源、安全状態機械、対象ledgerを独立Gateにする。

## トレードオフと不採用案

平均accuracyや温度調整の成功だけでLiveへ移行しない。最初はReportOnly/上限0。

## 検証条件

ScoreのRPS/CDF誤差/重大過小評価、workflow誤受理率とcoverage、独立reviewを含める。

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-014: Score/Choice後回しの決定を撤回する

版: v2.0 / 状態: 設計採用 / 前版から: 置換

## 背景

前版ADR-014の延期は、今回のLaya採用とScore必須要件に合わない。

## 決定

Choice・Noul・Scoreを同じ優先度でv2 coreとする。大量候補、任意tool生成、無制限dynamic schemaだけを対象外とする。

## トレードオフと不採用案

Scoreが重い/精度不足ならschema/profile/modelを見直す。Score削除によって三primitive対応完了とはしない。

## 検証条件

三primitiveが一つのengineと同じSDK型体系で使えることを受入条件にする。[S01]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-015: Scoreはrubricと分布を正本にする

版: v2.0 / 状態: 設計採用 / 前版から: 新規

## 背景

元LayaのScoreは0〜K−1の期待段階。正規化値や平均だけを返すと意味・尾の情報を失う。

## 決定

期待段階、正規化平均、mass/CDF、上側確率を提供。尺度の方向・段階説明・versionをschemaに固定し、Score<Risk>とScore<Urgency>を区別する。

## トレードオフと不採用案

unknownを中間binにしない。ordinalの等間隔表示を実世界の確率や損失へ読み替えない。

## 検証条件

同平均/異なるtail、3/5/7 bins、最大値、丸め、schema変更を数学fixtureで試験する。[S04]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。


---

# ADR-016: Compact128・1評価1問・最大3slotを固定する

版: v2.0 / 状態: 設計採用 / 前版から: 新規

## 背景

少数questionで三primitiveを合成したいが、大batchと長文でICP予算を超えさせたくない。

## 決定

1 engine update=1問。1workflow=1〜3問の逐次ジョブ。static prefix≤64、総長≤128。Score標準5/max7、Choice2〜5、Noul2。

## トレードオフと不採用案

短いheadへ無言でrubricを切らない。入りきらない場合のExtended256は別Gate。3評価の総コストは1評価分にならない。

## 検証条件

全primitiveの最大候補数でprefix余地と総instructions/heapを測り、early denyとrequired-slot completenessを試験する。[S03][S04][S07]

出典: [SOURCES.md](SOURCES.md)。正本: [最終仕様](FINAL_SPEC.md)。
