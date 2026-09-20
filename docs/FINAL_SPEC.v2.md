# IC-Laya — 最終仕様 v2.0

**確定日: 2026-09-19（JST）**  
**設計状態: 機能・API・責務分離を固定。モデル品質・ICP性能・本番Txは受入試験前。**  
**対象: 英語 / Rust / ICP内推論 / Choice・Noul・Score / 型付き関数連携 / 制約付きTx**

## 1. 作るもの

**Layaのoption-marker scoringをbackendにした、少数質問向けのオンチェーン型付き判断ランタイム**を作る。仮称はIC-Laya。Jevの内部アーキテクチャ再現・完全API互換は主張しない。

第一級のprimitiveを最初から三つ提供する。

| Primitive | Rust側の概念型 | 契約 |
|---|---|---|
| Choice | `Choice<T>` | 登録された有限候補への分布と、最上位候補 |
| Noul | `Noul<P>` | 登録された命題Pに対するfalse/trueの分布 |
| Score | `Score<S>` | 登録された順序尺度Sへの分布、期待レベル、正規化平均、上側確率 |

**削るのは大量質問・無制限入力・不要な計算であり、Scoreではない。** NLIへ戻してChoice/Scoreを擬似的に組み立てることも、既定のfallbackにしない。

ランタイムは送金専用ではない。routing、優先度判定、レビュー支援から使える。送金は同じ型付き判断を利用する最初の制約付きexecutor adapterと位置付ける。

## 2. 決定表

| 項目 | 固定する仕様 |
|---|---|
| Backendの第一参照 | `convaiinnovations/laya-typed-decisions`。独立repoをrevision固定。root `laya`との自動切替なし。 |
| 位置付け | 実装・移植の参照checkpoint。特定業務での実用採用は評価Gateを通す。 |
| 実装 | Rust / Candle CPUを初期実装に利用し、Laya独自headをadapterとして移植。 |
| 配備構成 | 資金を持たない`decision_engine`と、policy/永続workflow/権限を持つ`executor`。 |
| 推論単位 | engineの1 updateで1質問、batch size 1。 |
| 業務単位 | 1 workflowに登録済み質問を1〜3個。必要な分だけ逐次評価。 |
| 初期profile | `Compact128-v1`: 質問・候補・state・特殊tokenを全部合わせて128 tokens以下。 |
| Choice | 2〜5候補。候補ID・順序・説明をschemaで固定。 |
| Noul | 2候補。falseを0、trueを1に固定。 |
| Score | 3〜7段階。標準は5段階。各段階の意味をrubricに記述。 |
| 長文 | 無言の切詰め・自動要約なし。超過は明示的エラーまたはreview。 |
| schema | 管理者が登録するversion付きschema。質問文・候補・rubricを毎requestで自由に差し替えない。 |
| 数値計算 | F32で参照一致を確立。INT8を最初の最適化候補とする。採用は品質・速度の実測後。 |
| abstention | 校正済みの判定ruleとRust policyで決定。モデルのact headを実行権限にしない。 |
| Tx | executor所有の専用treasuryから、登録済みledger/accountへのICRC-1 transfer。 |
| 初期運用 | `ReportOnly`、送金上限0、実資金dispatch無効。 |

Layaの三つのprimitiveとoption-marker構造は公開仕様・実装で確認できる。一方typed-decisions版は四つの合成workflowに特化したcheckpointで、任意業務への品質を保証するものではない。[S01][S02][S03][S04]

## 3. 全体構成

```text
Application / Rust SDK
  │ typed proposal + workflow_id + caller nonce
  ▼
Executor canister
  ├ authentication / delegation / business-state validation
  ├ workflow registry / immutable evidence snapshot
  ├ q1 ──────────► Decision engine ──► Choice / Noul / Score
  ├ q2 (必要時) ─► Decision engine ──► Choice / Noul / Score
  ├ q3 (必要時) ─► Decision engine ──► Choice / Noul / Score
  ├ same-snapshot result binding / policy revalidation
  ├ budget reservation / internal AuthorizedTransfer
  └ durable dispatch ───────────────► Approved ICRC-1 ledger
```

decision_engineにはtreasury、送金API、任意method呼出し、秘密鍵、モデルを起点とした外部LLM fallbackを持たせない。executorはML演算へ直接依存せず、検証済みの型付きresultを扱う。

Decision-only利用では認証されたクライアントからengineを呼べるが、そのreceiptをexecutorへ持ち込んで送金許可の代用にするendpointは作らない。実行に使う評価はexecutorが発行し、保存したworkflowに束縛された応答だけとする。

## 4. 質問数・計算量・workflow

### 4.1 1質問と1業務を混同しない

`evaluate()`は常に1質問。`run_workflow()`は1〜3質問を扱う非同期ジョブである。3質問を一つの巨大forwardへまとめない。複数質問は独立したmodel evaluationであり、一つの潜在表現を無料で共有できると想定しない。公開Laya APIも質問別の入力を作ってbatch化している。[S04]

順番と実行条件は登録したworkflowに固定する。例えば必要十分な命題検査 → 対応候補 → リスク尺度という順にし、早い段階で拒否・reviewとなれば残りを省略できる。**早期の許可は不可**で、送金に必要な全signalが揃うまでdispatchしない。

三つの分布を独立と仮定して確率を掛け合わせない。同じmodel/stateに基づく相関したsignalとして扱い、最終policy全体の誤受理とcoverageを評価する。

### 4.2 同じstateに束縛する

workflow受付時に、元テキスト、信頼済み業務record、proposal、model bundle、schema群、calibration群、policy、委任のrevisionを固定する。各応答はworkflow ID、question slot、evaluation ID、snapshot hash、model/schema/calibration IDを持つ。

同じslotへの再送は同じevaluation IDを使う。engine側は重複要求を既存resultへ収束させ、異なるpayloadの同じIDを拒否する。推論中のtransport障害でも無制限に計算を繰り返さない。rate limit、期限、再送回数上限を設ける。

途中でstateやmodelが変われば、古い結果と新しい結果を混ぜない。該当workflowを`NeedsReview`または`Stale`へ移し、別versionの再評価は明示的な新attemptとして扱う。以前の結果が遅れて届いても、新attemptへ採用しない。

## 5. Compact128と入力コンパイラ

128は品質・資源を測るための製品profileであり、ModernBERTの数学的な特殊境界ではない。local attentionの幅が128でも、そのmaskが128-token入力で全結合になるとは限らない。Candle実装では幅の半分を距離制限にしている。[S05]

入力はLayaの形式を維持する。

```text
[CLS] <type> question: <instructions> [SEP]
[MASK] <option0> ... [MASK] <optionK-1> [SEP]
<state> [SEP]
```

`CompiledSchema`はstatic prefix token IDs、marker位置、qtype、候補ID、rubric、tokenizer hashを保持する。prefixは最終state用SEPを除き64 tokens以下。動的stateの最大長は`128 - prefix_len - 1`であり、prefixが上限でも63 tokensの余地を確保する。末尾まで含めた全長を必ず検査する。

**7段階のScoreでもrubricが収まるschemaだけを登録する。** 説明を削って意味を変えたり、勝手に7段階を5段階へ縮めたりしない。入りきらないschemaは登録拒否とし、後続の`Extended256`を別profileとして検証・校正してから有効化する。v2初期releaseではExtended256は無効。

Layaの元rendererにはinstruction/option/stateの切詰めがある。[S03] 移植では受理する入力を「切詰めが一切起きない部分集合」に限定する。各optionの元コード上の上限やhead budgetも満たすことを確認し、tokenizer fixtureで完全一致させる。max_lenだけ128へ下げ、head_max_lenをそのままにして正常だと判断しない。

raw stateは16KiB以下をtokenize前に検査。予約済み特殊tokenのリテラル注入はv2では拒否する。アプリのJSONを使う場合はrendererのfield順・空白・escapeをschemaに固定し、modelへ見せたbytesのhashも記録する。

固定prefixのtoken IDsは再利用するが、bidirectional encoderのhidden stateやdecoder用KV cacheは再利用しない。動的stateを変更したらencoder全体を評価する。

## 6. Scoreを第一級の値にする

### 6.1 必須の返り値

`Score<S>`は、単なるfloatではなく次の情報を持つ。

| Field | 内容 |
|---|---|
| `schema_id / schema_version / rubric_hash` | 尺度・段階の定義と方向 |
| `distribution_ppm` | 段階順に並べた確率質量。合計1,000,000 |
| `expected_level_microunits` | 0〜K−1の段階番号の期待値を10⁶倍した値 |
| `mean_ppm` | 期待段階をK−1で割って0〜1へ正規化した値を10⁶倍 |
| `tail_from(bin_id)` | 指定段階以上の確率質量の和。SDKで分布から計算 |
| `diagnostics` | top1、margin、分布entropy等。真の正答確率と呼ばない |
| `stamp` | workflow/snapshot/model/profile/calibrationとの結び付き |

Laya APIの元のScoreは段階番号の期待値であり、初めから0〜1の値ではない。[S04] 正規化の分母を明示し、段階数を変えても値の意味が同じだと誤解させない。

等間隔の段階番号を使う平均は表示用の規約であり、「low→medium」と「high→severe」の実世界の差が同じという主張ではない。別のutility weightが必要ならpolicyに明示する。Riskの0.2は実世界で20%の損失が起きるという意味ではない。

### 6.2 平均だけで判定しない

説明用の分布として、5段階のAとBを比較する。

```text
A = [0,   0, 1, 0,   0]   normalized mean = 0.5
B = [0.5, 0, 0, 0, 0.5]   normalized mean = 0.5

P(level >= 3): A = 0、B = 0.5
```

同じ平均でも上側の質量は違う。高リスク側を問題にするpolicyは`tail_from(high_bin)`を利用し、そのeventの校正も評価する。ただしその値もmodelがrubric上の段階へ割り当てた確率であり、現実の事故率ではない。

未知・情報不足を中間段階へ押し込まない。Scoreの`Undetermined/Abstained`は分布とは別の外側variantとして扱う。入力や校正の条件を満たさないときに「中くらい」を返すAPIにはしない。

## 7. 型付きAPIと実行権限

Rust SDKの公開概念型は以下に固定する。

```rust
Choice<Action>       // Actionの有限候補
Noul<RefundRequested> // 命題を型で区別
Score<Risk>           // RiskとUrgencyを取り違えない
Evaluation<T>         // Assessed(T) / Abstained(reason) / Error(reason)
```

Candid境界ではgenericをそのまま公開せず、schema ID、version、variant、候補IDを含むDTOを使う。SDKとexecutorはexpected schema/hash/candidate orderを検査してからローカル型へ戻す。`Score<Risk>`と`Score<Urgency>`の取り違えを型で防ぎ、同じRiskでもrubric versionの違いは実行時に検査する。[S11][S12]

概念上、アプリは次のように合成する。これは関数の関係を示す設計例であり、実装済みSDKではない。

```rust
// 同じsnapshotに対して必要なsignalをそろえる。
let intent: Noul<RefundRequested> = ...;
let action: Choice<PaymentAction> = ...;
let risk: Score<PaymentRisk> = ...;

// 型の一致だけでなく、stamp・校正・現stateも検査する。
let signals = policy.bind_required_signals(intent, action, risk)?;
let authorized = policy.authorize(proposal, signals, current_state)?;
executor.dispatch(authorized).await?;
```

`AuthorizedTransfer`はexecutor内部のprivate constructorでのみ生成し、外部DTO・Deserialize・Clone/Copyにしない。Rust型は意味的正しさやcontrollerの善意を証明しない。永続request state、caller認証、nonce、業務operation ID、予算検査と併用する。

モデルが選ぶ`PaymentAction::Execute`は提案に過ぎない。金額、宛先、fee、送金元、委任、残高・invoice等の機械的事実はRust/app stateで検証し、modelの「認める」という文章やscoreから生成しない。

## 8. Calibrationとact head

校正は`model + dtype/kernel + tokenizer + compiled schema + candidate/rubric order + profile + task domain`の組に対して行う。Choice、Noul、Scoreの一つだけが合格しても、別primitiveの合格とは扱わない。

typed-decisions checkpointには、継承した`temperature_by_options`がper-type temperatureを上書きする注意点が記載されている。[S02] 配備時は有効temperatureを一つに解決した校正artifactを作り、元設定を二重適用しない。まず元APIの出力を再現するparity試験、その後に新しい校正設定での受入試験を行う。

APIが返すentropy由来の集中度は診断値であり、普遍的な正答confidenceではない。[S03] 0.95/0.98といった閾値を設計者が根拠なく本番設定へ入れない。

act/escalate headは参照実装では保持して一致性を検証する。option logitsへ逆流しない出力枝であることを確認した上で、配備runtimeでは計算を省略できる。[S03] **削除の狙いは責務整理であり、主要な速度改善とは見積もらない。** learned abstentionを使わないpolicyは、独立にrisk–coverageを評価する。

## 9. Rust移植と最適化

CandleにはModernBERT backboneの実装があるが、Layaの独自headを含む完成済みcanisterは確認できていない。[S05] `ModernBertForSequenceClassification`をそのまま使うのではなく、backboneのhidden出力へqtype embedding、2層decision transformer、marker scorerを接続する。

**演算を似たものへ置換しない。** 特にLayaのdecision transformerはPyTorchの`TransformerEncoderLayer`をactivation未指定で構築しているため既定のReLUを確認する。scorerのGELU、ModernBERTのGeGLUと区別する。[S03][S05][S06] LayerNorm epsilon、bias、pre-norm、RoPE、padding/local mask、label順もfixtureに含める。

最適化の順序は固定する。

1. F32でPython → Rust native → ICPの同一入力に対する一致性を確立。
2. prefix token、attention mask、RoPE table等の再利用、scratch bufferの再利用、不要コピー削減。
3. 実測で重いlinear演算へINT8/SIMDを導入し、再校正・Scoreの上側誤りを再評価。
4. 元モデルが品質/資源を満たさない場合のみ、同じ三primitiveを保つ小型studentへ蒸留。自動切替しない。

INT8 weightsを保持してmatmul直前に全部F32へ展開するだけでは、速度改善を仮定できない。採用する量子化方式は実際のWasm kernel込みで記録する。F16の保存サイズとF32での常駐量、stable memoryとheap、一時bufferを別々に測定する。

weightsはstable memoryにchunk uploadし、manifest・shape・hashを検証。ロード時の二重保持を避け、必要なら管理された複数updateにwarm-upを分割する。active modelは一つに限定し、無停止切替のための二重常駐は初期要件にしない。

## 10. ICP受入予算

ICP公開値はupdateあたり40 billion instructions、wasm32 heap 4GiB。[S07] 以下は**実測値ではなく本設計の受入目標**である。

| 対象 | v2目標 |
|---|---|
| 1質問の総instructions | tokenize、推論、後処理、serialization込みで20B以下 |
| 1workflow | 最大3評価。評価分の上限見積り60B + orchestration/transport分。 |
| 通常のwarm heap | 2.5GiB以下 |
| cold-loadを含むpeak heap | 3.0GiB以下 |
| 入力 | 64 / 96 / 128 tokensで計測。128が配備上限。 |
| Primitive別 | Noul 2候補、Choice 2/3/5、Score 3/5/7を個別計測 |
| latency/cycles | 実replicaで測定し、別途運用SLOを承認。予測の速度を保証しない。 |

3質問は別updateで実行するため、workflowの総計算を一つの40B上限へ押し込む設計ではない。一方、分割しても総cyclesや待ち時間は消えない。workflow全体の予算と期限を設ける。

INT8でも目標を満たさなければ小型studentを検証する。Scoreを消す、rubricを隠れて短縮する、inputを切り捨てる、上限ぎりぎりで本番を有効化することは採用しない。

## 11. Tx安全性

初期adapterはexecutorが所有する単一treasuryからのICRC-1 transfer。ownerが、delegate、task、ledger、宛先、1回/累積/期間別上限、fee上限、期限を登録する。個人のwalletから無断でdebitできるとは扱わない。[S10]

workflow受付時にcaller nonceを消費し、信頼済み業務operation IDを固定する。別nonceで同じinvoiceを二重処理することも防ぐ。queryによる状態表示やクライアント持込みのdecisionを、送金許可にしない。

最後のevaluationが完了したら、委任、失効、policy、snapshot、予算を再検査。予算予約、operation予約、送信args凍結、Submitted記録、初回outcallの発行までの間に別の外部awaitを入れない。[S09]

```text
Received → Evaluating(1..3) → ReadyToDispatch → Submitted
                ├ NeedsReview / Rejected / Stale      ├ Succeeded
                └ Cancelled                          ├ FailedDefinitive
                                                     └ OutcomeUnknown
```

結果不明は失敗ではない。amount、to、fee、memo、created_at_timeを変えた盲目的再送を禁止し、予約を維持する。ICRC-1のdedupはledgerごとに確認する。unknownの先行attemptがある限り、後のTooOld/BadFeeだけで未実行と確定しない。[S08][S10]

失効やpause後は新しいdispatchを止めるが、既に送信したTxを取消できるとは約束しない。日付・期限を越えても不明予約を解放しない。upgradeでnonce、成功履歴、業務消費状態、予約を巻き戻さない。

## 12. 固定したこと・残る検証

**固定:** 三primitive、Laya型backend、二canister分離、少数逐次質問、Compact128、schema registry、Score分布と上側確率、内部認可型、永続送金状態機械。

**受入試験で決める:** 対象業務の学習/校正artifact、固定weight/runtime revision、F32/INT8の配備形式、実測latency/cycles、ledger固有profile、本番risk budget。

これらを区別し、最終仕様であることを「421MのLayaがすでにICPで動作・性能検証済み」とは表現しない。モデルweightsや本番資金に未検証の値を代入せず、未承認の状態ではReportOnlyのままにする。

## 13. 受入完了の定義

Choice・Noul・Scoreがすべて型付き値として返り、同じsnapshotの結果を普通のRust関数へ渡せること。Scoreは平均と分布・上側質量を区別できること。必要な質問を省いた許可ができないこと。モデルが最大スコアを誤って返してもhard policyを越えられず、結果不明から二重送金せずに回復できること。

**最終形は、Layaをそのままホストするサービスではなく、三つの判断primitiveをRustの制御フローと制限付き実行権限へ接続するIC上のランタイムである。**
