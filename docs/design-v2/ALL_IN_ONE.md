# IC-Laya

## 最終仕様・実装計画・ADR v2.0

**2026-09-19（JST） | 英語・Rust・ICP | Choice / Noul / Score**

> 設計・API・責務分離を固定した版です。実モデル推論、Rust/Wasm build、ICP性能、ledger接続、実資金Txは未検証です。

## 目次

1. [最終仕様](#part-spec)
2. [実装計画と受入Gate](#part-plan)
3. [Score・API・型の契約](#part-score)
4. [WorkflowとTx状態機械](#part-workflow)
5. [16件のArchitecture Decision Records](#part-adr)
6. [v1からの変更](#part-changes)
7. [自己レビューと修正](#part-review)
8. [一次資料と確認範囲](#part-sources)

---

<a id="part-spec"></a>

## 1. 最終仕様

**確定日: 2026-09-19（JST）**  
**設計状態: 機能・API・責務分離を固定。モデル品質・ICP性能・本番Txは受入試験前。**  
**対象: 英語 / Rust / ICP内推論 / Choice・Noul・Score / 型付き関数連携 / 制約付きTx**

### 1. 作るもの

**Layaのoption-marker scoringをbackendにした、少数質問向けのオンチェーン型付き判断ランタイム**を作る。仮称はIC-Laya。Jevの内部アーキテクチャ再現・完全API互換は主張しない。

第一級のprimitiveを最初から三つ提供する。

| Primitive | Rust側の概念型 | 契約 |
|---|---|---|
| Choice | `Choice<T>` | 登録された有限候補への分布と、最上位候補 |
| Noul | `Noul<P>` | 登録された命題Pに対するfalse/trueの分布 |
| Score | `Score<S>` | 登録された順序尺度Sへの分布、期待レベル、正規化平均、上側確率 |

**削るのは大量質問・無制限入力・不要な計算であり、Scoreではない。** NLIへ戻してChoice/Scoreを擬似的に組み立てることも、既定のfallbackにしない。

ランタイムは送金専用ではない。routing、優先度判定、レビュー支援から使える。送金は同じ型付き判断を利用する最初の制約付きexecutor adapterと位置付ける。

### 2. 決定表

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

Layaの三つのprimitiveとoption-marker構造は公開仕様・実装で確認できる。一方typed-decisions版は四つの合成workflowに特化したcheckpointで、任意業務への品質を保証するものではない。[S01](#source-s01)[S02](#source-s02)[S03](#source-s03)[S04](#source-s04)

### 3. 全体構成

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

### 4. 質問数・計算量・workflow

#### 4.1 1質問と1業務を混同しない

`evaluate()`は常に1質問。`run_workflow()`は1〜3質問を扱う非同期ジョブである。3質問を一つの巨大forwardへまとめない。複数質問は独立したmodel evaluationであり、一つの潜在表現を無料で共有できると想定しない。公開Laya APIも質問別の入力を作ってbatch化している。[S04](#source-s04)

順番と実行条件は登録したworkflowに固定する。例えば必要十分な命題検査 → 対応候補 → リスク尺度という順にし、早い段階で拒否・reviewとなれば残りを省略できる。**早期の許可は不可**で、送金に必要な全signalが揃うまでdispatchしない。

三つの分布を独立と仮定して確率を掛け合わせない。同じmodel/stateに基づく相関したsignalとして扱い、最終policy全体の誤受理とcoverageを評価する。

#### 4.2 同じstateに束縛する

workflow受付時に、元テキスト、信頼済み業務record、proposal、model bundle、schema群、calibration群、policy、委任のrevisionを固定する。各応答はworkflow ID、question slot、evaluation ID、snapshot hash、model/schema/calibration IDを持つ。

同じslotへの再送は同じevaluation IDを使う。engine側は重複要求を既存resultへ収束させ、異なるpayloadの同じIDを拒否する。推論中のtransport障害でも無制限に計算を繰り返さない。rate limit、期限、再送回数上限を設ける。

途中でstateやmodelが変われば、古い結果と新しい結果を混ぜない。該当workflowを`NeedsReview`または`Stale`へ移し、別versionの再評価は明示的な新attemptとして扱う。以前の結果が遅れて届いても、新attemptへ採用しない。

### 5. Compact128と入力コンパイラ

128は品質・資源を測るための製品profileであり、ModernBERTの数学的な特殊境界ではない。local attentionの幅が128でも、そのmaskが128-token入力で全結合になるとは限らない。Candle実装では幅の半分を距離制限にしている。[S05](#source-s05)

入力はLayaの形式を維持する。

```text
[CLS] <type> question: <instructions> [SEP]
[MASK] <option0> ... [MASK] <optionK-1> [SEP]
<state> [SEP]
```

`CompiledSchema`はstatic prefix token IDs、marker位置、qtype、候補ID、rubric、tokenizer hashを保持する。prefixは最終state用SEPを除き64 tokens以下。動的stateの最大長は`128 - prefix_len - 1`であり、prefixが上限でも63 tokensの余地を確保する。末尾まで含めた全長を必ず検査する。

**7段階のScoreでもrubricが収まるschemaだけを登録する。** 説明を削って意味を変えたり、勝手に7段階を5段階へ縮めたりしない。入りきらないschemaは登録拒否とし、後続の`Extended256`を別profileとして検証・校正してから有効化する。v2初期releaseではExtended256は無効。

Layaの元rendererにはinstruction/option/stateの切詰めがある。[S03](#source-s03) 移植では受理する入力を「切詰めが一切起きない部分集合」に限定する。各optionの元コード上の上限やhead budgetも満たすことを確認し、tokenizer fixtureで完全一致させる。max_lenだけ128へ下げ、head_max_lenをそのままにして正常だと判断しない。

raw stateは16KiB以下をtokenize前に検査。予約済み特殊tokenのリテラル注入はv2では拒否する。アプリのJSONを使う場合はrendererのfield順・空白・escapeをschemaに固定し、modelへ見せたbytesのhashも記録する。

固定prefixのtoken IDsは再利用するが、bidirectional encoderのhidden stateやdecoder用KV cacheは再利用しない。動的stateを変更したらencoder全体を評価する。

### 6. Scoreを第一級の値にする

#### 6.1 必須の返り値

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

Laya APIの元のScoreは段階番号の期待値であり、初めから0〜1の値ではない。[S04](#source-s04) 正規化の分母を明示し、段階数を変えても値の意味が同じだと誤解させない。

等間隔の段階番号を使う平均は表示用の規約であり、「low→medium」と「high→severe」の実世界の差が同じという主張ではない。別のutility weightが必要ならpolicyに明示する。Riskの0.2は実世界で20%の損失が起きるという意味ではない。

#### 6.2 平均だけで判定しない

説明用の分布として、5段階のAとBを比較する。

```text
A = [0,   0, 1, 0,   0]   normalized mean = 0.5
B = [0.5, 0, 0, 0, 0.5]   normalized mean = 0.5

P(level >= 3): A = 0、B = 0.5
```

同じ平均でも上側の質量は違う。高リスク側を問題にするpolicyは`tail_from(high_bin)`を利用し、そのeventの校正も評価する。ただしその値もmodelがrubric上の段階へ割り当てた確率であり、現実の事故率ではない。

未知・情報不足を中間段階へ押し込まない。Scoreの`Undetermined/Abstained`は分布とは別の外側variantとして扱う。入力や校正の条件を満たさないときに「中くらい」を返すAPIにはしない。

### 7. 型付きAPIと実行権限

Rust SDKの公開概念型は以下に固定する。

```rust
Choice<Action>       // Actionの有限候補
Noul<RefundRequested> // 命題を型で区別
Score<Risk>           // RiskとUrgencyを取り違えない
Evaluation<T>         // Assessed(T) / Abstained(reason) / Error(reason)
```

Candid境界ではgenericをそのまま公開せず、schema ID、version、variant、候補IDを含むDTOを使う。SDKとexecutorはexpected schema/hash/candidate orderを検査してからローカル型へ戻す。`Score<Risk>`と`Score<Urgency>`の取り違えを型で防ぎ、同じRiskでもrubric versionの違いは実行時に検査する。[S11](#source-s11)[S12](#source-s12)

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

### 8. Calibrationとact head

校正は`model + dtype/kernel + tokenizer + compiled schema + candidate/rubric order + profile + task domain`の組に対して行う。Choice、Noul、Scoreの一つだけが合格しても、別primitiveの合格とは扱わない。

typed-decisions checkpointには、継承した`temperature_by_options`がper-type temperatureを上書きする注意点が記載されている。[S02](#source-s02) 配備時は有効temperatureを一つに解決した校正artifactを作り、元設定を二重適用しない。まず元APIの出力を再現するparity試験、その後に新しい校正設定での受入試験を行う。

APIが返すentropy由来の集中度は診断値であり、普遍的な正答confidenceではない。[S03](#source-s03) 0.95/0.98といった閾値を設計者が根拠なく本番設定へ入れない。

act/escalate headは参照実装では保持して一致性を検証する。option logitsへ逆流しない出力枝であることを確認した上で、配備runtimeでは計算を省略できる。[S03](#source-s03) **削除の狙いは責務整理であり、主要な速度改善とは見積もらない。** learned abstentionを使わないpolicyは、独立にrisk–coverageを評価する。

### 9. Rust移植と最適化

CandleにはModernBERT backboneの実装があるが、Layaの独自headを含む完成済みcanisterは確認できていない。[S05](#source-s05) `ModernBertForSequenceClassification`をそのまま使うのではなく、backboneのhidden出力へqtype embedding、2層decision transformer、marker scorerを接続する。

**演算を似たものへ置換しない。** 特にLayaのdecision transformerはPyTorchの`TransformerEncoderLayer`をactivation未指定で構築しているため既定のReLUを確認する。scorerのGELU、ModernBERTのGeGLUと区別する。[S03](#source-s03)[S05](#source-s05)[S06](#source-s06) LayerNorm epsilon、bias、pre-norm、RoPE、padding/local mask、label順もfixtureに含める。

最適化の順序は固定する。

1. F32でPython → Rust native → ICPの同一入力に対する一致性を確立。
2. prefix token、attention mask、RoPE table等の再利用、scratch bufferの再利用、不要コピー削減。
3. 実測で重いlinear演算へINT8/SIMDを導入し、再校正・Scoreの上側誤りを再評価。
4. 元モデルが品質/資源を満たさない場合のみ、同じ三primitiveを保つ小型studentへ蒸留。自動切替しない。

INT8 weightsを保持してmatmul直前に全部F32へ展開するだけでは、速度改善を仮定できない。採用する量子化方式は実際のWasm kernel込みで記録する。F16の保存サイズとF32での常駐量、stable memoryとheap、一時bufferを別々に測定する。

weightsはstable memoryにchunk uploadし、manifest・shape・hashを検証。ロード時の二重保持を避け、必要なら管理された複数updateにwarm-upを分割する。active modelは一つに限定し、無停止切替のための二重常駐は初期要件にしない。

### 10. ICP受入予算

ICP公開値はupdateあたり40 billion instructions、wasm32 heap 4GiB。[S07](#source-s07) 以下は**実測値ではなく本設計の受入目標**である。

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

### 11. Tx安全性

初期adapterはexecutorが所有する単一treasuryからのICRC-1 transfer。ownerが、delegate、task、ledger、宛先、1回/累積/期間別上限、fee上限、期限を登録する。個人のwalletから無断でdebitできるとは扱わない。[S10](#source-s10)

workflow受付時にcaller nonceを消費し、信頼済み業務operation IDを固定する。別nonceで同じinvoiceを二重処理することも防ぐ。queryによる状態表示やクライアント持込みのdecisionを、送金許可にしない。

最後のevaluationが完了したら、委任、失効、policy、snapshot、予算を再検査。予算予約、operation予約、送信args凍結、Submitted記録、初回outcallの発行までの間に別の外部awaitを入れない。[S09](#source-s09)

```text
Received → Evaluating(1..3) → ReadyToDispatch → Submitted
                ├ NeedsReview / Rejected / Stale      ├ Succeeded
                └ Cancelled                          ├ FailedDefinitive
                                                     └ OutcomeUnknown
```

結果不明は失敗ではない。amount、to、fee、memo、created_at_timeを変えた盲目的再送を禁止し、予約を維持する。ICRC-1のdedupはledgerごとに確認する。unknownの先行attemptがある限り、後のTooOld/BadFeeだけで未実行と確定しない。[S08](#source-s08)[S10](#source-s10)

失効やpause後は新しいdispatchを止めるが、既に送信したTxを取消できるとは約束しない。日付・期限を越えても不明予約を解放しない。upgradeでnonce、成功履歴、業務消費状態、予約を巻き戻さない。

### 12. 固定したこと・残る検証

**固定:** 三primitive、Laya型backend、二canister分離、少数逐次質問、Compact128、schema registry、Score分布と上側確率、内部認可型、永続送金状態機械。

**受入試験で決める:** 対象業務の学習/校正artifact、固定weight/runtime revision、F32/INT8の配備形式、実測latency/cycles、ledger固有profile、本番risk budget。

これらを区別し、最終仕様であることを「421MのLayaがすでにICPで動作・性能検証済み」とは表現しない。モデルweightsや本番資金に未検証の値を代入せず、未承認の状態ではReportOnlyのままにする。

### 13. 受入完了の定義

Choice・Noul・Scoreがすべて型付き値として返り、同じsnapshotの結果を普通のRust関数へ渡せること。Scoreは平均と分布・上側質量を区別できること。必要な質問を省いた許可ができないこと。モデルが最大スコアを誤って返してもhard policyを越えられず、結果不明から二重送金せずに回復できること。

**最終形は、Layaをそのままホストするサービスではなく、三つの判断primitiveをRustの制御フローと制限付き実行権限へ接続するIC上のランタイムである。**


---

<a id="part-plan"></a>

## 2. 実装計画と受入Gate

2026-09-19（JST） / 正本は`FINAL_SPEC.md`。本書のタスクは未実装。

### 1. 実装順

| Phase | 実装内容 | 完了条件 |
|---|---|---|
| P0 — 固定仕様・task | 1つの英語workflow、三primitiveのschema、5段階Score rubric、誤りのコストを定義 | schema候補・順序・profileを固定。実送金はoff。 |
| P1 — 参照出力 | Laya typed-decisions、tokenizer、renderer、Python依存をrevision固定しfixture生成 | token IDs、markers、raw logits、元temperatureの有効値、三primitiveの参照出力を保存。 |
| P2 — Rust parity | Candle ModernBERT + Laya headを実装 | PythonとRustでtoken IDs完全一致。head activation・mask・dtype等を点検し、三primitiveで出力一致。 |
| P3 — Compact canister | chunk upload/warm-up、1質問update、登録schema、重複評価キャッシュ | 64/96/128 tokens、Choice/Noul/Score各候補数でinstructions/heapを計測。 |
| P4 — 数値最適化 | scratch再利用、静的prefix、SIMD、profile後にINT8 kernel | F32との品質差、上側Scoreの誤受理、calibrationを再検証。効果がなければINT8を採用しない。 |
| P5 — workflow/executor | 最大3質問の逐次ジョブ、same-snapshot binding、型付きsignal、委任・予算 | 一部質問欠落、別version、失効、再入で許可不可。 |
| P6 — ledger故障試験 | mock ledger、Submitted/OutcomeUnknown、同一args retry、late callback、upgrade | 予約・nonce・operation・成功記録の不変条件をすべて満たす。 |
| P7 — 校正・運用受入 | task別calibration、Shadow、対象ledger profile確認、独立review | Choice/Noul/Score/合成policyの各Gateを通し、ownerが上限・risk budgetを承認。 |

P5/P6はfake engineを使い、P1〜P4と並行して進めてよい。モデル移植が未完でも、認可と送金状態機械を独立に試験できる。日数や未計測の処理時間は約束しない。

### 2. モデル選択の境界

実装の第一参照checkpointは`convaiinnovations/laya-typed-decisions`に固定する。ただしこれは特定workflowのspecialistである。[S02](#source-s02) 本番taskに適合しなければ、同じarchitectureへtask別fine-tuningする。root Layaへの変更も評価・manifest変更を経て明示する。

421M版が資源目標を満たせない場合の次の候補は、ModernBERT-base級backboneとLaya式option-marker headを持つstudent。サイズ・速度・精度は現時点で保証しない。蒸留ではChoice、NoulだけでなくScoreの分布・順序・上側eventもtargetにする。NLIモデルへの無言の置換は禁止。

### 3. 推奨crate境界

```text
crates/
  decision-types/       # Choice<T>, Noul<P>, Score<S>, wire DTO
  schema-compiler/      # tokenizer、prefix/markers、rubric、hash
  laya-core/            # option logitsを返すbackend trait
  laya-candle/          # backbone + 2 decision layers + scorer
  decision-postprocess/# temperature、ppm、mean/CDF/tail、diagnostics
  decision-policy/     # 外部ML非依存の認可・校正/型検査
  workflow-core/       # snapshot、slots、attempts、状態遷移
  ledger-adapter/      # 固定したICRC-1送信・回復
  sdk/                 # schema照合とtyped復元
canisters/
  decision-engine/
  executor/
  mock-ledger/
xtask/
  reference-fixtures/
  model-pack/
  evaluation/
  instruction-bench/
```

`decision-policy`がCandleやmodel weightsをimportする依存構造にしない。ledgerのoutcall発行はadapterへ集約する。Pythonはoffline参照・学習・校正に限定し、canister runtimeへ持ち込まない。

### 4. Gate

#### G1 — 意味と品質

Choiceはaccuracy/macro-F1、NoulはBrier/NLL・false positive/negative、ScoreはMAEに加えてCDFの誤差・Ranked Probability Score・重大な過小評価・上側eventの校正を記録する。さらに最終workflowの危険な誤受理率とcoverageを測る。

例数より、業務レコード・template family単位でのtrain/calibration/test分割を優先する。modelcardのbenchmark値を自分のtaskの合格点として代入しない。small-sampleで過信しない。risk budgetと必要coverageが未承認なら実送金off。

Scoreは必須Gate。Choiceだけ成功してScoreは無効という状態を、三primitive対応の受入完了とは扱わない。5段階のうち中央へ集中する分布と両端へ分かれる分布をpolicyが区別する試験を含む。

#### G2 — 実装parity

token IDsとmarker位置の完全一致、schema変更検出、誤ったqtype・候補順拒否、softmaxの二重適用防止、temperatureの実効値、Scoreの単位を検証する。

logit比較の初期許容差はF32で`atol=1e-3, rtol=1e-3`を仮置きし、誤差源を記録して妥当性を判断する。閾値を跨ぐ危険な差を単なる許容誤差で済ませない。NaN/inf・不正分布・未知tensor・shape不一致は拒否。

元Laya APIのparityと、Compact128＋再校正後の製品parityを別fixtureで管理する。前者の切詰めまで本番APIにコピーしない。

#### G3 — ICP資源

1評価20B instructions以下、warm heap 2.5GiB以下、cold peak 3.0GiB以下を目標とする。これはICP上限ではなく余裕を持たせる設計値。[S07](#source-s07) tokenization、stable memoryコピー、model forward、後処理、response作成を含む。

質問数とScoreのbinsを分けて測る。5-bin Scoreは1質問だが、3-binよりstatic tokensが増える。stateの有効長を減らして速い値だけ出さない。全workflowのcycles、遅延、拒否率、cold restartも報告する。native秒数からICP秒数への換算はしない。

#### G4 — 実行安全性

必須試験: 他callerのrequest ID、nonce再利用、別nonceの同一operation、partial decision set、異なるsnapshot、model/schema変更、推論中revocation、2重process、quota枯渇、送信後callback trap、unknown後TooOld、遅延success、日次reset、upgrade、過去会計snapshot復元。

hard invariantはモデルが常に最大スコアを返すstubでも成立させる。型のcompile-fail試験と、Candid/runtimeの認証試験の両方が必要。

#### G5 — 本番有効化

特定ledgerのdedup/履歴照合仕様、上限、権限・controller管理、保持する原文、独立reviewを確認する。`ReportOnly → Mock → Shadow → LimitedLive`を順に進める。Liveを有効化しても、modelの誤推論リスクがゼロになったとは説明しない。

### 5. 最初の小さな完成形

1つの英語workflowを用意し、`Noul<RefundRequested>`、`Choice<PaymentAction>`、`Score<PaymentRisk>`を同じstateに対して返す。元から送金権限のあるtreasury operationのみを対象にする。

最初は値の表示・通常関数への受渡しまで。その後mock ledgerへ接続する。入力テキストだけを見て、recipient・amount・支払済み状態を捏造して送金するデモにはしない。


---

<a id="part-score"></a>

## 3. Score・API・型の契約

版: v2.0 / 2026-09-19（JST）

### 数値の正本

確率のwire単位はppmで、`M = 1_000_000`。softmaxの有限性・非負・合計を検証後、largest-remainder法で整数化する。端数同順位は固定bin index順。確率質量の合計を厳密にMにする。異常な値を丸めて正常化しない。

ScoreのKは3〜7、bin indexは0〜K−1。整数化したmassを`m_i`とする。

```text
expected_level_microunits = Σ i * m_i
mean_ppm = round_half_up(Σ i * m_i / (K - 1))
cdf_ppm(j) = Σ(i <= j) m_i
tail_ppm(j) = Σ(i >= j) m_i = M - cdf_ppm(j - 1)
```

演算はchecked u64以上を使い、overflowを拒否する。score indexの期待値は`expected_level_microunits / M`、正規化平均は`mean_ppm / M`。K=1は受理しない。bin indexが範囲外ならエラー。上側の尾は明示的に`>=`で、`>`と取り違えない。

wireは分布を正本とし、derived mean/CDF/tailは受信側でも再計算・照合できる。二つの異なる手順で計算したfloat平均と整数化平均を同じ権威ある値として保存しない。元logitsは評価用fixtureへ保存するが、本番の一般利用者へ常時返さない。

Noulの候補はfalse/trueの2つ。`p_true_ppm = mass[1]`。Choiceは候補IDを明示し、schema順で分布を並べる。argmaxが同点なら候補順で表示上のselectedを固定し、許可policyはtieをreviewへ回せる。

### 校正と診断

`CalibrationStatus`は`Uncalibrated`または、profile/schema/runtimeを含む署名対象ではないbundle識別子。単一の`confidence`だけを合否条件にしない。

entropy由来の`concentration`、top1、marginは分布の診断値である。既知のtask分布で選択したdecision ruleと組み合わせる。低entropyでも間違う可能性を残し、どのdomainでも正答確率と解釈しない。

Ordinal labelだけでScoreの真値を決めにくい場合は、ラベル規約を先に定義する。`unknown`をhigh/medium等と同じ軸のbinにしない。情報不足は`Evaluation::Abstained`かschema固有の別質問で扱う。

### 外部DTO

概念上のwire形は以下。これはCandidコンパイル済み定義ではなく、生成するIDLの契約である。

```text
DecisionRequest {
  evaluation_id,
  workflow_binding?,
  schema_id,
  schema_version,
  expected_model_bundle,
  profile_id,
  state_utf8,
  expires_at
}

DecisionResult {
  stamp: {
    evaluation_id, workflow_id?, question_slot?,
    state_hash, snapshot_hash?,
    model_bundle_hash, schema_hash, profile_id,
    calibration_id?, decision_rule_id?
  },
  outcome: Assessed(
    Choice { option_ids, mass_ppm, selected_id }
    | Noul { false_ppm, true_ppm }
    | Score { bin_ids, mass_ppm, expected_level_microunits, mean_ppm }
  ) | Abstained(reason) | Error(reason),
  usage: { input_tokens, measured_instructions }
}
```

clientが持ち込む`workflow_binding`は実行権限ではない。executorが使用するresultはexecutor自身が登録engineへ発行したcallの返答に限定する。SDKのgeneric型はDTOのschema/version/variantを検証してから構築する。[S11](#source-s11)[S12](#source-s12)

### 配備API

| 所属 | endpoint | 役割 |
|---|---|---|
| engine | `evaluate_one` update | 1質問。既知callerのquota、expiry、schema、expected bundleを検査 |
| engine | `lookup_evaluation` update | 同じcaller/evaluation IDの結果回復。新規推論しない |
| engine | `describe_schema` query | schemaの表示。信頼を要する用途はhash等を別途検証 |
| engine admin | stage/activate bundle、register schema | chunk/manifest検証後のみ有効化 |
| executor | `submit_workflow` update | 型付きproposal、operation ID、caller nonceを保存 |
| executor | `advance_workflow` update | 保存済みplanから次slot/認可へ。model outputを引数に取らない |
| executor | `get_workflow` query / verified update | 状態表示/確認。再送金しない |
| executor admin | delegate/revoke/pause/reconcile | callerの役割を明示し、通常model APIと分離 |

同じevaluation IDで違うstate/schemaを送る場合はconflict。期限切れのIDは、キャッシュが消えていても新規推論を起動しない。クライアントの再接続で同じものを何度も推論しないよう、live IDはexpiryまで保持し、caller別上限で保護する。

### TypedSDK

`Choice<T>`、`Noul<P>`、`Score<S>`は同じものを異なる名前で包むだけではなく、登録schemaと対応するdescriptorから生成する。型tagはRust側の誤用防止、version/hashは更新時の誤用防止を担当する。

検査済みのdecisionでも実行権限ではない。executor内部で、必要signalの完全性と同じsnapshotを検査した`BoundSignals`からのみ`AuthorizedTransfer`を作る。外部はどちらの内部型もdeserializeできない。

業務の閾値はこの文書に固定しない。risk budgetを未承認のまま`mean_ppm < 500_000`等の便宜的条件で送金を許可しない。


---

<a id="part-workflow"></a>

## 4. WorkflowとTx状態機械

v2.0 / 2026-09-19（JST）

### 1. 固定する記録

`WorkflowRecord`はcaller/nonce、operation ID、typed proposal、元evidence、snapshot hash、workflow plan、schema群、model bundle、calibration群、policy/delegation revision、slots、現在状態、送信args、予約、receiptを持つ。

`Snapshot`はapp側の真実性が検証されたstateと、単なる主張としての非信頼テキストを区別する。nonceやhashがあるだけで文中の事実を真実と認定しない。

### 2. 最大3slotの逐次評価

planは登録時に固定する。slotは`Pending → InFlight(evaluation_id) → Complete(result)`または`Abstained/Error`。順番を変えたり、望ましくない結果だけを無視したりしない。

各評価は独立したengine update。1slotに重複リクエストが来ても同じevaluation IDへ収束する。engineはlive IDについて同じpayloadのresultを返し、計算を再発行しない。キャッシュはexpiryとcaller別上限で管理し、expired IDには新規評価しない。期限・回数・cyclesの予算を予約して資源攻撃を防ぐ。

次slotを開始する条件はplanに明記。deny/reviewへのshort-circuitは可能だが、approveのために必須signalを省くことは不可。必要signal同士の独立性は仮定しない。

### 3. Workflow状態

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

### 4. 認可・資金・二重処理

初期はexecutorの専用treasuryのみ。ownerがdelegate、allowlist、task、期限、一回/累積/期間別予算、fee上限を登録する。modelの提案は委任を作らない。[S10](#source-s10)

callerごとのmonotonic nonceを永続化する。同一nonce/同一payloadは既存workflowへ収束。違うpayloadは拒否。compact後もwatermarkを保持し、低いnonceを再実行しない。

別nonceによる同じ業務を防ぐため、appで検証されたoperation状態を`Available → Reserved(workflow_id) → Consumed(workflow_id)`とする。任意の新IDをcallerが与えただけでは新規業務と認めない。同時proposalがあっても予約は一つ。

初回dispatch直前に、全signalのmodel/profile/schema/calibration/snapshot、policy/delegation revision、expiry、revocation、現在予算を確認する。金額+feeをchecked arithmeticで予約し、送信argsを固定、Submittedを記録して送信する。この間に他の外部awaitを挟まない。[S09](#source-s09)

### 5. 結果不明

ICRC-1 transfer argsはledger、caller identity、from_subaccount、to、amount、fee、memo、created_at_timeを固定する。unknown状態で新timestamp、memo、feeにして再送しない。[S08](#source-s08)[S10](#source-s10)

初回のみで確実な未実行エラーならFailedDefinitiveにできる。先行unknownがあるとき、後のTooOld/BadFeeはそのattemptの不受理を示しても、過去の成功を否定しない。予算とoperationの予約を保持する。

確認したledger profileのdedup期間・条件を満たす同一payloadだけ、回数を限定して再送を検討する。ICRC-1のdedupはSHOULDであり、任意ledgerのexactly-onceを保証しない。履歴に見つからないだけで未実行としない。先行callが後から実行され得ないことも確認する。[S10](#source-s10)[S08](#source-s08)

失効やpause後は受動照合を既定とする。不明だったcallを新しく成立させ得るretryは別途明示承認が必要。それでもdedup・同一args条件を省けない。

### 6. 会計と更新

不明予約を日次resetや期限で解放しない。元のbudget epochに紐づけ、累積委任上限とoutstanding reservation上限を併用する。Succeededは予約→spentを1回、FailedDefinitiveは予約解除を1回だけ行う。同一payload retryで二重に予約しない。

model/schemaを更新するときは新workflowから新bundleを使う。旧workflowが終わるまで旧bundleを保てない場合、drainまたは明示的にstaleへ移す。旧結果と新結果を合成しない。

executorのcode/model rollbackと会計state rollbackを混同しない。nonce、送金成功、消費済みoperation、不明予約は保全する。不明Txが残る通常upgradeは既定で停止。災害復旧の古いsnapshotからは送金を凍結してledgerと照合する。

### 7. 不変条件

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


---

<a id="part-adr"></a>

## 5. 16件のArchitecture Decision Records

2026-09-19（JST）

本版は会話中のv1設計を置き換える。ここでの確定は設計・実装方針であり、未実施の品質/性能Gateや本番運用承認を完了扱いしない。


---

### ADR-001: 三つのtyped decision primitiveを製品の核にする

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

### 背景

従来のNLI-only設計では、実行時候補のChoiceとordinal Scoreを不自然に代用することになる。

### 決定

Choice<T>・Noul<P>・Score<S>をv2の必須primitiveとして固定する。Jevの内部構造や完全互換は主張しない。

### トレードオフと不採用案

NLIの候補別確率を後から正規化する方式を既定backendにしない。モデル交換後もprimitive契約を維持する。

### 検証条件

三primitiveの入出力と失敗型を別々に試験する。[S01](#source-s01)[S04](#source-s04)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-002: Laya typed-decisionsを第一参照checkpointに固定する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

### 背景

小型NLIではなく、動的候補をmarkerで読むarchitectureが目的に合う。ただしspecialist checkpointの汎化は未確認。

### 決定

独立repo convaiinnovations/laya-typed-decisionsをrevision固定して移植する。四つの合成workflow以外への品質は独自評価する。

### トレードオフと不採用案

root Layaへのsilent routingや、モデルカードの数値だけによる本番採用はしない。必要なら同型のtask別fine-tuningまたはstudentへ移行する。

### 検証条件

品質・資源Gate前は配備適合と認定しない。将来のweights交換もschema/profile/calibrationを再固定する。[S02](#source-s02)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-003: Candle backboneとLaya固有headを分離する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

### 背景

CandleにModernBERTはあるが、Layaの完成済みcanisterではない。標準sequence classifierとも異なる。

### 決定

backbone hidden出力にqtype embedding、2層decision transformer、marker scorerを移植する薄いadapterを作る。

### トレードオフと不採用案

Candleだけで無修正動作するとは仮定しない。PyTorch既定のReLU、scorer GELU、backbone GeGLU、mask、normを混同しない。

### 検証条件

Python/native/Wasm/ICPのfixture一致とimport/数値検査を通す。[S03](#source-s03)[S05](#source-s05)[S06](#source-s06)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-004: 分布・診断値・校正・abstentionを区別する

版: v2.0 / 状態: 設計採用 / 前版から: 改訂

### 背景

entropyが低くても正しいとは限らず、Scoreの大きさは確信度ではない。付属temperatureには優先順位の注意点がある。

### 決定

全primitiveで分布を保存し、calibration artifactをmodel/dtype/schema/profile/taskに結び付ける。effective temperatureを一意に解決する。

### トレードオフと不採用案

普遍的confidence閾値や、Scoreの平均だけを承認根拠にする処理は採用しない。learned act headは参照で保持し、policyから独立させる。

### 検証条件

元出力parity後に再校正し、最終policyのrisk–coverageも測る。[S02](#source-s02)[S03](#source-s03)[S04](#source-s04)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-005: engineとexecutorの二canister境界を維持する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

### 背景

重い推論と資金権限を同じ変更範囲に入れない。

### 決定

engineは推論のみ、executorは認証・workflow・委任・予算・ledger adapterを担当する。

### トレードオフと不採用案

callが増え、非同期状態管理が必要になる。GPU的な一括低latencyは約束しない。

### 検証条件

engineから送金できず、クライアントの持込みscoreをexecutorが信用しないことを試験する。[S09](#source-s09)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-006: 委任と内部capabilityを送金の必須条件にする

版: v2.0 / 状態: 設計採用 / 前版から: 維持

### 背景

正しい型の候補を選ぶことと、資金所有者の許可は別。

### 決定

executor所有の専用treasuryへ明示委任を設定。AuthorizedTransferはprivateな内部型で、外部DTO/Deserialize/Clone/Copyにしない。

### トレードオフと不採用案

Rustの型だけでone-time実行や意味的正しさを証明しない。nonce、operation、現stateを併用する。

### 検証条件

偽造型、権限外caller、金額/宛先差替え、失効でdispatchできないことを確認する。[S10](#source-s10)[S12](#source-s12)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-007: 複数signalを同一snapshotに束縛する

版: v2.0 / 状態: 設計採用 / 前版から: 強化

### 背景

逐次評価中にstateやbundleが変わり、都合のよい結果だけ混ぜると誤った認可になる。

### 決定

全slotのworkflow/model/schema/profile/calibration/snapshotを照合。必須slot完了後に現stateと委任/予算を再検査する。

### トレードオフと不採用案

違うslotの確率を独立として掛けない。古い結果を新attemptへ使わず、失効/stale時はreviewへ。

### 検証条件

partial set、stale callback、model変更、budget競合、早期approveを拒否する。[S09](#source-s09)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-008: 結果不明を保持する永続Tx状態機械を維持する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

### 背景

外部ledgerの処理はcallbackの失敗で巻き戻らない。retryで二重送金が起き得る。

### 決定

FrozenTransfer、Submitted、OutcomeUnknown、nonce/operation消費状態を永続化する。不明予約は保持し、同一argsかつ確認済みdedup条件でのみ回復する。

### トレードオフと不採用案

unknown後のTooOld/BadFeeで過去の未実行を断定しない。新timestampによるblind retry、一般的なexactly-once保証は不採用。

### 検証条件

late success、callback trap、日次reset、dedup expiry、revocation後retryを故障注入する。[S08](#source-s08)[S10](#source-s10)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-009: モデルは一つだけ常駐しロードpeakを管理する

版: v2.0 / 状態: 設計採用 / 前版から: 維持

### 背景

保存ファイルが入っても、raw bytesとtensorの二重保持でheapが不足し得る。

### 決定

stable memoryにchunk uploadしhash/shapeを検査。管理されたwarm-upとactive bundle切替を行う。

### トレードオフと不採用案

無停止切替の二重常駐は初期要件にしない。init/upgradeで無制限の一括ロードをしない。

### 検証条件

warm/cold heap、ロード命令数、破損chunk、upgrade回復を測る。[S07](#source-s07)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-010: F32基準から実測でINT8へ進む

版: v2.0 / 状態: 設計採用 / 前版から: 維持

### 背景

INT8はサイズを減らしても、kernel次第では速くならず出力も変わる。

### 決定

F32でcorrectnessを確立し、prefix/scratch再利用後、hot linearへINT8/SIMDを適用する。

### トレードオフと不採用案

いきなりINT4化、全weightsの毎回F32展開、未検証fast mathによる速度主張は採用しない。

### 検証条件

三primitiveのparity/校正とScoreの上側誤りを再評価。速度とheapが改善した方式だけ採用する。[S05](#source-s05)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-011: 入口制限とevaluationの重複排除を必須にする

版: v2.0 / 状態: 設計採用 / 前版から: 強化

### 背景

少数質問でも巨大入力・無限retry・大量未完了ジョブで資源を消費する。

### 決定

認証、16KiB byte cap、token cap、slot cap、quota、expiry、live evaluation IDの重複排除を設ける。

### トレードオフと不採用案

無制限query推論、無言切詰め、expired IDを新規計算する処理は不採用。

### 検証条件

最大入力、予約特殊token、同じID別payload、transport再送、pending枯渇を試験する。

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-012: schema・model・calibrationの版を一体管理する

版: v2.0 / 状態: 設計採用 / 前版から: 強化

### 背景

同じRisk型でもrubricが変われば数値の意味は変わる。

### 決定

weights/config/tokenizer/runtime/renderer/profile/schema/calibration/ruleをhashで識別する。原文とeffective inputのhashも記録する。

### トレードオフと不採用案

main追従、無言モデル変更、ログへの無制限原文、会計snapshotの巻戻しは禁止。

### 検証条件

旧新schema混在、in-flight更新、nonce維持、原文保持方針、controllerの更新手続を確認する。

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-013: 全primitiveと合成policyを別Gateで受け入れる

版: v2.0 / 状態: 設計採用 / 前版から: 強化

### 背景

Choiceだけ良くてもScoreの重大な過小評価で危険な処理を許可し得る。

### 決定

Choice/Noul/Score、最終workflow、実装parity、資源、安全状態機械、対象ledgerを独立Gateにする。

### トレードオフと不採用案

平均accuracyや温度調整の成功だけでLiveへ移行しない。最初はReportOnly/上限0。

### 検証条件

ScoreのRPS/CDF誤差/重大過小評価、workflow誤受理率とcoverage、独立reviewを含める。

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-014: Score/Choice後回しの決定を撤回する

版: v2.0 / 状態: 設計採用 / 前版から: 置換

### 背景

前版ADR-014の延期は、今回のLaya採用とScore必須要件に合わない。

### 決定

Choice・Noul・Scoreを同じ優先度でv2 coreとする。大量候補、任意tool生成、無制限dynamic schemaだけを対象外とする。

### トレードオフと不採用案

Scoreが重い/精度不足ならschema/profile/modelを見直す。Score削除によって三primitive対応完了とはしない。

### 検証条件

三primitiveが一つのengineと同じSDK型体系で使えることを受入条件にする。[S01](#source-s01)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-015: Scoreはrubricと分布を正本にする

版: v2.0 / 状態: 設計採用 / 前版から: 新規

### 背景

元LayaのScoreは0〜K−1の期待段階。正規化値や平均だけを返すと意味・尾の情報を失う。

### 決定

期待段階、正規化平均、mass/CDF、上側確率を提供。尺度の方向・段階説明・versionをschemaに固定し、Score<Risk>とScore<Urgency>を区別する。

### トレードオフと不採用案

unknownを中間binにしない。ordinalの等間隔表示を実世界の確率や損失へ読み替えない。

### 検証条件

同平均/異なるtail、3/5/7 bins、最大値、丸め、schema変更を数学fixtureで試験する。[S04](#source-s04)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

### ADR-016: Compact128・1評価1問・最大3slotを固定する

版: v2.0 / 状態: 設計採用 / 前版から: 新規

### 背景

少数questionで三primitiveを合成したいが、大batchと長文でICP予算を超えさせたくない。

### 決定

1 engine update=1問。1workflow=1〜3問の逐次ジョブ。static prefix≤64、総長≤128。Score標準5/max7、Choice2〜5、Noul2。

### トレードオフと不採用案

短いheadへ無言でrubricを切らない。入りきらない場合のExtended256は別Gate。3評価の総コストは1評価分にならない。

### 検証条件

全primitiveの最大候補数でprefix余地と総instructions/heapを測り、early denyとrequired-slot completenessを試験する。[S03](#source-s03)[S04](#source-s04)[S07](#source-s07)

出典: [SOURCES.md](#part-sources)。正本: [最終仕様](#part-spec)。


---

<a id="part-changes"></a>

## 6. v1からの変更

2026-09-19（JST）。これは会話中に示された設計方針に対する変更記録であり、前版ファイルの機械的なdiffではない。

| 前版/直前の議論 | v2で固定 |
|---|---|
| NLIを基礎にしたengine | Laya式option-marker backend |
| DeBERTa-small/ModernBERT-NLI比較 | Laya typed-decisionsを第一参照に固定。資源/品質未達なら同じprimitiveを維持して適応/蒸留 |
| Choice/Scoreを後続へ | 三primitiveをcoreへ。ADR-014を置換 |
| NLI neutralをuncertaintyへ寄せる説明 | NLIラベルで代用しない。外側のabstentionと分布を分離 |
| 1質問/callだけの説明 | 1評価1質問、1workflow最大3質問の逐次実行を明確化 |
| Scoreを平均にまとめる | rubric ID、期待段階、正規化平均、分布、上側確率を維持 |
| 128をlocal attention由来の自然境界とする説明 | 製品の資源profileとする。attention maskの同値化はしない |
| act headを無条件で削る | 参照parityでは保持。実行権限に使わず、logit不変確認後に配備計算を省略 |
| 短いmax_lenだけ設定 | prefix≤64をコンパイル時検査。無言切詰め禁止 |
| temperatureは付属設定で十分という印象 | 上書き優先順位を確認し、task/profileごとに有効値を一意に固定 |
| 型付きdecisionをTxへ接続 | private認可型・委任・nonce・業務重複・不明予約を維持/強化 |

新設ADR-015はScoreの数値契約、ADR-016はCompact128/少数逐次workflow。既存の認可・ledger回復・監査・upgradeの境界は撤回しない。


---

<a id="part-review"></a>

## 7. 自己レビューと修正

2026-09-19（JST）。対象は会話中の設計説明と今回の最終仕様・ADR。公開一次資料との突合、型/数値契約、状態遷移、受入条件を自己レビューした。第三者監査ではない。

### 修正・明文化した事項

| ID | 指摘 | v2への反映 |
|---|---|---|
| V2-R01 | Scoreを後回し/削除すると要件を満たさない | core primitiveへ昇格。ADR-014置換、ADR-015新設。 |
| V2-R02 | root Layaとtyped-decisions specialistを同じ汎用性で扱う危険 | 独立repoを第一参照に固定し、対象外業務の品質を別Gateにする。 |
| V2-R03 | Scoreの元値は0〜1ではなく0〜K−1 | 期待段階と正規化平均の両方を単位付きで定義。 |
| V2-R04 | 平均が同じなら同じriskと扱ってしまう | 分布と上側確率を正本にする。中央集中/両端分裂の例を試験。 |
| V2-R05 | ordinalの等間隔化を実世界の事故率と誤解する | 表示上の規約と明記。rubric、方向、utilityを分離。 |
| V2-R06 | 1質問/callと3signalのTx合成が矛盾し得る | engineは1問、workflow最大3問の逐次ジョブ。必須slotが揃うまで許可しない。 |
| V2-R07 | 逐次質問で別snapshot/modelの結果を混ぜられる | immutable bindingと全slot一致、stale/late callback排除を規定。 |
| V2-R08 | 3質問へ分割すると総計算も1質問分になるという印象 | 最大60Bは評価分だけの目標と明記し、cycles/latencyを別集計。 |
| V2-R09 | max_lenだけ128にするとheadがstateを圧迫しうる | prefix≤64をコンパイル時検査。rubric/option/stateの無言切詰めを禁止。 |
| V2-R10 | local attention 128なら128token入力の全attentionが同じになるという解釈 | 実際のlocal maskを保持。128は製品profileとして選択。 |
| V2-R11 | headのactivationをすべてGELUとして移植する誤り | PyTorch default ReLU、scorer GELU、backbone GeGLUを分離して照合。 |
| V2-R12 | 付属temperatureをそのまま使えば校正済みという過信 | priorityを解決したeffective設定とtask/profile別再校正を必須化。 |
| V2-R13 | entropy集中度を正答確率として扱う | diagnosticsとして返し、calibrationとabstention ruleを別途評価。 |
| V2-R14 | act headを削ると大幅高速化できるという過大評価 | logits不変の確認後に省略。主要最適化とは見積もらない。 |
| V2-R15 | 型付き出力なら直接Txへ渡して安全という混同 | 外部DTOと内部AuthorizedTransferを分離。runtime権限/予算検査を維持。 |
| V2-R16 | transport再送で同じ評価を何度も計算 | evaluation IDによるlive重複排除、expiry、quotaを規定。 |
| V2-R17 | unknown送金を後続TooOld等だけで失敗扱い | 不明予約とoperation予約を保持。ledger固有の確実な照合を要求。 |
| V2-R18 | INT8化すれば必ず速いという仮定 | kernel込みで実測。F32全展開だけの「最適化」を採用しない。 |
| V2-R19 | 型名が同じなら違うrubricでも使える | schema/version/hashとcandidate順をSDK/runtimeで検査。 |
| V2-R20 | 品質未達時にScoreを無効化して受入完了にする | 三primitive別Gateと最終policy Gateをすべて必須化。 |

### 第2パス

本文、ADR、profileの上限（1/3質問、128tokens、2〜5Choice、3〜7Score、標準5段階）、Scoreの単位、同じsnapshot、ReportOnly、unknown予約保持を読み合わせる。文書構造と数値fixtureを`validate.py`で確認する。

validate.pyは実モデル、Rustの型実装、認可コード、ledger接続を試験するものではない。文書の整合と設計上の数値例を検査する補助である。

### 残る受入作業

固定したrevision/hash、実tokenizerのschema token数、raw logitsの一致、ICPのinstructions/heap、INT8の実利、Scoreの過小評価と校正、最終workflowの誤受理率、特定ledgerの回復、独立レビューは未実施。

設計方針は固定したが、これらの未実測値を推測で合格にしない。未合格のまま実資金Txを有効化しない。


---

<a id="part-sources"></a>

## 8. 一次資料と確認範囲

確認日: 2026-09-19（JST）。以下のURLは確認した公開資料。main/stableは変わり得るため、実装時はcheckpoint、ソース、toolchainのrevisionとhashを固定する。今回はweightsの取得・実モデル推論・Rust/Wasm build・ICP deployは実施していない。

| ID | 資料 | 確認範囲 |
|---|---|---|
| <a id="source-s01"></a>S01 | [Laya model card](https://huggingface.co/convaiinnovations/laya) | 三primitive、option-marker architecture、モデル系列。ベンチ速度をICP性能へ転用しない。 |
| <a id="source-s02"></a>S02 | [Laya typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions) | 特化対象の限界、temperature_by_optionsの優先による校正注意点。本番taskの精度保証ではない。 |
| <a id="source-s03"></a>S03 | [Laya rl_common.py](https://huggingface.co/convaiinnovations/laya/blob/main/rl_common.py) | rendering、marker、head、act branch、entropy由来診断。元rendererの無言切詰めをそのまま配備しない。 |
| <a id="source-s04"></a>S04 | [Laya rl_agent_api.py](https://huggingface.co/convaiinnovations/laya/blob/main/rl_agent_api.py) | 質問ごとの入力作成、temperature解決、Scoreの期待段階、Noulのp[1]。 |
| <a id="source-s05"></a>S05 | [Candle modernbert.rs](https://raw.githubusercontent.com/huggingface/candle/main/candle-transformers/src/models/modernbert.rs) | backbone、GeGLU、local mask、出力経路。Laya checkpointのICP互換を証明するものではない。 |
| <a id="source-s06"></a>S06 | [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) | activation等の既定値。採用する参照環境のversionと一致させる。 |
| <a id="source-s07"></a>S07 | [ICP Resource limits](https://docs.internetcomputer.org/references/resource-limits/) | update 40B instructions、wasm32 heap 4GiB等。20B/3GiBは本設計の目標であり公式上限ではない。 |
| <a id="source-s08"></a>S08 | [ICP Safe retries and idempotency](https://docs.internetcomputer.org/guides/canister-calls/idempotency/) | nonce、dedup、結果不明、時間窓後の回復と照合。 |
| <a id="source-s09"></a>S09 | [ICP security — Inter-canister calls](https://docs.internetcomputer.org/guides/security/inter-canister-calls/) | awaitをまたぐstateと権限の再検査、障害処理。 |
| <a id="source-s10"></a>S10 | [ICRC-1 specification](https://github.com/dfinity/ICRC-1/blob/main/standards/ICRC-1/README.md) | caller所有accountのdebit、transfer args、dedupのSHOULD要件。ledger固有検証は別途必要。 |
| <a id="source-s11"></a>S11 | [Candid type reference](https://docs.internetcomputer.org/references/candid-spec/) | wireでのrecord/variant/principal等とRust SDKの型復元の境界。 |
| <a id="source-s12"></a>S12 | [Rust visibility and privacy](https://doc.rust-lang.org/reference/visibility-and-privacy.html) | module/private fieldの保護。意味的正しさや分散した一回性を保証するものではない。 |

この文書のcrate構成、Compact128、質問数、上限、型名、API名、状態機械、計測目標は設計上の選択であり、上記資料が公式に推奨した値ではない。


---
