# 実装計画 v2.0

2026-09-19（JST） / 正本は`FINAL_SPEC.md`。本書のタスクは未実装。

## 1. 実装順

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

## 2. モデル選択の境界

実装の第一参照checkpointは`convaiinnovations/laya-typed-decisions`に固定する。ただしこれは特定workflowのspecialistである。[S02] 本番taskに適合しなければ、同じarchitectureへtask別fine-tuningする。root Layaへの変更も評価・manifest変更を経て明示する。

421M版が資源目標を満たせない場合の次の候補は、ModernBERT-base級backboneとLaya式option-marker headを持つstudent。サイズ・速度・精度は現時点で保証しない。蒸留ではChoice、NoulだけでなくScoreの分布・順序・上側eventもtargetにする。NLIモデルへの無言の置換は禁止。

## 3. 推奨crate境界

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

## 4. Gate

### G1 — 意味と品質

Choiceはaccuracy/macro-F1、NoulはBrier/NLL・false positive/negative、ScoreはMAEに加えてCDFの誤差・Ranked Probability Score・重大な過小評価・上側eventの校正を記録する。さらに最終workflowの危険な誤受理率とcoverageを測る。

例数より、業務レコード・template family単位でのtrain/calibration/test分割を優先する。modelcardのbenchmark値を自分のtaskの合格点として代入しない。small-sampleで過信しない。risk budgetと必要coverageが未承認なら実送金off。

Scoreは必須Gate。Choiceだけ成功してScoreは無効という状態を、三primitive対応の受入完了とは扱わない。5段階のうち中央へ集中する分布と両端へ分かれる分布をpolicyが区別する試験を含む。

### G2 — 実装parity

token IDsとmarker位置の完全一致、schema変更検出、誤ったqtype・候補順拒否、softmaxの二重適用防止、temperatureの実効値、Scoreの単位を検証する。

logit比較の初期許容差はF32で`atol=1e-3, rtol=1e-3`を仮置きし、誤差源を記録して妥当性を判断する。閾値を跨ぐ危険な差を単なる許容誤差で済ませない。NaN/inf・不正分布・未知tensor・shape不一致は拒否。

元Laya APIのparityと、Compact128＋再校正後の製品parityを別fixtureで管理する。前者の切詰めまで本番APIにコピーしない。

### G3 — ICP資源

1評価20B instructions以下、warm heap 2.5GiB以下、cold peak 3.0GiB以下を目標とする。これはICP上限ではなく余裕を持たせる設計値。[S07] tokenization、stable memoryコピー、model forward、後処理、response作成を含む。

質問数とScoreのbinsを分けて測る。5-bin Scoreは1質問だが、3-binよりstatic tokensが増える。stateの有効長を減らして速い値だけ出さない。全workflowのcycles、遅延、拒否率、cold restartも報告する。native秒数からICP秒数への換算はしない。

### G4 — 実行安全性

必須試験: 他callerのrequest ID、nonce再利用、別nonceの同一operation、partial decision set、異なるsnapshot、model/schema変更、推論中revocation、2重process、quota枯渇、送信後callback trap、unknown後TooOld、遅延success、日次reset、upgrade、過去会計snapshot復元。

hard invariantはモデルが常に最大スコアを返すstubでも成立させる。型のcompile-fail試験と、Candid/runtimeの認証試験の両方が必要。

### G5 — 本番有効化

特定ledgerのdedup/履歴照合仕様、上限、権限・controller管理、保持する原文、独立reviewを確認する。`ReportOnly → Mock → Shadow → LimitedLive`を順に進める。Liveを有効化しても、modelの誤推論リスクがゼロになったとは説明しない。

## 5. 最初の小さな完成形

1つの英語workflowを用意し、`Noul<RefundRequested>`、`Choice<PaymentAction>`、`Score<PaymentRisk>`を同じstateに対して返す。元から送金権限のあるtreasury operationのみを対象にする。

最初は値の表示・通常関数への受渡しまで。その後mock ledgerへ接続する。入力テキストだけを見て、recipient・amount・支払済み状態を捏造して送金するデモにはしない。
