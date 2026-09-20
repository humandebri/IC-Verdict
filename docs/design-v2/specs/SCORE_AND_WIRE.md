# Score・wire・型の契約

版: v2.0 / 2026-09-19（JST）

## 数値の正本

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

## 校正と診断

`CalibrationStatus`は`Uncalibrated`または、profile/schema/runtimeを含む署名対象ではないbundle識別子。単一の`confidence`だけを合否条件にしない。

entropy由来の`concentration`、top1、marginは分布の診断値である。既知のtask分布で選択したdecision ruleと組み合わせる。低entropyでも間違う可能性を残し、どのdomainでも正答確率と解釈しない。

Ordinal labelだけでScoreの真値を決めにくい場合は、ラベル規約を先に定義する。`unknown`をhigh/medium等と同じ軸のbinにしない。情報不足は`Evaluation::Abstained`かschema固有の別質問で扱う。

## 外部DTO

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

clientが持ち込む`workflow_binding`は実行権限ではない。executorが使用するresultはexecutor自身が登録engineへ発行したcallの返答に限定する。SDKのgeneric型はDTOのschema/version/variantを検証してから構築する。[S11][S12]

## 配備API

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

## TypedSDK

`Choice<T>`、`Noul<P>`、`Score<S>`は同じものを異なる名前で包むだけではなく、登録schemaと対応するdescriptorから生成する。型tagはRust側の誤用防止、version/hashは更新時の誤用防止を担当する。

検査済みのdecisionでも実行権限ではない。executor内部で、必要signalの完全性と同じsnapshotを検査した`BoundSignals`からのみ`AuthorizedTransfer`を作る。外部はどちらの内部型もdeserializeできない。

業務の閾値はこの文書に固定しない。risk budgetを未承認のまま`mean_ppm < 500_000`等の便宜的条件で送金を許可しない。
