# ADR索引 — v2.0

| ADR | 決定 | 前版からの変更 |
|---|---|---|
| [ADR-001](ADR-001.md) | 三つのtyped decision primitiveを製品の核にする | 改訂 |
| [ADR-002](ADR-002.md) | Laya typed-decisionsを第一参照checkpointに固定する | 改訂 |
| [ADR-003](ADR-003.md) | Candle backboneとLaya固有headを分離する | 改訂 |
| [ADR-004](ADR-004.md) | 分布・診断値・校正・abstentionを区別する | 改訂 |
| [ADR-005](ADR-005.md) | engineとexecutorの二canister境界を維持する | 維持 |
| [ADR-006](ADR-006.md) | 委任と内部capabilityを送金の必須条件にする | 維持 |
| [ADR-007](ADR-007.md) | 複数signalを同一snapshotに束縛する | 強化 |
| [ADR-008](ADR-008.md) | 結果不明を保持する永続Tx状態機械を維持する | 維持 |
| [ADR-009](ADR-009.md) | モデルは一つだけ常駐しロードpeakを管理する | 維持 |
| [ADR-010](ADR-010.md) | F32基準から実測でINT8へ進む | 維持 |
| [ADR-011](ADR-011.md) | 入口制限とevaluationの重複排除を必須にする | 強化 |
| [ADR-012](ADR-012.md) | schema・model・calibrationの版を一体管理する | 強化 |
| [ADR-013](ADR-013.md) | 全primitiveと合成policyを別Gateで受け入れる | 強化 |
| [ADR-014](ADR-014.md) | Score/Choice後回しの決定を撤回する | 置換 |
| [ADR-015](ADR-015.md) | Scoreはrubricと分布を正本にする | 新規 |
| [ADR-016](ADR-016.md) | Compact128・1評価1問・最大3slotを固定する | 新規 |
| [ADR-017](ADR-017.md) | INT8/SIMD最適化の実行計画を実測で確定する | 新規（ADR-010の追補） |
