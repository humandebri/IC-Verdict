# Laya W8A8: executed Wasm instructions after Binaryen optimization

This experiment addresses **the number of Wasm instructions charged during execution**, not a change to the IC instruction tariff. The input was the existing `/Volumes/KINGSTON/ICP/IC-Laya-Standalone/build/decision-engine.wasm`, SHA-256 `0298b36221d5676f7721640b56f15ad7bd7cb81b765fba1f9c88333494007030`, 5,893,156 bytes. This is a newer local build than the Wasm hashes in `INT8_OPTIMIZATION_V4.md`. No Laya source, model checkpoint, or canister was changed.

[Binaryen v131](https://github.com/WebAssembly/binaryen/releases/tag/version_131) `wasm-opt` was applied at `-O3`, `-Os`, and `-O4`. The official Mac arm64 release archive SHA-256 was `e441b48dc22163d209b4f05e44dc7210909b01237642b6c9ae48fd710a3ef83e`. Each Wasm was gzip-compressed for installation into its own isolated PocketIC canister. The canister's owner-only `benchmark_int8_kernel` method constructed the same W8A8 input and weights for every variant. Three calls per shape were made; the table reports the median. The output checksum matched across all variants and repeats at each shape.

| Tokens × rows × columns | Original instructions | `-O3` | `-Os` | `-O4` |
|---|---:|---:|---:|---:|
| 28 × 3072 × 1024 | 64,893,163 | 64,859,817 (−0.051%) | 64,859,394 (−0.052%) | 65,128,202 (+0.362%) |
| 128 × 3072 × 1024 | 274,433,498 | 273,522,136 (−0.332%) | 273,522,380 (−0.332%) | 274,759,142 (+0.119%) |
| 128 × 5248 × 1024 | 464,863,354 | 463,376,169 (−0.320%) | 463,376,685 (−0.320%) | 465,465,895 (+0.130%) |
| 128 × 1024 × 2624 | 230,635,651 | 230,099,841 (−0.232%) | 230,099,829 (−0.232%) | 230,585,487 (−0.022%) |

`-O3` reduced module size to 4,524,782 bytes (−23.22%), but its executed kernel instruction reduction remained only 0.05–0.33% on the measured shapes. Module byte size is not an estimate of dynamic instructions. `-O4` increased instruction use on three of four tested shapes. These results do not measure full model inference, compilation overhead, or production replica resource use.

The related Laya source-side optimization was already measured separately on the real model: 42,841,760,152 to 39,274,507,249 instructions at 128 tokens (8.327% reduction), with identical logits on 96 inputs. That improvement came from kernel loop unrolling and removing a redundant attention copy, not from changing the tariff. The present experiment shows that a generic post-build optimizer adds little to that result. A large further reduction would need fewer arithmetic operations or a materially better kernel representation, validated on the real inference path.

Reproduce with `tools/bench_laya_wasm_optimizer.py --baseline PATH_TO_LAYA_WASM --wasm-opt PATH_TO_WASM_OPT --out REPORT_JSON`, with `POCKET_IC_BIN` set. The [raw results](laya-wasm-opt-20260925.json) include module hashes, sizes, each instruction count, each output checksum, and wall times. Python packages: `pocket-ic==3.1.2`, `ic-py`, Binaryen v131. The script uses a fresh PocketIC instance and does not mutate an installed network canister.
