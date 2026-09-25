# Native W8A16 prototype versus IC Wasm

Run: https://github.com/humandebri/IC-Verdict/actions/runs/36126178526

Code: ca1ac54088aa0ec633c327e7215b8e8bd2430261. Both jobs passed, including scalar sample checks for every weight bank and the full-output bank-zero checksum matching PocketIC.

Shape: 120 x 2304 x 768. Signed i16 activations and i8 weights. Same synthetic input and scaling; private buffers per worker. Native uses SSE128, 4x2 tiles, GCC -O3, no AVX and no floating-point contraction. Production Wasm uses its existing tile strategy. This compares two complete implementation paths, not the isolated effect of metering.

| Host | Weights | Workers | Wasm GMAC/s/worker (median) | Native GMAC/s/worker (median) | Ratio |
|---|---|---:|---:|---:|---:|
| AMD EPYC 7763 64-Core Processor | gemm_hot | 1 | 14.707 | 29.464 | 2.003 |
| AMD EPYC 7763 64-Core Processor | gemm_hot | 4 | 6.502 | 14.729 | 2.265 |
| AMD EPYC 7763 64-Core Processor | gemm_rotating_32 | 1 | 13.034 | 29.475 | 2.261 |
| AMD EPYC 7763 64-Core Processor | gemm_rotating_32 | 4 | 6.110 | 14.865 | 2.433 |
| Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | gemm_hot | 1 | 16.243 | 28.091 | 1.729 |
| Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | gemm_hot | 4 | 8.372 | 10.726 | 1.281 |
| Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | gemm_rotating_32 | 1 | 14.869 | 27.364 | 1.840 |
| Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz | gemm_rotating_32 | 4 | 7.896 | 10.686 | 1.353 |

## What can be reduced?

The smallest native-minimum / Wasm-maximum ratio across the four-worker cases is 1.275789. With an ideal fully integrated primitive retaining that ratio, the same measured wall-time budget could support 1.276 times the dense work; the corresponding dense-operation count reduction is 21.62%. This is an experimental ceiling, not an approved cost schedule.

Sensitivity to integration overhead / performance margin relative to native time:

| Assumed additional time | Effective work ratio | Dense instruction reduction at unchanged observed time budget |
|---|---:|---:|
| 0% | 1.276 | 21.62% |
| 10% | 1.160 | 13.78% |
| 25% | 1.021 | 2.02% |

These margins are scenarios, not measured integration costs. A roughly 10% dense-count reduction is a useful first prototype target, conditional on integrated node tests reproducing sufficient speedup. No universal 2x or 11x increase is established.

## Limits and required integration

- Native timing covers worker execution and joining, with a checksum once per group. Wasm timing includes PocketIC update handling and a checksum per eight kernels. Native has no IC boundary validation, sandbox transition, dirty-page tracking, or deterministic time-slicing integration. The standalone native ratio therefore overstates the speedup of a finished IC primitive by an unknown amount.
- Measurements are three groups on two GitHub Actions VMs (four logical CPUs, two physical cores exposed), not production replica hardware or consensus. Native runs before Wasm, not randomized interleaving. Bank rotation is about 56 MB per worker; this does not represent every large-model memory pattern.
- Hardware perf counters are unavailable on both hosts, despite perf returning success. Wall throughput is the primary metric; no PMU cycles/cache claims are made.
- A production primitive needs shape/bounds validation, deterministic integer overflow and floating-point behavior, bounded work units with interruption/accounting, single-thread execution, memory accounting and a conservative static charge calibrated on supported nodes under competing workloads. No global local.get exemption is justified.
- The 0.595291 instructions/MAC audit is for the original W8A16 bench kernel. It is not the current W8A8 Laya inference coefficient. Dense speedup cannot be directly converted into model/token multipliers; non-dense work and memory remain.

For dense instruction fraction p and dense charge reduction d, whole-inference count ratio is 1-p*d. At p=0.9 and d=0.1 the whole inference reduction is 9%, giving at most 1.099x work under a fixed instruction budget if work scales linearly.
