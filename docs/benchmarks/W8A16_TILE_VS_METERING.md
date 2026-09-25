# W8A16 tile choice versus IC instruction accounting

Run: https://github.com/humandebri/IC-Verdict/actions/runs/36130248756

The only source change was the production `matmul_i8` dispatch for K=768, selecting R x C tiles 16x16, 8x16, 4x8, or 4x2. The synthetic input, weights, W8A16 arithmetic, output shape (120x2304x768), and PocketIC canister wrapper were identical. `prepare` compared every output bit with the scalar reference for each variant. Run outputs shared the exact full-output checksum `641785196358105`.

`kernel_metered_per_iteration` is the slope between 1 and 4 kernel iterations within the same update. The first nonzero execution has 5,040,212 fixed metered instructions, absent at zero iterations; this fixed amount is excluded from the slope. The slope was also checked against 8 iterations. The reported wall time is the median of three warmed PocketIC update calls, each running eight kernels. It includes update overhead and is not a direct CPU-time measurement.

| Tile | IC instructions per kernel | Relative to 16x16 | EPYC 7763 wall (8 kernels) | EPYC 9V74 wall (8 kernels) |
|---|---:|---:|---:|---:|
| 16x16 | 126,401,828 | -0.00% | 117.36 ms | 119.95 ms |
| 8x16 | 137,710,079 | +8.95% | 106.39 ms | 91.07 ms |
| 4x8 | 158,659,867 | +25.52% | 84.67 ms | 89.61 ms |
| 4x2 | 205,186,393 | +62.33% | 97.93 ms | 101.42 ms |

The 4x8 variant was 25.52% more expensive in counted instructions, while its warmed update wall time was 27.86% lower on EPYC 7763 and 25.29% lower on EPYC 9V74. The current metering therefore ranks these two semantically equivalent kernels in the opposite order from observed wall time on both hosts. The 8x16 variant showed the same direction; 4x2 illustrates that simply shrinking the tile is not always faster.

Interpretation: the W8A16 Wasm implementation has a genuine performance opportunity in smaller tiling; the current runtime instruction tariff creates a penalty for using it under an instruction limit. Redundant `local` charging shown in the separate [local audit](WASM_LOCALS_VERIFICATION.md) explains why Wasm instruction totals are not a direct physical-work measure, but this experiment does not isolate a single opcode as the cause of the 4x8 difference.

Network limits: these are two GitHub Actions virtual CPUs with PocketIC, not production replicas or four-way concurrent execution. A lower-count schedule for 4x8 must be calibrated against CPU time, memory bandwidth, consensus scheduling, and the slowest supported node under contention. The earlier [sustained capacity comparison](W8A16_NATIVE_CAPACITY.md) tested four independent PocketIC workers, but not this 4x8 variant. No tariff change is justified solely by these VM wall times.

Applicability: this W8A16 kernel is not the Laya W8A8 inference kernel. Laya already achieved an 8.327% whole-inference instruction reduction by changing its own tile and removing a redundant attention copy; its 128-token 39.27B count requires separate profiling.
