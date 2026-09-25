# Are Wasm local operations real matrix-multiplication overhead?

## Finding

Local instructions are a normal part of Wasm's stack-based representation. Their frequency is not, by itself, evidence of a bad matrix kernel. Charging each one does not measure an equally frequent CPU load/store, however. We confirmed this in source, generated machine code, and a PocketIC metering experiment. The original kernel also has substantial real register spills; these must not be dismissed as free bookkeeping.

## Reproduction and scope

- [Successful x86 audit](https://github.com/humandebri/IC-Verdict/actions/runs/36128545768), code commit `c13d352`.
- Original measured Wasm SHA-256: `3bbb0359cd243985ccff6879ba003563cc7f7646f52b9d035deaac971c03799f`.
- Official Wasmtime C library **48.0.1**, matching the referenced IC dependency. Python bindings are 48.0.0; the workflow replaces their native library with the official 48.0.1 release. The loaded library SHA is recorded in the report.
- Cranelift optimization level `none`, NaN canonicalization enabled, host ISA defaults. AMD EPYC 9V74 x86 Linux. A previous run with native library 48.0.0 produced the same results.
- Disassembly is of the original computation compiled in standalone Wasmtime, **before IC instrumentation**. It is not a dump of PocketIC's instrumented machine code. The separate PocketIC probe checks actual IC instruction accounting. No runtime speedup is inferred from assembly instruction counts.
- Full assembly is in the CI artifact `wasm-locals-machine-code`; [report and excerpts](w8a16-metering/locals-36128545768/) are committed here.

## 1. Why many locals are normal

An ordinary vector multiply-accumulate can be represented as:

```wat
local.get $acc
local.get $a
local.get $w
i32x4.dot_i16x8_s
i32x4.add
local.set $acc
```

Four of these six Wasm instructions are local operations. They pass operands and name a result; this fraction is not a CPU-time fraction. The production kernel's roughly 52% local-operation charge is therefore not surprisingly large solely as a consequence of representing arithmetic in Wasm. Its unrolling and operand reuse produce a different ratio from this simple example.

## 2. Source contradicts a literal local-equals-stack-access interpretation

[IC's cost comment](https://github.com/dfinity/ic/blob/d26cd031176beec51b39fbb9e39e80a3a46a748e/rs/embedders/src/wasm_utils/instrumentation.rs#L396) motivates the local price of 1 using a nearby stack-memory load/store model.

[Cranelift 48.0.1 translation](https://github.com/bytecodealliance/wasmtime/blob/v48.0.1/crates/cranelift/src/translate/code_translator.rs#L145) handles `local.get` with `use_var` and compiler-stack bookkeeping; `local.set/tee` define the corresponding compiler variable. These operators disappear as standalone operations in the intermediate representation. Vector type casts and later register assignment can still produce moves or spills. This translation happens even at optimization level `none`; it does not require aggressive optimization.

Thus the IC comment is not a generally accurate description of current generated code. The price can still function as a conservative accounting proxy, but the comment does not establish that each local instruction incurs a cached memory access.

## 3. Controlled test: metering changes, arithmetic machine code does not

A function adding two integer parameters was compiled with and without 20 redundant `local.set/local.get` pairs. Both generated exactly the same 12-byte x86 function, SHA-256 `c505a4c01758d981f6dcda86544c95cb5a2beb76eaec0f7a8772dacb61a819e2`:

```asm
push rbp
mov rbp, rsp
lea eax, [rdx + rcx]
mov rsp, rbp
pop rbp
ret
```

A separate PocketIC update-call experiment returned 579 in both variants, while `performance_counter(0)` increased from **40,234 to 40,274**: exactly the added 40 local operations. Those totals include fixed IC overhead; the controlled difference is the relevant quantity.

Reproduction: `POCKET_IC_BIN=... python tools/probe_local_metering.py output.json` (tested with PocketIC network-launcher v16.0.0-2026-09-18-03-28 and pocket-ic Python 3.1.2).

## 4. The same result in the actual production matrix functions

After each vector `local.get $x` in the two production tile functions, we inserted `local.tee $x`. This preserves the operand stack and simply assigns the same value back to the same local. Nothing else in the computation changes.

| Function | Added static local operations | Original x86 bytes | After adding locals | Machine code |
|---|---:|---:|---:|---|
| 16-row tile | 5,510 | 70,936 | 70,936 | Byte-for-byte identical |
| 8-row tile | 2,779 | 44,064 | 44,064 | Byte-for-byte identical |

The prior dynamic execution trace contains **54,487,296** vector local.get executions in these functions for 120x2304x768. Applying the existing unit price to the added tees predicts that many extra counted instructions, with unchanged uninstrumented compute code. This full-matrix delta is a cost-table prediction, not an additional PocketIC measurement; the small probe above directly tests metering.

This artificial-insertion experiment proves representation dependence. It does **not** prove that all original local operations can be removed from valid Wasm, nor that the original 52% charge can safely be refunded.

## 5. Real spill traffic exists in the original kernel

The Rust kernel declares a 16x16 array of vector accumulators and heavily unrolls the inner work. It needs many live intermediate values. The disassembly confirms real native stack access rather than merely assuming it:

| Innermost loop, one iteration | 16-row tile | 8-row tile |
|---|---:|---:|
| SIMD dot instructions | 2,048 | 1,024 |
| SIMD adds | 2,048 | 1,024 |
| Native stack reads | 2,174 | 1,039 |
| Native stack writes | 408 | 205 |
| All native instructions in loop | 7,240 | 3,876 |

Stack accesses here are `movdqu` instructions with RSP/RBP-relative operands. In the 16-row loop, Wasm has 4,933 local operations, while native stack reads/writes total 2,582. These are different quantities, not a one-to-one mapping. Neither count is a CPU-time percentage. All these figures describe this uninstrumented compilation; IC instrumentation can alter register allocation.

## Implication

The earlier phrase "overhead of variable operations needed to express matrix multiplication in Wasm" conflated representation/accounting overhead with physical execution overhead. A precise statement is: **about 52% of the counted instructions are local operators; the compiler resolves them into value dependencies, while register pressure separately produces real spill traffic.**

The next implementation questions are distinct: reassess local-operation pricing or recognized arithmetic-pattern pricing using real execution costs, and reduce the kernel's excessive live-value pressure through tile/unroll choices. A smaller tile may reduce physical spills while increasing Wasm loads and counted instructions. The previous native-versus-Wasm throughput experiment changed both compiler path and tiling, so it cannot isolate the cost of locals.

Global free locals or an unconditional 52% discount are not established by these tests. The source rationale for treating every local as a stack access does warrant revisiting.
