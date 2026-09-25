#!/usr/bin/env python3
"""Static cost attribution for the exact measured 120x2304x768 Wasm.

Input is wabt's flat wasm2wat output with debug names. Only branch-free inner
loops are attributed; surrounding code is an explicit remainder, not guessed.
Costs follow dfinity/ic d26cd031176beec51b39fbb9e39e80a3a46a748e,
rs/embedders/src/wasm_utils/instrumentation.rs::instruction_to_cost (Wasm32).
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

COST = {op: 1 for op in (
    "local.get", "local.set", "local.tee", "i32.const", "i32.shl", "i32.add", "i32.ne",
    "v128.load", "v128.load8x8_s", "i32x4.dot_i16x8_s", "i32x4.add",
)} | {"br_if": 2, "end": 0}
FUNCTIONS = ("matmul_i8_simd_tileKj300_Kj10_KBW_", "matmul_i8_simd_tileKj300_Kj8_Kj10_")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wat", type=Path, required=True)
    p.add_argument("--wasm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    active, stack, loops, name = False, [], [], ""
    with args.wat.open() as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if line.startswith("  (func "):
                active = any(n in s for n in FUNCTIONS)
                name, stack = s.split()[1], []
            if not active or not s or s.startswith("("):
                continue
            op = s.split()[0].rstrip(")")
            for ctx in stack:
                if ctx["kind"] == "loop":
                    ctx["counts"][op] += 1
            if op in ("loop", "block", "if"):
                stack.append({"kind": op, "start_line": lineno, "function": name, "counts": Counter()})
            elif op == "end" and stack:
                ctx = stack.pop()
                if ctx["kind"] == "loop" and ctx["counts"]["i32x4.dot_i16x8_s"] and not ctx["counts"]["loop"]:
                    ctx["end_line"] = lineno
                    loops.append(ctx)
    assert len(loops) == 2, "expected exactly two flat hot loops in the measured binary"
    loops.sort(key=lambda x: x["counts"]["i32x4.dot_i16x8_s"], reverse=True)
    totals = Counter()
    for ctx, repetitions, dots in zip(loops, (7*144*12, 144*12), (2048, 1024)):
        c = ctx["counts"]
        assert c["i32x4.dot_i16x8_s"] == dots
        assert c.keys() <= COST.keys(), "unknown opcode: do not silently assume its cost"
        ctx["cost_per_loop"] = sum(COST[op]*n for op,n in c.items())
        ctx["loop_executions_per_kernel"] = repetitions
        for op,n in c.items():
            totals[op] += COST[op]*n*repetitions
    assert totals["i32x4.dot_i16x8_s"]*8 == 120*2304*768
    measured = 126_402_135
    groups = {
        "inner_loop_locals": sum(totals[op] for op in ("local.get", "local.set", "local.tee")),
        "inner_loop_dot_and_add": totals["i32x4.dot_i16x8_s"] + totals["i32x4.add"],
        "inner_loop_vector_loads": totals["v128.load"] + totals["v128.load8x8_s"],
    }
    groups["other_inner_loop"] = sum(totals.values()) - sum(groups.values())
    groups["outside_attributed_inner_loops"] = measured - sum(totals.values())
    report = {"shape_m_n_k": [120,2304,768], "wasm_sha256": hashlib.sha256(args.wasm.read_bytes()).hexdigest(),
        "ic_cost_source_revision": "d26cd031176beec51b39fbb9e39e80a3a46a748e", "measured_kernel_cost": measured,
        "attribution": {k:{"cost":v,"percent_of_measured":100*v/measured} for k,v in groups.items()},
        "loops": loops, "weighted_opcode_costs": totals}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["attribution"], indent=2))


if __name__ == "__main__":
    main()
