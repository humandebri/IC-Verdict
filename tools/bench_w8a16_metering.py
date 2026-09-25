#!/usr/bin/env python3
"""Compare W8A16 node service demand with an equal-metered Wasm control.

Runs against an isolated PocketIC server on the current host. The W8A16 endpoint
contains fixed setup and validation; varying its iteration count and fitting a
slope removes that fixed work. The control uses a tight integer loop, calibrated
to consume the same number of IC instructions per extra kernel iteration.
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import ic
from ic.candid import Types
import psutil
import wasmtime
from pocket_ic import PocketIC
from ic.candid import labelHash

OWNER = ic.Principal.from_str("f4yey-rejpn-tijgk-4mjjc-rsxkx-odbv7-nr5ne-2zmbp-imi6v-sac2h-vqe")
LEVELS = (1, 4, 8, 16)
CONTROL_WAT = r"""
(module
  (import "ic0" "msg_arg_data_copy" (func $arg_copy (param i32 i32 i32)))
  (import "ic0" "performance_counter" (func $counter (param i32) (result i64)))
  (import "ic0" "msg_reply_data_append" (func $append (param i32 i32)))
  (import "ic0" "msg_reply" (func $reply))
  (memory 1)
  (global $value (mut i32) (i32.const 123456789))
  (global $salt (mut i32) (i32.const 987654321))
  (func (export "canister_update run")
    (local $i i32) (local $target i32)
    (call $arg_copy (i32.const 16) (i32.const 0) (i32.const 4))
    (local.set $target (i32.load (i32.const 16)))
    (if (i32.gt_u (local.get $target) (i32.const 0))
      (then
        (loop $again
          __REPEAT_XOR__
          (local.set $i (i32.add (local.get $i) (i32.const 1)))
          (br_if $again (i32.lt_u (local.get $i) (local.get $target)))
        )
      )
    )
    (global.set $salt (i32.add (global.get $salt) (local.get $target)))
    (i64.store (i32.const 0) (call $counter (i32.const 0)))
    (call $append (i32.const 0) (i32.const 8))
    (call $reply)
  )
)
"""


def field(record, name):
    return record["_" + str(labelHash(name))]


def slope(rows, kind, metric):
    subset = [r for r in rows if r["kind"] == kind]
    xs = [r["charged_instructions"] for r in subset]
    ys = [r[metric] for r in subset]
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--m", type=int, default=40)
    parser.add_argument("--n", type=int, default=2304)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 3 or any(x <= 0 for x in (args.m, args.n, args.k)):
        parser.error("positive dimensions and at least three repeats required")

    wasm = args.wasm.read_bytes()
    install_wasm = wasm if wasm.startswith(b"\x1f\x8b") else gzip.compress(wasm, compresslevel=9)
    if len(install_wasm) >= 2_000_000:
        raise RuntimeError("compressed canister exceeds PocketIC ingress limit")
    pic = PocketIC()
    pic.set_sender(OWNER)
    gemm = pic.create_canister()
    control = pic.create_canister()
    for canister in (gemm, control):
        pic.add_cycles(canister, 20_000_000_000_000)
    pic.install_code(gemm, install_wasm, [{"type": Types.Principal, "value": OWNER.bytes}])
    xor_body = "\n".join(
        "(global.set $value (i32.xor (global.get $value) (global.get $salt)))"
        for _ in range(257)
    )
    pic.install_code(control, bytes(wasmtime.wat2wasm(CONTROL_WAT.replace("__REPEAT_XOR__", xor_body))), [])
    port = urlparse(pic.server.url).port
    pids = subprocess.check_output(
        ["lsof", "-t", "-nP", "-iTCP:" + str(port), "-sTCP:LISTEN"], text=True
    ).splitlines()
    server = psutil.Process(int(pids[0]))

    def process_tree():
        try:
            return [server] + server.children(recursive=True)
        except (PermissionError, psutil.AccessDenied):
            return [server]

    def cpu_seconds():
        total = 0.0
        for proc in process_tree():
            try:
                times = proc.cpu_times()
                total += times.user + times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def call_gemm(iterations):
        payload = ic.encode([
            {"type": Types.Nat32, "value": value}
            for value in (args.m, args.n, args.k, iterations)
        ])
        before_cpu, before_wall = cpu_seconds(), time.perf_counter()
        reply = pic.update_call(gemm, "bench_int8", payload)
        wall, cpu = time.perf_counter() - before_wall, cpu_seconds() - before_cpu
        variant = ic.decode(reply)[0]["value"]
        record = field(variant, "Ok")
        if not field(record, "simd_used"):
            raise RuntimeError("Wasm SIMD path was not used")
        return {
            "kind": "gemm", "level": iterations,
            "charged_instructions": field(record, "instructions"),
            "instructions_per_mac": field(record, "instructions_per_mac"),
            "wall_s": wall, "process_tree_cpu_s": cpu,
        }

    def call_control(loops):
        before_cpu, before_wall = cpu_seconds(), time.perf_counter()
        reply = pic.update_call(control, "run", loops.to_bytes(4, "little"))
        wall, cpu = time.perf_counter() - before_wall, cpu_seconds() - before_cpu
        return {
            "kind": "control", "loops": loops,
            "charged_instructions": int.from_bytes(reply[:8], "little"),
            "wall_s": wall, "process_tree_cpu_s": cpu,
        }

    gemm_one = call_gemm(1)
    control_zero = call_control(0)
    control_calibration = call_control(10_000)
    control_per_loop = (control_calibration["charged_instructions"] - control_zero["charged_instructions"]) / 10_000
    if control_per_loop <= 0:
        raise RuntimeError("control did not consume instructions")
    loops_per_gemm = round(gemm_one["charged_instructions"] / control_per_loop)
    if loops_per_gemm * max(LEVELS) >= 2**32:
        raise RuntimeError("control loop count exceeds u32")
    # A second call warms up compiled Wasm and the execution path.
    call_gemm(16)
    call_control(16 * loops_per_gemm)

    rows = []
    for block in range(args.repeats):
        plan = [("gemm", level) for level in LEVELS] + [("control", level) for level in LEVELS]
        random.Random(20260925 + block).shuffle(plan)
        for kind, level in plan:
            row = call_gemm(level) if kind == "gemm" else call_control(level * loops_per_gemm)
            row["block"] = block
            row["level"] = level
            rows.append(row)
            print(f"{block:02d} {kind:7s} x{level:2d} I={row['charged_instructions']:,} "
                  f"wall={row['wall_s']:.6f}s cpu={row['process_tree_cpu_s']:.6f}s", flush=True)

    ratios = {}
    blocks = list(range(args.repeats))
    for metric in ("wall_s", "process_tree_cpu_s"):
        gemm_slope = slope(rows, "gemm", metric)
        control_slope = slope(rows, "control", metric)
        bootstrap = []
        rng = random.Random(20260925)
        for _ in range(1000):
            selected = [rng.choice(blocks) for _ in blocks]
            sample = [row for block in selected for row in rows if row["block"] == block]
            gs = slope(sample, "gemm", metric)
            cs = slope(sample, "control", metric)
            if gs > 0 and cs > 0:
                bootstrap.append(gs / cs)
        bootstrap.sort()
        interval = (
            [bootstrap[int(0.025 * len(bootstrap))], bootstrap[int(0.975 * len(bootstrap))]]
            if len(bootstrap) >= 950 else None
        )
        ratios[metric] = {
            "gemm_seconds_per_billion_charged": gemm_slope * 1e9,
            "control_seconds_per_billion_charged": control_slope * 1e9,
            "gemm_over_control_load_per_charge": gemm_slope / control_slope if gemm_slope > 0 and control_slope > 0 else None,
            "ratio_bootstrap_95pct": interval,
            "positive_bootstrap_samples": len(bootstrap),
        }
    report = {
        "host": platform.platform(),
        "uname": platform.uname()._asdict(),
        "cpu_model": subprocess.check_output(["sh", "-c", "lscpu 2>/dev/null | grep 'Model name' | head -1 || true"], text=True).strip(),
        "wasm_sha256": hashlib.sha256(wasm).hexdigest(),
        "shape": [args.m, args.n, args.k],
        "control_instructions_per_loop": control_per_loop,
        "control_loops_per_kernel": loops_per_gemm,
        "pocketic_pid": server.pid,
        "process_tree": [{"pid": p.pid, "name": p.name()} for p in process_tree()],
        "warmup": {"gemm": gemm_one, "control_zero": control_zero, "control_calibration": control_calibration},
        "ratios": ratios,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps(ratios, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
