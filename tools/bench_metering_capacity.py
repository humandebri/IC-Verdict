#!/usr/bin/env python3
"""Measure sustained metered execution capacity; setup is outside every sample.

Concurrency uses independent PocketIC instances on one server to measure host
contention. It does not emulate a production subnet's consensus or scheduler.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time
from urllib.parse import urlparse

import psutil
import wasmtime
from pocket_ic import PocketIC
from bench_w8a16_metering import CONTROL_WAT


def control_wasm(kind):
    if kind == "xor":
        op = "(global.set $value (i32.xor (global.get $value) (global.get $salt)))"
    elif kind == "multiply_add":
        op = "(global.set $value (i32.add (i32.mul (global.get $value) (i32.const 1664525)) (global.get $salt)))"
    else:
        op = "(global.set $value (i32.load (i32.shl (global.get $value) (i32.const 2))))"
    wat = CONTROL_WAT.replace("__REPEAT_XOR__", "\n".join([op] * 257))
    if kind == "pointer_chase":
        wat = wat.replace("(memory 1)", "(memory 257)")
        wat = wat.replace("(i32.const 123456789)", "(i32.const 17)")
        # LCG with a full period modulo 2**22; every pointer remains in bounds.
        init = """
        (func (export "canister_update prepare") (local $i i32)
          (loop $init
            (i32.store (i32.shl (local.get $i) (i32.const 2))
              (i32.and (i32.add (i32.mul (local.get $i) (i32.const 1664525))
                                  (i32.const 1013904223)) (i32.const 4194303)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br_if $init (i32.lt_u (local.get $i) (i32.const 4194304))))
          (call $reply))
        """
        # Argument and reply buffers must not overwrite the pointer array.
        wat = wat.replace("(i32.const 16)", "(i32.const 16777232)")
        wat = wat.replace("(i64.store (i32.const 0)", "(i64.store (i32.const 16777216)")
        wat = wat.replace("(call $append (i32.const 0)", "(call $append (i32.const 16777216)")
        wat = wat.rstrip()[:-1] + init + ")"
    return bytes(wasmtime.wat2wasm(wat))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=3)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pics = [PocketIC() for _ in range(4)]
    port = urlparse(pics[0].server.url).port
    server = psutil.Process(int(subprocess.check_output(
        ["lsof", "-t", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], text=True).splitlines()[0]))

    def processes():
        return [server] + server.children(recursive=True)

    def resources():
        cpu = 0.0
        rss = 0
        for p in processes():
            try:
                t = p.cpu_times()
                cpu += t.user + t.system
                rss += p.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return cpu, rss

    def run_one(pic, canister, count):
        start = time.perf_counter()
        data = pic.update_call(canister, "run", count.to_bytes(4, "little"))
        wall = time.perf_counter() - start
        values = [int.from_bytes(data[i:i+8], "little") for i in range(0, len(data), 8)]
        return {"wall_s": wall, "metered": values[0], "kernel_metered": values[1] if len(values)>1 else None,
                "checksum": values[2] if len(values)>2 else None}

    perf_probe = subprocess.run(["sh", "-c", "perf stat -e cycles,instructions,cache-misses -- true"], capture_output=True, text=True)
    perf_ok = perf_probe.returncode == 0
    report = {
        "lscpu": subprocess.check_output(["lscpu"], text=True),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text() if Path("/sys/fs/cgroup/cpu.max").exists() else None,
        "perf_probe": {"returncode": perf_probe.returncode, "stderr": perf_probe.stderr},
        "method": "whole-update sustained throughput; four independent instances share a host, not a subnet scheduler",
        "cases": {}, "samples": [],
    }
    cases = [("gemm_hot", 1), ("gemm_rotating_32", 32), ("xor", None), ("multiply_add", None), ("pointer_chase", None)]
    random.Random(20260925).shuffle(cases)
    for name, banks in cases:
        is_gemm = banks is not None
        wasm = args.wasm.read_bytes() if is_gemm else control_wasm(name)
        canisters = []
        for pic in pics:
            cid = pic.create_canister()
            pic.add_cycles(cid, 100_000_000_000_000)
            pic.install_code(cid, wasm, [])
            canisters.append(cid)
            if is_gemm:
                payload = b"".join(x.to_bytes(4, "little") for x in (120, 2304, 768, banks))
                pic.update_call(cid, "prepare", payload)
            elif name == "pointer_chase":
                pic.update_call(cid, "prepare", b"")
        zero = run_one(pics[0], canisters[0], 0)
        calibration_count = 1 if is_gemm else 1000
        cal = run_one(pics[0], canisters[0], calibration_count)
        per = (cal["metered"] - zero["metered"]) / calibration_count
        target = 100_000_000 if name == "pointer_chase" else 1_000_000_000
        count = max(1, round(target / per))
        if is_gemm:
            count = min(count, 64)
        report["cases"][name] = {"count": count, "calibration": cal, "empty": zero, "metered_per_iteration": per, "weight_banks": banks}
        # Warm every sandbox and touch every weight bank before sustained samples.
        for pic, cid in zip(pics, canisters):
            run_one(pic, cid, max(count, banks or 0))
        for group in range(args.groups):
            for workers in (1, 4):
                calls = 8 if workers == 1 else 6
                barrier = threading.Barrier(workers)
                def worker(index):
                    barrier.wait()
                    return [run_one(pics[index], canisters[index], count) for _ in range(calls)]
                perf = None
                perf_file = args.out.parent / f"perf-{name}-{group}-{workers}.txt"
                if perf_ok:
                    perf = subprocess.Popen(["perf", "stat", "-x", ";", "-e", "cycles,instructions,cache-misses,context-switches,page-faults",
                        "-p", ",".join(str(p.pid) for p in processes()), "-o", str(perf_file)], stderr=subprocess.DEVNULL)
                    time.sleep(0.1)
                before_cpu, _ = resources()
                start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                    batches = list(executor.map(worker, range(workers)))
                elapsed = time.perf_counter() - start
                after_cpu, rss = resources()
                if perf:
                    perf.send_signal(signal.SIGINT)
                    perf.wait(timeout=10)
                rows = [r for batch in batches for r in batch]
                total = sum(r["metered"] for r in rows)
                times = sorted(r["wall_s"] for r in rows)
                sample = {"case": name, "group": group, "workers": workers, "calls": len(rows),
                    "wall_s": elapsed, "cpu_s": after_cpu-before_cpu, "metered": total,
                    "effective_billion_per_worker_s": total/elapsed/workers/1e9,
                    "cpu_s_per_billion": (after_cpu-before_cpu)/total*1e9,
                    "used_cpu_cores": (after_cpu-before_cpu)/elapsed,
                    "median_call_s": statistics.median(times), "p95_call_s": times[min(len(times)-1,int(len(times)*0.95))],
                    "rss_bytes_summed": rss, "rows": rows,
                    "perf": perf_file.read_text() if perf and perf_file.exists() else None}
                report["samples"].append(sample)
                args.out.write_text(json.dumps(report, indent=2)+"\n")
                print(json.dumps({k:v for k,v in sample.items() if k not in ("rows", "perf")}), flush=True)
        # Canisters from earlier cases remain idle; summed RSS includes them.
    args.out.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
