#!/usr/bin/env python3
"""Report where inference instructions are actually spent, phase by phase.

Why: the acceptance targets are instruction budgets, and the only figure available
so far was a single total per question. `measure_inference.py` measured 4.73B
instructions for a 13M-parameter tier, against roughly 151M theoretical MACs, so
about 31 instructions per MAC. Until the breakdown is known there is no basis for
choosing what to optimise, and ADR-017 explicitly refuses to start on kernels
before this measurement exists.

It loads a sized pack, registers the three schemas, and calls the engine's
`measure_phases` (measurement-only; it bypasses the cache and cannot move funds).

Usage:
    python3 tools/measure_phases.py --tier measure-s --tier measure-m --chunk-kib 256
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_common import (BUILD, ROOT, decision_request, register_schemas,  # noqa: E402
                            sha256, upload_pack)
# The replica plumbing and the schema/state fixtures live in measure_inference; it
# has a __main__ guard, so importing it is side-effect free.
from measure_inference import (SCHEMAS, STATE, Failure, Icp, ensure_canister,  # noqa: E402
                               ensure_cycles, ensure_identity, network_status, principal,
                               require_local_network)


def parse_phases(output: str) -> list[dict]:
    """Pull `vec record { name : text; instructions : nat64 }` out of an icp reply."""
    import re

    entries = []
    for match in re.finditer(r'name\s*=\s*"([^"]+)"\s*;\s*instructions\s*=\s*([\d_]+)', output):
        entries.append({"name": match.group(1),
                        "instructions": int(match.group(2).replace("_", ""))})
    return entries


def measure_tier(icp: Icp, tier: str, args) -> dict:
    record, manifest_raw = upload_pack(icp, tier, args.chunk_kib, args.timeout)
    tokenizer_raw = (ROOT / "fixtures" / tier / "tokenizer.json").read_bytes()
    compiled, bundle, now_ns = register_schemas(icp, manifest_raw, tokenizer_raw, SCHEMAS, args.timeout)

    phases_per_schema = []
    for schema in compiled:
        evaluation_id = sha256(f"phases-{tier}-{schema['schema_id']}".encode())
        request = decision_request(schema, bundle, evaluation_id, STATE, now_ns)
        started = time.monotonic()
        out = icp.call("decision-engine", "measure_phases", request,
                       timeout=args.timeout, expect_ok=False)
        wall = round(time.monotonic() - started, 3)
        phases = parse_phases(out)
        if not phases:
            phases_per_schema.append({"schema": schema["schema_id"], "error": out.strip()[:400]})
            print(f"    {schema['schema_id']:<18} ERROR {out.strip()[:160]}")
            continue
        total = sum(p["instructions"] for p in phases)
        # `share` is derived here, not by the caller: the original `phases` entries
        # only carry name and instructions.
        detailed = [dict(p, share=round(p["instructions"] / total, 4)) for p in phases]
        phases_per_schema.append({
            "schema": schema["schema_id"], "primitive": schema["primitive"],
            "wall_seconds": wall, "total": total, "phases": detailed,
        })
        print(f"    {schema['schema_id']:<18} total={total:>12,}  " +
              "  ".join(f"{p['name']}={p['share']:.0%}" for p in detailed))

    # What the arithmetic alone would cost, for an efficiency ratio.
    config = record["config"]
    layers = config["layers"]
    tokens = 128
    hidden = config["hidden_size"]
    window = min(tokens, config["local_attention"])
    per_layer = (4 * tokens * hidden * hidden                      # qkv + out projections
                 + 2 * tokens * 2 * config["intermediate_size"] * hidden  # wi + wo
                 + 2 * tokens * window * hidden)                   # attention scores + weighted sum
    theoretical_macs = per_layer * layers
    record["theoretical_macs_encoder"] = theoretical_macs
    record["phases"] = phases_per_schema
    for entry in phases_per_schema:
        if "total" in entry and theoretical_macs:
            entry["instructions_per_mac"] = round(entry["total"] / theoretical_macs, 1)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", action="append", default=[])
    parser.add_argument("--identity", default="ic-laya-measure")
    parser.add_argument("--env", default="local")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--chunk-kib", type=int, default=256)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    tiers = args.tier or ["measure-s", "measure-m"]

    if not args.skip_build:
        print("building decision-engine with the candle feature ...")
        target_dir = os.environ.get("CARGO_TARGET_DIR", "/tmp/target")
        subprocess.run(["bash", "tools/build_one.sh", "decision-engine"], cwd=ROOT, check=True,
                       env={**os.environ, "IC_LAYA_CANDLE": "1", "CARGO_TARGET_DIR": target_dir,
                            "PATH": f"{Path.home()}/.cargo/bin:" + os.environ["PATH"]})
    if not (BUILD / "decision-engine.wasm").exists():
        raise Failure("build/decision-engine.wasm missing")

    icp = Icp(ROOT, args.env, args.identity)
    require_local_network(icp)
    started_here = network_status(icp) is None
    if started_here:
        print("starting local network ...")
        icp.run(["network", "start", "-d", "-e", args.env])
    try:
        owner = ensure_identity(icp, args.identity)
        ensure_cycles(icp)
        ensure_canister(icp, "decision-engine")
        print("installing decision-engine (reinstall)")
        icp.run(["canister", "install", "decision-engine", "-e", icp.env, "-y", "-m", "reinstall",
                 "--wasm", str(BUILD / "decision-engine.wasm"), "--args", f"({principal(owner)})"],
                timeout=max(args.timeout, 600))
        admission = icp.run(["canister", "call", "decision-engine", "allow_caller",
                             f"({principal(owner)}, 1000 : nat32)", "-e", icp.env,
                             "--candid", icp.did["decision-engine"]], expect_ok=False)
        if "Ok" not in admission:
            raise Failure(f"could not admit the measurement identity: {admission.strip()[:200]}")

        results = []
        for tier in tiers:
            print(f"\n=== {tier} ===")
            try:
                results.append(measure_tier(icp, tier, args))
            except (Failure, RuntimeError) as error:
                results.append({"tier": tier, "harness_error": str(error)[:400]})
                print(f"  harness error: {error}")

        out = ROOT / "artifacts/phase_measurements.json"
        out.write_text(json.dumps({
            "kind": "per-phase instruction cost on a local replica",
            "note": "measure_phases is measurement-only; it bypasses the engine cache and "
                    "cannot move funds. instructions are counted by the canister",
            "tiers": results}, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")
    finally:
        if started_here:
            print("stopping local network ...")
            icp.run(["network", "stop", "-e", args.env], expect_ok=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)
