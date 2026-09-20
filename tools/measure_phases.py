#!/usr/bin/env python3
"""Report where inference instructions are actually spent, phase by phase.

Why: the acceptance targets are instruction budgets, and the only figure available
so far was a single total per question. `measure_inference.py` measured 4.73B
instructions for a 13M-parameter tier. This script reports the same runs split by
phase and normalised by the arithmetic they contain, so the next optimisation step
is chosen from the breakdown (encoder matmuls dominate) rather than from intuition
about which op looks expensive. `instructions_per_mac` is the software-stack cost
coefficient: it is a property of the kernel path, not of the hardware, which is why
the SIMD dispatch has to be verified in the built Wasm and not assumed.

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


def build_state(prefix_tokens: int, profile_tokens: int) -> str:
    """Fill the state so the rendered input reaches the Compact128 profile.

    The short STATE below renders to only 27-38 tokens under the fixture tokenizer,
    which is not the length the design budgets for. Measuring at both lengths matters
    because attention is quadratic in tokens while everything else is linear.
    """
    filler = ("The customer reports a duplicate payment and requests a refund for the "
              "second charge on the same invoice reference number")
    words = filler.split()
    budget = profile_tokens - prefix_tokens - 1     # one trailing separator
    if budget <= 0:
        raise RuntimeError("profile too small for this schema")
    out = []
    index = 0
    while len(out) < budget:
        out.append(words[index % len(words)])
        index += 1
    return " ".join(out[:budget])


# IC per-update-call instruction ceiling (docs.internetcomputer.org, resource limits).
# The forum's in-consensus inference study reports capacity as tokens per call against
# this same ceiling, so reporting it makes the two directly comparable.
UPDATE_INSTRUCTION_LIMIT = 40_000_000_000


def encoder_macs(config: dict, tokens: int) -> int:
    """MACs for one encoder pass over `tokens` rendered tokens.

    Shapes come from `laya_candle::expected_tensors`; one token costs

        4 * h * h        qkv (3*h*h) + out (h*h)
        3 * inter * h    wi (2*inter*h: gate and up) + wo (inter*h)
        2 * window * h   QK^T + weighted sum inside the sliding window

    `wo.weight` is [hidden, intermediate], so charging `2*inter*h` for wi *and* wo was
    wrong, but only in the MLP term: the other terms were right, so the denominator was
    overstated by 1.242x at hidden 512 / intermediate 2048 (not 4/3 -- the MLP is about
    73% of the total, which dilutes a 25% error in that term). The previous version did
    that on top of a hardcoded 128-token length.
    """
    h = config["hidden_size"]
    inter = config["intermediate_size"]
    window = min(tokens, config["local_attention"])
    return (4 * h * h + 3 * inter * h + 2 * window * h) * tokens * config["layers"]


def sweep_arguments(args) -> list[int]:
    if not args.token_sweep:
        return [args.profile_tokens] if args.full_profile else [0]
    return [int(x) for x in args.token_sweep.split(",") if x.strip()]


def measure_sweep(icp: Icp, tier: str, args, record: dict, compiled, bundle: bytes,
                  now_ns: int) -> dict:
    """Measure one schema at several rendered lengths.

    The short-input measurements suggested most of the cost was per-token, but fitting
    two points implies a large constant: 3.1B instructions per question at the 13M
    tier, 66% of the total at 27 tokens. Three lengths are needed to tell a genuine
    constant from an artifact of the fit, and token length is a design parameter, so
    the distinction has to be measured rather than modelled.
    """
    schema = compiled[2]          # Score, the widest option set
    points = []
    for length in sweep_arguments(args):
        state = build_state(len(schema["prefix"]), length)
        evaluation_id = sha256(f"sweep-{tier}-{length}".encode())
        out = icp.call("decision-engine", "measure_phases",
                       decision_request(schema, bundle, evaluation_id, state, now_ns),
                       timeout=args.timeout, expect_ok=False)
        probe = icp.call("decision-engine", "evaluate",
                         decision_request(schema, bundle, sha256(b"len-" + evaluation_id),
                                          state, now_ns),
                         timeout=args.timeout, expect_ok=False)
        import re as _re
        found = _re.search(r"input_tokens = ([\d_]+) : nat32", probe)
        rendered = int(found.group(1).replace("_", "")) if found else None
        phases = parse_phases(out)
        if not phases or rendered is None:
            points.append({"requested": length, "error": out.strip()[:200]})
            continue
        total = sum(p["instructions"] for p in phases)
        points.append({"requested": length, "rendered": rendered, "total": total,
                       "phases": {p["name"]: p["instructions"] for p in phases}})
        print(f"    rendered={rendered:<4} total={total:>14,}")
    record["sweep"] = points
    return record


def measure_tier(icp: Icp, tier: str, args) -> dict:
    record, manifest_raw = upload_pack(icp, tier, args.chunk_kib, args.timeout)
    tokenizer_raw = (ROOT / "fixtures" / tier / "tokenizer.json").read_bytes()
    compiled, bundle, now_ns = register_schemas(icp, manifest_raw, tokenizer_raw, SCHEMAS, args.timeout)

    if len(sweep_arguments(args)) > 1:
        return measure_sweep(icp, tier, args, record, compiled, bundle, now_ns)

    phases_per_schema = []
    # The rendered length is measured, never assumed. A previous version divided by a
    # hardcoded 128-token profile while the actual inputs were 27-38 tokens, which
    # understated instructions/MAC by about 4x. The three schemas differ from each
    # other, so the length is keyed per schema rather than reduced to one number.
    rendered_tokens: dict[str, int] = {}
    for schema in compiled:
        evaluation_id = sha256(f"phases-{tier}-{schema['schema_id']}".encode())
        state = build_state(len(schema["prefix"]), args.profile_tokens) if args.full_profile else STATE
        request = decision_request(schema, bundle, evaluation_id, state, now_ns)
        started = time.monotonic()
        out = icp.call("decision-engine", "measure_phases", request,
                       timeout=args.timeout, expect_ok=False)
        wall = round(time.monotonic() - started, 3)
        probe = icp.call("decision-engine", "evaluate",
                         decision_request(schema, bundle, sha256(b"len-" + evaluation_id),
                                          state, now_ns),
                         timeout=args.timeout, expect_ok=False)
        import re as _re
        found = _re.search(r"input_tokens = ([\d_]+) : nat32", probe)
        if found:
            rendered_tokens[schema["schema_id"]] = int(found.group(1).replace("_", ""))
        phases = parse_phases(out)
        if not phases:
            phases_per_schema.append({"schema": schema["schema_id"], "error": out.strip()[:400]})
            print(f"    {schema['schema_id']:<18} ERROR {out.strip()[:160]}")
            continue
        total = sum(p["instructions"] for p in phases)
        # `share` is derived here, not by the caller: the original `phases` entries
        # only carry name and instructions.
        # Encoder sub-phases repeat once per layer, so aggregate by name. The first
        # occurrence of each name fixes the report order.
        aggregated: dict[str, int] = {}
        for entry in phases:
            aggregated[entry["name"]] = aggregated.get(entry["name"], 0) + entry["instructions"]
        detailed = [{"name": name, "instructions": value, "share": round(value / total, 4)}
                    for name, value in aggregated.items()]
        phases_per_schema.append({
            "schema": schema["schema_id"], "primitive": schema["primitive"],
            "wall_seconds": wall, "total": total, "phases": detailed,
        })
        print(f"    {schema['schema_id']:<18} total={total:>12,}  " +
              "  ".join(f"{p['name']}={p['share']:.0%}" for p in detailed))

    # What the arithmetic alone would cost. Uses each evaluation's *measured* rendered
    # length; the Compact128 profile is what the design allows, not what this sends.
    config = record["config"]
    record["rendered_tokens"] = rendered_tokens or None
    record["profile_tokens"] = args.profile_tokens if args.full_profile else None
    record["phases"] = phases_per_schema
    record["phase_names_observed"] = len({p["name"] for e in phases_per_schema for p in e.get("phases", [])})
    if not rendered_tokens:
        return record
    record["notes"] = ("instructions_per_mac, instructions_per_token and "
                       "tokens_per_update_budget use each evaluation's own measured "
                       "rendered length, not the 128-token profile")
    for entry in phases_per_schema:
        tokens = rendered_tokens.get(entry.get("schema"))
        if "total" not in entry or not tokens:
            continue
        macs = encoder_macs(config, tokens)
        per_token = entry["total"] / tokens
        # Encoder-only as well as the whole call: the decision head and scorer are
        # included in `total` but are not in the MAC denominator, so the two ratios
        # answer different questions ("cost of the budget" vs "cost of the kernel").
        encoder = sum(p["instructions"] for p in entry["phases"]
                      if p["name"].startswith("layer.") or p["name"] in ("embedding", "final_norm", "encoder"))
        entry["tokens"] = tokens
        entry["macs"] = macs
        entry["encoder_instructions"] = encoder
        entry["instructions_per_mac"] = round(entry["total"] / macs, 2)
        entry["instructions_per_mac_encoder"] = round(encoder / macs, 2)
        entry["instructions_per_token"] = round(per_token, 1)
        # tokens per update call, i.e. the capacity the DFINITY forum's study reports:
        # how many tokens of this model fit under the 40B instruction ceiling.
        entry["tokens_per_update_budget"] = round(UPDATE_INSTRUCTION_LIMIT / per_token, 1)
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
    parser.add_argument("--full-profile", action="store_true",
                        help="pad the state so the rendered input reaches --profile-tokens "
                             "(the short STATE only renders to 27-38 tokens)")
    parser.add_argument("--profile-tokens", type=int, default=128)
    parser.add_argument("--token-sweep", default=None,
                        help="comma-separated rendered lengths to sweep (e.g. 32,64,128). "
                             "Two lengths cannot separate a constant per-call cost from a "
                             "per-token one; three can. Implies --full-profile.")
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
