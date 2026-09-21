#!/usr/bin/env python3
"""Measure the openJev (GLiClass) canister against *real* prompt lengths.

Why this exists
---------------
`tools/verdict_canister.py` proves the canister runs, but it infers on a handful of
hard-coded token ids. `docs/VERDICT_ENGINE.md` says so explicitly: the 123-token
ceiling is a *first-order extrapolation* from T=2..70, and the real benchmark inputs
run to 150 tokens. An extrapolation is not a measurement, and this is the number the
whole "is 151M viable on ICP" decision rests on.

What it measures
----------------
One case from the author's own benchmark, tokenized by the same Rust code path the
canister and the parity gate use (`verdict-infer tokens`, which is the `check` prompt
contract). That sequence is padded with neutral filler up to each requested length,
so every point in the sweep is a real prompt of a known token count -- not synthetic
ids, and not a truncated prompt.

The canister's own `measured_instructions` is the measurement. Nothing here converts
native timing into instruction counts.

Usage
-----
    # Drives the whole thing: local network, install, upload, sweep.
    python3 tools/measure_verdict.py --sweep 32,64,96,128,150

    # Reuse a chain that is already installed and warm (skips the 605 MiB upload).
    python3 tools/measure_verdict.py --sweep 128,150,160 --skip-upload --keep

The upload is the slow part and is charged at 2,000 cycles/byte, so a resumed run is
almost always what you want after the first one.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from icp_guard import require_local_network, require_local_replica  # noqa: E402

from verdict_canister import (  # noqa: E402  (same directory, shared plumbing)
    BUILD,
    ROOT,
    UPLOADER,
    Failure,
    Icp,
    canister_principal,
    ensure_cycles,
    ensure_identity,
    owner_pem,
    owner_principal,
    upload_pack,
)

INFER = ROOT / "target" / "debug" / "verdict-infer"
DEFAULT_CASES = ROOT / "models" / "verdict-parity" / "cases.jsonl"
DEFAULT_TOKENIZER = ROOT / "models" / "verdict-151m" / "tokenizer.json"


def load_case(path: Path, index: int) -> dict:
    with path.open() as handle:
        for position, line in enumerate(handle):
            if position == index:
                return json.loads(line)
    raise Failure(f"{path} has no case at index {index}")


def case_tokens(case: dict, tokenizer: Path, want: int | None) -> tuple[list[int], int]:
    """`verdict-infer tokens` -- the prompt contract lives in Rust, not here."""
    if not INFER.exists():
        raise Failure(f"{INFER} is missing; run: cargo build -p verdict-candle --bin verdict-infer")
    command = [str(INFER), "tokens", "--pack", str(ROOT / "fixtures" / "verdict-tiny"),
               "--tokenizer", str(tokenizer), "--case-json", json.dumps(case)]
    if want is not None:
        command += ["--tokens", str(want)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        raise Failure(f"verdict-infer tokens:\n{completed.stdout}{completed.stderr}")
    header = completed.stdout.splitlines()[0]
    natural = int(re.search(r"natural=(\d+)", header).group(1))
    classes = len(re.findall(r"\d+", re.search(r"class_positions=\[([^\]]*)\]", header).group(1)))
    ids = [int(v) for v in re.search(r"^ids=(.+)$", completed.stdout, re.M).group(1).split(",")]
    return ids, natural


def infer(icp: Icp, args, canister: str, principal: str, owner: Path, ids: list[int],
          profile: bool = False, detailed: bool = False) -> dict:
    """One measured `infer_tokens` update, made by the canister owner.

    A call that trips the replica's own 40B instruction limit is a *result*, not a
    failure: it is the most direct evidence available about where the ceiling is.
    The instruction count is not returned in that case (the call never completes),
    so the point is recorded with `over_budget` and used only as a bound.

    With `profile`, the call is `infer_profiled` instead, which is owner-only and
    returns the same total plus a per-phase breakdown.
    """
    command = [str(UPLOADER), "--url", args.replica, "--canister", canister,
               "--pem", str(owner), "--no-upload", "--allow-caller", principal]
    command += ["--profile" if profile else "--infer", ",".join(str(i) for i in ids)]
    if profile and detailed:
        command.append("--profile-detailed")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        if "instruction limit" in output or "IC0522" in output:
            return {"over_budget": True, "input_tokens": len(ids), "instructions": None}
        raise Failure(f"{'infer_profiled' if profile else 'infer_tokens'} failed:\n{output}")
    if profile:
        total = re.search(r"PROFILE tokens=(\d+) measured=(\d+)", output)
        if not total:
            raise Failure(f"no PROFILE line in output:\n{output}")
        phases = []
        for name, instructions, share in re.findall(
                r"PHASE (\S+)\s+(\d+)\s+([\d.]+)%(?: x\d+)?", output):
            phases.append({"name": name, "instructions": int(instructions), "percent": float(share)})
        return {"over_budget": False, "instructions": int(total.group(2)),
                "input_tokens": int(total.group(1)), "phases": phases}
    match = re.search(r"MEASURED_INSTRUCTIONS (\d+) tokens=(\d+)", output)
    if not match:
        raise Failure(f"no MEASURED_INSTRUCTIONS in output:\n{output}")
    return {"over_budget": False, "instructions": int(match.group(1)), "input_tokens": int(match.group(2))}


def critical_length(points: list[dict], budget: int) -> float | None:
    """Largest token count that still fits `budget`, by linear interpolation.

    Two points are enough to fit a line because the cost is dominated by the dense
    matmuls, which are linear in T. A point that tripped the instruction limit has no
    measured value; it still bounds the answer from above, so the reported crossing
    is never taken above the longest length known to be over budget.
    """
    measured = sorted((p for p in points if p.get("instructions")), key=lambda p: p["input_tokens"])
    over = [p["input_tokens"] for p in points if p.get("over_budget")]
    crossing = None
    for previous, current in zip(measured, measured[1:]):
        if previous["instructions"] <= budget <= current["instructions"]:
            span = current["instructions"] - previous["instructions"]
            if span <= 0:
                crossing = float(previous["input_tokens"])
                break
            share = (budget - previous["instructions"]) / span
            crossing = previous["input_tokens"] + share * (current["input_tokens"] - previous["input_tokens"])
            break
    if crossing is not None and over:
        crossing = min(crossing, float(min(over) - 1))
    return crossing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, default=ROOT / "models" / "verdict-pack")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case-index", type=int, default=0, help="which benchmark case to use as the prompt")
    parser.add_argument("--sweep", default="32,64,96,128,150", help="comma-separated token lengths")
    parser.add_argument("--budget", type=int, default=40_000_000_000, help="instructions budget to solve for")
    parser.add_argument("--profile", type=int, default=0, metavar="T",
                        help="also call infer_profiled at T tokens and record the phase breakdown")
    parser.add_argument("--profile-detailed", action="store_true",
                        help="include every encoder sub-phase in the profile (many entries)")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "verdict_sweep.json")
    parser.add_argument("--env", default="local")
    parser.add_argument("--replica", default="")
    parser.add_argument("--identity", default="ic-verdict-local")
    parser.add_argument("--home", type=Path, default=ROOT / ".icphome")
    parser.add_argument("--chunk", type=int, default=1024 * 1024)
    parser.add_argument("--top-up", default="", help="cycles to add before an upload")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if not (BUILD / "verdict-engine.wasm").exists():
        raise Failure("missing build/verdict-engine.wasm; run bash tools/build_one.sh verdict-engine")
    if not UPLOADER.exists():
        raise Failure(f"{UPLOADER} is missing; run: cargo build -p verdict-upload")

    lengths = [int(v) for v in args.sweep.split(",") if v.strip()]
    case = load_case(args.cases, args.case_index)

    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    icp = Icp(args.env, args.identity, args.home)
    # This tool mints cycles, reinstalls the canister and uploads 600 MiB, so it must
    # never reach a real network. Refuse before any of that.
    require_local_network(icp, Failure)
    status = icp.run(["network", "status", "-e", args.env, "--json"], expect_ok=False)
    started_here = "api_url" not in status
    if started_here:
        print("starting the local network ...")
        icp.run(["network", "start", "-d", "-e", args.env])
        status = icp.run(["network", "status", "-e", args.env, "--json"])
    match = re.search(r'"api_url":\s*"([^"]+)"', status)
    args.replica = args.replica or (match.group(1) if match else "")
    require_local_replica(args.replica, Failure)
    print(f"  network: {args.replica}")

    try:
        payer = ensure_identity(icp)
        ensure_cycles(icp, payer)
        owner = owner_pem(args.home)
        # upload_pack reads these off the namespace; this tool has no deploy() step
        # to attach them, so they are set here instead of duplicating the upload code.
        args.owner_pem = owner
        principal = owner_principal(owner)
        if canister_principal(icp) is None:
            icp.run(["canister", "create", "verdict-engine", "-e", icp.env, "-q"])
        canister = canister_principal(icp)
        print(f"  canister: {canister}  owner: {principal}")

        if not args.skip_upload:
            print("installing ...")
            icp.run(["canister", "install", "verdict-engine", "-e", icp.env, "-y", "-m", "reinstall",
                     "--wasm", str(BUILD / "verdict-engine.wasm"), "--args", f'(principal "{principal}")'])
            upload_pack(icp, args, canister)
        print("  canister info:", " ".join(icp.call("info").split())[:200])

        natural_ids, natural = case_tokens(case, args.tokenizer, None)
        print(f"case {case.get('id')} native length: {natural} tokens, {len(natural_ids)} ids")

        points: list[dict] = []
        for want in lengths:
            if want < natural:
                print(f"  skip T={want}: below the case's own {natural} tokens (this tool only pads)")
                continue
            ids, _ = case_tokens(case, args.tokenizer, want)
            if len(ids) != want:
                print(f"  skip T={want}: tool produced {len(ids)} ids")
                continue
            measured = infer(icp, args, canister, payer, owner, ids)
            if measured.get("over_budget"):
                point = {"requested_tokens": want, "input_tokens": measured["input_tokens"],
                         "over_budget": True, "instructions": None}
                print(f"  T={measured['input_tokens']:4d}  OVER BUDGET: the replica rejected the call at the "
                      f"{args.budget:,}-instruction limit")
            else:
                point = {"requested_tokens": want, "over_budget": False, **measured,
                         "instructions_per_token": measured["instructions"] / measured["input_tokens"]}
                print(f"  T={measured['input_tokens']:4d}  {measured['instructions']:>16,} instructions"
                      f"  ({point['instructions_per_token']:,.0f}/token)")
            points.append(point)

        crossing = critical_length(points, args.budget) if points else None
        phases = None
        if args.profile:
            ids, _ = case_tokens(case, args.tokenizer, args.profile)
            if len(ids) != args.profile:
                raise Failure(f"--profile {args.profile}: tool produced {len(ids)} ids")
            print(f"  profiling at T={args.profile} ...")
            measured = infer(icp, args, canister, payer, owner, ids,
                             profile=True, detailed=args.profile_detailed)
            if measured.get("over_budget"):
                raise Failure(f"--profile {args.profile} is over the instruction limit; "
                              f"profile a length that completes")
            phases = {"tokens": measured["input_tokens"],
                      "measured_instructions": measured["instructions"],
                      "phases": measured["phases"]}
            for phase in measured["phases"]:
                print(f"    {phase['name']:<16} {phase['instructions']:>14,} {phase['percent']:5.1f}%")
        if not points:
            # A sweep that measured nothing must not look like a successful run: the
            # first version of this tool silently produced an empty report when the
            # padding helper was off by one.
            raise Failure("no measurement points: every requested length was skipped (see the skip lines above)")
        report = {
            "case_id": case.get("id"),
            "natural_tokens": natural,
            "budget": args.budget,
            "points": points,
            "max_tokens_within_budget": crossing,
            "profile": phases,
            "note": ("measured on a local replica through the canister's own measured_instructions; "
                     "lengths above the case's natural length are the same prompt padded with neutral "
                     "filler, so they price a longer sequence, not a longer document"),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

        print()
        measured_points = [p for p in points if p.get("instructions")]
        if crossing is not None:
            print(f"largest measured length within {args.budget:,} instructions: T ≈ {crossing:.1f} tokens "
                  f"(benchmark inputs run to 150)")
        elif measured_points:
            print(f"no crossing inside the sweep; longest completed call was "
                  f"{max(p['input_tokens'] for p in measured_points)} tokens")
        over = [p["input_tokens"] for p in points if p.get("over_budget")]
        if over:
            print(f"over budget at T={min(over)} (the replica rejected it at the instruction limit)")
        print(f"wrote {args.out.relative_to(ROOT)}")
        return 0
    finally:
        if started_here and not args.keep:
            print("stopping the local network ...")
            icp.run(["network", "stop", "-e", args.env], expect_ok=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)
