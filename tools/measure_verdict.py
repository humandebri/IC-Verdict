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

    # Measure the 5B *query* path instead: its own sweep and its own artifact.
    python3 tools/measure_verdict.py --query --skip-upload --keep

The upload is the slow part and is charged at 2,000 cycles/byte, so a resumed run is
almost always what you want after the first one.

`--query` is a different measurement, not a faster one. A query call gets 5B instructions
instead of 40B, so it can only score a handful of tokens: the JevBench prompt used by the
update sweep is 118 tokens and cannot fit a query at all. That mode therefore pads the
canonical short id list (`--query-ids`) with neutral filler and records where the
canister's own guard refuses.
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
# The query path's defaults. The lengths deliberately probe past the ceiling the current
# cost model implies for F32 (14 tokens) so the sweep records a refusal, not just a pass;
# `--query-ids` is the canonical short prompt (`cls,<<LABEL>>,2000,<<LABEL>>,3000,sep`)
# and `--query-filler` is the tokenizer's `[PAD]`, taken from the real pack's manifest.
DEFAULT_QUERY_SWEEP = "2,6,10,12,14,15,16,20"
DEFAULT_QUERY_IDS = "50281,50368,2000,50368,3000,50282"
DEFAULT_QUERY_FILLER = 50283


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
          profile: bool = False, detailed: bool = False, query: bool = False) -> dict:
    """One measured inference, made by the canister owner.

    A call the canister's own budget guard refuses, and a call the replica cuts off at
    its instruction limit, are both *results*, not failures: they are the most direct
    evidence available about where the ceiling is. Neither returns an instruction count
    (the call never completes the forward pass), so the point is recorded as a bound.
    The two are told apart by the classifiers `verdict-upload` prints before failing
    (`REJECTED` for a decoded `Err`, `TRAPPED` for a transport-level trap), not by
    parsing English prose.

    With `profile`, the call is `infer_profiled` instead, which is owner-only and
    returns the same total plus a per-phase breakdown. With `query`, it is
    `infer_tokens_query` under the 5B query budget.
    """
    command = [str(UPLOADER), "--url", args.replica, "--canister", canister,
               "--pem", str(owner), "--no-upload", "--allow-caller", principal]
    if query:
        command += ["--query-infer", ",".join(str(i) for i in ids)]
    else:
        command += ["--profile" if profile else "--infer", ",".join(str(i) for i in ids)]
        if profile and detailed:
            command.append("--profile-detailed")
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        if "REJECTED" in output:
            variant = output.split("REJECTED", 1)[1].strip().splitlines()[0].strip()
            if variant.startswith("Capacity"):
                return {"over_budget": True, "guard_refused": True, "variant": variant,
                        "input_tokens": len(ids), "instructions": None}
            # Unauthorized / ModelUnavailable / TooLong are real failures, not bounds.
            raise Failure(f"the canister refused the call: {variant}\n{output}")
        if "TRAPPED" in output or "instruction limit" in output or "IC0522" in output:
            return {"over_budget": True, "replica_trapped": True,
                    "input_tokens": len(ids), "instructions": None}
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


def parse_ids(text: str) -> list[int]:
    """The canonical short id list, e.g. `cls,<<LABEL>>,2000,<<LABEL>>,3000,sep`."""
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values:
        raise Failure("--query-ids is empty")
    return values


def pad_ids(ids: list[int], want: int, filler: int) -> list[int] | None:
    """Grow the canonical id list to `want` by inserting filler before the separator.

    Mirrors what the update sweep does to a real prompt: the class slots stay where the
    prompt contract put them and neutral tokens are added inside the sequence. Returns
    None when `want` is below the list's own length, because this only pads.
    """
    if want < len(ids):
        return None
    if want == len(ids):
        return list(ids)
    return ids[:-1] + [filler] * (want - len(ids)) + ids[-1:]


def query_limits(icp: Icp, args, canister: str, owner: Path) -> dict:
    """The canister's own answer to "how long an input can a query score?".

    Read from the canister, not recomputed here: the ceiling follows from the cost model
    the owner installed, and a client that hardcoded 14 would be wrong the moment
    `set_cost_model` corrects the slope.
    """
    command = [str(UPLOADER), "--url", args.replica, "--canister", canister,
               "--pem", str(owner), "--no-upload", "--query-limits"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    output = completed.stdout + completed.stderr
    match = re.search(r"QUERY_BUDGET (\d+) MAX_TOKENS (\d+) MARGIN (\d+) MAX_INPUT (\d+) "
                      r"COST_FIXED (\d+) PER_TOKEN (\d+)", output)
    if completed.returncode != 0 or not match:
        raise Failure(f"query_limits failed:\n{output}")
    return {"budget": int(match.group(1)), "max_tokens": int(match.group(2)),
            "margin_permille": int(match.group(3)), "max_input_tokens": int(match.group(4)),
            "cost_fixed": int(match.group(5)), "cost_per_token": int(match.group(6))}


def query_run(icp: Icp, args, canister: str, payer: str, owner: Path,
              lengths: list[int]) -> int:
    """Sweep short inputs through the query path and record where it stops.

    These lengths are far below the update sweep's: a query gets 5B instructions, so the
    usable range is a handful of tokens. A point the guard refuses is evidence, not a
    failure, and the sweep stops there -- the guard is monotone in T, so every longer
    length would produce the same verdict at the cost of one call each.
    """
    base = parse_ids(args.query_ids)
    limits = query_limits(icp, args, canister, owner)
    print(f"  query limits: budget={limits['budget']:,} max_tokens={limits['max_tokens']} "
          f"margin={limits['margin_permille']} max_input={limits['max_input_tokens']}")
    points: list[dict] = []
    refused: list[int] = []
    for want in lengths:
        ids = pad_ids(base, want, args.query_filler)
        if ids is None:
            print(f"  skip T={want}: below the canonical {len(base)}-token list (this tool only pads)")
            continue
        measured = infer(icp, args, canister, payer, owner, ids, query=True)
        if measured.get("over_budget"):
            point = {"requested_tokens": want, "input_tokens": measured["input_tokens"],
                     "over_budget": True, "instructions": None,
                     "guard_refused": bool(measured.get("guard_refused")),
                     "replica_trapped": bool(measured.get("replica_trapped"))}
            if point["guard_refused"]:
                point["variant"] = measured.get("variant", "")
            how = ("the canister's budget guard refused it before spending anything"
                   if point["guard_refused"] else "the replica cut the call off at its instruction limit")
            print(f"  T={want:4d}  OVER BUDGET: {how}")
            points.append(point)
            refused.append(measured["input_tokens"])
            break
        point = {"requested_tokens": want, "over_budget": False, **measured,
                 "instructions_per_token": measured["instructions"] / measured["input_tokens"]}
        print(f"  T={measured['input_tokens']:4d}  {measured['instructions']:>16,} instructions"
              f"  ({point['instructions_per_token']:,.0f}/token)")
        points.append(point)
    measured_points = [p for p in points if p.get("instructions")]
    if not measured_points:
        raise Failure("no query measurement points: every requested length was skipped or refused")
    within = max(p["input_tokens"] for p in measured_points)
    report = {
        "kind": "query",
        "budget": limits["budget"],
        "query_limits": limits,
        # The ceiling only means something together with the model that produced it, and
        # the int8 run needs its own fit to be readable at all.
        "cost_model": {"cost_fixed": limits["cost_fixed"], "cost_per_token": limits["cost_per_token"],
                       "margin_permille": limits["margin_permille"], "budget": limits["budget"]},
        "base_ids": base,
        "filler": args.query_filler,
        "points": points,
        "max_tokens_measured": within,
        "guard_refused_at": min(refused) if refused else None,
        "note": ("5B query path (`infer_tokens_query`). Inputs are the canonical short id list "
                 "padded with neutral filler, not the JevBench prompt: that prompt is 118 tokens "
                 "and cannot fit a query call at all. The sweep stops at the first refusal because "
                 "the budget guard is monotone in T, so no longer length is measured. The ceiling "
                 "is exact for the `cost_model` recorded above, not for the hardware: if that "
                 "kernel's cost depends on the data (int8), install a fit from the worst observed "
                 "cost, otherwise a call at the ceiling can still be trapped by the replica."),
    }
    args.query_out.parent.mkdir(parents=True, exist_ok=True)
    args.query_out.write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(f"query path: longest measured T={within} tokens, budget {limits['budget']:,}")
    if refused:
        print(f"refused from T={min(refused)}")
    try:
        shown = args.query_out.relative_to(ROOT)
    except ValueError:
        shown = args.query_out
    print(f"wrote {shown}")
    return 0


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
    parser.add_argument("--query", action="store_true",
                        help="measure the 5B query path (infer_tokens_query) instead of the update sweep")
    parser.add_argument("--query-sweep", default=DEFAULT_QUERY_SWEEP,
                        help="comma-separated token lengths for --query")
    parser.add_argument("--query-ids", default=DEFAULT_QUERY_IDS,
                        help="canonical short id list (cls,<<LABEL>>,content,...,sep) padded for --query; "
                             "the defaults fit models/verdict-pack, so a run against fixtures/verdict-tiny "
                             "must override both --query-ids and --query-filler")
    parser.add_argument("--query-filler", type=int, default=DEFAULT_QUERY_FILLER,
                        help="neutral/pad token id used to pad --query-ids")
    parser.add_argument("--query-out", type=Path,
                        default=ROOT / "artifacts" / "verdict_query_sweep.json")
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
    if args.query and args.profile:
        raise Failure("--profile measures infer_profiled, which is not part of the query path")

    lengths = [int(v) for v in args.sweep.split(",") if v.strip()]
    query_lengths = [int(v) for v in args.query_sweep.split(",") if v.strip()]
    # The query sweep pads the canonical short id list instead of the benchmark case, so a
    # query-only run must not require `models/verdict-parity/cases.jsonl` to exist.
    case = None if args.query else load_case(args.cases, args.case_index)

    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    icp = Icp(args.env, args.identity, args.home)
    # Start the replica before demanding proof that it is local: `network start` is
    # not a guarded command, while everything past this point mints cycles,
    # reinstalls the canister and uploads 600 MiB. Checking first made the documented
    # cold start impossible -- with nothing running, `network status` is unreadable
    # and the fail-closed guard refused before the start could happen.
    status = icp.run(["network", "status", "-e", args.env, "--json"], expect_ok=False)
    started_here = "api_url" not in status
    if started_here:
        print("starting the local network ...")
        icp.run(["network", "start", "-d", "-e", args.env])
        status = icp.run(["network", "status", "-e", args.env, "--json"])
    require_local_network(icp, Failure)
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

        if args.query:
            return query_run(icp, args, canister, payer, owner, query_lengths)

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
        try:
            shown = args.out.relative_to(ROOT)
        except ValueError:
            # --out outside the repository: the write succeeded, so printing the
            # absolute path must not turn a completed measurement into a traceback.
            shown = args.out
        print(f"wrote {shown}")
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
