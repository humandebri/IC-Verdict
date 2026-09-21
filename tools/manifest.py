#!/usr/bin/env python3
"""Write or check MANIFEST.sha256, the delivery hash list.

Why this exists: the file shipped with 89 entries, no generator, no verifier, nothing
that read it (only a docstring claimed it pinned the fixtures), 24 entries that no longer
matched the tree, and no coverage of the verdict subsystem. An integrity claim that
cannot be checked is worse than no claim.

Coverage is `git ls-files`, so the list describes what a checkout receives: build
outputs, downloaded checkpoints and the generated fixture packs cannot make it depend on
the machine. That also means the verdict subsystem only enters the list once it is
tracked.

    python3 tools/manifest.py --write     # last step of a delivery, after every edit
    python3 tools/manifest.py --check     # or: python3 tools/verify.py --manifest

`--check` is opt-in in verify.py: while the tree is being edited the list is expected to
be stale, and a delivery run is the thing that must be able to prove otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MANIFEST.sha256"
SELF = "MANIFEST.sha256"
# Outputs that `tools/verify.py` rewrites on every run. Pinning them would make the list
# unsatisfiable by construction: the check would always find files that changed because
# the check itself ran. The manifest pins delivery inputs; these are the evidence a run
# produces from them. Deliberate measurement records (`verdict_sweep.json`,
# `inference_measurements.json`, `phase_measurements.json`) stay in the list.
EXCLUDE_EXACT = {"artifacts/verification.json", "artifacts/VALIDATION.txt",
                 "artifacts/synthetic_neural_checks.json"}


def excluded(path: str) -> bool:
    if path in EXCLUDE_EXACT:
        return True
    return path.startswith("artifacts/") and path.endswith(".log")


def tracked() -> list[str]:
    listed = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if listed.returncode != 0:
        raise SystemExit("manifest: `git ls-files` failed; the list covers tracked files only")
    return sorted(p for p in listed.stdout.splitlines() if p and p != SELF and not excluded(p))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()

    paths = tracked()
    current = {p: digest(ROOT / p) for p in paths}
    if args.write:
        body = "".join(f"{current[p]}  {p}\n" for p in paths)
        OUT.write_text(body)
        print(f"wrote {OUT.name} ({len(paths)} tracked files)")
        return 0

    if not OUT.exists():
        print(f"{SELF} is missing; run tools/manifest.py --write", file=sys.stderr)
        return 1
    recorded: dict[str, str] = {}
    for line in OUT.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            recorded[parts[1].strip()] = parts[0].strip().lower()
    stale = sorted(p for p, h in recorded.items() if p not in current or current[p] != h)
    missing = sorted(p for p in current if p not in recorded)
    for path in stale:
        print(f"STALE   {path}", file=sys.stderr)
    for path in missing:
        print(f"MISSING {path}", file=sys.stderr)
    print(f"{SELF}: {len(current)} tracked files, {len(stale)} stale, {len(missing)} missing")
    return 1 if stale or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
