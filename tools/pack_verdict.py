#!/usr/bin/env python3
"""Compatibility entrypoint for the Rust INT8 packer.

All quantisation and validation live in `tools/verdict-pack`; this wrapper preserves
the documented command line while ensuring no canonical F32 pack is written.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    command = ["cargo", "run", "--quiet", "-p", "verdict-pack", "--", *sys.argv[1:]]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
