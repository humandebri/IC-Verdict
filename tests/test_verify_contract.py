"""Contract test for `tools/verify.py` (review finding P1-4).

Before this, `--rust` on a machine without a toolchain, or `--verdict` without the
pack, produced a report whose verdict was "PASS" and exited 0: `NOT_RUN` was not a
failure unless `--require-rust` was *also* passed. This test pins the new contract:

  * a check that was requested and did not run => exit 1, and both the stdout and
    `artifacts/{verification.json,VALIDATION.txt}` say so;
  * the no-flag invocation stays exit 0 -- it is a source-delivery validation --
    but must announce that nothing behavioural was verified.

Everything is in-process: `verify.shutil`/`verify.subprocess` are replaced and
`verify.ROOT` points at a temporary directory, so the real `artifacts/` are never
touched and no toolchain is invoked.
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify  # noqa: E402


class FakeCompleted:
    def __init__(self, code: int = 0, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, err


def _sandbox(tmp: Path) -> None:
    """Minimal tree the source-only checks can scan without finding problems."""
    (tmp / "artifacts").mkdir(parents=True, exist_ok=True)
    (tmp / "dfx.json").write_text("{}\n")


class VerifyContractTests(unittest.TestCase):
    def _main(self, argv: list[str], toolchain_available: bool = False):
        """Run `verify.main()` in-process against a temporary ROOT."""
        recorded: list[list[str]] = []

        def fake_run(command, **kwargs):
            recorded.append(list(command))
            return FakeCompleted()

        def fake_which(name):
            if name in {"cargo", "rustc", "rustup", "dfx"}:
                return f"/usr/bin/{name}" if toolchain_available else None
            return None

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            _sandbox(tmp)
            saved = (verify.ROOT, verify.shutil, verify.subprocess, sys.argv)
            verify.ROOT = tmp
            verify.shutil = types.SimpleNamespace(which=fake_which)
            verify.subprocess = types.SimpleNamespace(run=fake_run,
                                                     TimeoutExpired=subprocess.TimeoutExpired)
            sys.argv = ["verify.py", *argv]
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    code = verify.main()
            finally:
                verify.ROOT, verify.shutil, verify.subprocess, sys.argv = saved
            report = json.loads((tmp / "artifacts" / "verification.json").read_text())
            validation = (tmp / "artifacts" / "VALIDATION.txt").read_text()
            return code, buffer.getvalue(), report, validation, recorded

    def test_rust_requested_without_a_toolchain_is_not_a_pass(self) -> None:
        code, out, report, validation, _ = self._main(["--rust"])
        self.assertEqual(code, 1, out)
        self.assertIn("REQUESTED BUT NOT VERIFIED: rust_workspace_tests", out)
        self.assertIn("rust_workspace_tests", report["requested_but_not_verified"])
        self.assertIn("REQUESTED BUT NOT VERIFIED rust_workspace_tests", validation)
        statuses = {c["check"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses["rust_workspace_tests"], "NOT_RUN")

    def test_verdict_requested_without_the_pack_is_not_a_pass(self) -> None:
        code, out, report, _, _ = self._main(["--verdict"])
        self.assertEqual(code, 1, out)
        self.assertIn("REQUESTED BUT NOT VERIFIED: openjev_checkpoint_parity", out)
        self.assertIn("openjev_checkpoint_parity", report["requested_but_not_verified"])

    def test_no_flags_stays_green_but_announces_nothing_was_verified(self) -> None:
        code, out, report, _, _ = self._main([])
        self.assertEqual(code, 0, out)
        self.assertIn("NOTHING VERIFIED", out)
        self.assertEqual(report["requested_but_not_verified"], [])

    def test_rust_requested_with_a_toolchain_runs_the_checks(self) -> None:
        code, out, report, _, recorded = self._main(["--rust"], toolchain_available=True)
        self.assertEqual(code, 0, out)
        self.assertEqual(report["requested_but_not_verified"], [])
        commands = [" ".join(c) for c in recorded]
        self.assertTrue(any("cargo test --workspace" in c for c in commands), commands)
        self.assertTrue(any("verdict-candle" in c for c in commands), commands)
        self.assertTrue(any("verdict-engine" in c for c in commands), commands)


if __name__ == "__main__":
    unittest.main(verbosity=2)
