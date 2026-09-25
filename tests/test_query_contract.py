"""Contract tests for the 5B query path (canister `infer_tokens_query` + its tooling).

The query path exists for a different reason than the update path: it needs no consensus
round and costs no cycles, but it gets 5B instructions instead of 40B. These tests pin the
parts that are easy to get quietly wrong:

  * a refusal by the canister's own budget guard (`REJECTED Capacity`) and a replica
    instruction-limit trap (`TRAPPED`) are *bounds*, not failures -- while any other
    decoded rejection (`ModelUnavailable`, `Unauthorized`, ...) still is a failure;
  * a measured reply is parsed from the same line the update path uses;
  * padding only ever grows the canonical id list, and inserts filler before the
    separator, so the class slots stay where the prompt contract put them;
  * `query_limits` is read from the canister rather than recomputed here;
  * the default sweep probes past the F32 ceiling instead of only measuring successes;
  * `verify.py --verdict-query` obeys the existing "requested but not run is not a pass"
    contract.

Everything is in-process: the imported modules' `subprocess` handles are swapped and the
real `artifacts/` are never touched.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import measure_verdict  # noqa: E402
import verify  # noqa: E402


class FakeCompleted:
    def __init__(self, code: int = 0, out: str = "", err: str = "") -> None:
        self.returncode, self.stdout, self.stderr = code, out, err


@contextlib.contextmanager
def fake_subprocess(module, completed):
    """Swap one module's `subprocess` handle, leaving the stdlib module alone."""
    def run(command, **kwargs):
        return completed(command) if callable(completed) else completed

    saved = module.subprocess
    module.subprocess = types.SimpleNamespace(run=run)
    try:
        yield
    finally:
        module.subprocess = saved


ARGS = types.SimpleNamespace(replica="http://127.0.0.1:8000")


class QueryInferenceTests(unittest.TestCase):
    def call(self, completed, ids=(10, 11, 12)):
        with fake_subprocess(measure_verdict, completed):
            return measure_verdict.infer(
                None, ARGS, "canister", "principal", Path("owner.pem"), list(ids), query=True)

    def test_a_budget_guard_refusal_is_a_bound_not_a_failure(self) -> None:
        point = self.call(FakeCompleted(1, out="REJECTED Capacity\n"))
        self.assertTrue(point["over_budget"])
        self.assertTrue(point["guard_refused"])
        self.assertFalse(point.get("replica_trapped", False))
        self.assertIsNone(point["instructions"])
        self.assertEqual(point["input_tokens"], 3)

    def test_a_replica_instruction_limit_is_also_a_bound(self) -> None:
        point = self.call(FakeCompleted(
            1, out="TRAPPED CanisterError: instruction limit exceeded (IC0522)\n"))
        self.assertTrue(point["over_budget"])
        self.assertTrue(point["replica_trapped"])
        self.assertFalse(point.get("guard_refused", False))

    def test_other_rejections_stay_failures(self) -> None:
        # A cold model or a caller that is not admitted is not a ceiling: recording it as
        # `over_budget` would silently turn a broken setup into "the query path is short".
        for output in ('REJECTED ModelUnavailable("warm-up required")\n', "REJECTED Unauthorized\n"):
            with self.assertRaises(measure_verdict.Failure):
                self.call(FakeCompleted(1, out=output))

    def test_a_measured_reply_is_parsed(self) -> None:
        point = self.call(FakeCompleted(
            0, out="QUERY tokens=6 class_positions=[1, 3]\nMEASURED_INSTRUCTIONS 4567 tokens=6\n"))
        self.assertFalse(point["over_budget"])
        self.assertEqual(point["instructions"], 4567)
        self.assertEqual(point["input_tokens"], 6)

    def test_padding_grows_the_list_before_the_separator(self) -> None:
        base = [50281, 50368, 2000, 50368, 3000, 50282]
        self.assertEqual(measure_verdict.pad_ids(base, 8, 50283),
                         [50281, 50368, 2000, 50368, 3000, 50283, 50283, 50282])
        self.assertEqual(measure_verdict.pad_ids(base, 6, 50283), base)
        self.assertIsNone(measure_verdict.pad_ids(base, 4, 50283))

    def test_query_limits_come_from_the_canister(self) -> None:
        completed = FakeCompleted(
            0, out="QUERY_BUDGET 5000000000 MAX_TOKENS 14 MARGIN 1005 MAX_INPUT 128 "
                   "COST_FIXED 254400000 PER_TOKEN 327600000\n")
        with fake_subprocess(measure_verdict, completed):
            limits = measure_verdict.query_limits(None, ARGS, "canister", Path("owner.pem"))
        self.assertEqual(limits["budget"], 5_000_000_000)
        self.assertEqual(limits["max_tokens"], 14)
        self.assertEqual(limits["max_input_tokens"], 128)
        self.assertEqual(limits["cost_fixed"], 254_400_000)
        self.assertEqual(limits["cost_per_token"], 327_600_000)

    def test_the_report_records_the_cost_model_that_produced_the_ceiling(self) -> None:
        # Without this the int8 artifact is indistinguishable from the F32 one: the same
        # keys with a different `max_tokens` and nothing saying why.
        limits_line = ("QUERY_BUDGET 5000000000 MAX_TOKENS 14 MARGIN 1005 MAX_INPUT 128 "
                       "COST_FIXED 254400000 PER_TOKEN 327600000\n")
        measured_line = "QUERY tokens=3 class_positions=[1]\nMEASURED_INSTRUCTIONS 1000000 tokens=3\n"

        def fake(command):
            return FakeCompleted(0, out=limits_line if "--query-limits" in command else measured_line)

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "verdict_query_sweep.json"
            args = types.SimpleNamespace(replica="http://127.0.0.1:8000", query_ids="1,2,3",
                                        query_filler=9, query_out=out)
            with fake_subprocess(measure_verdict, fake):
                code = measure_verdict.query_run(
                    None, args, "canister", "principal", Path("owner.pem"), [3])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text())
        self.assertEqual(report["cost_model"]["cost_fixed"], 254_400_000)
        self.assertEqual(report["cost_model"]["cost_per_token"], 327_600_000)
        self.assertEqual(report["cost_model"]["margin_permille"], 1005)
        self.assertEqual(report["base_ids"], [1, 2, 3])
        self.assertEqual(report["max_tokens_measured"], 3)

    def test_a_failed_limit_probe_is_a_failure(self) -> None:
        with fake_subprocess(measure_verdict, FakeCompleted(1, out="no such method\n")):
            with self.assertRaises(measure_verdict.Failure):
                measure_verdict.query_limits(None, ARGS, "canister", Path("owner.pem"))

    def test_the_default_sweep_probes_past_the_f32_ceiling(self) -> None:
        lengths = [int(v) for v in measure_verdict.DEFAULT_QUERY_SWEEP.split(",") if v.strip()]
        self.assertIn(14, lengths)
        self.assertGreater(max(lengths), 14,
                           "the sweep must ask for a length the guard refuses, or the artifact "
                           "only ever shows successes")


class VerifyQueryFlagTests(unittest.TestCase):
    def _main(self, argv: list[str]):
        """Run `verify.main()` in-process, failing any `measure_verdict.py` invocation."""
        recorded: list[list[str]] = []

        def fake_run(command, **kwargs):
            recorded.append(list(command))
            query = any("measure_verdict.py" in str(part) for part in command)
            return FakeCompleted(1 if query else 0, err="boom\n" if query else "")

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            (tmp / "artifacts").mkdir(parents=True)
            (tmp / "dfx.json").write_text("{}\n")
            saved = (verify.ROOT, verify.shutil, verify.subprocess, sys.argv)
            verify.ROOT = tmp
            verify.shutil = types.SimpleNamespace(which=lambda name: None)
            verify.subprocess = types.SimpleNamespace(
                run=fake_run, TimeoutExpired=subprocess.TimeoutExpired)
            sys.argv = ["verify.py", *argv]
            buffer = __import__("io").StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    code = verify.main()
            finally:
                verify.ROOT, verify.shutil, verify.subprocess, sys.argv = saved
            report = json.loads((tmp / "artifacts" / "verification.json").read_text())
            return code, buffer.getvalue(), report, recorded

    def test_a_requested_query_check_that_fails_is_not_a_pass(self) -> None:
        code, out, report, recorded = self._main(["--verdict-query"])
        self.assertEqual(code, 1, out)
        self.assertIn("REQUESTED BUT NOT VERIFIED: openjev_query_canister", out)
        self.assertIn("openjev_query_canister", report["requested_but_not_verified"])
        statuses = {c["check"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses["openjev_query_canister"], "FAIL")
        invocations = [c for c in recorded if any("measure_verdict.py" in p for p in c)]
        self.assertTrue(invocations, "verify.py never invoked measure_verdict.py")
        self.assertIn("--query", invocations[0])
        self.assertIn("--skip-upload", invocations[0])

    def test_the_query_row_stays_not_run_when_it_is_not_requested(self) -> None:
        code, out, report, _ = self._main([])
        self.assertEqual(code, 0, out)
        statuses = {c["check"]: c["status"] for c in report["checks"]}
        self.assertEqual(statuses["openjev_query_canister"], "NOT_RUN")


if __name__ == "__main__":
    unittest.main()
