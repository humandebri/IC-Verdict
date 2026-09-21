"""Regression test for the local-network guard (review finding P0-1).

Before this guard existed, `tools/verdict_canister.py` and
`tools/measure_verdict.py` ran `icp token transfer`, `icp cycles mint` and
`icp canister install -m reinstall` against whatever `--env` named, and
`--replica` was handed straight to ic-agent. The scripts that mint cycles and
wipe canisters now route every state-changing call through `icp_guard`.

These tests do not need the `icp` CLI or a network: `subprocess.run` is replaced,
so the assertion is exactly "the state-changing command was never executed".
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import icp_guard  # noqa: E402
import verdict_canister  # noqa: E402


class StubFailure(Exception):
    pass


class StubIcp:
    """Minimal duck type: `env` plus the `run` signature the guard uses."""

    def __init__(self, env: str, status: str) -> None:
        self.env = env
        self.status = status
        self.calls: list[list[str]] = []

    def run(self, args, expect_ok: bool = True, identity: bool = True) -> str:
        self.calls.append(list(args))
        return self.status


LOCAL = json.dumps({"managed": True, "api_url": "http://127.0.0.1:8000"})
REMOTE = json.dumps({"managed": False, "api_url": "https://ic0.app"})


class NetworkGuardTests(unittest.TestCase):
    def test_local_replica_is_allowed(self) -> None:
        icp = StubIcp("local", LOCAL)
        self.assertIsNone(icp_guard.local_network_violation(icp))
        icp_guard.require_local_network(icp, StubFailure)   # must not raise

    def test_non_managed_environment_is_refused(self) -> None:
        icp = StubIcp("ic", REMOTE)
        violation = icp_guard.local_network_violation(icp)
        self.assertIsNotNone(violation)
        self.assertIn("https://ic0.app", violation)
        with self.assertRaises(StubFailure):
            icp_guard.require_local_network(icp, StubFailure)

    def test_unknown_status_is_refused(self) -> None:
        # Fails closed: an unreadable status is not evidence of a local replica. `network
        # start` is not a guarded command, so starting a stopped replica still works.
        icp = StubIcp("local", "no json here")
        violation = icp_guard.local_network_violation(icp)
        self.assertIsNotNone(violation)
        self.assertIn("cannot confirm", violation)
        with self.assertRaises(StubFailure):
            icp_guard.require_local_network(icp, StubFailure)

    def test_state_changing_commands_are_guarded_once(self) -> None:
        icp = StubIcp("local", LOCAL)
        icp.failure_type = StubFailure
        icp._local_network_checked = False
        for args in (["token", "transfer", "100", "p", "-e", "local"],
                     ["cycles", "mint", "--cycles", "100t", "-e", "local"],
                     ["canister", "install", "x", "-e", "local", "-m", "reinstall"],
                     ["canister", "delete", "x", "-e", "local"],
                     ["canister", "settings", "update", "x", "-e", "local"],
                     ["canister", "snapshot", "create", "x", "-e", "local"],
                     ["canister", "call", "x", "info", "-e", "local"]):
            icp_guard.guard_command(icp, args)
        # One probe, not six: the decision is memoised per Icp instance.
        self.assertEqual(len(icp.calls), 1)
        self.assertEqual(icp.calls[0][:2], ["network", "status"])

    def test_read_only_commands_do_not_probe(self) -> None:
        icp = StubIcp("local", LOCAL)
        for args in (["network", "status", "-e", "local"],
                     ["cycles", "balance", "-e", "local"],
                     ["identity", "list", "--json"],
                     ["canister", "status", "x", "-e", "local", "--json"]):
            icp_guard.guard_command(icp, args)
        self.assertEqual(icp.calls, [])

    def test_remote_environment_refuses_before_the_probe_repeats(self) -> None:
        icp = StubIcp("ic", REMOTE)
        icp.failure_type = StubFailure
        icp._local_network_checked = False
        with self.assertRaises(StubFailure):
            icp_guard.guard_command(icp, ["canister", "install", "x", "-e", "ic", "-m", "reinstall"])


class ReplicaUrlTests(unittest.TestCase):
    def test_loopback_urls_are_allowed(self) -> None:
        for url in ("http://127.0.0.1:8000", "http://localhost:4943", "http://[::1]:8000"):
            self.assertIsNone(icp_guard.replica_violation(url), url)

    def test_non_loopback_urls_are_refused(self) -> None:
        for url in ("https://ic0.app", "https://boundary.example", "", "ftp://127.0.0.1"):
            self.assertIsNotNone(icp_guard.replica_violation(url), url)
        with self.assertRaises(StubFailure):
            icp_guard.require_local_replica("https://ic0.app", StubFailure)


class IcpRunnerIntegrationTests(unittest.TestCase):
    """The guard must sit inside `Icp.run`, not only in a caller's pre-flight line."""

    def _icp(self):
        # verdict_canister's Icp shells out; subprocess is replaced below.
        return verdict_canister.Icp("local", "ic-verdict-local", Path("/tmp/ic-laya-test-home"))

    def test_transfer_is_never_executed_against_a_remote_environment(self) -> None:
        recorded: list[list[str]] = []
        real_run = subprocess.run

        def fake_run(command, **kwargs):
            recorded.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=REMOTE, stderr="")

        subprocess.run = fake_run
        try:
            icp = self._icp()
            with self.assertRaises(verdict_canister.Failure):
                icp.run(["token", "transfer", "100", "payer", "-e", "local"])
        finally:
            subprocess.run = real_run

        self.assertEqual(len(recorded), 1, recorded)
        self.assertIn("network", recorded[0])
        self.assertEqual([c for c in recorded if "transfer" in c], [])

    def test_transfer_runs_against_a_managed_local_replica(self) -> None:
        recorded: list[list[str]] = []

        def fake_run(command, **kwargs):
            recorded.append(list(command))
            if command[1:3] == ["network", "status"]:
                return subprocess.CompletedProcess(command, 0, stdout=LOCAL, stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

        real_run = subprocess.run
        subprocess.run = fake_run
        try:
            icp = self._icp()
            self.assertEqual(icp.run(["token", "transfer", "100", "payer", "-e", "local"]), "ok")
        finally:
            subprocess.run = real_run
        self.assertEqual(len([c for c in recorded if "transfer" in c]), 1, recorded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
