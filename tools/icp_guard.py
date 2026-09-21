"""Fail-closed refusal to touch anything that is not a locally launched replica.

Why this exists as one module: `icp canister create/install/top-up/call`,
`icp token transfer` and `icp cycles mint` are irreversible or spend real value.
Pointed at a connected network they do exactly that, and a fresh `--identity`
means the operator would not even notice their own identity was not used. The
scripts that mint cycles or `-m reinstall` canisters (`verdict_canister.py`,
`measure_verdict.py`, `local_integration.py`) all route through this check.

`managed: true` is the only reliable local/remote discriminator -- icp-cli sets it
for an environment it launched itself. There is deliberately no override flag:
`tools/verdict_canister.py --env <mainnet>` must fail, not warn.

The `failure_type` argument lets each script raise its own `Failure` class so its
existing `except Failure` handler keeps working; nothing here imports a script.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

# Loopback hosts a locally launched replica may listen on. `127.0.0.1` is what
# icp-cli reports for the managed network; the others cover an explicit --replica.
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
# `icp <group> <verb>` pairs that can spend ICP/cycles or overwrite canister state.
STATE_CHANGING = {
    # Every `icp canister` verb that writes state, spends cycles or moves the canister
    # (verified against `icp canister --help`): `list`, `logs`, `metadata` and `status`
    # are read-only and stay unguarded.
    ("canister", "call"),
    ("canister", "create"),
    ("canister", "delete"),
    ("canister", "install"),
    ("canister", "migrate-id"),
    ("canister", "settings"),
    ("canister", "snapshot"),
    ("canister", "start"),
    ("canister", "stop"),
    ("canister", "top-up"),
    ("token", "transfer"),
    ("cycles", "mint"),
}


def any_network_status(icp) -> dict | None:
    """Return the configured environment's status, local or not.

    Read-only: uses `network status`, which is never guarded, so this can be called
    from inside the guard itself.
    """
    raw = icp.run(["network", "status", "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except ValueError:
        return None


def local_network_violation(icp) -> str | None:
    """Explain why `icp.env` must not be used, or None when it is a local replica.

    Fails closed: an unreadable or unparsable `network status` is not evidence of a
    local replica. Treating it as one (the previous behaviour) meant that a change in
    the CLI's output format silently disarmed the only barrier in front of
    `token transfer`, `cycles mint` and `canister install`. `network start` is not a
    guarded command, so a stopped replica can still be started.
    """
    status = any_network_status(icp)
    if status is None:
        return (f"cannot confirm that environment '{icp.env}' is a locally launched replica: "
                f"`network status --json` was unreadable or did not parse. State-changing commands "
                f"are refused rather than run against an unverified network.")
    if not status.get("managed"):
        return (f"environment '{icp.env}' points at {status.get('api_url')}, which is not a local "
                f"network. This tool mints cycles, installs canisters, and wipes state, so it only "
                f"runs against a locally launched replica.")
    return None


def require_local_network(icp, failure_type: type[Exception] = RuntimeError) -> None:
    """Refuse anything that could touch a real network."""
    violation = local_network_violation(icp)
    if violation:
        raise failure_type(violation)


def replica_violation(url: str) -> str | None:
    """Explain why `url` is not a local replica gateway, or None when it is.

    Separate from `require_local_network` on purpose: `--replica` is passed straight
    to ic-agent, so `--env local --replica https://ic0.app` would otherwise satisfy
    the environment check and then upload 600 MiB to a real network.
    """
    if not url:
        return "no gateway URL: refusing to talk to an unknown endpoint"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return f"gateway URL {url!r} is not http(s)"
    host = parsed.hostname or ""
    if host not in LOOPBACK_HOSTS:
        return (f"gateway URL {url!r} is not a loopback address. This tool uploads packs and "
                f"issues inference calls, so it only runs against a local replica.")
    return None


def require_local_replica(url: str, failure_type: type[Exception] = RuntimeError) -> None:
    violation = replica_violation(url)
    if violation:
        raise failure_type(violation)


def guard_command(icp, args: list[str]) -> None:
    """Guard state-changing icp invocations once per `Icp` instance.

    Called from `Icp.run` so that *every* path -- including a future script that
    forgets its own pre-flight call -- is covered. Read-only commands and the
    guard's own `network status` pass through.
    """
    if len(args) < 2 or (args[0], args[1]) not in STATE_CHANGING:
        return
    if getattr(icp, "_local_network_checked", False):
        return
    require_local_network(icp, getattr(icp, "failure_type", RuntimeError))
    icp._local_network_checked = True
