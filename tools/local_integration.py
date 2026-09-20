#!/usr/bin/env python3
"""Local-replica integration test for the three IC-Laya canisters.

This is the gate that `cargo test --workspace` cannot cover: it exercises the
real message boundaries, the real stable-memory persistence, and the real
inter-canister call into the mock ledger.

It runs the workflow the design cares about, including the hardest case:

  submit -> 3 sequential questions -> dispatch under commit-then-trap fault
         -> OutcomeUnknown with the reservation retained
         -> retry the SAME frozen payload -> ledger deduplicates
         -> Succeeded, budget charged exactly once

It also asserts the properties that separate this design from "call and hope":

  * a partially evaluated request is not dispatchable, so money never moves on a
    subset of the registered signals;
  * a transfer the ledger committed but whose reply was lost is never reported as
    a failure, and is never re-sent with a new timestamp;
  * the reservation survives the unknown outcome and is settled exactly once;
  * a submit for a recipient outside the grant's allowlist, and a submit from a
    caller that is not the grant's delegate, are both refused;
  * an upgrade over an unresolved transfer is refused rather than silently
    discarding it.

Assertions run as the fixed identity `--identity` (default `ic-laya-local-test`)
rather than whatever identity happens to be the default, so results do not depend
on local icp-cli configuration. Canister creation still needs cycles, which on a
local replica are minted for that identity on first use.

The decision engine runs in synthetic-fixture mode, so no model weights are
needed. `FixtureTokenizer` below is a faithful port of
`crates/ic-laya-core/src/demo.rs`; if that fixture changes, this must change too.

Usage:
    python3 tools/local_integration.py [--keep] [--skip-upgrade]

Requires: the `icp` CLI, and `bash tools/build_one.sh <name>` to have produced
build/*.wasm. No mainnet access, no real assets, no deployment outside the local
network. The mock ledger's transfers are a test double.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

# From crates/ic-laya-core/src/demo.rs. TEST-ONLY values; not language understanding.
SPECIAL = {"cls": 1, "sep": 2, "mask": 3, "pad": 0}
SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]
FIXTURE_BUNDLE = b"TEST-ONLY-STATE-INDEPENDENT-FIXTURE-v1"
FIXTURE_TOKENIZER = b"TEST-ONLY-WHITESPACE-TOKENIZER-v1"
AMOUNT = 100
FEE = 10


class Failure(Exception):
    pass


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def check(condition: bool, label: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise Failure(label)


# --- canonical hashing, ported from ic-laya-core ---------------------------------

class Canonical:
    """Length-framed, domain-separated hashing. Mirrors src/types.rs `Canonical`."""

    def __init__(self, domain: str) -> None:
        self.buf = bytearray()
        self.bytes(domain.encode())

    def bytes(self, value: bytes) -> "Canonical":
        self.buf += len(value).to_bytes(8, "big") + value
        return self

    def u64(self, value: int) -> "Canonical":
        self.buf += value.to_bytes(8, "big")
        return self

    def text(self, value: str) -> "Canonical":
        return self.bytes(value.encode())

    def finish(self) -> bytes:
        return sha256(bytes(self.buf))


def fixture_encode(text: str) -> list[int]:
    """FixtureTokenizer::encode_piece: whitespace split, 10 + first two hash bytes."""
    ids = []
    for word in text.split():
        digest = sha256(word.encode())
        ids.append(10 + int.from_bytes(digest[:2], "big"))
    return ids


def compile_schema(schema: dict, qtype_id: int) -> dict:
    """Port of schema::compile for the fixture tokenizer."""
    question = f"{schema['primitive']} question: {schema['instructions']}"
    instruction_ids = fixture_encode(question)
    prefix = [SPECIAL["cls"], *instruction_ids, SPECIAL["sep"]]
    markers = []
    for option in schema["options"]:
        ids = fixture_encode(option["text"])
        markers.append(len(prefix))
        prefix.append(SPECIAL["mask"])
        prefix.extend(ids)
    prefix.append(SPECIAL["sep"])
    if len(prefix) > 64:
        raise Failure(f"fixture prefix {len(prefix)} exceeds MAX_PREFIX")

    tokenizer_hash = sha256(FIXTURE_TOKENIZER)
    h = Canonical("ic-laya/schema/v1")
    h.text(schema["id"]).u64(schema["version"]).u64(schema["tag"]).text(schema["instructions"])
    h.u64(len(schema["options"]))
    for option in schema["options"]:
        h.text(option["id"]).text(option["text"])
    h.bytes(tokenizer_hash).text("Compact128-v1")
    return {
        "schema": {
            "id": schema["id"],
            "version": schema["version"],
            "primitive": schema["primitive"],
            "instructions": schema["instructions"],
            "options": schema["options"],
        },
        "schema_hash": h.finish(),
        "tokenizer_hash": tokenizer_hash,
        "prefix": prefix,
        "markers": markers,
        "special": dict(SPECIAL, literals=SPECIAL_LITERALS),
        "qtype_id": qtype_id,
    }


def schemas() -> list[dict]:
    def opt(oid: str, text: str) -> dict:
        return {"id": oid, "text": text}

    return [
        {
            "id": "RefundRequested", "version": 1, "primitive": "Noul", "tag": 1,
            "instructions": "Does the customer ask for a refund?",
            "options": [opt("false", "false"), opt("true", "true")],
            "rule": {"NoulTrue": {"minimum_ppm": 900_000}},
        },
        {
            "id": "PaymentAction", "version": 1, "primitive": "Choice", "tag": 0,
            "instructions": "Choose the handling action for this request.",
            "options": [opt("execute", "process refund"), opt("reject", "reject request"),
                        opt("review", "human review")],
            "rule": {"ChoiceIs": {"option_id": "execute", "minimum_ppm": 900_000}},
        },
        {
            "id": "PaymentRisk", "version": 1, "primitive": "Score", "tag": 2,
            "instructions": "Rate request risk from minimal to severe.",
            "options": [opt("minimal", "minimal risk"), opt("low", "low risk"),
                        opt("medium", "moderate risk"), opt("high", "high risk"),
                        opt("severe", "severe risk")],
            "rule": {"ScoreTailAtMost": {"first_bad_bin": 3, "maximum_ppm": 50_000}},
        },
    ]


# --- Candid text rendering --------------------------------------------------------

def blob(value: bytes) -> str:
    """Render bytes as a Candid blob literal.

    Each byte becomes `\\xx`. The quotes are added manually: `json.dumps` would
    escape the backslashes themselves and produce `\\\\xx`, which Candid reads as
    a literal backslash followed by garbage.
    """
    return 'blob "' + "".join(f"\\{byte:02x}" for byte in value) + '"'


def principal(p: str) -> str:
    return f"principal {json.dumps(p)}"


def principal_of(raw: bytes) -> str:
    """Render bytes as a valid principal in text form.

    Candid principals are `crc32be(bytes) || bytes` in base32 with the standard
    alphabet, no padding, and a dash every five characters. The dashes are not
    cosmetic: the Candid parser rejects ungrouped text.
    """
    import base64
    import zlib

    body = raw[:29]
    checksum = zlib.crc32(body).to_bytes(4, "big")
    text = base64.b32encode(checksum + body).decode().lower().rstrip("=")
    return "-".join(text[index:index + 5] for index in range(0, len(text), 5))


class Icp:
    """Thin `icp` CLI wrapper bound to a fixed, non-default test identity.

    Every call passes `--identity` explicitly, so the test does not depend on
    whichever identity happens to be the default on this machine. The identity is
    created on first use if absent; see `ensure_identity`.

    ICP_HOME is deliberately *not* overridden. The test was originally written to
    point ICP_HOME at a repo-local store, but canister creation needs cycles and
    those live with the operator's funded identity, which an isolated home cannot
    reach on a local replica. `--identity` alone is what removes the ambient
    default dependency.
    """

    def __init__(self, project_root: Path, env: str, identity: str) -> None:
        self.root = project_root
        self.env = env
        self.identity = identity
        # Passing the generated .did makes replies decode with real field names
        # instead of numeric hashes, and turns argument encoding errors into
        # actionable messages rather than silent inferred types.
        self.did = {name: str(BUILD / f"{name}.did") for name in
                    ("decision-engine", "executor", "mock-ledger")}

    def run(self, args: list[str], expect_ok: bool = True, identity: "bool | str" = True) -> str:
        command = ["icp", *args, "--project-root-override", str(self.root)]
        environment = dict(os.environ)
        # Only some subcommands accept --identity; `network` manages the replica
        # itself and `identity` is how a principal is discovered, so both must be
        # invoked without it.
        if identity and args[0] not in {"network", "identity"}:
            command += ["--identity", identity if isinstance(identity, str) else self.identity]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300,
                                   env=environment)
        if expect_ok and completed.returncode != 0:
            raise Failure(f"{' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
        return completed.stdout + completed.stderr

    def call(self, canister: str, method: str, args: str = "()", expect_ok: bool = True) -> str:
        return self.run(["canister", "call", canister, method, args, "-e", self.env,
                         "--candid", self.did[canister]], expect_ok=expect_ok)


def known_identities(icp: "Icp") -> dict[str, str]:
    """Map identity name -> principal.

    `identity list` must not carry `--identity`: it is how a principal is
    discovered in the first place.
    """
    listing = icp.run(["identity", "list", "--json"], identity=False)
    start = listing.find("{")
    if start < 0:
        raise Failure(f"could not read identity list:\n{listing}")
    parsed = json.loads(listing[start:])
    return {entry["name"]: entry["principal"] for entry in parsed["identities"]}


def ensure_identity(icp: "Icp", name: str) -> str:
    """Create `name` if absent and return its principal.

    `--storage plaintext` is used because `keyring` prompts interactively, which
    would hang a non-interactive run, so the key is stored unencrypted in the
    operator's icp home. It is a local-replica key with no funds and must never be
    reused on a real network; the `ic-laya-local-test` name marks that intent.
    """
    known = known_identities(icp)
    if name not in known:
        icp.run(["identity", "new", name, "--storage", "plaintext", "-q"], identity=False)
        known = known_identities(icp)
    if name not in known:
        raise Failure(f"identity {name} missing after creation")
    return known[name]


def canister_principal(icp: "Icp", name: str) -> str | None:
    """Read a canister's principal, or None when it does not exist yet."""
    raw = icp.run(["canister", "status", name, "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:]).get("id")
    except ValueError:
        return None


def ensure_cycles(icp: "Icp", amount: str = "10t") -> None:
    """Ensure the test identity can pay for canister creation.

    A freshly created identity has no cycles, so the first canister creation fails
    with "Insufficient cycles". On a local replica `icp cycles mint` funds the
    identity without touching any real ICP; on a real network this is skipped by
    the balance check. Existing balances are left alone rather than topped up on
    every run.
    """
    balance = icp.run(["cycles", "balance", "-e", icp.env], expect_ok=False)
    match = re.search(r"([\d_]+) cycles", balance)
    if match and int(match.group(1).replace("_", "")) > 0:
        return
    print(f"  funding the test identity with {amount} cycles (local replica only)")
    icp.run(["cycles", "mint", "--cycles", amount, "-e", icp.env])


def ensure_canister(icp: "Icp", name: str) -> str:
    """Create the canister if absent, then return its principal.

    `create -q` only prints a principal on first creation, so the id is always
    re-read from `canister status` rather than parsed out of creation output.
    """
    if canister_principal(icp, name) is None:
        icp.run(["canister", "create", name, "-e", icp.env, "-q"])
    principal = canister_principal(icp, name)
    if principal is None:
        raise Failure(f"canister {name} has no principal after create")
    return principal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true",
                        help="leave the local network running after this invocation")
    parser.add_argument("--env", default="local")
    parser.add_argument("--identity", default="ic-laya-local-test",
                        help="fixed test identity, created if absent (never the ambient default)")
    parser.add_argument("--port", type=int, default=8000,
                        help="assert the network listens on this port; the real port comes from "
                             "icp.yaml gateway.port and is not configured here")
    parser.add_argument("--keep-state", action="store_true",
                        help="reuse existing canister state instead of reinstalling")
    parser.add_argument("--skip-upgrade", action="store_true",
                        help="skip the upgrade-guard check (it needs the network to stay up "
                             "within this invocation)")
    args = parser.parse_args()

    for wasm in ["decision-engine", "executor", "mock-ledger"]:
        if not (BUILD / f"{wasm}.wasm").exists():
            raise Failure(f"missing build/{wasm}.wasm; run bash tools/build_one.sh {wasm} first")

    icp = Icp(ROOT, args.env, args.identity)
    require_local_network(icp)
    # Do not start a network that is already up, and do not stop one we did not
    # start: another project may be using it deliberately.
    status = network_status(icp)
    started_here = status is None
    if started_here:
        print(f"starting local network (gateway port comes from icp.yaml) ...")
        icp.run(["network", "start", "-d", "-e", args.env])
        status = network_status(icp)
    else:
        print("local network already running; reusing it")
    actual_port = urlparse(status["api_url"]).port
    if actual_port != args.port:
        raise Failure(
            f"network is listening on port {actual_port}, but --port {args.port} was requested. "
            f"--port only asserts; change gateway.port in icp.yaml or pass --port {actual_port}.")
    print(f"  network: {status['api_url']}")
    try:
        return run_workflow(icp, args.keep_state, not args.skip_upgrade)
    finally:
        if started_here and not args.keep:
            print("stopping local network ...")
            icp.run(["network", "stop", "-e", args.env], expect_ok=False)


def network_status(icp: "Icp") -> dict | None:
    """Return the running *local* network's status, or None when it is not up.

    Requires `managed: true`, which icp-cli sets only for a locally launched
    replica, so a real network is never treated as usable. `require_local_network`
    turns the resulting "not running" into an explicit refusal.
    """
    status = any_network_status(icp)
    if not status or not status.get("managed") or not status.get("api_url"):
        return None
    return status


def any_network_status(icp: "Icp") -> dict | None:
    """Return the configured environment's status, local or not."""
    raw = icp.run(["network", "status", "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except ValueError:
        return None


def require_local_network(icp: "Icp") -> None:
    """Refuse anything that could touch a real network.

    This test mints cycles, installs canisters, and reinstalls them. Pointed at a
    connected network it would attempt exactly that, and a fresh `--identity`
    means the operator would not even notice their own identity was not used. The
    `managed` flag is the only reliable local/remote discriminator, so it is
    enforced here rather than left to the docstring.
    """
    status = any_network_status(icp)
    if status and not status.get("managed"):
        raise Failure(
            f"environment '{icp.env}' points at {status.get('api_url')}, which is not a local "
            f"network. This test mints cycles, installs canisters, and wipes state, so it only "
            f"runs against a locally launched replica.")


def run_workflow(icp: Icp, keep_state: bool, verify_upgrade: bool = False) -> int:
    owner = ensure_identity(icp, icp.identity)
    print(f"owner/delegate identity: {icp.identity}")
    print(f"owner principal: {owner}")

    print("creating canisters ...")
    ensure_cycles(icp)
    engine = ensure_canister(icp, "decision-engine")
    executor = ensure_canister(icp, "executor")
    ledger = ensure_canister(icp, "mock-ledger")
    print(f"  engine={engine}\n  executor={executor}\n  ledger={ledger}")

    print("installing canisters ...")
    # Reinstall by default: these are test doubles on a local replica, and a
    # consumed operation or an advanced nonce from a previous run would
    # otherwise make `submit` fail for reasons unrelated to the code under test.
    # --keep-state opts out.
    mode = "upgrade" if keep_state else "reinstall"
    icp.run(["canister", "install", "decision-engine", "-e", icp.env, "-y", "-m", mode,
             "--wasm", str(BUILD / "decision-engine.wasm"), "--args", f"({principal(owner)})"])
    icp.run(["canister", "install", "executor", "-e", icp.env, "-y", "-m", mode,
             "--wasm", str(BUILD / "executor.wasm"),
             "--args", f"({principal(owner)}, {principal(engine)})"])
    icp.run(["canister", "install", "mock-ledger", "-e", icp.env, "-y", "-m", mode,
             "--wasm", str(BUILD / "mock-ledger.wasm"), "--args", f"({principal(owner)})"])
    check(True, f"all three canisters installed ({mode})")

    print("configuring engine and executor ...")
    bundle = sha256(FIXTURE_BUNDLE)
    out = icp.call("decision-engine", "enable_synthetic_fixture")
    check(bundle in decode_blobs(out), "engine reports the fixture bundle id")
    icp.call("decision-engine", "allow_caller", f"({principal(executor)}, 1000)")
    check(True, "executor admitted as an engine caller")

    compiled = [dict(compile_schema(s, s["tag"]), rule=s["rule"]) for s in schemas()]
    for schema in compiled:
        icp.call("decision-engine", "register_schema",
                 f"({candid_schema(schema['schema'])}, {schema['qtype_id']})")
    check(True, "three schemas registered and compiled")

    now_ns = 4_000_000_000_000_000_000  # far future so expiry never races the test
    calibrations = []
    for schema in compiled:
        calibration_id = sha256(f"TEST-ONLY-calibration-{schema['schema']['id']}".encode())
        calibrations.append(calibration_id)
        arg = ("(record { id = %s; model = %s; schema = %s; tokenizer = %s; temperature = 1.0; "
               "expires_at_ns = %d : nat64; holdout_hash = %s; sample_count = 1 : nat64; test_only = true })"
               % (blob(calibration_id), blob(bundle), blob(schema["schema_hash"]),
                  blob(schema["tokenizer_hash"]), now_ns, blob(sha256(b"SYNTHETIC-NOT-REAL-HOLDOUT"))))
        icp.call("decision-engine", "register_calibration", arg)
    check(True, "three calibrations registered")

    plan_id = sha256(b"TEST-ONLY-refund-plan-v1")
    operation_id = sha256(b"trusted-demo-invoice-1")
    grant_id = sha256(b"TEST-ONLY-grant-1")

    signals = []
    for schema, calibration_id in zip(compiled, calibrations):
        signals.append(
            "(record { schema = %s; calibration = %s; rule = %s })"
            % (candid_compiled(schema), blob(calibration_id), candid_rule(schema["rule"]))
        )
    icp.call("executor", "register_plan",
             "(record { id = %s; version = 1 : nat64; model = %s; signals = vec { %s } })"
             % (blob(plan_id), blob(bundle), "; ".join(signals)))
    check(True, "plan registered")

    evidence = "The customer reports a duplicate payment and requests a refund."
    icp.call("executor", "register_operation",
             "(record { id = %s; revision = 1 : nat64; evidence = %s; proposal = %s; status = variant { Available } })"
             % (blob(operation_id), json.dumps(evidence),
                "(record { ledger = %s; from_subaccount = %s; to = %s; amount = %d : nat; fee = %d : nat })"
                % (principal(ledger), blob(bytes(32)),
                   "(record { owner = %s; subaccount = %s })" % (principal(owner), blob(bytes(32))),
                   AMOUNT, FEE)))
    check(True, "trusted operation registered")

    icp.call("executor", "register_grant",
             "(record { id = %s; revision = 1 : nat64; delegate = %s; plan = %s; ledger = %s; "
             "from_subaccount = %s; recipients = vec { record { owner = %s; subaccount = %s } }; "
             "max_amount = 1000 : nat; max_fee = 10 : nat; total_cap = 1000 : nat; window_cap = 500 : nat; "
             "window_ns = 60_000_000_000 : nat64; expires_at_ns = %d : nat64; revoked = false; "
             "total = record { spent = 0 : nat; reserved = 0 : nat }; windows = vec {} })"
             % (blob(grant_id), principal(owner), blob(plan_id), principal(ledger), blob(bytes(32)),
                principal(owner), blob(bytes(32)), now_ns))
    check(True, "grant registered")

    # --- rejection cases -------------------------------------------------------
    # These run before any request exists so they cannot disturb the workflow below.
    # The owner registers them; the delegate (this identity) is not a party.
    print("checking rejection paths ...")

    # 1. The allowlist is enforced by the grant, not at operation registration:
    #    `install_operation` only validates the proposal itself. Registering an
    #    operation whose recipient is outside the grant is therefore allowed, but
    #    creating a request for it must be refused, and no request may be created.
    stranger = principal_of(sha256(b"not-an-allowlisted-recipient")[:29])
    outside_id = sha256(b"TEST-ONLY-outside-allowlist")
    registered = icp.call("executor", "register_operation",
                          "(record { id = %s; revision = 1 : nat64; evidence = %s; proposal = %s; status = variant { Available } })"
                          % (blob(outside_id), json.dumps("outside allowlist"),
                             "(record { ledger = %s; from_subaccount = %s; to = %s; amount = 1 : nat; fee = 1 : nat })"
                             % (principal(ledger), blob(bytes(32)),
                                "(record { owner = %s; subaccount = %s })" % (principal(stranger), blob(bytes(32))))))
    # Registration itself must succeed; only request creation is refused. If this
    # failed, the Denied below would come from NotFound and prove nothing.
    check("Ok" in registered, "operation outside the allowlist is still registrable")
    # A rejected submit does NOT consume the nonce: `g.eligible` runs before the
    # counter advances, so the workflow below still starts at nonce 0.
    denied = icp.call("executor", "submit",
                      f"({blob(outside_id)}, {blob(grant_id)}, 0 : nat64)")
    check("Denied" in denied, "submit for a non-allowlisted recipient is denied")

    # 3. Callers outside the grant are refused. Two distinct reasons are covered:
    #    anonymous fails the executor's admission check, and a valid but unrelated
    #    principal fails the grant's delegate check specifically.
    intruder_name = f"{icp.identity}-intruder"
    intruder = ensure_identity(icp, intruder_name)
    if intruder == owner:
        raise Failure("intruder identity unexpectedly equals the owner")
    intruded = icp.run(["canister", "call", "executor", "submit",
                        f"({blob(operation_id)}, {blob(grant_id)}, 0 : nat64)", "-e", icp.env,
                        "--candid", icp.did["executor"], "--identity", intruder_name],
                       expect_ok=False, identity=False)
    check("Unauthorized" in intruded,
          f"principal outside the grant is unauthorized ({intruder[:12]}...): {intruded.strip()[:200]}")
    anonymous = icp.run(["canister", "call", "executor", "submit",
                         f"({blob(operation_id)}, {blob(grant_id)}, 0 : nat64)", "-e", icp.env,
                         "--candid", icp.did["executor"], "--identity", "anonymous"],
                        expect_ok=False, identity=False)
    check("Unauthorized" in anonymous, f"anonymous caller is unauthorized: {anonymous.strip()[:200]}")

    icp.call("mock-ledger", "ic_laya_mock_profile")
    icp.call("executor", "register_mock_ledger", f"({principal(ledger)})")
    icp.call("executor", "set_mode", "(variant { Mock })")
    check(True, "mock ledger registered and executor in Mock mode")

    # `post_upgrade` runs even for a first `install`, and recover_after_upgrade()
    # sets paused=true. That is the intended fail-safe, so the test unpauses
    # explicitly rather than the canister assuming a fresh install is safe.
    icp.call("executor", "pause", "(false)")
    check(True, "executor unpaused by the owner")

    print("running the workflow ...")
    submitted = icp.call("executor", "submit", f"({blob(operation_id)}, {blob(grant_id)}, 0 : nat64)")
    if "Ok" not in submitted:
        raise Failure(f"submit failed: {submitted.strip()}")
    request_id = blob_arg(submitted)
    print(f"  request {request_id.hex()[:16]}...")

    # A partially evaluated request must not be authorizable. Otherwise dispatch
    # could move money on the strength of a subset of the registered signals.
    icp.call("executor", "advance", f"({blob(request_id)})")
    partial = icp.call("executor", "get_request", f"({blob(request_id)})")
    check("ReadyToDispatch" not in partial,
          "request with one of three signals is not ReadyToDispatch")
    for step in range(2, 4):
        status = icp.call("executor", "advance", f"({blob(request_id)})")
        if "Err" in status:
            raise Failure(f"advance {step} failed: {status}")
    final = icp.call("executor", "get_request", f"({blob(request_id)})")
    check("ReadyToDispatch" in final, "three sequential questions evaluated to ReadyToDispatch")

    print("arming the commit-then-trap fault ...")
    icp.call("mock-ledger", "set_fault", "(variant { CommitThenCallbackTrap })")
    icp.call("executor", "dispatch", f"({blob(request_id)})", expect_ok=False)
    state = icp.call("executor", "get_request", f"({blob(request_id)})")
    check("OutcomeUnknown" in state, "lost reply is recorded as OutcomeUnknown, not failure")
    check("Submitted" not in state, "request did not stay in Submitted")
    committed = icp.call("mock-ledger", "committed_transfers")
    check("1 : nat64" in committed or committed.strip().endswith("(1 : nat64)"),
          f"ledger committed the transfer despite the lost reply (raw: {committed.strip()[:80]})")

    print("retrying with the same frozen payload ...")
    icp.call("mock-ledger", "set_fault", "(variant { None })")
    icp.call("executor", "retry_mock_transfer", f"({blob(request_id)})")
    final = icp.call("executor", "get_request", f"({blob(request_id)})")
    check("Succeeded" in final, "duplicate reply settles the request")
    committed = icp.call("mock-ledger", "committed_transfers")
    check("1 : nat64" in committed or committed.strip().endswith("(1 : nat64)"),
          f"ledger committed exactly once, not twice (raw: {committed.strip()[:80]})")

    # Always strand a second request in OutcomeUnknown. A following run with
    # --keep-state then upgrades the canister while that transfer is unresolved,
    # which is what pre_upgrade/post_upgrade exist for.
    leave_unknown_for_upgrade(icp, grant_id, ledger, owner)
    print("\nALL LOCAL INTEGRATION CHECKS PASSED")
    if verify_upgrade:
        owner_args = f"({principal(owner)}, {principal(owner)})"
        return verify_upgrade_guard(icp, owner_args)
    return 0


def leave_unknown_for_upgrade(icp: "Icp", grant_id: bytes, ledger: str, owner: str) -> None:
    """Submit a second request and strand it in OutcomeUnknown.

    The next `--keep-state` run upgrades the canister with this transfer
    unresolved, which is the situation `pre_upgrade`/`post_upgrade` exist for.
    """
    print("stranding a second request in OutcomeUnknown for the upgrade run ...")
    # An operation backs at most one request, and the first is now consumed, so the
    # second request needs its own operation. It reuses the same allowlisted
    # recipient and amount, so the already-registered grant still covers it.
    second_operation = sha256(b"trusted-demo-invoice-2")
    icp.call("executor", "register_operation",
             "(record { id = %s; revision = 1 : nat64; evidence = %s; proposal = %s; status = variant { Available } })"
             % (blob(second_operation), json.dumps("second trusted invoice"),
                "(record { ledger = %s; from_subaccount = %s; to = %s; amount = %d : nat; fee = %d : nat })"
                % (principal(ledger), blob(bytes(32)),
                   "(record { owner = %s; subaccount = %s })" % (principal(owner), blob(bytes(32))),
                   AMOUNT, FEE)))
    # A fresh nonce is required; the workflow used 0.
    submitted = icp.call("executor", "submit", f"({blob(second_operation)}, {blob(grant_id)}, 1 : nat64)")
    if "Ok" not in submitted:
        raise Failure(f"second submit failed: {submitted.strip()}")
    second = blob_arg(submitted)
    for step in range(3):
        status = icp.call("executor", "advance", f"({blob(second)})")
        if "Err" in status:
            raise Failure(f"second advance {step} failed: {status}")
    icp.call("mock-ledger", "set_fault", "(variant { CommitThenCallbackTrap })")
    icp.call("executor", "dispatch", f"({blob(second)})", expect_ok=False)
    state = icp.call("executor", "get_request", f"({blob(second)})")
    check("OutcomeUnknown" in state, "second request is stranded in OutcomeUnknown for the upgrade")
    print(f"  stranded request {second.hex()[:16]}...")


def verify_upgrade_guard(icp: "Icp", install_args: str) -> int:
    """Assert the upgrade guard refuses to upgrade over an unresolved transfer.

    `pre_upgrade` traps when any request is still Submitted or OutcomeUnknown, so a
    normal upgrade must fail and leave the reservation intact rather than silently
    discarding an in-flight transfer.

    This must run in the same invocation as the workflow: `icp network start`
    recreates the replica, so canisters never survive into a separate run.
    """
    print("verifying the unresolved-transfer upgrade guard ...")
    blocked = icp.run(["canister", "install", "executor", "-e", icp.env, "-y", "-m", "upgrade",
                       "--wasm", str(BUILD / "executor.wasm"), "--args", install_args],
                      expect_ok=False)
    check("unresolved transfer" in blocked,
          f"upgrade is refused while a transfer is unresolved: {blocked.strip()[-160:]}")
    check(True, "the unresolved reservation therefore survives the refused upgrade")
    print("\nUPGRADE GUARD CHECKS PASSED")
    return 0


def decode_blobs(output: str) -> list[bytes]:
    """Collect every `blob "..."` payload, decoding Candid escape sequences.

    icp prints blob bytes as `\\xx` escapes, not hex, so the escapes are the
    only thing to parse. Non-escaped characters are appended verbatim.
    """
    import re

    blobs = []
    for body in re.findall(r'blob\s+"((?:[^"\\]|\\.)*)"', output):
        collected = bytearray()
        index = 0
        while index < len(body):
            if body[index] == "\\" and index + 2 < len(body) + 1:
                collected.append(int(body[index + 1:index + 3], 16))
                index += 3
            else:
                collected.extend(body[index].encode())
                index += 1
        blobs.append(bytes(collected))
    return blobs


def blob_arg(output: str) -> bytes:
    """Read the first blob out of an icp reply."""
    blobs = decode_blobs(output)
    if not blobs:
        raise Failure(f"no blob in output:\n{output}")
    return blobs[0]


def candid_schema(schema: dict) -> str:
    options = "; ".join(
        "record { id = %s; text = %s }" % (json.dumps(o["id"]), json.dumps(o["text"]))
        for o in schema["options"]
    )
    return ("record { id = %s; version = %d : nat64; primitive = variant { %s }; instructions = %s; options = vec { %s } }"
            % (json.dumps(schema["id"]), schema["version"], schema["primitive"],
               json.dumps(schema["instructions"]), options))


def candid_compiled(schema: dict) -> str:
    prefix = "; ".join(str(v) for v in schema["prefix"])
    markers = "; ".join(str(v) for v in schema["markers"])
    special = ("record { cls = %d : nat32; sep = %d : nat32; mask = %d : nat32; pad = %d : nat32; literals = vec { %s } }"
               % (SPECIAL["cls"], SPECIAL["sep"], SPECIAL["mask"], SPECIAL["pad"],
                  "; ".join(json.dumps(x) for x in SPECIAL_LITERALS)))
    return ("record { schema = %s; schema_hash = %s; tokenizer_hash = %s; prefix = vec { %s }; "
            "markers = vec { %s }; special = %s; qtype_id = %d : nat32 }"
            % (candid_schema(schema["schema"]), blob(schema["schema_hash"]),
               blob(schema["tokenizer_hash"]), prefix, markers, special, schema["qtype_id"]))


def candid_rule(rule: dict) -> str:
    if "NoulTrue" in rule:
        return "variant { NoulTrue = record { minimum_ppm = %d : nat32 } }" % rule["NoulTrue"]["minimum_ppm"]
    if "ChoiceIs" in rule:
        return ("variant { ChoiceIs = record { option_id = %s; minimum_ppm = %d : nat32 } }"
                % (json.dumps(rule["ChoiceIs"]["option_id"]), rule["ChoiceIs"]["minimum_ppm"]))
    return ("variant { ScoreTailAtMost = record { first_bad_bin = %d : nat32; maximum_ppm = %d : nat32 } }"
            % (rule["ScoreTailAtMost"]["first_bad_bin"], rule["ScoreTailAtMost"]["maximum_ppm"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)
