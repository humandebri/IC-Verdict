#!/usr/bin/env python3
"""Drive the `verdict-engine` canister on a local replica.

Creates the canister if needed, installs `build/verdict-engine.wasm`, uploads a
pack and warms it up through `verdict-upload` (an agent speaking the ingress
protocol, because the CLI cannot carry a 600 MiB pack), then runs one inference
and prints the measured instruction count. Everything is local: the icp-cli state
lives in this repository, cycles are minted on the replica, and no mainnet
endpoint is ever reached. That is enforced, not asserted: `tools/icp_guard.py`
refuses a non-managed environment and a non-loopback `--replica` before any
transfer, mint or install.

    python3 tools/verdict_canister.py --pack fixtures/verdict-tiny \
        --tokenizer fixtures/verdict-tiny/tokenizer.json --ids 1,3,11,3,12,2

`--skip-upload` reuses a canister that is already warm, which is how a run against
the real 605 MiB pack is resumed without pushing the bytes again.

Two identities are in play, and they are deliberately different:
  * the icp-cli identity pays for the canister and is its controller;
  * an Ed25519 key written to `--home` is the canister's *owner*, and is the only
    identity allowed to upload a pack or call `infer_tokens`.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from icp_guard import guard_command, require_local_network, require_local_replica

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
UPLOADER = ROOT / "target" / "debug" / "verdict-upload"
SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]
# PKCS#8 v1 Ed25519: SEQUENCE { INTEGER 0, SEQUENCE { OID 1.3.101.112 }, OCTET STRING { seed } }
ED25519_PKCS8_PREFIX = "302e020100300506032b657004220420"


class Failure(Exception):
    pass


class Icp:
    """Thin icp-cli wrapper; every path it touches stays inside the repository."""

    def __init__(self, env: str, identity: str, home: Path) -> None:
        self.env = env
        self.identity = identity
        self.home = home
        self.did = str(BUILD / "verdict-engine.did")
        # `Failure` so the guard raises the exception this script already handles, and
        # the memo so `guard_command` shells out to `network status` at most once.
        self.failure_type = Failure
        self._local_network_checked = False

    def run(self, args: list[str], expect_ok: bool = True, identity: bool = True) -> str:
        # Refuse state-changing commands against a real network before the subprocess
        # starts: `token transfer`, `cycles mint`, `canister create/install/top-up/call`.
        # Read-only commands (network status, cycles balance, identity list) pass.
        guard_command(self, args)
        command = ["icp", *args, "--project-root-override", str(ROOT)]
        if identity and args[0] not in {"network", "identity"}:
            command += ["--identity", self.identity]
        environment = dict(os.environ)
        # icp-cli keeps settings, identities and caches under $HOME/Library, so a
        # repo-local HOME makes a run self-contained and trivially reversible.
        environment["HOME"] = str(self.home)
        environment["DO_NOT_TRACK"] = "1"
        completed = subprocess.run(command, capture_output=True, text=True, timeout=3600,
                                   env=environment)
        output = completed.stdout + completed.stderr
        if expect_ok and completed.returncode != 0:
            raise Failure(f"{' '.join(command)}\n{output}")
        return output

    def call(self, method: str, args: str = "()", expect_ok: bool = True) -> str:
        return self.run(["canister", "call", "verdict-engine", method, args,
                         "-e", self.env, "--candid", self.did], expect_ok=expect_ok)


def identity_principal(icp: Icp) -> str:
    listing = icp.run(["identity", "list", "--json"], identity=False)
    start = listing.find("{")
    if start < 0:
        raise Failure(f"could not read identity list:\n{listing}")
    for entry in json.loads(listing[start:])["identities"]:
        if entry["name"] == icp.identity:
            return entry["principal"]
    raise Failure(f"identity {icp.identity} is missing")


def ensure_identity(icp: Icp) -> str:
    try:
        return identity_principal(icp)
    except Failure:
        pass
    icp.run(["identity", "new", icp.identity, "--storage", "plaintext", "-q"], identity=False)
    return identity_principal(icp)


def canister_principal(icp: Icp) -> str | None:
    raw = icp.run(["canister", "status", "verdict-engine", "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:]).get("id")
    except ValueError:
        return None


def ensure_cycles(icp: Icp, principal_name: str) -> None:
    """Fund the test identity on a local replica.

    A local replica pre-funds the *anonymous* identity, not the one this script
    uses, so the first run moves ICP from that faucet to ours and only then
    converts it to cycles. These are a real `icp token transfer` and a real
    `icp cycles mint`, so `icp_guard` refuses to reach this function unless the
    environment is a locally launched replica (`managed: true`). No `icp.yaml`
    mapping is involved: icp-cli's own `managed` flag is the discriminator.
    """
    balance = icp.run(["cycles", "balance", "-e", icp.env], expect_ok=False)
    match = re.search(r"([\d_]+) cycles", balance)
    if match and int(match.group(1).replace("_", "")) > 0:
        return
    print("funding the test identity from the local anonymous faucet, then minting cycles")
    icp.run(["token", "transfer", "100", principal_name, "-e", icp.env], identity=False)
    icp.run(["cycles", "mint", "--cycles", "100t", "-e", icp.env])


def special_ids(meta: dict, tokenizer: bytes) -> dict[str, int]:
    """cls/sep come from the pack config; mask/pad are read off the tokenizer, so a
    fixture with its own small id space works unchanged."""
    config = meta["config"]
    added = {t["content"]: t["id"] for t in json.loads(tokenizer)["added_tokens"]}
    return {"cls": config["cls_token_id"], "sep": config["sep_token_id"],
            "mask": added["[MASK]"], "pad": added["[PAD]"]}


def owner_pem(home: Path) -> Path:
    """A dedicated Ed25519 owner key.

    icp-cli's identities are secp256k1 PKCS#8 v2, which ic-agent's `BasicIdentity`
    does not parse. An Ed25519 PKCS#8 v1 key is only a SEQUENCE around the 32-byte
    seed, so it is written directly, and the same key is the canister owner and the
    uploader.
    """
    path = home / "verdict-owner.pem"
    if not path.exists():
        der = bytes.fromhex(ED25519_PKCS8_PREFIX) + os.urandom(32)
        body = base64.b64encode(der).decode()
        wrapped = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
        # 0600, created with O_EXCL: this key is the canister's only owner (pack upload and
        # infer_tokens), and `write_text` would have left it 0644 under the usual umask.
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"-----BEGIN PRIVATE KEY-----\n{wrapped}\n-----END PRIVATE KEY-----\n")
    else:
        # A key written by an earlier version keeps the mode it was created with.
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return path


def owner_principal(pem: Path) -> str:
    out = subprocess.run([str(UPLOADER), "--pem", str(pem), "--print-principal"],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise Failure(f"verdict-upload --print-principal: {out.stdout}{out.stderr}")
    return out.stdout.strip().splitlines()[-1].strip()


def upload_pack(icp: Icp, args: argparse.Namespace, canister: str) -> None:
    # Ingress bytes are charged to the receiving canister at 2,000 cycles/byte, so a
    # 600 MiB pack costs more than 1 T cycles before any execution.
    if args.top_up:
        icp.run(["canister", "top-up", "verdict-engine", "--amount", args.top_up, "-e", icp.env])
    ids = special_ids(json.loads((args.pack / "manifest.json").read_bytes()),
                      args.tokenizer.read_bytes())
    command = [str(UPLOADER), "--url", args.replica, "--canister", canister,
               "--pem", str(args.owner_pem), "--pack", str(args.pack), "--tokenizer", str(args.tokenizer),
               "--cls", str(ids["cls"]), "--sep", str(ids["sep"]),
               "--mask", str(ids["mask"]), "--pad", str(ids["pad"]),
               "--chunk", str(args.chunk)]
    print("uploading pack through the agent ...")
    if subprocess.run(command, text=True, timeout=7200).returncode != 0:
        raise Failure("verdict-upload failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ids", default="1,3,11,3,12,2",
                        help="token ids for infer_tokens. The default fits fixtures/verdict-tiny only: "
                             "against the real pack the canister rejects it with Invalid(\"no class tokens\"), "
                             "because those ids contain no <<LABEL>> (50368). Use tools/measure_verdict.py "
                             "to derive ids from a real case.")
    parser.add_argument("--env", default="local")
    parser.add_argument("--replica", default="", help="gateway URL; defaults to the running network's api_url")
    parser.add_argument("--chunk", type=int, default=1024 * 1024, help="upload chunk size (canister caps at 1 MiB)")
    parser.add_argument("--identity", default="ic-verdict-local")
    parser.add_argument("--home", type=Path, default=ROOT / ".icphome")
    parser.add_argument("--top-up", default="30t", help="cycles to add before an upload; empty disables")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--keep", action="store_true", help="leave the local network running")
    parser.add_argument("--decide", action="store_true", help="also run one decide() request")
    parser.add_argument("--query", action="store_true",
                        help="run the smoke inference through infer_tokens_query (5B budget) instead of "
                             "the update path; the ids must then be short enough for the query ceiling")
    args = parser.parse_args()

    if not (BUILD / "verdict-engine.wasm").exists():
        raise Failure("missing build/verdict-engine.wasm; run bash tools/build_one.sh verdict-engine")
    if not UPLOADER.exists():
        raise Failure(f"{UPLOADER} is missing; run: cargo build -p verdict-upload")

    args.home.mkdir(parents=True, exist_ok=True, mode=0o700)
    icp = Icp(args.env, args.identity, args.home)
    # Start the replica before demanding proof that it is local: `network start` is
    # not a guarded command, while everything past this point mints cycles and
    # `-m reinstall`s a canister, which wipes its state irreversibly. Checking first
    # made the documented cold start impossible -- with nothing running, `network
    # status` is unreadable and the fail-closed guard refused before the start.
    status = icp.run(["network", "status", "-e", args.env, "--json"], expect_ok=False)
    started_here = "api_url" not in status
    if started_here:
        print("starting the local network ...")
        icp.run(["network", "start", "-d", "-e", args.env])
        status = icp.run(["network", "status", "-e", args.env, "--json"])
    require_local_network(icp, Failure)
    match = re.search(r'"api_url":\s*"([^"]+)"', status)
    args.replica = args.replica or (match.group(1) if match else "")
    # `--env local` with `--replica https://ic0.app` would otherwise pass the
    # environment check and then hand 600 MiB to a real gateway.
    require_local_replica(args.replica, Failure)
    print(f"  network: {args.replica}")
    try:
        return deploy(icp, args)
    finally:
        if started_here and not args.keep:
            print("stopping the local network ...")
            icp.run(["network", "stop", "-e", args.env], expect_ok=False)


def deploy(icp: Icp, args: argparse.Namespace) -> int:
    payer = ensure_identity(icp)
    print(f"payer identity (cycles/controller): {icp.identity} ({payer})")
    ensure_cycles(icp, payer)
    args.owner_pem = owner_pem(args.home)
    owner = owner_principal(args.owner_pem)
    print(f"canister owner (uploads and infer_tokens): {owner}")

    if canister_principal(icp) is None:
        icp.run(["canister", "create", "verdict-engine", "-e", icp.env, "-q"])
    canister = canister_principal(icp)
    if canister is None:
        raise Failure("canister has no principal after create")
    print(f"  canister: {canister}")

    print("installing ...")
    icp.run(["canister", "install", "verdict-engine", "-e", icp.env, "-y", "-m", "reinstall",
             "--wasm", str(BUILD / "verdict-engine.wasm"), "--args", f'(principal "{owner}")'])

    if not args.skip_upload:
        upload_pack(icp, args, canister)

    print("info:", " ".join(icp.call("info").split())[:300])

    # The measured inference is made by the owner key. The payer is allowlisted in
    # the same invocation so decide() can also be issued from the CLI below.
    method = "infer_tokens_query" if args.query else "infer_tokens"
    print(f"{method} {args.ids} ...")
    command = [str(UPLOADER), "--url", args.replica, "--canister", canister,
               "--pem", str(args.owner_pem), "--no-upload",
               "--allow-caller", payer, "--query-infer" if args.query else "--infer", args.ids]
    if subprocess.run(command, text=True, timeout=1800).returncode != 0:
        raise Failure("verdict-upload inference failed")

    if args.decide:
        print("decide ...")
        record = ("record { state = %s; question = %s; options = vec { record { id = \"a\"; text = %s }; "
                  "record { id = \"b\"; text = %s } }; abstention = true; temperature = 1.4265148639678955 }"
                  % (json.dumps("I lost my wallet yesterday and need to stop my debit card immediately."),
                     json.dumps("What is the primary customer inquiry?"),
                     json.dumps("Reporting a lost or stolen card"),
                     json.dumps("Disputing an unrecognized charge")))
        reply = icp.call("decide", "(%s)" % record)
        print(" ", " ".join(reply.split())[:900])
        if "Err" in reply:
            raise Failure("decide returned an error")
    print("\nVERDICT ENGINE LOCAL RUN OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)
