#!/usr/bin/env python3
"""Record PASS / FAIL / NOT_RUN independently. Never substitute Python for Rust."""
from __future__ import annotations
import argparse,ast,datetime,importlib.metadata,json,os,re,shutil,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

# Vendored/cache trees that must never be scanned as project source. `.cargohome`
# holds the Cargo registry used by tools/verdict-upload (thousands of .rs files).
SKIP={".venv","target",".cargohome",".icphome",".icp","__pycache__"}
def skipped(path):
    return any(part in SKIP for part in path.parts)

def cargo_environment():
    """Point Cargo at the in-repo registry when the default one is not writable.

    `.cargohome/` exists so a run never writes outside the repository. That matters
    beyond tidiness: under a read-only-home sandbox, `cargo test --workspace` dies
    with `failed to open .../registry/cache/...: Operation not permitted` as soon as
    any crate is missing from the default cache, which looks like a test failure and
    is not one. A caller that has already chosen CARGO_HOME keeps it.
    """
    if os.environ.get("CARGO_HOME"):
        return {}
    default=Path.home()/".cargo"
    if os.access(default,os.W_OK) or not (ROOT/".cargohome").is_dir():
        return {}
    return {"CARGO_HOME":str(ROOT/".cargohome")}

CARGO_ENV=cargo_environment()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust",action="store_true",help="also execute cargo test/check if available")
    p.add_argument("--require-rust",action="store_true",help="return failure if Rust tests cannot run")
    p.add_argument("--local-integration",action="store_true",
                   help="also run tools/local_integration.py (starts a local replica and installs the three canisters)")
    p.add_argument("--manifest",action="store_true",
                   help="also check MANIFEST.sha256 (regenerate it with tools/manifest.py --write)")
    p.add_argument("--verdict",action="store_true",
                   help="also run the openJev/GLiClass parity check against the pack in models/verdict-pack")
    p.add_argument("--verdict-canister",action="store_true",
                   help="also sweep the openJev canister on an already-warm local replica (tools/measure_verdict.py --skip-upload)")
    args=p.parse_args();records=[]
    artifacts=ROOT/"artifacts";artifacts.mkdir(exist_ok=True)
    def run(name,cmd,log,timeout=300):
        try:
            completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout,
                                     env={**os.environ,**CARGO_ENV})
            text=completed.stdout+completed.stderr
            (artifacts/log).write_text(text)
            records.append(dict(check=name,status="PASS" if completed.returncode==0 else "FAIL",command=cmd,log=log,returncode=completed.returncode))
            print(name,records[-1]["status"])
        except (OSError,subprocess.TimeoutExpired) as error:
            records.append(dict(check=name,status="FAIL",command=cmd,error=str(error)))
            print(name,"FAIL",error)
    # Opt-in, and deliberately the FIRST check: it describes the tree as it was handed over.
    # `cargo test` without `--locked` can rewrite Cargo.lock, and the artifact logs are
    # regenerated below, so a check placed after them would report the run's own side
    # effects as staleness. A stale list is a failed claim, not a warning.
    if args.manifest:
        run("manifest_integrity",[sys.executable,"tools/manifest.py","--check"],"manifest_check.log")
    else:
        records.append(dict(check="manifest_integrity",status="NOT_RUN",
          reason="Pass --manifest (regenerate with tools/manifest.py --write after the last edit)"))
    run("python_reference_and_export_tests",[sys.executable,"-m","unittest","discover","-s","tests","-v"],"python_tests.log")
    errors=[]
    for path in ROOT.rglob("*.py"):
        if skipped(path):continue
        try:ast.parse(path.read_text(),filename=str(path))
        except Exception as e:errors.append(str(e))
    for path in ROOT.rglob("Cargo.toml"):
        if skipped(path):continue
        try:tomllib.loads(path.read_text())
        except Exception as e:errors.append(str(e))
    try:json.loads((ROOT/"dfx.json").read_text())
    except Exception as e:errors.append(str(e))
    records.append(dict(check="python_toml_json_syntax",status="FAIL" if errors else "PASS",errors=errors,note="Does not parse or compile Rust source."))
    link_errors=[]
    for path in ROOT.rglob("*.md"):
        if skipped(path):continue
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if link.startswith(("http://","https://","#","mailto:")):continue
            if not (path.parent/link.split("#",1)[0]).exists():link_errors.append(f"{path.relative_to(ROOT)}: {link}")
    records.append(dict(check="local_markdown_links",status="FAIL" if link_errors else "PASS",errors=link_errors))
    run("shell_syntax",["bash","-n","tools/build_one.sh"],"shell_syntax.log")
    # Regenerates the tiny openJev fixture, which exercises tools/pack_verdict.py end
    # to end without the 605 MiB checkpoint. The weights are random: this proves the
    # pack pipeline, never the model.
    run("openjev_fixture_pack",[sys.executable,"tools/make_verdict_fixture.py"],"verdict_fixture.log")
    binaries={name:shutil.which(name) for name in ["cargo","rustc","rustup","dfx"]}
    want_rust=args.rust or args.require_rust
    if want_rust and binaries["cargo"] and binaries["rustc"]:
        run("rust_workspace_tests",["cargo","test","--workspace"],"rust_workspace_tests.log",900)
        run("rust_candle_canister_native_check",["cargo","check","-p","decision-engine","--features","candle"],"rust_candle_check.log",900)
        # Debug, not release: this records that the wasm32 target still compiles and
        # links (including the getrandom custom backend). Release artifacts and
        # Candid come from tools/build_one.sh.
        run("wasm_build",["cargo","build","--target","wasm32-unknown-unknown","-p","decision-engine","--lib","--features","candle"],"wasm_build.log",900)
        run("verdict_candle_tests",["cargo","test","-p","verdict-candle"],"verdict_candle_tests.log",900)
        run("verdict_engine_wasm_build",["cargo","build","--target","wasm32-unknown-unknown","-p","verdict-engine","--lib"],"verdict_wasm_build.log",900)
    else:
        why="Rust toolchain not installed" if not binaries["cargo"] or not binaries["rustc"] else "Pass --rust to run Rust checks"
        records.append(dict(check="rust_workspace_tests",status="NOT_RUN",reason=why))
        records.append(dict(check="rust_candle_canister_native_check",status="NOT_RUN",reason=why))
        records.append(dict(check="wasm_build",status="NOT_RUN",reason=why))
        records.append(dict(check="verdict_candle_tests",status="NOT_RUN",reason=why))
        records.append(dict(check="verdict_engine_wasm_build",status="NOT_RUN",reason=why))
    # Opt-in: needs the 605 MiB pack (tools/pack_verdict.py) and the release binary.
    parity_pack=ROOT/"models"/"verdict-pack"
    parity_bin=ROOT/"target"/"release"/"verdict-infer"
    parity_cases=ROOT/"models"/"verdict-parity"
    if args.verdict and parity_pack.exists() and parity_bin.exists() and (parity_cases/"cases.jsonl").exists():
        run("openjev_checkpoint_parity",[str(parity_bin),"check","--pack",str(parity_pack),
            "--tokenizer",str(ROOT/"models"/"verdict-151m"/"tokenizer.json"),
            "--cases",str(parity_cases/"cases.jsonl"),
            "--predictions",str(parity_cases/"predictions.jsonl"),
            "--limit","1000"],"verdict_parity.log",7200)
    else:
        records.append(dict(check="openjev_checkpoint_parity",status="NOT_RUN",
          reason="Pass --verdict with models/verdict-pack and models/verdict-parity populated; see docs/VERDICT_ENGINE.md"))
    # Opt-in: the replica must already hold the 605 MiB pack, so this is a
    # measurement of a warm chain, not a deployment step. 120 is the longest length
    # measured to fit the 40B update limit (tools/measure_verdict.py --sweep).
    #
    # No --keep: this leaves the network stopped so the local-integration check below
    # can start its own. With --keep the two checks collide on port 8000, and the
    # integration run fails with "network 'local' is already running" -- a failure
    # that says nothing about either check.
    if args.verdict_canister:
        run("openjev_canister_instructions",[sys.executable,"tools/measure_verdict.py",
            "--skip-upload","--sweep","120"],"verdict_sweep.log",3600)
    else:
        records.append(dict(check="openjev_canister_instructions",status="NOT_RUN",
          reason="Pass --verdict-canister with a warm local replica; see docs/VERDICT_ENGINE.md"))
    for check in ["real_ledger_transfer"]:
        records.append(dict(check=check,status="NOT_RUN",reason="Not performed by this validation script"))
    # Opt-in: this starts a local replica and installs canisters into it.
    if args.local_integration:
        run("icp_canister_integration",[sys.executable,"tools/local_integration.py"],"local_integration.log",1500)
    else:
        records.append(dict(check="icp_canister_integration",status="NOT_RUN",
          reason="Pass --local-integration to run it (starts a local replica via icp CLI)"))
    versions={}
    for name in ["torch","numpy","safetensors"]:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    rust_sources=[p for p in ROOT.rglob("*.rs") if not skipped(p)]
    # A check that was requested and did not run is not a pass. Without this, `--rust`
    # on a machine without a toolchain, or `--verdict` without the pack, wrote a report
    # whose verdict was "PASS" and exited 0 -- exactly the failure mode this file exists
    # to prevent. The no-flag invocation still exits 0: it is explicitly a
    # source-delivery validation, and it now says so out loud.
    requested={"rust_workspace_tests":want_rust,"rust_candle_canister_native_check":want_rust,
      "wasm_build":want_rust,"verdict_candle_tests":want_rust,"verdict_engine_wasm_build":want_rust,
      "openjev_checkpoint_parity":args.verdict,"openjev_canister_instructions":args.verdict_canister,
      "icp_canister_integration":args.local_integration,"manifest_integrity":args.manifest}
    def reason_for(name):
        return next((r.get("reason","") for r in records
                     if r.get("check")==name and r["status"]!="PASS"),"")
    unverified=sorted(name for name,want in requested.items() if want and
                      any(r.get("check")==name and r["status"]!="PASS" for r in records))
    if not any(requested.values()):
        print("NOTHING VERIFIED: source-delivery checks only. Pass --rust / --verdict / "
              "--verdict-canister / --local-integration to verify behaviour.")
    for name in unverified:
        print(f"REQUESTED BUT NOT VERIFIED: {name} ({reason_for(name)})")
    report=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      scope="Source delivery validation, NOT proof of Rust/ICP/pretrained-model operation",python=sys.version,
      packages=versions,toolchain=binaries,rust_source_files=len(rust_sources),
      rust_test_functions_written=sum(len(re.findall(r"#\[test\]",f.read_text())) for f in rust_sources),
      rust_test_functions_executed_here=None if not want_rust else "see cargo result/log",
      requested_but_not_verified=unverified,checks=records)
    (artifacts/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["IC-Laya validation report",report["scope"],""]+[f'{x["status"]:8} {x["check"]}' for x in records]
    if unverified:
        lines += [""]+[f'REQUESTED BUT NOT VERIFIED {name}: {reason_for(name)}' for name in unverified]
    lines += ["",f'Rust test functions written: {report["rust_test_functions_written"]}; not a passed-test count.',
      "Synthetic weights have no language-understanding capability. Python reference tests do not execute Rust."]
    (artifacts/"VALIDATION.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    failed=any(r["status"]=="FAIL" for r in records)
    return 1 if failed or unverified else 0
if __name__=="__main__":raise SystemExit(main())
