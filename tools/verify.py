#!/usr/bin/env python3
"""Record PASS / FAIL / NOT_RUN independently. Never substitute Python for Rust."""
from __future__ import annotations
import argparse,ast,datetime,importlib.metadata,json,os,re,shutil,subprocess,sys,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust",action="store_true",help="also execute cargo test/check if available")
    p.add_argument("--require-rust",action="store_true",help="return failure if Rust tests cannot run")
    p.add_argument("--local-integration",action="store_true",
                   help="also run tools/local_integration.py (starts a local replica and installs the three canisters)")
    args=p.parse_args();records=[]
    artifacts=ROOT/"artifacts";artifacts.mkdir(exist_ok=True)
    def run(name,cmd,log,timeout=300):
        try:
            completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=timeout)
            text=completed.stdout+completed.stderr
            (artifacts/log).write_text(text)
            records.append(dict(check=name,status="PASS" if completed.returncode==0 else "FAIL",command=cmd,log=log,returncode=completed.returncode))
            print(name,records[-1]["status"])
        except (OSError,subprocess.TimeoutExpired) as error:
            records.append(dict(check=name,status="FAIL",command=cmd,error=str(error)))
            print(name,"FAIL",error)
    run("synthetic_pytorch_numpy",[sys.executable,"tools/generate_fixtures.py"],"synthetic_generation.log")
    run("python_reference_and_export_tests",[sys.executable,"-m","unittest","discover","-s","tests","-v"],"python_tests.log")
    errors=[]
    for path in ROOT.rglob("*.py"):
        if any(x in {".venv","target"} for x in path.parts):continue
        try:ast.parse(path.read_text(),filename=str(path))
        except Exception as e:errors.append(str(e))
    for path in ROOT.rglob("Cargo.toml"):
        try:tomllib.loads(path.read_text())
        except Exception as e:errors.append(str(e))
    try:json.loads((ROOT/"dfx.json").read_text())
    except Exception as e:errors.append(str(e))
    records.append(dict(check="python_toml_json_syntax",status="FAIL" if errors else "PASS",errors=errors,note="Does not parse or compile Rust source."))
    link_errors=[]
    for path in ROOT.rglob("*.md"):
        for link in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if link.startswith(("http://","https://","#","mailto:")):continue
            if not (path.parent/link.split("#",1)[0]).exists():link_errors.append(f"{path.relative_to(ROOT)}: {link}")
    records.append(dict(check="local_markdown_links",status="FAIL" if link_errors else "PASS",errors=link_errors))
    run("shell_syntax",["bash","-n","tools/build_one.sh"],"shell_syntax.log")
    binaries={name:shutil.which(name) for name in ["cargo","rustc","rustup","dfx"]}
    want_rust=args.rust or args.require_rust
    if want_rust and binaries["cargo"] and binaries["rustc"]:
        run("rust_workspace_tests",["cargo","test","--workspace"],"rust_workspace_tests.log",900)
        run("rust_candle_canister_native_check",["cargo","check","-p","decision-engine","--features","candle"],"rust_candle_check.log",900)
        # Debug, not release: this records that the wasm32 target still compiles and
        # links (including the getrandom custom backend). Release artifacts and
        # Candid come from tools/build_one.sh.
        run("wasm_build",["cargo","build","--target","wasm32-unknown-unknown","-p","decision-engine","--lib","--features","candle"],"wasm_build.log",900)
    else:
        why="Rust toolchain not installed" if not binaries["cargo"] or not binaries["rustc"] else "Pass --rust to run Rust checks"
        records.append(dict(check="rust_workspace_tests",status="NOT_RUN",reason=why))
        records.append(dict(check="rust_candle_canister_native_check",status="NOT_RUN",reason=why))
        records.append(dict(check="wasm_build",status="NOT_RUN",reason=why))
    for check in ["upstream_laya_checkpoint_parity","real_ledger_transfer"]:
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
    rust_sources=list(ROOT.rglob("*.rs"))
    report=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      scope="Source delivery validation, NOT proof of Rust/ICP/pretrained-model operation",python=sys.version,
      packages=versions,toolchain=binaries,rust_source_files=len(rust_sources),
      rust_test_functions_written=sum(len(re.findall(r"#\[test\]",f.read_text())) for f in rust_sources),
      rust_test_functions_executed_here=None if not want_rust else "see cargo result/log",checks=records)
    (artifacts/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["IC-Laya validation report",report["scope"],""]+[f'{x["status"]:8} {x["check"]}' for x in records]
    lines += ["",f'Rust test functions written: {report["rust_test_functions_written"]}; not a passed-test count.',
      "Synthetic weights have no language-understanding capability. Python reference tests do not execute Rust."]
    (artifacts/"VALIDATION.txt").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    failed=any(r["status"]=="FAIL" for r in records)
    missing_required=args.require_rust and any(r["check"]=="rust_workspace_tests" and r["status"]!="PASS" for r in records)
    return 1 if failed or missing_required else 0
if __name__=="__main__":raise SystemExit(main())
