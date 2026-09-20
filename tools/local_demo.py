#!/usr/bin/env python3
"""Create and install LOCAL scripted/mock canisters. Never uses the IC network.
Run `dfx start --background` first. Refuses to reinstall existing canisters.
This integration path is supplied but was NOT executed in the delivery environment.
"""
from __future__ import annotations
import json,subprocess,time,shutil,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args, capture=False):
    print("$ "+" ".join(args),flush=True)
    return subprocess.run(args,cwd=ROOT,check=True,text=True,stdout=subprocess.PIPE if capture else None).stdout

def main():
    for tool in ["cargo","dfx"]:
        if not shutil.which(tool): raise SystemExit(f"{tool} is required")
    run(["dfx","canister","--network","local","create","--all"])
    owner=run(["dfx","identity","get-principal"],True).strip()
    ids={name:run(["dfx","canister","--network","local","id",name],True).strip() for name in ["decision_engine","executor","mock_ledger"]}
    for package in ["decision-engine","executor","mock-ledger"]:run(["bash","tools/build_one.sh",package])
    for name,arguments in [("decision_engine",f'(principal "{owner}")'),("mock_ledger",f'(principal "{owner}")'),
                            ("executor",f'(principal "{owner}", principal "{ids["decision_engine"]}")')]:
        run(["dfx","canister","--network","local","install",name,"--mode","install","--argument",arguments])
    text=run(["cargo","run","--quiet","-p","ic-laya-core","--example","bootstrap_args","--",owner,ids["decision_engine"],ids["executor"],ids["mock_ledger"],str(time.time_ns())],True)
    setup=json.loads(text)
    if setup.get("test_only") is not True: raise SystemExit("not a test-only bootstrap")
    for step in setup["steps"]:
        output=run(["dfx","canister","--network","local","call",step["canister"],step["method"],step["argument_hex"],"--argument-type","raw"],True)
        print(output,flush=True)
        # dfx transport success does not imply a Result::Ok payload.
        if re.search(r"\bErr\s*=",output):
            raise SystemExit("canister returned Err; stop rather than continuing the workflow")
    print("LOCAL mock workflow completed. Predictions were scripted, not language-model outputs.")
if __name__=="__main__":main()
