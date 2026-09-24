#!/usr/bin/env python3
"""Local-only smoke/upgrade test for an already installed Tetris-capable canister.

Requires a local owner identity and anonymous controller on a disposable target.
Never installs over mainnet. Leaves the demo enabled and warm for UI verification.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--canister',required=True)
    p.add_argument('--owner',required=True)
    p.add_argument('--pem',required=True)
    a=p.parse_args()
    report={'canister':a.canister,'checks':[]}
    def run(cmd):
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        if r.returncode: raise RuntimeError(r.stdout+r.stderr)
        return r.stdout
    def call(method,args='()',owner=False,query=True):
        cmd=['icp','canister','call',a.canister,method,args,'-e','local','--identity',a.owner if owner else 'anonymous','--candid','build/verdict-engine.did']
        if query:cmd.append('--query')
        return run(cmd)
    def check(name,condition):
        if not condition: raise AssertionError(name)
        report['checks'].append(name)
        print('PASS',name,flush=True)
    def warm():
        # Endpoint is obtained from the selected local project, not guessed.
        status=run(['icp','network','status','-e','local'])
        url=next(s.split(': ',1)[1] for s in status.splitlines() if s.startswith('Api Url:'))
        if not url.startswith(('http://localhost:','http://127.0.0.1:')):raise ValueError('not local')
        run(['target/release/verdict-upload','--url',url,'--canister',a.canister,'--pem',a.pem,'--warm-only'])
    report['initial_status']=call('tetris_status')
    call('set_tetris_enabled','(false)',owner=True,query=False)
    check('disabled when switched off','enabled = false' in call('tetris_status'))
    check('anonymous cannot enable','Unauthorized' in call('set_tetris_enabled','(true)',query=False))
    check('owner enables','Ok' in call('set_tetris_enabled','(true)',owner=True,query=False))
    warm()
    check('warm without reupload','warmed = true' in call('tetris_status'))
    req='(vec {record {holes=0:nat16;height=4:nat8;lines=1:nat8};record {holes=200:nat16;height=20:nat8;lines=4:nat8}})'
    reply=call('tetris_decide_query',req)
    check('anonymous bounded query','Ok' in reply)
    report['reply']=reply
    check('empty candidate list rejected','Invalid' in call('tetris_decide_query','(vec {})'))
    check('out-of-bounds rejected','Invalid' in call('tetris_decide_query',req.replace('holes=200','holes=201')))
    check('generic query remains protected','Unauthorized' in call('infer_tokens_query','(vec {50281:nat32;50282:nat32})'))
    run(['icp','canister','install',a.canister,'-e','local','--identity','anonymous','-m','upgrade','--wasm','build/verdict-engine.wasm'])
    status=call('tetris_status')
    check('enabled survives upgrade','enabled = true' in status)
    check('heap model resets','warmed = false' in status)
    warm()
    after=call('tetris_decide_query',req)
    report['reply_after_upgrade']=after
    # Instruction counters include allocator/cache effects, not semantic output.
    semantic=lambda s:re.sub(r'measured_instructions = [\d_]+ : nat64;', '', s)
    check('query output identical after upgrade',semantic(after)==semantic(reply))
    call('set_tetris_enabled','(false)',owner=True,query=False)
    check('operator shutdown rejects query','Unauthorized' in call('tetris_decide_query',req))
    call('set_tetris_enabled','(true)',owner=True,query=False)
    report['passed']=True
    Path('artifacts/tetris_smoke.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
