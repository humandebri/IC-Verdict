#!/usr/bin/env python3
"""Upgrade one existing disposable local canister and verify both Tetris query APIs."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister',required=True)
    parser.add_argument('--owner',required=True)
    parser.add_argument('--pem',required=True)
    parser.add_argument('--wasm',default='build/verdict-engine.wasm')
    parser.add_argument('--candid',default='build/verdict-engine.did')
    args=parser.parse_args()
    def run(command):
        result=subprocess.run(command,capture_output=True,text=True,timeout=600)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
        return result.stdout
    status=json.loads(run(['icp','network','status','-e','local','--json']))
    url=status['api_url']
    if not status.get('managed') or urlparse(url).hostname not in {'localhost','127.0.0.1','::1'}:
        raise ValueError('A managed loopback network is required')
    report={'canister':args.canister,'endpoint':url,'wasm_sha256':hashlib.sha256(Path(args.wasm).read_bytes()).hexdigest(),'checks':[]}
    def call(method,value='()',owner=False,query=True):
        command=['icp','canister','call',args.canister,method,value,'-e','local','--identity',args.owner if owner else 'anonymous','--candid',args.candid]
        if query:command.append('--query')
        return run(command)
    def check(name,condition):
        if not condition:raise AssertionError(name)
        report['checks'].append(name);print('PASS',name,flush=True)
    feature=lambda h,y,l:f'record {{holes={h}:nat16;height={y}:nat8;lines={l}:nat8}}'
    old='(vec {'+feature(0,4,1)+';'+feature(200,20,4)+'})'
    request=lambda fs,piece=2,nxt=0,rough=180:'(record {piece='+str(piece)+':nat8;next='+str(nxt)+':nat8;roughness='+str(rough)+':nat16;options=vec {'+';'.join(fs)+'}})'
    fs=[feature(200,20,0)]*3+[feature(0,1,4)]
    before=call('tetris_status');report['status_before']=before
    check('existing test target enabled and warm','enabled = true' in before and 'warmed = true' in before)
    report['legacy_before']=call('tetris_decide_query',old)
    run(['icp','canister','install',args.canister,'-e','local','--identity','anonymous','-m','upgrade','--wasm',args.wasm])
    after=call('tetris_v2_status');check('enabled flag retained and warm-up required','enabled = true' in after and 'warmed = false' in after)
    run(['target/release/verdict-upload','--url',url,'--canister',args.canister,'--pem',args.pem,'--warm-only'])
    after=call('tetris_v2_status');report['status_after']=after
    check('model retained without reupload',re.search(r'model = blob "([^"]+)"',before)[1]==re.search(r'model = blob "([^"]+)"',after)[1])
    check('v2 advertises 40-token ceiling','max_tokens = 40' in after and 'warmed = true' in after)
    report['legacy_after']=call('tetris_decide_query',old)
    check('legacy API still accepts original request','Ok = record' in report['legacy_after'] and 'input_tokens = 24' in report['legacy_after'])
    report['v2_replies']=[]
    for n in [2,3,4]:
        reply=call('tetris_decide_v2_query',request(fs[:n-1]+[fs[-1]]));report['v2_replies'].append(reply)
        scores=re.search(r'scores = vec \{([^}]+)',reply)
        check(f'{n} candidates return {n} scores',bool(scores) and len(re.findall(r': float32',scores[1]))==n)
        tokens=int(re.search(r'input_tokens = (\d+)',reply)[1]);count=int(re.search(r'measured_instructions = ([\d_]+)',reply)[1].replace('_',''))
        check(f'{n} candidates fit query token and instruction limits',tokens<=39 and count<5_000_000_000)
    for name,bad in [('one option',request(fs[:1])),('five options',request(fs+[fs[0]])),('piece bounds',request(fs,piece=7)),('NEXT bounds',request(fs,nxt=7)),('roughness bounds',request(fs,rough=181)),('holes bounds',request([feature(201,20,4)]*4)),('height bounds',request([feature(0,21,4)]*4)),('lines bounds',request([feature(0,20,5)]*4))]:
        check(name+' rejected','Invalid' in call('tetris_decide_v2_query',bad))
    check('anonymous cannot enable or disable','Unauthorized' in call('set_tetris_enabled','(false)',query=False))
    call('set_tetris_enabled','(false)',owner=True,query=False)
    try:
        check('operator switch disables both adapters',all('Unauthorized' in call(method,req) for method,req in [('tetris_decide_query',old),('tetris_decide_v2_query',request(fs))]))
    finally:call('set_tetris_enabled','(true)',owner=True,query=False)
    check('generic inference remains protected','Unauthorized' in call('infer_tokens_query','(vec {})'))
    report['passed']=True
    Path('artifacts/tetris_query_v2_smoke.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
