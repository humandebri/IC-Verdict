#!/usr/bin/env python3
"""Compare two already-warm canisters on a managed loopback replica; never installs."""
import argparse
import billing_cli
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--identity',required=True)
    p.add_argument('--baseline',default='baseline')
    p.add_argument('--candidate',default='verdict-engine')
    p.add_argument('--did',type=Path,default=Path('build/verdict-engine.did'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--diagnostic-case',type=Path)
    p.add_argument('--resume',action='store_true',help='keep completed rows in --out')
    p.add_argument('--unpaid-baseline',action='store_true',help='explicitly compare against a pre-billing baseline via direct calls')
    p.add_argument('--lengths',default='8,16,32,64,96,120,128',help='synthetic token lengths')
    billing_cli.add_arguments(p)
    args=p.parse_args()
    billing_cli.require_payment(args)
    common=['-e','local','--project-root-override',str(args.project)]
    status=json.loads(subprocess.check_output(['icp','network','status','--json',*common],text=True))
    if not status.get('managed') or urlparse(status['api_url']).hostname not in {'localhost','127.0.0.1','::1'}:
        raise ValueError('only a managed loopback replica is permitted')
    def call(name,method,argument):
        flags=['--query'] if method=='cycles_pricing' else []
        if method in billing_cli.PAID_METHODS and not (args.unpaid_baseline and name==args.baseline):
            flags+=billing_cli.proxy_flags(args,call(name,'cycles_pricing','()'))
        r=subprocess.run(['icp','canister','call',name,method,argument,'--identity',args.identity,'--candid',str(args.did),*common,*flags],capture_output=True,text=True)
        text=r.stdout+r.stderr
        if r.returncode or 'Err =' in text:
            if 'Capacity' in text:return {'guard_refused':True}
            if 'instruction limit' in text or 'IC0522' in text:return {'instruction_limit':True}
            raise RuntimeError(text)
        return text
    if args.execution_pricing:
        for name in [args.baseline,args.candidate]:
            if not (args.unpaid_baseline and name==args.baseline):call(name,'set_execution_pricing',billing_cli.pricing_arg(args.execution_pricing))
    report=json.loads(args.out.read_text()) if args.resume else {'endpoint':status['api_url'],'heap':{},'rows':[]}
    if report['endpoint'] != status['api_url']:raise ValueError('resume endpoint differs')
    for name in [args.baseline,args.candidate]:
        if name in report['heap']:continue
        text=call(name,'info','()')
        report['heap'][name]=int(re.search(r'heap_bytes = ([\d_]+)',text)[1].replace('_',''))
    lengths=[int(n) for n in args.lengths.split(',')]
    if any(n<7 or n>128 for n in lengths):raise ValueError('lengths must be in 7..128')
    inputs=[('synthetic', [50281]+[50368]*5+[1234]*(n-7)+[50282]) for n in lengths]
    if args.diagnostic_case:
        case=json.loads(args.diagnostic_case.read_text())
        text=subprocess.check_output(['target/release/verdict-infer','tokens','--pack','models/verdict-pack',
            '--tokenizer','models/verdict-151m/tokenizer.json','--case-json',json.dumps(case)],text=True)
        ids=[int(x) for x in next(x[4:] for x in text.splitlines() if x.startswith('ids=')).split(',')]
        inputs.append((case['id'],ids))
    for label,ids in inputs:
        if any(row['input']==label and row['tokens']==len(ids) for row in report['rows']):continue
        row={'input':label,'tokens':len(ids),'results':{}}
        for name in [args.baseline,args.candidate]:
            text=call(name,'infer_tokens','(vec {'+';'.join(map(str,ids))+'})')
            if isinstance(text,dict):
                row['results'][name]=text
                if text.get('guard_refused'):
                    profiled=call(name,'infer_profiled','(vec {'+';'.join(map(str,ids))+'},false)')
                    if isinstance(profiled,dict):row['results'][name]['profile']=profiled
                    else:row['results'][name]['profile_instructions']=int(re.search(r'measured_instructions = ([\d_]+)',profiled)[1].replace('_',''))
            else:
                measured=int(re.search(r'measured_instructions = ([\d_]+)',text)[1].replace('_',''))
                logits=re.search(r'logits = vec \{([^}]+)\}',text)[1].strip()
                row['results'][name]={'instructions':measured,'logits':logits}
        a,b=[row['results'][name] for name in [args.baseline,args.candidate]]
        if 'logits' in a and 'logits' in b:
            row['exact_logits']=a['logits']==b['logits']
            if not row['exact_logits']:raise ValueError(f'logits changed: {row}')
            row['instruction_ratio']=b['instructions']/a['instructions']
        report['rows'].append(row)
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(label,len(ids),row.get('instruction_ratio','refused'),flush=True)


if __name__=='__main__':
    main()
