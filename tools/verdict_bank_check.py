#!/usr/bin/env python3
"""Read-only 1000-case author parity check with a pinned executable and four shards."""
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/model-quality-diagnosis/bank-full'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=(ROOT/'models/verdict-parity/cases.jsonl').read_text().splitlines()
    assert len(cases)==1000
    with tempfile.TemporaryDirectory(prefix='verdict-bank-check-') as directory:
        tmp=Path(directory);exe=tmp/'verdict-infer'
        shutil.copy2(ROOT/'target/release/verdict-infer',exe)
        protocol=dict(binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),cases=1000,shards=4,canister_calls=0)
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        def run(index):
            subset=cases[index*250:(index+1)*250]
            path=tmp/f'cases-{index}.jsonl';path.write_text('\n'.join(subset)+'\n')
            command=[str(exe),'check','--pack',str(ROOT/'models/verdict-pack'),'--tokenizer',str(ROOT/'models/verdict-151m/tokenizer.json'),'--cases',str(path),'--predictions',str(ROOT/'models/verdict-parity/predictions.jsonl'),'--temp','1.4265148639678955','--min-cases','250','--min-argmax-ratio','0','--quiet']
            log=OUT/f'shard-{index}.txt'
            with log.open('w') as stream:
                completed=subprocess.run(command,text=True,stdout=stream,stderr=subprocess.STDOUT)
            output=log.read_text()
            if completed.returncode:raise RuntimeError(f'shard {index}: exit {completed.returncode}')
            match=re.search(r'cases=(\d+) skipped_no_truth=(\d+) argmax_match=(\d+)/(\d+)',output)
            assert match and int(match[1])==int(match[4])==250
            result=dict(shard=index,cases=250,missing=int(match[2]),matched=int(match[3]),
                        unsafe_abstention_escape=int(re.search(r'unsafe_abstention_escape=(\d+)',output)[1]),
                        cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            print(json.dumps(result),flush=True);return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results=list(executor.map(run,range(4)))
        summary=dict(cases=1000,matched=sum(r['matched'] for r in results),missing=sum(r['missing'] for r in results),unsafe_abstention_escape=sum(r['unsafe_abstention_escape'] for r in results),shards=results,**{k:v for k,v in protocol.items() if k!='shards' and k!='cases'})
        summary['ratio']=summary['matched']/1000
        (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
