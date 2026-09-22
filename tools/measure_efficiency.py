#!/usr/bin/env python3
"""Compare production block32 against measurement-only tiles on a managed local replica."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--canister', default='verdict-engine')
    p.add_argument('--did', type=Path, default=Path('build/verdict-engine.did'))
    p.add_argument('--wasm', type=Path, default=Path('build/verdict-engine.wasm'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--attention', action='store_true')
    args = p.parse_args()
    common = ['-e', 'local', '--project-root-override', str(args.project)]
    status = json.loads(subprocess.check_output(['icp', 'network', 'status', '--json', *common], text=True))
    if not status.get('managed') or urlparse(status['api_url']).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('only a managed loopback replica is permitted')
    rows = []
    with args.wasm.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if args.attention:
        for t in [8,16,32,64,96,120,128]:
            for candidate in [False,True]:
                output=subprocess.check_output(['icp','canister','call',args.canister,'bench_local_attention',
                    f'({t}:nat32,64:nat32,{str(candidate).lower()})','--candid',str(args.did),
                    '--identity',args.identity,*common],text=True)
                found=re.search(r'per_iteration = ([\d_]+)',output)
                if not found:raise ValueError(output)
                instructions=int(found[1].replace('_',''))
                rows.append(dict(tokens=t,distance=64,candidate=candidate,instructions=instructions))
                args.out.write_text(json.dumps(dict(wasm_sha256=digest,endpoint=status['api_url'],rows=rows),indent=2)+'\n')
                print(rows[-1],flush=True)
        return
    for m in [8, 16, 32, 64, 96, 120, 128]:
        for n, k in [(2304, 768), (768, 768), (768, 1152)]:
            for tile in [0, 2, 4, 8]:
                output = subprocess.check_output(['icp', 'canister', 'call', args.canister, 'bench_int8_block32',
                    f'({m}:nat32,{n}:nat32,{k}:nat32,3:nat32,{tile}:nat32)', '--candid', str(args.did),
                    '--identity', args.identity, *common], text=True)
                found = re.search(r'per_iteration = ([\d_]+)', output)
                if not found:
                    raise ValueError(output)
                instructions = int(found[1].replace('_', ''))
                rows.append(dict(m=m, n=n, k=k, tile=tile, instructions=instructions))
                args.out.write_text(json.dumps(dict(wasm_sha256=digest, endpoint=status['api_url'],
                    comparison='tile 0 is production; candidates assert exact agreement before timing', rows=rows), indent=2)+'\n')
                print(f'm={m} n={n} k={k} tile={tile} instructions={instructions}', flush=True)


if __name__ == '__main__':
    main()
