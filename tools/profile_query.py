#!/usr/bin/env python3
"""Measure query boundaries and instruction phases on a managed loopback replica.

Query mode temporarily relaxes the application's estimate, never the replica's
instruction limit. The original cost model is restored and checked in finally.
Use a dedicated local measurement canister; no mainnet or deployment support.
"""
import argparse
import billing_cli
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse


def integer(reply, name):
    match = re.search(r'\b' + re.escape(name) + r' = ([\d_]+)', reply)
    if not match:
        raise ValueError(f'missing {name}: {reply[:500]}')
    return int(match[1].replace('_', ''))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canister', required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--controller-proxy')
    p.add_argument('--mode', choices=['query', 'decide', 'update', 'profile', 'kernels'], required=True)
    p.add_argument('--requests', type=Path, help='decide mode: JSON cases with pre-tokenized lengths')
    p.add_argument('--lengths', default='32,38,39,40,41,42,43,44,45,46,47,48,56,64')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--classes', type=int, default=2)
    p.add_argument('--compact-classes', action='store_true', help='adjacent class tokens to stress the maximum class count')
    p.add_argument('--respect-guard', action='store_true', help='verify the installed policy without relaxing its estimate')
    p.add_argument('--mixed', action='store_true', help='vary ordinary token IDs instead of PAD filler')
    p.add_argument('--vary-inputs', action='store_true', help='use different ordinary token IDs for each repeat, avoiding identical-query cache samples')
    p.add_argument('--out', type=Path, required=True)
    billing_cli.add_arguments(p)
    a = p.parse_args()
    if a.mode == "update": billing_cli.require_payment(a)
    if not 1 <= a.classes <= 25 or a.repeats < 1:
        p.error('classes must be 1..25 and repeats must be positive')
    if a.mode == 'decide' and not a.requests:
        p.error('--requests is required for decide mode')
    common = ['-e', 'local']

    def run(command):
        result = subprocess.run(command, text=True, capture_output=True, timeout=240)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        return result.stdout

    status = json.loads(run(['icp', 'network', 'status', '--json', *common]))
    if not status.get('managed') or urlparse(status['api_url']).hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise ValueError('a managed loopback replica is required')

    def call(method, args='()', query=False):
        command = ['icp', 'canister', 'call', a.canister, method, args,
                   '--identity', a.identity, '--candid', 'build/verdict-engine.did', *common]
        if method in billing_cli.PAID_METHODS:
            command += billing_cli.proxy_flags(a, call('cycles_pricing', query=True)['raw'])
        if query:
            command.append('--query')
        start = time.monotonic()
        result = subprocess.run(command, text=True, capture_output=True, timeout=240)
        elapsed = time.monotonic() - start
        raw = result.stdout + result.stderr
        if result.returncode:
            if 'IC0522' in raw or 'instruction limit' in raw.lower():
                return {'status': 'instruction_limit', 'wall_seconds': elapsed, 'raw': raw}
            raise RuntimeError(raw)
        if 'Err =' in raw:
            if 'Capacity' in raw:
                return {'status': 'guard_refused', 'wall_seconds': elapsed, 'raw': raw}
            raise RuntimeError(raw)
        return {'status': 'ok', 'wall_seconds': elapsed, 'raw': raw}

    if a.execution_pricing:
        call('set_execution_pricing', billing_cli.pricing_arg(a.execution_pricing))
    limits = call('query_limits', query=True)['raw']
    info = call('info', query=True)['raw']
    status_cmd = ['icp', 'canister', 'status', a.canister, '--identity', 'anonymous', '--json', *common]
    if a.controller_proxy:
        status_cmd += ['--proxy', a.controller_proxy]
    deployed = json.loads(run(status_cmd))
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'canister': a.canister,
              'endpoint': status['api_url'], 'mode': a.mode, 'classes': a.classes,
              'compact_classes': a.compact_classes,
              'input_kind': 'mixed synthetic IDs' if a.mixed or a.vary_inputs else 'canonical short prompt padded with PAD',
              'vary_inputs': a.vary_inputs,
              'module_hash': deployed['module_hash'], 'initial_info': info,
              'initial_query_limits': limits, 'rows': [],
              'source_hashes_note': 'Working-tree snapshot only; module_hash identifies the installed Wasm. Sources can differ during candidate builds.',
              'query_repeats_note': ('Each repeat uses distinct filler IDs; compare matching IDs across candidate modules.' if a.vary_inputs else
                                     'Identical queries can hit the replica cache; repetitions are not independent timing samples.'),
              'source_hashes': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                  map(Path, ['crates/verdict-simd/src/lib.rs', 'crates/modernbert-candle/src/lib.rs',
                             'crates/verdict-candle/src/lib.rs', 'canisters/verdict-engine/src/lib.rs'])}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.out.write_text(json.dumps(report, indent=2) + '\n')

    original = (integer(limits, 'cost_fixed'), integer(limits, 'cost_per_token'), integer(info, 'budget'))
    relax_guard = a.mode in {'query', 'decide'} and not a.respect_guard
    report['respect_guard'] = a.respect_guard
    save()
    def set_cost(values):
        return call('set_cost_model', '(' + ','.join(f'{n}:nat64' for n in values) + ')')

    try:
        if relax_guard:
            report['guard_relaxation'] = 'local estimate only: fixed=1, per_token=1; replica limit unchanged'
            set_cost((1, 1, original[2]))
        if a.mode == 'decide':
            report['input_kind'] = 'un-padded English requests, 2 options plus abstention'
            report['classes'] = 3
            for request in json.loads(a.requests.read_text()):
                case = request['case']
                assert case['candidates'][-1]['id'] == '__insufficient_evidence__'
                options = ';'.join('record {id=' + json.dumps(c['id']) + ';text=' + json.dumps(c['description']) + '}'
                                   for c in case['candidates'][:-1])
                arg = '(record {state=' + json.dumps(case['text']) + ';question=' + json.dumps(case['question'])
                arg += ';options=vec {' + options + '};abstention=true;temperature=1.0:float64})'
                row = call('decide_query', arg, query=True)
                row.update(tokens=request['tokens'], case=case)
                if row['status'] == 'ok':
                    assert integer(row['raw'], 'input_tokens') == row['tokens']
                    row['instructions'] = integer(row['raw'], 'measured_instructions')
                report['rows'].append(row)
                save()
                print(f'decide T={row["tokens"]}: {row["status"]} {row.get("instructions", "")}', flush=True)
            return
        for t in map(int, a.lengths.split(',')):
            if a.mode == 'kernels':
                for n, k in [(2304, 768), (768, 768), (768, 1152)]:
                    row = call('bench_int8', f'({t}:nat32,{n}:nat32,{k}:nat32,3:nat32)')
                    row.update(tokens=t, n=n, k=k)
                    for name in ['per_iteration', 'quantize_activations_instructions', 'quantize_weights_instructions']:
                        row[name] = integer(row['raw'], name)
                    row['instructions_per_mac'] = row['per_iteration'] / (t*n*k)
                    report['rows'].append(row)
                    save()
                    print(f'kernel T={t} N={n} K={k}: {row["per_iteration"]:,}', flush=True)
                continue
            base = [50281] + ([50368]*a.classes if a.compact_classes else
                              sum(([50368, 2000 + i*1000] for i in range(a.classes)), []))
            if t < len(base) + 1:
                continue
            if a.vary_inputs and t == len(base) + 1:
                raise ValueError('--vary-inputs requires at least one ordinary filler token')
            for repeat in range(a.repeats):
                filler = ([100 + (i*137 + (repeat+1)*7919) % 20000 for i in range(t-len(base)-1)] if a.vary_inputs else
                          [100 + (i*137) % 20000 for i in range(t-len(base)-1)] if a.mixed else [50283]*(t-len(base)-1))
                ids = base + filler + [50282]
                arg = '(vec {' + ';'.join(map(str, ids)) + '}' + (',true)' if a.mode == 'profile' else ')')
                method = {'profile': 'infer_profiled', 'update': 'infer_tokens', 'query': 'infer_tokens_query'}[a.mode]
                row = call(method, arg, a.mode == 'query')
                row.update(tokens=t, repeat=repeat, ids=ids)
                if row['status'] == 'ok':
                    row['instructions'] = integer(row['raw'], 'measured_instructions')
                    if a.mode == 'profile':
                        phases = [{'name': name, 'instructions': int(count.replace('_', ''))}
                                  for name, count in re.findall(r'name = "([^"]+)";\s*instructions = ([\d_]+)', row['raw'])]
                        assert phases and sum(x['instructions'] for x in phases) == row['instructions']
                        totals = defaultdict(int)
                        for phase in phases:
                            totals[phase['name']] += phase['instructions']
                        row.update(phases=phases, phase_totals=dict(totals))
                report['rows'].append(row)
                save()
                print(f'{a.mode} T={t} repeat={repeat}: {row["status"]} {row.get("instructions", "")}', flush=True)
    finally:
        if relax_guard:
            set_cost(original)
            after = call('query_limits', query=True)['raw']
            report['restored_query_limits'] = after
            report['guard_restored'] = after == limits
            save()
            assert report['guard_restored'], 'cost model restoration mismatch'
        else:
            save()


if __name__ == '__main__':
    main()
