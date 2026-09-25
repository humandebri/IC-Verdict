#!/usr/bin/env python3
"""Measure *executed IC instructions*, not Wasm byte size, after wasm-opt.

Requires Binaryen wasm-opt, PocketIC, and Python packages pocket-ic and ic-py.
No model pack or shared canister is used: the canister's owner-only synthetic
INT8 benchmark constructs matching inputs inside each isolated PocketIC canister.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from ic import Principal
from ic.candid import Types, decode, encode, labelHash
from pocket_ic import PocketIC

SHAPES = ((28, 3072, 1024), (128, 3072, 1024), (128, 5248, 1024), (128, 1024, 2624))
KEY_OK = '_' + str(labelHash('Ok'))
KEY_INSTRUCTIONS = '_' + str(labelHash('instructions'))
KEY_CHECKSUM = '_' + str(labelHash('checksum'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--wasm-opt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'binaryen_version': subprocess.check_output([str(a.wasm_opt), '--version'], text=True).strip(),
              'method': 'isolated PocketIC, same synthetic W8A8 matrix inputs, canister benchmark_int8_kernel',
              'shape_tokens_rows_cols': SHAPES, 'variants': {}, 'samples': []}
    original = a.baseline.read_bytes()
    with tempfile.TemporaryDirectory(prefix='laya-wasm-opt-') as tmp:
        variants = {'original': original}
        for flag in ('-O3', '-Os', '-O4'):
            path = Path(tmp) / f'{flag}.wasm'
            subprocess.run([str(a.wasm_opt), str(a.baseline), flag, '-o', str(path)], check=True)
            variants[flag] = path.read_bytes()
        owner = Principal.from_hex('0102030405')
        pic = PocketIC()
        pic.set_sender(owner)
        ids = {}
        for name, data in variants.items():
            report['variants'][name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
            cid = pic.create_canister()
            pic.add_cycles(cid, 100_000_000_000_000)
            # Gzip keeps management ingress below 2 MiB. IC decodes it at install.
            pic.install_code(cid, gzip.compress(data, compresslevel=9, mtime=0),
                             [{'type': Types.Principal, 'value': owner.bytes}])
            ids[name] = cid
        for shape in SHAPES:
            request = bytes(encode([{'type': Types.Nat32, 'value': v} for v in shape]))
            for name, cid in ids.items():
                for rep in range(3):
                    started = time.perf_counter()
                    reply = bytes(pic.update_call(cid, 'benchmark_int8_kernel', request))
                    elapsed = time.perf_counter() - started
                    value = decode(reply)[0]['value']
                    if KEY_OK not in value:
                        raise RuntimeError(f'{name}, {shape}: {value}')
                    ok = value[KEY_OK]
                    report['samples'].append({'variant': name, 'shape': shape, 'rep': rep,
                                              'instructions': ok[KEY_INSTRUCTIONS],
                                              'checksum': bytes(ok[KEY_CHECKSUM]).hex(),
                                              'update_wall_s': elapsed})
                    a.out.write_text(json.dumps(report, indent=2) + '\n')
            subset = [row for row in report['samples'] if tuple(row['shape']) == shape]
            assert len({row['checksum'] for row in subset}) == 1, f'output changed at {shape}'
        for shape in SHAPES:
            baseline = statistics.median(row['instructions'] for row in report['samples']
                                         if tuple(row['shape']) == shape and row['variant'] == 'original')
            for name in variants:
                median = statistics.median(row['instructions'] for row in report['samples']
                                           if tuple(row['shape']) == shape and row['variant'] == name)
                report.setdefault('summary', []).append({'shape': shape, 'variant': name,
                                                          'median_instructions': median,
                                                          'reduction_percent': 100 * (1 - median / baseline)})
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
