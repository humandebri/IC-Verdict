#!/usr/bin/env python3
"""Summarize the explicitly selected local measurements without editing raw files.

Run --write after a measurement run, or --check to compare existing summaries
with locally available raw files. Missing raw files are expected in a fresh clone.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'artifacts/summary-sources.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compact(value, key='', path='$', omitted=None):
    # Preserve metrics, totals, provenance and individual game summaries. Omit
    # detailed boards, per-turn traces, per-phase traces and raw Candid replies.
    detail = key in {'turns', 'phases', 'cases', 'raw'}
    long_list = isinstance(value, list) and len(value) > 32
    if detail or long_list:
        omitted.append({'path': path, 'count': len(value) if hasattr(value, '__len__') else None,
                        'sha256': digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())})
        return {'omitted': True, 'see_raw_source': True}
    if isinstance(value, dict):
        return {k: compact(v, k, f'{path}.{k}', omitted) for k, v in value.items()}
    if isinstance(value, list):
        return [compact(v, '', f'{path}[{i}]', omitted) for i, v in enumerate(value)]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    checked = missing = stale = 0
    for source in json.loads(SOURCES.read_text()):
        raw_path = ROOT / source
        destination = ROOT / 'artifacts/summaries' / Path(source).relative_to('artifacts')
        if not raw_path.exists():
            missing += 1
            if not destination.exists():
                print(f'MISSING summary: {destination.relative_to(ROOT)}')
                stale += 1
            continue
        raw = raw_path.read_bytes()
        omitted = []
        summary = compact(json.loads(raw), omitted=omitted)
        result = {'source': source, 'source_sha256': digest(raw), 'source_bytes': len(raw),
                  'note': 'Derived summary, not full evidence. Omitted details require the local raw source.',
                  'summary': summary, 'omitted': omitted}
        text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text)
        elif not destination.exists() or destination.read_text() != text:
            print(f'STALE: {destination.relative_to(ROOT)}')
            stale += 1
        checked += 1
    print(f'{checked} local sources, {missing} raw sources absent, {stale} missing/stale summaries')
    return bool(stale)


if __name__ == '__main__':
    raise SystemExit(main())
