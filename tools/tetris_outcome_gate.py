#!/usr/bin/env python3
"""Prerequisite for Tetris v4: frozen 100-question outcome comparison gate.
Only queries an explicitly selected existing managed loopback canister. Never deploys.
"""
import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/tetris-outcome-gate'
ACTIONS = ('left', 'right', 'rotate', 'down', 'wait')
STATE = 'Tetris. Choose next move.'
WASM = '0962574cae166a5a438e62a588d3351408613f52b867e6e18ed7311ba3d8074b'
MODEL = '0757d774c8c44397d189c5b479853897acef8f1e2b55ebf1f5a46969063225e0'

def label(action, outcome):
    if action not in range(5):
        raise ValueError('action bounds')
    if outcome is None:
        return ACTIONS[action] + ' game over'
    for key, maximum in [('lines', 4), ('holes', 200), ('height', 20)]:
        if type(outcome.get(key)) is not int or not 0 <= outcome[key] <= maximum:
            raise ValueError(key + ' bounds')
    return f"{ACTIONS[action]} lines{outcome['lines']} holes{outcome['holes']} height{outcome['height']}"

def cases():
    """50 pairs; each action is uniquely correct 20 times across 100 questions."""
    result = []
    for pair in range(50):
        metric = ('lines', 'holes', 'height')[pair % 3]
        level = pair // 3
        if metric == 'lines':
            ordinary = dict(lines=level % 4, holes=level % 11, height=12)
            better = dict(ordinary, lines=ordinary['lines'] + 1)
        elif metric == 'holes':
            ordinary = dict(lines=0, holes=(level * 7) % 50 + 1, height=12)
            better = dict(ordinary, holes=max(0, ordinary['holes'] - (1 + level % 5)))
        else:
            ordinary = dict(lines=0, holes=0, height=2 + level % 19)
            better = dict(ordinary, height=ordinary['height'] - 1)
        first = pair % 5
        second = (first + 1 + (pair // 5) % 4) % 5
        for version, winner in enumerate((first, second)):
            outcomes = [dict(better if a == winner else ordinary) for a in range(5)]
            result.append(dict(id=f'{pair:02d}-{version}', pair=pair, version=version,
                               metric=metric, outcomes=outcomes, expected=winner))
    return result

def request(case):
    # Expected answer, pair ID and metric selector are never part of inference.
    return dict(text=STATE, labels=[label(a, o) for a, o in enumerate(case['outcomes'])])

def summarize(rows):
    if len(rows) != 100 or {r['id'] for r in rows} != {c['id'] for c in cases()}:
        raise ValueError('gate requires all 100 unique predeclared questions')
    frozen = {c['id']: c for c in cases()}
    for row in rows:
        if any(row.get(k) != v for k, v in frozen[row['id']].items()) or type(row.get('action')) is not int or row['action'] not in range(5):
            raise ValueError('fixture metadata or action changed')
    correct = sum(r['action'] == r['expected'] for r in rows)
    per_action = {ACTIONS[a]: dict(total=sum(r['expected'] == a for r in rows),
                    correct=sum(r['expected'] == a and r['action'] == a for r in rows)) for a in range(5)}
    pairs = sum(all(r['action'] == r['expected'] for r in rows if r['pair'] == p) for p in range(50))
    gates = dict(accuracy=correct >= 90,
                 every_action=all(r['total'] == 20 and r['correct'] >= 16 for r in per_action.values()),
                 both_swapped_questions=pairs >= 40)
    return dict(correct=correct, total=100, accuracy=correct / 100, per_action=per_action,
                pairs_correct=pairs, pairs_total=50, paired_accuracy=pairs / 50,
                selected_counts={ACTIONS[a]: sum(r['action'] == a for r in rows) for a in range(5)},
                per_metric={m: dict(total=sum(r['metric'] == m for r in rows),
                    correct=sum(r['metric'] == m and r['action'] == r['expected'] for r in rows))
                    for m in ('lines','holes','height')},
                gates=gates, passed=all(gates.values()))

class Tokenizer:
    def __init__(self):
        self.process = subprocess.Popen([str(ROOT / 'target/release/tetris_probe'), str(ROOT)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.cache = {}
    def raw(self, text):
        if text not in self.cache:
            self.process.stdin.write(json.dumps(dict(raw=text)) + '\n')
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('tokenizer probe exited')
            self.cache[text] = json.loads(line)
        return self.cache[text]
    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.process.stdout.close()

def token_bound(tok):
    config = json.loads((ROOT / 'models/verdict-151m/tokenizer.json').read_text())
    assert config['model']['type'] == 'BPE' and config['normalizer'] == {'type': 'NFC'}
    assert config['pre_tokenizer'] == dict(type='ByteLevel', add_prefix_space=False, trim_offsets=True, use_regex=True)
    # All generated strings are ASCII, so NFC is the identity.
    # Each variable is an entire numeric ByteLevel pretoken; separators cannot
    # interact across numeric/letter boundaries. Check every value at every slot.
    parts = []
    for name in ACTIONS:
        parts.extend([['<<LABEL>>'], [name + ' lines'], list(map(str, range(5))),
                      [' holes'], list(map(str, range(201))), [' height'], list(map(str, range(21)))])
    parts.append(['<<SEP>>' + STATE])
    encoded = [[(s, tok.raw(s)['ids']) for s in choices] for choices in parts]
    worst = [max(choices, key=lambda x: len(x[1])) for choices in encoded]
    checks = 0
    for index, choices in enumerate(encoded):
        for value in choices:
            joined = [value if i == index else w for i, w in enumerate(worst)]
            assert tok.raw(''.join(s for s, _ in joined))['ids'] == [n for _, ids in joined for n in ids]
            checks += 1
    normal = ''.join(s for s, _ in worst)
    maximum = len(tok.raw(normal)['ids']) + 2
    # Cover all omitted/safe/game-over combinations, including every legal subset.
    for kinds in itertools.product(range(3), repeat=5):
        if not any(kinds):
            continue
        text = ''.join('<<LABEL>>' + (''.join(s for s, _ in worst[a * 7 + 1:a * 7 + 7])
                                    if kind == 1 else label(a, None))
                       for a, kind in enumerate(kinds) if kind)
        maximum = max(maximum, len(tok.raw(text + '<<SEP>>' + STATE)['ids']) + 2)
        checks += 1
    return dict(max_tokens=maximum, checks=checks, worst_prompt=normal,
                proof='Every numeric domain exhausted at fixed ByteLevel pretoken boundaries; concatenated IDs checked, plus all omitted/safe/game-over combinations.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canister', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--candid', default='build/tetris-query-v3-history/verdict-engine.did')
    args = parser.parse_args()
    def run(command):
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
        return p.stdout
    network = json.loads(run(['icp','network','status','-e','local','--json']))
    if not network.get('managed') or urlparse(network['api_url']).hostname not in {'localhost','127.0.0.1','::1'}:
        raise ValueError('An existing managed loopback network is required')
    status = json.loads(run(['icp','canister','status',args.canister,'-e','local','--identity','anonymous','--json']))
    if status['module_hash'].removeprefix('0x') != WASM:
        raise ValueError('Unexpected Wasm; do not upgrade automatically')
    base = ['icp','canister','call',args.canister,'-e','local','--identity',args.owner,'--candid',args.candid,'--query']
    ready = run(base + ['tetris_v3_status','()'])
    model_blob = 'blob "' + ''.join('\\' + MODEL[i:i + 2] for i in range(0,64,2)) + '"'
    assert model_blob in ready and 'warmed = true' in ready and 'max_tokens = 52' in ready
    fixtures = cases()
    OUT.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        path = OUT / name
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2) + '\n')
        temp.replace(path)
    provenance = dict(started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=network['api_url'],
                      canister=args.canister, wasm=WASM, model=MODEL,
                      tokenizer=hashlib.sha256((ROOT / 'models/verdict-151m/tokenizer.json').read_bytes()).hexdigest(),
                      fixtures=hashlib.sha256(json.dumps(fixtures,sort_keys=True).encode()).hexdigest())
    save('result.json', dict(provenance=provenance, status='running', passed=False))
    save('fixtures.json',dict(provenance=provenance, thresholds=dict(correct=90,per_action=16,swapped_pairs=40),cases=fixtures))
    tok = Tokenizer()
    rows = []
    try:
        bound = token_bound(tok)
        assert bound['max_tokens'] <= 52, bound
        save('token-bound.json',dict(provenance=provenance, **bound))
        print('PASS token bound',bound['max_tokens'],'checks',bound['checks'],flush=True)
        def query(req):
            labels = req['labels']
            options = ';'.join('record {id=' + json.dumps(str(i)) + ';"text"=' + json.dumps(s) + '}' for i,s in enumerate(labels))
            argument = '(record {question="";state=' + json.dumps(req['text']) + ';abstention=false;temperature=1.0:float64;options=vec {' + options + '}})'
            tokens = len(tok.raw(''.join('<<LABEL>>' + s for s in labels) + '<<SEP>>' + req['text'])['ids']) + 2
            assert tokens <= 52
            start = time.monotonic()
            reply = run(base + ['decide_query',argument])
            latency = (time.monotonic() - start) * 1000
            assert 'Ok = record' in reply and model_blob in reply, reply
            actual_tokens = int(re.search(r'input_tokens = (\d+)',reply)[1])
            instructions = int(re.search(r'measured_instructions = ([\d_]+)',reply)[1].replace('_',''))
            action = int(re.search(r'selected = "(\d+)"',reply)[1])
            values = re.search(r'probabilities = vec \{([^}]+)',reply)[1]
            scores = [float(v.strip().split(':')[0].replace('_','')) for v in values.split(';') if v.strip()]
            assert actual_tokens == tokens and instructions < 5_000_000_000 and len(scores)==len(labels)
            assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores) and abs(sum(scores)-1)<1e-5
            assert action == max(range(len(scores)), key=lambda i:scores[i])
            return dict(action=action, scores=scores, input_tokens=tokens, instructions=instructions, latency_ms=latency)
        prefix, text = bound['worst_prompt'].split('<<SEP>>')
        boundary = query(dict(text=text, labels=prefix.split('<<LABEL>>')[1:]))
        save('boundary-query.json',dict(provenance=provenance, **boundary))
        for case in fixtures:
            req=request(case)
            answer=query(req)
            rows.append(dict(**case, request=req, **answer))
            save('responses.json',dict(provenance=provenance, rows=rows))
            if len(rows)%10==0:
                print('query',len(rows),'/100 correct',sum(r['action']==r['expected'] for r in rows),flush=True)
        result=summarize(rows)
        result.update(provenance=provenance,status='complete',actual_queries=101,query_errors=0,
                      max_tokens=max(r['input_tokens'] for r in [boundary,*rows]),
                      max_instructions=max(r['instructions'] for r in [boundary,*rows]),
                      production_changed=False,
                      next_step='implement bounded time-aware search' if result['passed'] else 'stop before search, v4 and UI implementation')
        save('result.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as error:
        save('result.json',dict(provenance=provenance,status='failed',passed=False,completed_questions=len(rows),error=str(error)))
        raise
    finally:
        tok.close()

if __name__ == '__main__':
    main()
