"""Validate document consistency and score examples, not runtime correctness."""
from pathlib import Path
from fractions import Fraction
import json
import re
import sys

root = Path(__file__).resolve().parents[1]
checks = []

def check(label, value):
    checks.append((label, bool(value)))

profile = json.loads((root / 'profiles/compact128.json').read_text())
p = profile['request']
check('Core contains Choice, Noul, Score', set(profile['primitives']) == {'choice', 'noul', 'score'})
check('1 question per evaluation, 3 per workflow', p['max_questions_per_engine_update'] == 1 and p['max_question_slots_per_workflow'] == 3)
check('Compact128 prefix budget leaves at least 63 state tokens', p['max_total_tokens'] - p['max_static_prefix_tokens'] - p['final_separator_tokens'] == 63)
check('Choice 2..5; Score 3..7, default 5', (p['choice_min_options'],p['choice_max_options'],p['score_min_bins'],p['score_default_bins'],p['score_max_bins']) == (2,5,3,5,7))
check('Silent truncation disabled', not p['silent_truncation'])
check('Live transfer disabled; amount cap zero', not profile['safety']['live_transfer_enabled'] and profile['safety']['autonomous_amount_limit_atoms'] == 0)
check('Unverified revisions/thresholds are not invented', profile['model_revision'] is None and profile['safety']['approved_policy_thresholds'] is None)

M = 1_000_000

def metrics(masses):
    if not (3 <= len(masses) <= 7) or any(not isinstance(v,int) or v < 0 for v in masses) or sum(masses) != M:
        raise ValueError('Invalid score distribution')
    numerator = sum(i * m for i,m in enumerate(masses))
    denom = len(masses)-1
    normalized = (2*numerator + denom)//(2*denom)  # positive round-half-up
    return numerator, normalized

a = [0,0,M,0,0]
b = [M//2,0,0,0,M//2]
check('Equal normalized means can have different upper tails', metrics(a)[1] == metrics(b)[1] == 500_000 and sum(a[3:]) == 0 and sum(b[3:]) == 500_000)
for k in range(3,8):
    lo = [M] + [0]*(k-1)
    hi = [0]*(k-1) + [M]
    check(f'{k} bins: score endpoint units correct', metrics(lo) == (0,0) and metrics(hi) == ((k-1)*M, M))
    for i in range(k):
        masses = [0]*k; masses[i] = M
        for j in range(k):
            tail = sum(masses[j:])
            cdf_before = sum(masses[:j])
            check(f'{k} bins: exact CDF/tail complement at point {i}, threshold {j}', tail + cdf_before == M)
for bad in [[500_000,500_000], [M,0,1], [M,-1,1], [0,0,0]]:
    try: metrics(bad); rejected = False
    except ValueError: rejected = True
    check(f'Invalid distribution rejected: {bad}', rejected)

# Strict exact rational test of the rounding rule for representative values.
for d in range(2,7):
    for n in [0,1,d//2,d,M*d-1,M*d]:
        want = int(Fraction(n,d) + Fraction(1,2))
        check(f'Positive rounding n={n}, d={d}', (2*n+d)//(2*d) == want)

adr_files = sorted((root/'adr').glob('ADR-*.md'))
check('16 ADRs exist', len(adr_files) == 16)
for i,f in enumerate(adr_files,1):
    text = f.read_text()
    check(f'ADR-{i:03} complete', f'ADR-{i:03}' in text and all(h in text for h in ['## 背景','## 決定','## トレードオフと不採用案','## 検証条件']))
source_ids = set(re.findall(r'\| (S\d{2}) \|', (root/'SOURCES.md').read_text()))
for f in root.rglob('*.md'):
    if f.name.startswith('ALL_IN_ONE'): continue
    text=f.read_text()
    check(f'{f.relative_to(root)}: sources resolve', set(re.findall(r'\[(S\d{2})\]',text)) <= source_ids)
    check(f'{f.relative_to(root)}: fences paired', text.count('```') % 2 == 0)
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if target.startswith(('https://','http://','#')): continue
        check(f'{f.relative_to(root)}: local link {target}', (f.parent/target.split('#')[0]).exists())
    check(f'{f.relative_to(root)}: no Hangul typo', not re.search('[\uac00-\ud7af]',text))
review=(root/'review/REVIEW.md').read_text()
check('All 20 review records present', all(f'V2-R{i:02}' in review for i in range(1,21)))
state=(root/'specs/WORKFLOW_AND_TX.md').read_text()
check('All 10 invariants present', all(f'I{i:02}:' in state for i in range(1,11)))
check('Unknown and TooOld not conflated', '先行unknown' in state and 'TooOld/BadFee' in state)
check('Score mean/tail gate present', '上側' in state and 'I07' in state)

out=['IC-Laya v2 DOCUMENT / NUMERIC EXAMPLE CHECKS',
     'No model inference, Rust compilation, canister benchmark, security proof or ledger integration.', '']
out += [f'{"PASS" if v else "FAIL"}  {label}' for label,v in checks]
out += ['', f'{sum(v for _,v in checks)}/{len(checks)} checks passed.']
(root/'review/CHECKS.txt').write_text('\n'.join(out)+'\n')
print(out[-1])
for label,value in checks:
    if not value: print('FAILED:',label)
sys.exit(0 if all(v for _,v in checks) else 1)
