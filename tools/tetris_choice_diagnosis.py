#!/usr/bin/env python3
"""Offline four-choice diagnosis: exact Rust candidate corpus and unchanged model pack."""
import collections
import hashlib
import itertools
import json
import time
from pathlib import Path
from game_feasibility import Probe, ROOT

OUT=ROOT/'artifacts/tetris-choice-diagnosis'
TEXT='Tetris. Clear lines, avoid holes and height.'
def features(c):return (c['lines'],c['holes'],c['height'])
def label(c):return f"lines{c['lines']} holes{c['holes']} height{c['height']}"
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
    corpus=json.loads((OUT/'corpus.json').read_text())
    eligible=[r for r in corpus if len(r['offered'])==4 and len({features(c) for c in r['offered']})==4]
    # Fixed evenly spaced sample, chosen before any model inference.
    real=[eligible[i*(len(eligible)-1)//7] for i in range(8)]
    cases=[]
    for i,r in enumerate(real):cases.append(dict(id=f'real-{i}',kind='real',source={k:r[k] for k in ['seed','policy','turn','piece']},options=r['offered']))
    # Each synthetic winner weakly dominates all alternatives in every feature,
    # and strictly dominates them in at least one feature. These are input probes,
    # not a claim that each feature tuple is a reachable placement on one board.
    synthetic=[[(0,0,4),(0,1,4),(0,5,4),(0,20,4)],[(0,0,2),(0,0,3),(0,0,7),(0,0,15)],
               [(4,0,4),(0,0,4),(1,0,4),(2,0,4)],[(1,0,2),(0,2,6),(0,5,10),(0,10,15)],
               [(0,2,8),(0,3,8),(0,7,8),(0,15,8)],[(2,1,4),(1,2,5),(0,3,7),(0,9,12)]]
    for i,opts in enumerate(synthetic):cases.append(dict(id=f'dominance-{i}',kind='dominance',options=[dict(lines=l,holes=h,height=y,evaluation=10*l-8*h-y) for l,h,y in opts]))
    protocol=dict(text=TEXT,diagnostic_script_sha256=sha('tools/tetris_choice_diagnosis.py'),rust_source_sha256=sha('canisters/verdict-engine/src/tetris_demo.rs'),model_sha256=sha('models/verdict-pack/manifest.json'),pack_sha256=sha('models/verdict-pack/model.bin'),tokenizer_sha256=sha('models/verdict-151m/tokenizer.json'),probe_sha256=sha('target/release/tetris_probe'),corpus_sha256=sha('artifacts/tetris-choice-diagnosis/corpus.json'),cases=cases,orders='all 24 permutations per case',max_tokens=52,canister_calls=0,limitations='Heuristic score is a proxy, not a proven optimal Tetris move; corpus uses two deterministic heuristic trajectories, not model play.')
    save('protocol.json',protocol)
    probe=Probe();results=[];parity=[]
    try:
        frozen=json.loads((ROOT/'artifacts/tetris-placement-query/local-smoke.json').read_text())['results']
        frozen+=[{'receipt':json.loads((ROOT/'artifacts/tetris-placement-query/production-turn-1.json').read_text())['response']}]
        for item in frozen:
            r=item['receipt']
            if r['mode']!=0:continue
            selected=probe.ask(TEXT,[label(c) for c in r['candidates']])
            parity.append(dict(turn=r['turn'],recorded=r['selected'],native=selected,match=selected==r['selected']))
        if not all(r['match'] for r in parity):raise RuntimeError('Native parity mismatch; inspect before interpreting results')
        save('parity.json',parity)
        for case in cases:
            opts=case['options'];rows=[]
            for order in itertools.permutations(range(4)):
                slot=probe.ask(TEXT,[label(opts[i]) for i in order]);chosen=order[slot]
                rows.append(dict(order=order,selected_slot=slot,selected_candidate=chosen,heuristic_best=opts[chosen]['evaluation']==max(c['evaluation'] for c in opts),dominated=any(c['lines']>=opts[chosen]['lines'] and c['holes']<=opts[chosen]['holes'] and c['height']<=opts[chosen]['height'] and features(c)!=features(opts[chosen]) for c in opts)))
            tally=collections.Counter(r['selected_candidate'] for r in rows)
            results.append(dict(id=case['id'],kind=case['kind'],source=case.get('source'),chosen_counts=dict(tally),same_candidate_all_orders=len(tally)==1,majority_consistency=max(tally.values())/24,heuristic_best=sum(r['heuristic_best'] for r in rows),dominated=sum(r['dominated'] for r in rows),rows=rows))
            save('permutations.json',results);print(case['id'],dict(tally),'best',sum(r['heuristic_best'] for r in rows),'/24',flush=True)
        save('inferences.json',probe.rows)
    finally:probe.close()
    losses=[]
    for r in corpus:
        full=max(c['evaluation'] for c in r['all']);four=max(c['evaluation'] for c in r['offered'])
        losses.append(dict(seed=r['seed'],policy=r['policy'],turn=r['turn'],legal_count=len(r['all']),best_all=full,best_four=four,loss=full-four,feature_aliasing=len({features(c) for c in r['offered']})<len(r['offered'])))
    save('candidate-losses.json',losses)
    summary=dict(parity=dict(matches=sum(r['match'] for r in parity),total=len(parity)),permutation={},candidates={})
    for kind in ['real','dominance']:
        rs=[r for r in results if r['kind']==kind];slots=collections.Counter(x['selected_slot'] for r in rs for x in r['rows'])
        summary['permutation'][kind]=dict(cases=len(rs),tests=24*len(rs),same_candidate_all_orders=sum(r['same_candidate_all_orders'] for r in rs),heuristic_best=sum(r['heuristic_best'] for r in rs),dominated_choices=sum(r['dominated'] for r in rs),selected_slots=dict(slots),mean_majority_consistency=sum(r['majority_consistency'] for r in rs)/len(rs))
    for policy in ['all-heuristic','four-heuristic']:
        rs=[r for r in losses if r['policy']==policy]
        summary['candidates'][policy]=dict(boards=len(rs),missed_best=sum(r['loss']>0 for r in rs),mean_score_loss=sum(r['loss'] for r in rs)/len(rs),max_score_loss=max(r['loss'] for r in rs),feature_aliasing=sum(r['feature_aliasing'] for r in rs))
    summary['unique_inferences']=len(probe.rows);summary['max_tokens']=max(r['result']['tokens'] for r in probe.rows)
    save('result.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
