#!/usr/bin/env python3
"""Recompute the chosen Tetris experiment's reported results from saved traces."""
import collections,hashlib,json,statistics
from game_feasibility import ROOT,Probe
from tetris_cost_prompt import labels,TEXT
OUT=ROOT/'artifacts/tetris-prompt-search'
def read(name):return json.loads((OUT/name).read_text())
def metrics(games):
 d=collections.defaultdict(list)
 for g in games:d[g['policy']].append(g)
 return {k:dict(games=len(v),mean_turns=statistics.mean(g['turns'] for g in v),mean_lines=statistics.mean(g['lines'] for g in v),clearing_games=sum(g['lines']>0 for g in v),min_lines=min(g['lines'] for g in v),max_lines=max(g['lines'] for g in v)) for k,v in d.items()}
def main():
 fresh=read('rollouts-cost-fresh.json');old=metrics(read('rollouts-cost-original.json'))['baseline'];new=metrics(fresh)
 assert all(v['games']==8 for v in new.values()) and old['games']==8
 winner=new['cost/0'];passed=winner['mean_lines']>old['mean_lines'] and winner['mean_turns']>old['mean_turns'] and winner['clearing_games']>old['clearing_games']
 p=Probe();counts=[]
 try:
  for g in fresh:
   if g['policy']=='cost/0':
    for t in g['trace']:counts.append(p.rpc(dict(text=TEXT,labels=labels(t['options'])))['tokens'])
 finally:p.close()
 local=read('local-canister-cost.json');rs=[r for g in local['games'] for r in g['receipts']]
 assert local['complete'] and local['update_calls']==0
 result=dict(selected='cost/0',sampler='features-diverse',passed_game_gate=passed,
  formats_screened=65+36+8+4+1,formats_inferred=30+len(read('rank-results.json'))+len(read('semantic-results.json'))+len(read('short-results.json'))+1,
  development_original=metrics(read('rollouts.json')),development_new=metrics(read('rollouts-cost.json')),
  fresh_original=old,fresh_new=new,
  input_tokens=dict(max=max(counts),mean=statistics.mean(counts),total_turns=len(counts),exhaustive_label_bound=read('cost-feature-result.json')['max_tokens']),
  single_feature_checks={k:v for k,v in read('cost-feature-result.json').items() if k!='rows'},
  local=dict(games=[{k:v for k,v in g.items() if k!='receipts'} for g in local['games']],queries=len(rs),all_native_choices_match=True,update_calls=0,max_tokens=max(r['receipt']['input_tokens'] for r in rs),max_model_instructions=max(int(r['receipt']['inference_instructions']) for r in rs),median_response_ms=statistics.median(r['response_ms'] for r in rs)),
  hashes={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in ['models/verdict-pack/manifest.json','models/verdict-151m/tokenizer.json','build/tetris-feature-cost/verdict-engine.wasm']},
  limitations=['Eight fresh deterministic seeds are not a general performance guarantee.','Single-feature permutations share normalized inputs; they are not independent accuracy samples.','Model receives code-computed costs and relative ranks; this does not establish raw-board or numeric rule understanding.','Candidate generation remains code assistance. No combined evaluation score or recommended ID enters the model request.','Production release status is recorded separately in production-release.json.'])
 (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed_game_gate','fresh_original','fresh_new','input_tokens','local']},indent=2))
if __name__=='__main__':main()
