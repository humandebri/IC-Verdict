#!/usr/bin/env python3
"""Fresh feature cases and token boundary checks after final prompt selection."""
import itertools,json,random
from game_feasibility import Probe,ROOT
from tetris_prompt_search import labels,feature,summarize
from tetris_rank_search import encode
OUT=ROOT/'artifacts/tetris-prompt-search'
def main():
 protocol=json.loads((OUT/'protocol.json').read_text());old=protocol['development_cases']+protocol['fresh_holdout']
 used={tuple(sorted(feature(c) for c in r['options'])) for r in old}
 cases=[]
 for r in json.loads((ROOT/'artifacts/tetris-choice-diagnosis/corpus.json').read_text()):
  key=tuple(sorted(feature(c) for c in r['offered']))
  if len(set(key))==4 and key not in used:
   used.add(key);cases.append(dict(id=f'final-real-{len(cases)}',kind='real',options=r['offered']))
  if len(cases)==12:break
 rng=random.Random(20260925)
 for metric in ['lines','holes','height']:
  for i in range(4):
   vals=sorted(rng.sample(range(5) if metric=='lines' else range(20),4));opts=[dict(lines=0,holes=0,height=8) for _ in vals]
   for o,v in zip(opts,vals):o[metric]=v
   for o in opts:o['evaluation']=10*o['lines']-8*o['holes']-o['height']
   cases.append(dict(id=f'final-{metric}-{i}',kind='dominance',options=opts))
 (OUT/'final-feature-protocol.json').write_text(json.dumps(dict(cases=cases,selection='Previously frozen final-selection.json; do not retune on these cases.'),indent=2))
 p=Probe();result={}
 try:
  bounds=[]
  for ranks in itertools.product(['min','mid','max','equal'],repeat=3):
   label=','.join(f'{r} {k}' for r,k in zip(ranks,['holes','height','clears']))
   for n in range(1,5):bounds.append(p.rpc(dict(text='Min holes',labels=[label]*n))['tokens'])
  for name in ['baseline','full_holes/5']:
   rows=[]
   for c in cases:
    opts=c['options']
    for j in range(4):
     order=[(i+j)%4 for i in range(4)];ls=labels([opts[i] for i in order],'named') if name=='baseline' else encode([opts[i] for i in order],'full_holes')
     text='Tetris. Clear lines, avoid holes and height.' if name=='baseline' else 'Min holes'
     sel=order[p.ask(text,ls)];chosen=opts[sel]
     rows.append(dict(case=c['id'],kind=c['kind'],selected=sel,best=chosen['evaluation']==max(o['evaluation'] for o in opts),dominated=any(o['lines']>=chosen['lines'] and o['holes']<=chosen['holes'] and o['height']<=chosen['height'] and feature(o)!=feature(chosen) for o in opts)))
   result[name]=dict(metrics=summarize(rows),rows=rows);print(name,result[name]['metrics'],flush=True)
  result['token_bound']=dict(max=max(bounds),checks=len(bounds),method='All 64 relative-feature labels repeated at each candidate count 1..4; fixed label separators and ASCII tokenizer boundaries.')
  (OUT/'final-feature-result.json').write_text(json.dumps(result,indent=2)+'\n')
  (OUT/'final-feature-inferences.json').write_text(json.dumps(p.rows,indent=2)+'\n')
 finally:p.close()
if __name__=='__main__':main()
