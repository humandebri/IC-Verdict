#!/usr/bin/env python3
"""Per-feature cost encoding. No joint score or preferred move is supplied."""
import itertools,json
from game_feasibility import Probe,ROOT
TEXT='Min holes'
def labels(opts):
 maximum=max(c['lines'] for c in opts)
 features=[(c['holes'],c['height'],maximum-c['lines']) for c in opts]
 def rank(i,value):
  a=min(v[i] for v in features);b=max(v[i] for v in features)
  return 'equal' if a==b else 'min' if value==a else 'max' if value==b else 'mid'
 return [','.join(f'{rank(i,v)} {name}' for i,(v,name) in enumerate(zip(row,['holes','height','missed']))) for row in features]
def main():
 p=Probe();rows=[];out=ROOT/'artifacts/tetris-prompt-search'
 try:
  for metric in ['lines','holes','height']:
   opts=[dict(lines=0,holes=0,height=8) for _ in range(4)]
   for i,c in enumerate(opts):c[metric]=i
   expected=3 if metric=='lines' else 0
   for order in itertools.permutations(range(4)):
    selected=order[p.ask(TEXT,labels([opts[i] for i in order]))]
    rows.append(dict(metric=metric,order=order,selected=selected,expected=expected,correct=selected==expected))
  bounds=[]
  for rs in itertools.product(['min','mid','max','equal'],repeat=3):
   label=','.join(f'{r} {k}' for r,k in zip(rs,['holes','height','missed']))
   for n in range(1,5):bounds.append(p.rpc(dict(text=TEXT,labels=[label]*n))['tokens'])
  summary=dict(correct=sum(r['correct'] for r in rows),total=len(rows),per_metric={m:sum(r['correct'] for r in rows if r['metric']==m) for m in ['lines','holes','height']},max_tokens=max(bounds),token_checks=len(bounds),rows=rows,unique_inferences=len(p.rows),note='Exhaustive single-feature rank/order checks, not independent accuracy samples. Values collapse to the same rank labels.')
  (out/'cost-feature-result.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'cost-feature-inferences.json').write_text(json.dumps(p.rows,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='rows'})
 finally:p.close()
if __name__=='__main__':main()
