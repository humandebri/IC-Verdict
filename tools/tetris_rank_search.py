#!/usr/bin/env python3
"""Follow-up search on development data only; game trials remain a separate gate."""
import json,itertools
from game_feasibility import Probe,ROOT
from tetris_prompt_search import labels,feature,summarize
OUT=ROOT/'artifacts/tetris-prompt-search'
TEXTS=['Max clears,min holes,height','Min holes,height;max clears','Min holes,then height','Few holes,low height','Avoid holes. Keep low.','Min holes','Least holes','Tetris: few holes, low stack','More clears,fewer holes,lower height']
def encode(opts,style):
 ls=labels(opts,'ranks')
 cols=[s.split(',') for s in ls]
 keys=['lines','holes','height']
 keep=[j for j,k in enumerate(keys) if len({c[k] for c in opts})>1] if style.startswith('omit') else list(range(3))
 if not keep:keep=[1]
 if style.endswith('holes'):keep=sorted(keep,key=lambda j:[1,2,0].index(j))
 return [','.join(row[j] for j in keep) for row in cols]
def main():
 protocol=json.loads((OUT/'protocol.json').read_text());cases=protocol['development_cases'];p=Probe();rs=[];bounds={}
 variants={f'{s}/{i}':dict(style=s,text=t) for s in ['full','full_holes','omit','omit_holes'] for i,t in enumerate(TEXTS)}
 (OUT/'rank-protocol.json').write_text(json.dumps(dict(variants=variants,cases=cases),indent=2))
 try:
  for cache in ['cache.json','rollout-inferences.json']:
   path=OUT/cache
   if path.exists():
    for r in json.loads(path.read_text()):p.cache[json.dumps([r['request']['text'],r['request']['labels']])]=r['result']
  for name,v in variants.items():
   bounds[name]=max(p.rpc(dict(text=v['text'],labels=encode(c['options'],v['style'])))['tokens'] for c in cases)
   if bounds[name]>52:continue
   rows=[]
   for c in cases:
    opts=c['options']
    for order in [list(range(4)),list(reversed(range(4)))]:
     ls=encode([opts[i] for i in order],v['style']);sel=order[p.ask(v['text'],ls)];chosen=opts[sel]
     rows.append(dict(case=c['id'],kind=c['kind'],selected=sel,best=chosen['evaluation']==max(o['evaluation'] for o in opts),dominated=any(o['lines']>=chosen['lines'] and o['holes']<=chosen['holes'] and o['height']<=chosen['height'] and feature(o)!=feature(chosen) for o in opts)))
   m=summarize(rows);rs.append(dict(name=name,variant=v,tokens=bounds[name],metrics=m,rows=rows));print(name,m,flush=True)
   (OUT/'rank-results.json').write_text(json.dumps(rs,indent=2))
  (OUT/'rank-inferences.json').write_text(json.dumps(p.rows,indent=2))
 finally:p.close()
if __name__=='__main__':main()
