#!/usr/bin/env python3
"""Offline prompt/token study; choose on development cases, confirm on held-out cases."""
import collections, hashlib, itertools, json, random
from pathlib import Path
from game_feasibility import Probe,ROOT
OUT=ROOT/'artifacts/tetris-prompt-budget'
VARIANTS={
 'baseline':('Tetris. Clear lines, avoid holes and height.','named'),
 'explicit':('More lines, fewer holes, lower height','named'),
 'compact':('Max lines,min holes,height','named'),
 'tuple':('lines/holes/height: max/min/min','tuple'),
}
def labels(opts,kind):
 return [f"{c['lines']}/{c['holes']}/{c['height']}" if kind=='tuple' else f"lines{c['lines']} holes{c['holes']} height{c['height']}" for c in opts]
def feat(c):return (c['lines'],c['holes'],c['height'])
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 dev=json.loads((ROOT/'artifacts/tetris-choice-diagnosis/protocol.json').read_text())['cases']
 corpus=json.loads((ROOT/'artifacts/tetris-choice-diagnosis/corpus.json').read_text())
 seen={tuple(sorted(feat(c) for c in case['options'])) for case in dev}
 available=[]
 for r in corpus:
  key=tuple(sorted(feat(c) for c in r['offered']))
  if len(key)==4 and len(set(key))==4 and key not in seen:
   available.append(r);seen.add(key)
 hold=[]
 for i in range(6):
  r=available[i*(len(available)-1)//5]
  hold.append(dict(id=f'hold-real-{i}',kind='real',options=r['offered']))
 rng=random.Random(9301)
 for i in range(6):
  good=(rng.randrange(1,5),rng.randrange(1,10),rng.randrange(2,9))
  opts=[good]+[(rng.randrange(good[0]+1),good[1]+rng.randrange(1,10),good[2]+rng.randrange(1,8)) for _ in range(3)]
  hold.append(dict(id=f'hold-dominance-{i}',kind='dominance',options=[dict(lines=l,holes=h,height=y,evaluation=10*l-8*h-y) for l,h,y in opts]))
 save('protocol.json',dict(variants=VARIANTS,development_cases=dev,held_out_cases=hold,development_orders='four cyclic rotations',held_out_orders='all 24 permutations',selection='Highest dominance accuracy among token-saving prompts; then lowest real-case dominated-choice rate; then fewer tokens. Require development dominance improvement over baseline.',canister_calls=0,model_sha256=hashlib.sha256((ROOT/'models/verdict-pack/manifest.json').read_bytes()).hexdigest(),probe_sha256=hashlib.sha256((ROOT/'target/release/tetris_probe').read_bytes()).hexdigest()))
 p=Probe();token_bounds={};rows=[]
 try:
  for name,(text,kind) in VARIANTS.items():
   counts=[]
   for axis,limit in [('lines',4),('holes',200),('height',20)]:
    for value in range(limit+1):
     c=dict(lines=4,holes=200,height=20);c[axis]=value
     counts.append(p.rpc(dict(text=text,labels=labels([c]*4,kind)))['tokens'])
   token_bounds[name]=dict(min=min(counts),max=max(counts))
  save('token-bounds.json',token_bounds);print('TOKENS',token_bounds,flush=True)
  def evaluate(stage,names,cases,orders):
   for name in names:
    text,kind=VARIANTS[name]
    for case in cases:
     opts=case['options']
     for order in orders:
      pos=p.ask(text,labels([opts[i] for i in order],kind));chosen=order[pos];c=opts[chosen]
      rows.append(dict(stage=stage,variant=name,case=case['id'],kind=case['kind'],order=order,selected=chosen,best=c['evaluation']==max(v['evaluation'] for v in opts),dominated=any(v['lines']>=c['lines'] and v['holes']<=c['holes'] and v['height']<=c['height'] and feat(v)!=feat(c) for v in opts)))
    save('responses.json',rows);print(stage,name,summary(stage,name),flush=True)
  def summary(stage,name):
   out={}
   for kind in ['real','dominance']:
    rs=[r for r in rows if r['stage']==stage and r['variant']==name and r['kind']==kind];groups=collections.defaultdict(list)
    for r in rs:groups[r['case']].append(r['selected'])
    out[kind]=dict(tests=len(rs),best=sum(r['best'] for r in rs),dominated=sum(r['dominated'] for r in rs),stable_cases=sum(len(set(v))==1 for v in groups.values()),cases=len(groups))
   return out
  evaluate('development',list(VARIANTS),dev,[tuple((j+i)%4 for j in range(4)) for i in range(4)])
  ds={name:summary('development',name) for name in VARIANTS}
  eligible=[n for n in VARIANTS if token_bounds[n]['max']<token_bounds['baseline']['max'] and ds[n]['dominance']['best']>ds['baseline']['dominance']['best']]
  winner=min(eligible,key=lambda n:(-ds[n]['dominance']['best'],ds[n]['real']['dominated'],token_bounds[n]['max'])) if eligible else None
  save('selection.json',dict(winner=winner,development=ds,token_bounds=token_bounds))
  if winner:evaluate('held_out',['baseline',winner],hold,list(itertools.permutations(range(4))))
  hs={n:summary('held_out',n) for n in ['baseline',winner]} if winner else {}
  accepted=bool(winner and hs[winner]['dominance']['best']>hs['baseline']['dominance']['best'] and hs[winner]['real']['dominated']<=hs['baseline']['real']['dominated'])
  result=dict(selected=winner,accepted=accepted,tokens=token_bounds,development=ds,held_out=hs,unique_inferences=len(p.rows))
  save('result.json',result);save('inferences.json',p.rows);print(json.dumps(result,indent=2),flush=True)
 finally:p.close()
if __name__=='__main__':main()
