#!/usr/bin/env python3
"""Progressive offline search of representations; never embeds a joint heuristic score."""
import argparse,collections,hashlib,itertools,json,random
from pathlib import Path
from game_feasibility import Probe,ROOT
OUT=ROOT/'artifacts/tetris-prompt-search'
PROMPTS={
 'task':'Tetris. Clear lines, avoid holes and height.',
 'direction':'More clears, fewer holes, lower height.',
 'safe':'A safe low stack with few holes and cleared lines.',
 'priority':'Avoid holes first; then clear lines and stay low.',
 'question':'Question: Best move?\n\nContext:\nTetris: clear lines, avoid holes.',
 'target':'Many cleared lines, few holes, low height.',
 'short':'Best Tetris move.',
 'compact':'Max clears,min holes,height',
}
KINDS=['named','cleared','words','units','relative','ranks','relative_holes_first','relative_natural']
SMALL=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']
def words(n):
 if n<=20:return SMALL[n]
 if n<100:return ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'][n//10]+(' '+SMALL[n%10] if n%10 else '')
 return SMALL[n//100]+' hundred'+(' '+words(n%100) if n%100 else '')
def labels(opts,kind):
 out=[]
 for c in opts:
  l,h,y=c['lines'],c['holes'],c['height']
  def rel(key,lo,mid,hi,tied):
   vals=[v[key] for v in opts];a,b=min(vals),max(vals)
   return tied if a==b else lo if c[key]==a else hi if c[key]==b else mid
  if kind=='named':s=f'lines{l} holes{h} height{y}'
  elif kind=='cleared':s=f'cleared{l} holes{h} height{y}'
  elif kind=='words':s=f'lines{words(l)} holes{words(h)} height{words(y)}'
  elif kind=='units':s=f'{l} lines {h} holes {y} high'
  elif kind.startswith('relative'):
   a=rel('lines','fewer clears','some clears','more clears','no clears' if l==0 else 'equal clears')
   b=rel('holes','fewer holes','some holes','more holes','no holes' if h==0 else 'equal holes')
   d=rel('height','lower','medium','higher','equal height')
   if kind=='relative_natural':s=f'{a}, {b}, {d} stack'
   else:s=','.join([b,a,d] if kind=='relative_holes_first' else [a,b,d])
  elif kind=='ranks':
   a=rel('lines','min clears','mid clears','max clears','equal clears')
   b=rel('holes','min holes','mid holes','max holes','equal holes')
   d=rel('height','min height','mid height','max height','equal height');s=','.join([a,b,d])
  else:raise ValueError(kind)
  out.append(s)
 return out

def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def feature(c):return c['lines'],c['holes'],c['height']
def summarize(rows):
 out={}
 for kind in ['real','dominance']:
  rs=[r for r in rows if r['kind']==kind];g=collections.defaultdict(list)
  for r in rs:g[r['case']].append(r['selected'])
  out[kind]=dict(tests=len(rs),best=sum(r['best'] for r in rs),dominated=sum(r['dominated'] for r in rs),stable=sum(len(set(v))==1 for v in g.values()),cases=len(g))
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 original=json.loads((ROOT/'artifacts/tetris-choice-diagnosis/protocol.json').read_text())['cases']
 variants={f'{k}/{p}':dict(kind=k,text=text) for k in KINDS for p,text in PROMPTS.items()}
 variants['baseline']=dict(kind='named',text=PROMPTS['task'])
 # Freeze a new holdout before inference. Previous studies' holdouts now count as development data.
 prior=json.loads((ROOT/'artifacts/tetris-prompt-budget/protocol.json').read_text())
 corpus=json.loads((ROOT/'artifacts/tetris-choice-diagnosis/corpus.json').read_text())
 seen={tuple(sorted(feature(c) for c in x['options'])) for x in original+prior['held_out_cases']}
 hold=[]
 for r in reversed(corpus):
  key=tuple(sorted(feature(c) for c in r['offered']))
  if len(set(key))==4 and key not in seen:
   seen.add(key);hold.append(dict(id=f"fresh-real-{len(hold)}",kind='real',options=r['offered'],seed=r['seed'],turn=r['turn']))
  if len(hold)==12:break
 rng=random.Random(20260924)
 for i in range(12):
  good=(rng.randrange(1,5),rng.randrange(0,12),rng.randrange(2,12));opts=[good]+[(rng.randrange(good[0]+1),good[1]+rng.randrange(1,10),good[2]+rng.randrange(1,9)) for _ in range(3)]
  hold.append(dict(id=f'fresh-dominance-{i}',kind='dominance',options=[dict(lines=l,holes=h,height=y,evaluation=10*l-8*h-y) for l,h,y in opts]))
 screening=[original[i] for i in [0,3,8,9,10,13]]
 protocol=dict(variants=variants,screening_cases=screening,development_cases=original+prior['held_out_cases'],fresh_holdout=hold,screening_orders=[[0,1,2,3],[3,2,1,0]],development_orders='four cyclic rotations',confirmation_orders='four cyclic rotations',selection='Screen all token-valid variants, expand top 6 by dominance success then real dominated count then shorter tokens; choose one finalist before fresh holdout. Require >=80% fresh dominance success and fewer real dominated choices than baseline.',code_assistance='Relative representations compute per-feature comparisons only. No combined score, recommended ID, or best/worst move label is passed.',max_tokens=52,model_sha256=hashlib.sha256((ROOT/'models/verdict-pack/manifest.json').read_bytes()).hexdigest(),canister_calls=0)
 save('protocol.json',protocol)
 probe=Probe();rows=[];bounds={};cache_path=OUT/'cache.json'
 # Reuse prior exact request/response pairs. All cached studies pin the same checkpoint and tokenizer.
 for path in [ROOT/'artifacts/tetris-choice-diagnosis/inferences.json',ROOT/'artifacts/tetris-prompt-budget/inferences.json',cache_path]:
  if path.exists():
   for r in json.loads(path.read_text()):probe.cache[json.dumps([r['request']['text'],r['request']['labels']])]=r['result']
 cache_records=[]
 def snapshot():
  seen={}
  for r in cache_records+probe.rows:seen[json.dumps(r['request'],sort_keys=True)]=r
  save('cache.json',list(seen.values()));save('responses.json',rows)
 def run(stage,name,cases,orders):
  v=variants[name];batch=[]
  for case in cases:
   opts=case['options']
   for order in orders:
    subset=[opts[i] for i in order];ls=labels(subset,v['kind']);slot=probe.ask(v['text'],ls);selected=order[slot];c=opts[selected]
    batch.append(dict(stage=stage,variant=name,case=case['id'],kind=case['kind'],order=order,selected=selected,best=c['evaluation']==max(o['evaluation'] for o in opts),dominated=any(o['lines']>=c['lines'] and o['holes']<=c['holes'] and o['height']<=c['height'] and feature(o)!=feature(c) for o in opts)))
  rows.extend(batch);snapshot();m=summarize(batch);print(stage,name,m,flush=True);return m
 try:
  if cache_path.exists():cache_records=json.loads(cache_path.read_text())
  # Include adversarial numeric lengths and tie/mid/min/max patterns in the token screen.
  extremes=[dict(lines=l,holes=h,height=y) for l,h,y in [(0,0,0),(1,99,9),(3,199,19),(4,200,20)]]
  check=original+hold+[dict(options=extremes),dict(options=[dict(lines=4,holes=200,height=20)]*4)]
  valid=[]
  for name,v in variants.items():
   counts=[probe.rpc(dict(text=v['text'],labels=labels(c['options'],v['kind'])))['tokens'] for c in check]
   bounds[name]=max(counts)
   if max(counts)<=52:valid.append(name)
  save('token-bounds.json',bounds);print('TOKEN_VALID',len(valid),'/',len(variants),flush=True)
  scores={name:run('screen',name,screening,[(0,1,2,3),(3,2,1,0)]) for name in valid}
  def rank(name,metrics):return (-metrics[name]['dominance']['best'],metrics[name]['real']['dominated'],bounds[name])
  finalists=sorted([n for n in valid if n!='baseline'],key=lambda n:rank(n,scores))[:6]
  development={name:run('development',name,original+prior['held_out_cases'],[tuple((i+j)%4 for j in range(4)) for i in range(4)]) for name in ['baseline']+finalists}
  winner=min(finalists,key=lambda n:rank(n,development));save('selection.json',dict(winner=winner,finalists=finalists,screen=scores,development=development))
  confirmation={name:run('confirmation',name,hold,[tuple((i+j)%4 for j in range(4)) for i in range(4)]) for name in ['baseline',winner]}
  b=confirmation['baseline'];w=confirmation[winner]
  accepted=w['dominance']['best']>=.8*w['dominance']['tests'] and w['real']['dominated']<b['real']['dominated']
  save('result.json',dict(winner=winner,accepted=accepted,tokens=bounds[winner],baseline_tokens=bounds['baseline'],confirmation=confirmation,development=development,unique_new_inferences=len(probe.rows)));snapshot();print('RESULT',winner,accepted,confirmation,flush=True)
 finally:probe.close()
if __name__=='__main__':main()
