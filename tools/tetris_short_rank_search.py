#!/usr/bin/env python3
"""Token ablation of holes-first relative features on development cases."""
import json
from game_feasibility import Probe,ROOT
from tetris_rank_search import encode
from tetris_prompt_search import feature,summarize
OUT=ROOT/'artifacts/tetris-prompt-search'
VARIANTS={f'{style}/{i}':dict(style=style,text=text) for style in ['space','same'] for i,text in enumerate(['Min holes','Min holes,then height'])}
def labels(opts,style):
 ls=[s.replace(',',' ') for s in encode(opts,'full_holes')]
 return [s.replace('equal','same') for s in ls] if style=='same' else ls

def main():
 cases=json.loads((OUT/'protocol.json').read_text())['development_cases'];p=Probe();results=[]
 (OUT/'short-protocol.json').write_text(json.dumps(dict(cases=cases,variants=VARIANTS),indent=2))
 try:
  for name,v in VARIANTS.items():
   counts=[p.rpc(dict(text=v['text'],labels=labels(c['options'],v['style'])))['tokens'] for c in cases];rows=[]
   for c in cases:
    opts=c['options']
    for order in [list(range(4)),list(reversed(range(4)))]:
     ls=labels([opts[i] for i in order],v['style']);sel=order[p.ask(v['text'],ls)];chosen=opts[sel]
     rows.append(dict(case=c['id'],kind=c['kind'],selected=sel,best=chosen['evaluation']==max(o['evaluation'] for o in opts),dominated=any(o['lines']>=chosen['lines'] and o['holes']<=chosen['holes'] and o['height']<=chosen['height'] and feature(o)!=feature(chosen) for o in opts)))
   m=summarize(rows);results.append(dict(name=name,variant=v,tokens=max(counts),metrics=m,rows=rows));print(name,m,flush=True)
   (OUT/'short-results.json').write_text(json.dumps(results,indent=2))
  (OUT/'short-inferences.json').write_text(json.dumps(p.rows,indent=2))
 finally:p.close()
if __name__=='__main__':main()
