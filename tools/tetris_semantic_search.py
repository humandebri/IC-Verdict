#!/usr/bin/env python3
"""Matched feature descriptions for the frozen classifier, no joint scores."""
import json
from game_feasibility import Probe,ROOT
from tetris_prompt_search import feature,summarize
OUT=ROOT/'artifacts/tetris-prompt-search'
TEXTS=['many clears few holes low stack','Few holes. Low stack. Clear lines.','few holes low stack','Tetris. Clear lines, avoid holes and height.']
def encode(opts,omit):
 keys=['lines','holes','height'];nouns=['clears','holes','stack'];adjectives=[['few','some','many'],['few','some','many'],['low','medium','high']];out=[]
 for c in opts:
  words=[]
  for j,k in enumerate(keys):
   a,b=min(o[k] for o in opts),max(o[k] for o in opts)
   if omit and a==b:continue
   adj='equal' if a==b else adjectives[j][0 if c[k]==a else 2 if c[k]==b else 1]
   words.extend([adj,nouns[j]])
  out.append(' '.join(words) or 'equal')
 return out

def main():
 cases=json.loads((OUT/'protocol.json').read_text())['development_cases'];p=Probe();results=[]
 variants={f'semantic-{omit}/{i}':dict(omit=omit,text=t) for omit in [False,True] for i,t in enumerate(TEXTS)}
 (OUT/'semantic-protocol.json').write_text(json.dumps(dict(cases=cases,variants=variants),indent=2))
 try:
  for name,v in variants.items():
   counts=[p.rpc(dict(text=v['text'],labels=encode(c['options'],v['omit'])))['tokens'] for c in cases]
   if max(counts)>52:continue
   rows=[]
   for c in cases:
    opts=c['options']
    for order in [list(range(4)),list(reversed(range(4)))]:
     ls=encode([opts[i] for i in order],v['omit']);sel=order[p.ask(v['text'],ls)];chosen=opts[sel]
     rows.append(dict(case=c['id'],kind=c['kind'],selected=sel,best=chosen['evaluation']==max(o['evaluation'] for o in opts),dominated=any(o['lines']>=chosen['lines'] and o['holes']<=chosen['holes'] and o['height']<=chosen['height'] and feature(o)!=feature(chosen) for o in opts)))
   m=summarize(rows);results.append(dict(name=name,variant=v,tokens=max(counts),metrics=m,rows=rows));print(name,m,flush=True)
   (OUT/'semantic-results.json').write_text(json.dumps(results,indent=2))
  (OUT/'semantic-inferences.json').write_text(json.dumps(p.rows,indent=2))
 finally:p.close()
if __name__=='__main__':main()
