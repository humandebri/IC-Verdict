#!/usr/bin/env python3
"""JSONL inference bridge; Rust owns all game physics and candidates."""
import json,sys,os
from game_feasibility import Probe,ROOT
from tetris_prompt_search import labels,PROMPTS
from tetris_rank_search import encode,TEXTS
from tetris_short_rank_search import labels as short_labels,VARIANTS
from tetris_cost_prompt import labels as cost_labels
p=Probe()
try:
 for line in sys.stdin:
  r=json.loads(line);kind,text=('named',PROMPTS['task']) if r['policy']=='baseline' else ('ranks',PROMPTS['compact'])
  ls=labels(r['options'],kind)
  if r['policy'].startswith(('full/','full_holes/','omit/','omit_holes/')):
   style,i=r['policy'].split('/');text=TEXTS[int(i)];ls=encode(r['options'],style)
  if r['policy'] in VARIANTS:
   v=VARIANTS[r['policy']];text=v['text'];ls=short_labels(r['options'],v['style'])
  if r['policy']=='cost/0':
   ls=cost_labels(r['options']);text='Min holes'
  print(p.ask(text,ls),flush=True)
finally:
 (ROOT/('artifacts/tetris-prompt-search/rollout-inferences-'+os.environ.get('TETRIS_RUN_TAG',os.environ.get('TETRIS_SAMPLER','uniform'))+'.json')).write_text(json.dumps(p.rows,indent=2)+'\n')
 p.close()
