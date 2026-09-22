#!/usr/bin/env python3
"""Regression checks for a freshly installed, warmed verdict-tiny test canister.

This explicitly upgrades the named test canister and attempts an incompatible
upgrade to verify rollback. Use a disposable local project without user state.
The fixed fixture IDs require a fresh canister for each run.
"""
import sys,json,re,time,hashlib,subprocess
from pathlib import Path
import argparse
from urllib.parse import urlparse
from local_integration import decode_blobs,blob
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',type=Path,required=True)
parser.add_argument('--identity',required=True)
parser.add_argument('--controller',required=True)
parser.add_argument('--canister',default='verdict-engine')
args=parser.parse_args()
PROJECT=args.project
CANISTER=args.canister
if PROJECT.resolve()==ROOT.resolve():raise ValueError('use a disposable test project')
common=['-e','local','--project-root-override',str(PROJECT)]
def run(args,identity=args.identity):
 return subprocess.check_output(['icp',*args,'--identity',identity,*common],text=True)
def call(method,args='()'):
 out=run(['canister','call',CANISTER,method,args,'--candid',str(ROOT/'build/verdict-engine.did')])
 if 'Err =' in out:raise RuntimeError(out)
 return out
status=json.loads(subprocess.check_output(['icp','network','status','--json',*common],text=True))
if not status.get('managed') or urlparse(status['api_url']).hostname not in {'localhost','127.0.0.1','::1'}:
 raise ValueError('only a managed loopback replica is permitted')
owner=subprocess.check_output(['icp','identity','principal','--identity',args.identity],text=True).strip()
expected_model=hashlib.sha256((ROOT/'fixtures/verdict-tiny/manifest.json').read_bytes()).digest()
info=call('info')
model_blob=re.search(r'model = (blob\s+"(?:[^"\\]|\\.)*")',info)[1]
if decode_blobs(model_blob)[0]!=expected_model:raise ValueError('regression requires the verdict-tiny fixture')
call('allow_caller',f'(principal "{owner}")')
schema='record {id="Efficiency";version=1:nat64;primitive=variant {Noul};instructions="choose";options=vec {record {id="false";text="false"};record {id="true";text="true"}}}'
out=call('register_schema',f'({schema},1:nat32)')
def field(name):
 fragment=re.search(name+r' = (blob\s+"(?:[^"\\]|\\.)*")',out)[1]
 return decode_blobs(fragment)[0]
schema_hash=field('schema_hash');tokenizer_hash=field('tokenizer_hash')
model=hashlib.sha256((ROOT/'fixtures/verdict-tiny/manifest.json').read_bytes()).digest()
cal=bytes([7])*32
now=time.time_ns()
call('register_calibration',f'(record {{id={blob(cal)};model={blob(model)};schema={blob(schema_hash)};tokenizer={blob(tokenizer_hash)};temperature=1.0;expires_at_ns={now+600_000_000_000}:nat64;holdout_hash={blob(bytes([8])*32)};sample_count=1:nat64;test_only=true}})')
req=f'(record {{evaluation_id={blob(bytes([9])*32)};model={blob(model)};schema_hash={blob(schema_hash)};calibration=opt {blob(cal)};binding=null;state="evidence";expires_at_ns={now+300_000_000_000}:nat64}})'
first=call('evaluate',req)
assert 'Assessed' in first
print('PASS registered calibrated evaluation',flush=True)
def warm():
 call('start_warmup')
 for _ in range(200):
  out=call('warmup_next')
  if 'Ok = true' in out:return
 raise RuntimeError('warmup did not complete')
run(['canister','install',args.canister,'-m','upgrade','--wasm',str(ROOT/'build/verdict-engine.wasm')],args.controller)
warm()
assert first==call('evaluate',req)
print('PASS upgrade + rewarm preserve snapshot and cached receipt',flush=True)
warm()
assert first==call('evaluate',req)
print('PASS rewarm preserves schema calibration and cached receipt',flush=True)
option='record {id="yes";text="yes"}'
qs=';'.join('record {id="'+i+'";question="choose";options=vec {'+option+'};abstention=true}' for i in ['one','two'])
out=call('decide_batch','(record {state="evidence";questions=vec {'+qs+'};temperature=1.0})')
assert 'one' in out and 'two' in out
print('PASS batch with two abstentions',flush=True)
# An incompatible upgrade must roll back both the pre_upgrade snapshot and heap.
failed=subprocess.run(['icp','canister','install',args.canister,'-m','upgrade','--wasm',str(ROOT/'build/executor.wasm'),'--identity',args.controller,*common],capture_output=True,text=True)
assert failed.returncode!=0
assert first==call('evaluate',req)
print('PASS failed upgrade leaves active heap and cache usable',flush=True)
