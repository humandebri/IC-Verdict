// Authoritative, read-only study against an explicitly selected managed loopback canister.
import {readFile,writeFile} from 'node:fs/promises';
import {createPrivateKey,createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
import {Actor,HttpAgent} from '@icp-sdk/core/agent';
import {Ed25519KeyIdentity} from '@icp-sdk/core/identity';
import {IDL} from '@icp-sdk/core/candid';
import {connect,hex} from '../src/api';
import {RealtimeGame,FRAME_MS,type Action} from '../src/realtime';
import {prompt,fixtures,type Format,type Labels,type Situation,type Fixture} from './input-study-core';
const root=resolve(import.meta.dirname,'../../..'),dir=resolve(root,'artifacts/tetris-input-study');
const host=process.env.IC_HOST,canister=process.env.CANISTER_ID,pem=process.env.STUDY_OWNER_PEM;
if(!host||!canister||!pem||process.env.IC_LOCAL!=='true'||!['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname))throw new Error('Explicit loopback IC_HOST, CANISTER_ID, IC_LOCAL=true and STUDY_OWNER_PEM required');
const network=JSON.parse(execFileSync('icp',['network','status','-e','local','--json'],{cwd:root,encoding:'utf8'}));
if(!network.managed||new URL(network.api_url).origin!==new URL(host).origin)throw new Error('Target must be the existing managed local network');
const status=JSON.parse(execFileSync('icp',['canister','status',canister,'-e','local','--identity','anonymous','--json'],{cwd:root,encoding:'utf8'}));
if(status.module_hash!=='0x0962574cae166a5a438e62a588d3351408613f52b867e6e18ed7311ba3d8074b')throw new Error('Unexpected local Wasm; never upgrade automatically');
const development=JSON.parse(await readFile(resolve(dir,'development.json'),'utf8'));
const budget=JSON.parse(await readFile(resolve(dir,'budgets.json'),'utf8'));
const frozen=JSON.parse(await readFile(resolve(dir,'fixtures.json'),'utf8'));
const all=fixtures(),digest=(s:string|Uint8Array)=>createHash('sha256').update(s).digest('hex');
if(digest(JSON.stringify(all))!==frozen.provenance.fixtures||digest(await readFile(resolve(root,'models/verdict-pack/manifest.json')))!==frozen.provenance.model)throw new Error('Frozen study provenance changed');
const publicClient=await connect(host,canister,true),ready=await publicClient.status();
if(!ready.warmed||!ready.enabled||ready.max_tokens!==52||hex(ready.model)!==frozen.provenance.model)throw new Error('Unexpected model/status');
// Load the existing local-only key in memory. Never export, print or persist it.
const key=createPrivateKey(await readFile(pem)),jwk=key.export({format:'jwk'});
if(jwk.crv!=='Ed25519'||!jwk.d)throw new Error('Expected existing local Ed25519 owner key');
const identity=Ed25519KeyIdentity.fromSecretKey(new Uint8Array(Buffer.from(jwk.d,'base64url')));
const agent=await HttpAgent.create({host,identity});await agent.fetchRootKey();
const errors=Object.fromEntries(['Unauthorized','Storage','Capacity','Numeric','TooLong','Transition','BindingMismatch','NonceGap','IdConflict','Expired','NotFound','Uncalibrated','Busy','NonceConsumed','OperationUsed','Budget','Stale','ReportOnly','LiveDisabled','OutcomeUnknown'].map(k=>[k,IDL.Null]));
const factory=()=>IDL.Service({decide_query:IDL.Func([IDL.Record({question:IDL.Text,state:IDL.Text,abstention:IDL.Bool,temperature:IDL.Float64,options:IDL.Vec(IDL.Record({id:IDL.Text,text:IDL.Text}))})],[IDL.Variant({Ok:IDL.Record({ids:IDL.Vec(IDL.Text),model:IDL.Vec(IDL.Nat8),probabilities:IDL.Vec(IDL.Float32),logits:IDL.Vec(IDL.Float32),selected:IDL.Text,input_tokens:IDL.Nat32,measured_instructions:IDL.Nat64,confidence:IDL.Float32}),Err:IDL.Variant({...errors,Invalid:IDL.Text,ModelUnavailable:IDL.Text,Denied:IDL.Text})})],['query'])});
const actor=Actor.createActor(factory,{agent,canisterId:canister});
type Variant={format:Format;labels:Labels;id:string};
type Answer={action:Action;actions:Action[];scores:number[];tokens:number;instructions:string;latency_ms:number};
let calls=0,queryErrors=0,maxInstructions=0n,maxTokens=0;
const cache=new Map<string,Promise<Answer>>();
async function request(text:string,labels:readonly string[],actions:Action[]):Promise<Answer>{
 const start=performance.now();calls++;
 try{
 const r=await actor.decide_query({question:'',state:text,abstention:false,temperature:1,options:labels.map((text,i)=>({id:String(actions[i]),text}))}) as {Ok?:{selected:string;ids:string[];model:number[];probabilities:number[];input_tokens:number;measured_instructions:bigint};Err?:unknown};
 if(!r.Ok)throw new Error(JSON.stringify(r.Err));const q=r.Ok,action=Number(q.selected) as Action;
 if(hex(q.model)!==frozen.provenance.model||!actions.includes(action)||q.input_tokens>52||q.measured_instructions>=5000000000n||JSON.stringify(q.ids)!==JSON.stringify(actions.map(String)))throw new Error('Query contract/model/budget mismatch');
 maxTokens=Math.max(maxTokens,q.input_tokens);maxInstructions=q.measured_instructions>maxInstructions?q.measured_instructions:maxInstructions;
 return {action,actions,scores:q.probabilities,tokens:q.input_tokens,instructions:String(q.measured_instructions),latency_ms:performance.now()-start};
 }catch(e){queryErrors++;throw e;}
}
async function decide(s:Situation,v:Variant,cached=true){const p=prompt(s,v.format,v.labels),k=JSON.stringify(p);if(!cached)return request(p.text,p.labels,p.actions);let r=cache.get(k);if(!r){r=request(p.text,p.labels,p.actions);cache.set(k,r);}return r;}
const save=(name:string,data:unknown)=>writeFile(resolve(dir,name),JSON.stringify(data,null,2)+'\n');
const provenance={...frozen.provenance,host,canister,wasm:status.module_hash,clock:'real local canister query results; fixed-latency games charge exactly 1000ms per choice (cached identical prompts permitted); live-latency games use actual uncached RTT; one in-flight query per game, at most four independent games/fixtures concurrently'};
async function map4<T,R>(items:T[],f:(x:T)=>Promise<R>){const out:R[]=[];for(let i=0;i<items.length;i+=4)out.push(...await Promise.all(items.slice(i,i+4).map(f)));return out;}
async function evaluate(fs:Fixture[],v:Variant){const rows=await map4(fs,async f=>({id:f.id,category:f.category,acceptable:f.acceptable,previous:f.input.previous[0]?.action,...await decide(f,v)}));const correct=rows.filter(r=>r.acceptable.includes(r.action)).length,rotations=rows.filter(r=>r.category==='rotate');return {variant:v,rows,summary:{n:rows.length,accuracy:correct/rows.length,correct,rotation_accuracy:rotations.filter(r=>r.action===2).length/rotations.length,rotation_required:rotations.length,rotation_correct:rotations.filter(r=>r.action===2).length,rotation_offered:rows.filter(r=>r.actions.includes(2)).length,rotation_chosen:rows.filter(r=>r.action===2).length,repeated:rows.filter(r=>r.action===r.previous).length}};}
const variants:Variant[]=budget.budgets.filter((b:{eligible:boolean})=>b.eligible).map((b:{variant:Variant})=>b.variant),baseline=variants.find(v=>v.id==='history-short')!;
const boundary=[];
for(const b of budget.budgets.filter((b:{eligible:boolean})=>b.eligible)){
 const [prefix,text]=b.worst_prompt.split('<<SEP>>'),labels=prefix.split('<<LABEL>>').slice(1);
 const result=await request(text,labels,[0,1,2,3,4]);if(result.tokens!==b.max_tokens)throw new Error('Tokenizer bound differs on canister');boundary.push({variant:b.variant.id,...result});
}
await save('query-boundaries.json',{provenance,boundary});
const dev=[];
for(const v of variants){const e=await evaluate(all.filter(f=>f.split==='dev'),v);dev.push(e);await save('query-development.json',{provenance,development:dev});console.log('query dev',v.id,JSON.stringify(e.summary));}
const winner=dev.filter(e=>e.variant.id!==baseline.id).sort((a,b)=>b.summary.accuracy-a.summary.accuracy||budget.budgets.find((x:{variant:Variant})=>x.variant.id===a.variant.id).max_tokens-budget.budgets.find((x:{variant:Variant})=>x.variant.id===b.variant.id).max_tokens||a.variant.id.localeCompare(b.variant.id))[0].variant;
await save('query-selection.json',{provenance,winner,criterion:'local query development accuracy, then token bound, then variant ID; saved before querying held-out states'});
const heldout=[];for(const v of [baseline,winner]){const e=await evaluate(all.filter(f=>f.split==='test'),v);heldout.push(e);console.log('query test',v.id,JSON.stringify(e.summary));}
await save('query-heldout.json',{provenance,heldout});
const parity=[];
for(const d of dev){const native=development.development.find((e:{variant:Variant})=>e.variant.id===d.variant.id);for(const r of d.rows){const n=native.rows.find((e:{id:string})=>e.id===r.id);if(n.action!==r.action)parity.push({variant:d.variant.id,id:r.id,native_action:n.action,query_action:r.action,native_scores:n.scores,query_scores:r.scores});}}
await save('native-query-differences.json',{provenance,compared:dev.reduce((n,d)=>n+d.rows.length,0),differences:parity});
async function game(seed:number,v:Variant,actualLatency=false){
 const g=new RealtimeGame(seed,0),turns=[];const frameLimit=Math.ceil(3600000/FRAME_MS);
 while(!g.over&&g.count<100&&g.frame<frameLimit){const t=g.beginQuery(g.frame*FRAME_MS);if(!t){g.tick();continue;}
  const r=await decide({board:g.board,input:t.input},v,!actualLatency);const latency=actualLatency?r.latency_ms:1000;
  for(let i=0;i<Math.ceil(latency/FRAME_MS)&&!g.over&&g.frame<frameLimit;i++)g.tick();g.response(t.request,r.action,latency);do{g.tick();}while(g.queued&&!g.over&&g.frame<frameLimit);
  turns.push({request:t.request,piece:t.piece,input:t.input,...r,outcome:t.outcome,charged_latency_ms:latency});
  if(turns.length%100===0)console.log('query game progress',seed,v.id,g.count,'pieces',turns.length,'decisions');
 }
 return {seed,variant:v.id,actualLatency,lines:g.lines,pieces:g.count,truncated:!g.over,metrics:g.metrics,turns};
}
const games:Awaited<ReturnType<typeof game>>[]=[];
for(let start=1;start<=10;start+=2){games.push(...await map4([start,start+1].flatMap(seed=>[baseline,winner].map(v=>({seed,v}))),async({seed,v})=>{const r=await game(seed,v);console.log('query game',seed,v.id,r.lines,r.pieces);return r;}));await save('query-games.json',{provenance,games});}
const live=[];for(const v of [baseline,winner]){const r=await game(42,v,true);live.push(r);console.log('live-latency',v.id,r.lines,r.pieces);await save('query-live-latency.json',{provenance,games:live});}
const aggregate=(id:string)=>{const rs=games.filter(g=>g.variant===id);return {games:rs.length,lines:rs.reduce((s,r)=>s+r.lines,0),positive:rs.filter(r=>r.lines>0).length,mean_pieces:rs.reduce((s,r)=>s+r.pieces,0)/rs.length,truncated:rs.filter(r=>r.truncated).length};};
const current=aggregate(baseline.id),candidate=aggregate(winner.id),h=heldout[1].summary;
const gates={accuracy:h.accuracy>=.8,rotation:h.rotation_accuracy>=.8,positive_games:candidate.positive>=5,mean_lines:candidate.lines/10>=1,survival:candidate.mean_pieces>=current.mean_pieces,no_query_errors:queryErrors===0};
await save('query-result.json',{provenance,winner,current,candidate,heldout:heldout.map(e=>({variant:e.variant,summary:e.summary})),gates,quality_pass:Object.values(gates).every(Boolean),calls,queryErrors,maxTokens,maxInstructions:String(maxInstructions),native_query_differences:parity.length,production_changed:false});
console.log('FINAL',JSON.stringify({winner,current,candidate,gates,calls,queryErrors,maxTokens,maxInstructions:String(maxInstructions)}));
