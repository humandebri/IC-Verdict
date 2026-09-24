import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {fixtures,prompt,FORMATS,LABELS,type Format,type Labels,type Situation,type Fixture} from './input-study-core';
import {RealtimeGame,FRAME_MS,PIECE_NAMES,actionsFor,type Action} from '../src/realtime';
const root=resolve(import.meta.dirname,'../../..'),out=resolve(root,'artifacts/tetris-input-study');
await mkdir(out,{recursive:true});
type ProbeReply={tokens:number;ids:number[];prompt?:string;selected?:number;scores?:number[]};
function worker(){
 const child=spawn(resolve(root,'target/release/tetris_probe'),[root],{stdio:['pipe','pipe','inherit']});
 const lines=createInterface({input:child.stdout});
 const waiting:{resolve:(r:ProbeReply)=>void;reject:(e:Error)=>void}[]=[];
 let dead:Error|undefined;
 lines.on('line',line=>{const next=waiting.shift();if(!next)throw new Error('Unexpected probe response');try{next.resolve(JSON.parse(line));}catch(e){next.reject(e as Error);}});
 const fail=(error:Error)=>{dead=error;for(const job of waiting.splice(0))job.reject(error);};
 child.on('error',fail);child.on('exit',code=>fail(new Error(`Probe exited ${code}`)));
 return {child,pending:()=>waiting.length,call:(input:unknown)=>new Promise<ProbeReply>((resolve,reject)=>{if(dead){reject(dead);return;}waiting.push({resolve,reject});child.stdin.write(JSON.stringify(input)+'\n');})};
}
const workers=[worker()];
const probe=(input:unknown)=>workers.reduce((a,b)=>b.pending()<a.pending()?b:a).call(input);
const save=(name:string,data:unknown)=>writeFile(resolve(out,name),JSON.stringify(data,null,2)+'\n');
const hash=(b:Uint8Array|string)=>createHash('sha256').update(b).digest('hex');
const tokenizerConfig=JSON.parse(await readFile(resolve(root,'models/verdict-151m/tokenizer.json'),'utf8'));
if(tokenizerConfig.model.type!=='BPE'||tokenizerConfig.pre_tokenizer.type!=='ByteLevel'||tokenizerConfig.pre_tokenizer.use_regex!==true||tokenizerConfig.pre_tokenizer.add_prefix_space!==false||tokenizerConfig.normalizer?.type!=='NFC')throw new Error('Tokenizer changed: redo boundary proof before proceeding');
const all=fixtures();
const provenance={model:hash(await readFile(resolve(root,'models/verdict-pack/manifest.json'))),tokenizer:hash(await readFile(resolve(root,'models/verdict-151m/tokenizer.json'))),fixtures:hash(JSON.stringify(all)),clock:'native frozen INT8 weights; game gravity charged fixed 1000 ms per decision; measured native CPU time is not query RTT'};
type Variant={format:Format;labels:Labels;id:string};
const variants:Variant[]=FORMATS.flatMap(format=>(['short','explicit'] as Labels[]).map(labels=>({format,labels,id:`${format}-${labels}`})));
// ByteLevel/GPT-2 regex splits words and numeric runs before BPE. Segments below
// end exactly on these boundaries (or added special-token boundaries).
// Exhaust every value in each independent segment; concatenated IDs are checked
// against whole-prompt encoding for every substitution and the worst witness.
const range=(n:number)=>Array.from({length:n},(_,i)=>String(i));
const numeric=(n:number,space=false)=>range(n).map(v=>(space?' ':'')+v);
function segments(v:Variant):string[][]{
 const parts:string[][]=[];const fixed=(s:string)=>parts.push([s]),values=(xs:readonly string[])=>parts.push([...xs]);
 for(const label of LABELS[v.labels]){fixed('<<LABEL>>');fixed(label);}fixed('<<SEP>>');
 fixed('Clear lines.');if(v.format==='history'||v.format==='no-history')fixed(' Tetris');values(PIECE_NAMES.map(p=>' '+p));
 fixed(' x');values(numeric(10));fixed(' y');values(numeric(20));fixed(' r');values(numeric(4));fixed(' next');values(PIECE_NAMES.map(p=>' '+p));
 if(v.format==='rows'){fixed(' rows');for(let i=0;i<20;i++)values(numeric(1024,true));}
 else{fixed(' heights');for(let i=0;i<10;i++)values(numeric(21,true));fixed(' holes');
  if(v.format==='columns'){for(let i=0;i<10;i++)values(numeric(20,true));}else values(numeric(201));}
 if(v.format==='history'){
  fixed(' prev x');values(numeric(10));fixed(' y');values(numeric(20));fixed(' r');values(numeric(4));values(LABELS.short.map(a=>' '+a));values([' executed',' blocked',' discarded']);fixed(' history');values(LABELS.short.map(a=>' '+a));values(LABELS.short.map(a=>' '+a));
 }
 return parts;
}
const rawCache=new Map<string,ProbeReply>();
async function raw(s:string){let r=rawCache.get(s);if(!r){r=await probe({raw:s});rawCache.set(s,r);}return r;}
async function budget(v:Variant){
 const domains=segments(v),encoded=await Promise.all(domains.map(async ds=>Promise.all(ds.map(async text=>({text,...await raw(text)})))));
 const worst=encoded.map(ds=>ds.reduce((a,b)=>b.tokens>a.tokens?b:a));
 const joined=worst.map(x=>x.text).join(''),expected=worst.flatMap(x=>x.ids),whole=await raw(joined);
 if(JSON.stringify(whole.ids)!==JSON.stringify(expected))throw new Error(`Non-independent tokenizer segments: ${v.id}`);
 let checks=0;
 for(let i=0;i<encoded.length;i++)for(const value of encoded[i]){
  const parts=worst.map((x,j)=>i===j?value:x);
  const actual=await raw(parts.map(x=>x.text).join(''));
  if(JSON.stringify(actual.ids)!==JSON.stringify(parts.flatMap(x=>x.ids)))throw new Error(`Token boundary failed ${v.id}/${i}`);
  checks++;
 }
 return {variant:v,max_tokens:whole.tokens+2,eligible:whole.tokens+2<=52,checks,worst_prompt:joined,proof:'All numeric/label domains exhausted independently at fixed ByteLevel pretoken boundaries; concatenated token IDs checked. Five labels and full history dominate legal subsets/shorter histories.'};
}
function summary(rows:Record<string,any>[]){const rotations=rows.filter(r=>r.category==='rotate');return {n:rows.length,correct:rows.filter(r=>r.correct).length,accuracy:rows.filter(r=>r.correct).length/rows.length,rotation_required:rotations.length,rotation_correct:rotations.filter(r=>r.correct).length,rotation_accuracy:rotations.filter(r=>r.correct).length/rotations.length,rotation_offered:rows.filter(r=>r.actions.includes(2)).length,rotation_chosen:rows.filter(r=>r.action===2).length,repeated:rows.filter(r=>r.previous===r.action).length,repeat_eligible:rows.filter(r=>r.previous!==undefined&&r.actions.includes(r.previous)).length,max_tokens:Math.max(...rows.map(r=>r.tokens))};}
const inferenceCache=new Map<string,Promise<ProbeReply>>();
async function decide(s:Situation,v:Variant){const p=prompt(s,v.format,v.labels),key=JSON.stringify(p);let pending=inferenceCache.get(key);if(!pending){pending=probe({...p,infer:true});inferenceCache.set(key,pending);}const r=await pending;if(r.tokens>52||r.selected===undefined||!r.scores)throw new Error('Invalid probe result');return {...r,actions:p.actions,action:p.actions[r.selected]};}
async function evaluate(fs:Fixture[],v:Variant){const rows=[];for(const f of fs){const r=await decide(f,v);rows.push({id:f.id,category:f.category,acceptable:f.acceptable,previous:f.input.previous[0]?.action,...r,correct:f.acceptable.includes(r.action)});}return {variant:v,summary:summary(rows),rows};}
async function game(seed:number,v:Variant){
 const g=new RealtimeGame(seed,0),turns=[];const limit=Math.ceil(3600000/FRAME_MS);
 while(!g.over&&g.count<100&&g.frame<limit){const t=g.beginQuery(g.frame*FRAME_MS);if(!t){g.tick();continue;}
  const r=await decide({board:g.board,input:t.input},v);
  for(let i=0;i<Math.ceil(1000/FRAME_MS)&&!g.over;i++)g.tick();
  g.response(t.request,r.action,1000);do{g.tick();}while(g.queued&&!g.over);
  turns.push({request:t.request,piece:t.piece,input:t.input,action:r.action,actions:r.actions,tokens:r.tokens,outcome:t.outcome});
 }
 return {seed,variant:v.id,lines:g.lines,pieces:g.count,truncated:!g.over,metrics:g.metrics,turns};
}
let winner:Variant,heldout:Awaited<ReturnType<typeof evaluate>>[];
const baseline=variants[0];
try{
 if(!process.argv.includes('--resume-games')){
 await save('fixtures.json',{provenance,oracle:'test only: one control then gravity; line count priority; no acceptable action, oracle score or future board is present in inference input',split:'cavity pivot x <= 4 development; x >= 5 held out; boards disjoint',cases:all});
 const budgets:Awaited<ReturnType<typeof budget>>[]=[];for(const v of variants){const b=await budget(v);budgets.push(b);console.log('budget',v.id,b.max_tokens,b.eligible);await save('budgets.json',{provenance,budgets});}
 const development=[];
 for(const b of budgets.filter(b=>b.eligible)){const e=await evaluate(all.filter(f=>f.split==='dev'),b.variant);development.push(e);await save('development.json',{provenance,development});console.log('dev',e.variant.id,JSON.stringify(e.summary));}
 const candidates=development.filter(e=>e.variant.id!=='history-short').sort((a,b)=>b.summary.accuracy-a.summary.accuracy||budgets.find(x=>x.variant.id===a.variant.id)!.max_tokens-budgets.find(x=>x.variant.id===b.variant.id)!.max_tokens||a.variant.id.localeCompare(b.variant.id));
 if(!candidates.length)throw new Error('No eligible alternative');
 winner=candidates[0].variant;
 // Commit selection before touching held-out inference.
 await save('selection.json',{provenance,winner,criterion:'development accuracy; tie by domain token bound, then variant ID; held-out outcomes unavailable at selection'});
 heldout=[];
 for(const v of [baseline,winner]){const e=await evaluate(all.filter(f=>f.split==='test'),v);heldout.push(e);console.log('test',v.id,JSON.stringify(e.summary));}
 await save('heldout.json',{provenance,heldout});
 }else{
  const selection=JSON.parse(await readFile(resolve(out,'selection.json'),'utf8'));
  const held=JSON.parse(await readFile(resolve(out,'heldout.json'),'utf8'));
  for(const k of ['fixtures','model','tokenizer'] as const)if(selection.provenance[k]!==provenance[k]||held.provenance[k]!==provenance[k])throw new Error('Resume provenance mismatch');
  winner=selection.winner;heldout=held.heldout;
 }
 if(!process.argv.includes('--native-games')&&!process.argv.includes('--resume-games')){
  console.log('Native diagnostics complete. Use input-study-query.ts for authoritative gates.');
 }else{
 // Independent virtual games may run in parallel; each still has exactly one pending decision.
 while(workers.length<4)workers.push(worker());
 const games:Awaited<ReturnType<typeof game>>[]=[];
 for(let start=1;start<=10;start+=2){
  const batch=await Promise.all([start,start+1].flatMap(seed=>[baseline,winner].map(async v=>{const r=await game(seed,v);console.log('game',seed,v.id,r.lines,r.pieces);return r;})));
  games.push(...batch);await save('games.json',{provenance,games});
 }
 const aggregate=(id:string)=>{const rs=games.filter(g=>g.variant===id);return {games:rs.length,lines:rs.reduce((s,r)=>s+r.lines,0),positive:rs.filter(r=>r.lines>0).length,mean_pieces:rs.reduce((s,r)=>s+r.pieces,0)/rs.length,truncated:rs.filter(r=>r.truncated).length};};
 const current=aggregate(baseline.id),candidate=aggregate(winner.id),h=heldout[1].summary;
 const gates={accuracy:h.accuracy>=.8,rotation:h.rotation_accuracy>=.8,positive_games:candidate.positive>=5,mean_lines:candidate.lines/10>=1,survival:candidate.mean_pieces>=current.mean_pieces};
 await save('result.json',{provenance,winner,current,candidate,heldout:heldout.map(e=>({variant:e.variant,summary:e.summary})),gates,authoritative:false,quality_pass:Object.values(gates).every(Boolean),query_validation:'not yet run; native inference is not instruction/RTT evidence',production_changed:false});
 console.log('RESULT',JSON.stringify({winner,current,candidate,gates}));
 }
}finally{for(const w of workers)w.child.stdin.end();}
