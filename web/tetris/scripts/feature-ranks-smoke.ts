import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {connectQuery} from '../src/placement-query-api';
import {clearLines,features} from '../src/game';
const host='http://localhost:8011',canister='4fbx2-kt777-77775-aaabq-cai';
const out='artifacts/tetris-prompt-search/local-canister-cost.json';
const nativeFetch=globalThis.fetch;const paths:string[]=[];
globalThis.fetch=async(...args)=>{const url=String(args[0] instanceof Request?args[0].url:args[0]);paths.push(new URL(url).pathname);assert(!/\/call(?:\?|$)/.test(url),'update forbidden');return nativeFetch(...args);};
const client=await connectQuery(host,canister);const before=await client.status();const games=[];
for(const [seed,file] of [[11,'rollouts-cost.json'],[277,'rollouts-cost-fresh.json']] as const){
 const expected=JSON.parse(readFileSync('artifacts/tetris-prompt-search/'+file,'utf8')).find((g:any)=>g.seed===seed&&g.policy==='cost/0');
 assert(expected);const count=paths.length;let g=client.start(seed,0);assert.equal(paths.length,count);const receipts=[];
 for(const turn of expected.trace){
  const begin=performance.now();const r=await client.step(g);
  assert.equal(r.selected,turn.selected,'Wasm/native selection parity');
  assert.deepEqual(r.before,turn.board);assert.deepEqual(r.candidates,turn.options);
  assert.equal(r.query_count,1);assert.equal(r.rounds.length,1);assert(r.input_tokens<=41);assert(BigInt(r.inference_instructions)<5_000_000_000n);
  assert(r.prompt.endsWith('Min holes'));
  for(const c of r.candidates){
   const board=Array.from({length:20},(_,y)=>r.before.slice(y*10,y*10+10));
   for(const [x,y] of c.cells){assert.equal(board[y][x],0);board[y][x]=r.piece+1;}
   const a=clearLines(board);assert.deepEqual(features(a.board,a.lines),{holes:c.holes,height:c.height,lines:c.lines});
   if(c===r.candidates[r.selected])assert.deepEqual(a.board.flat(),r.after);
  }
  receipts.push({response_ms:performance.now()-begin,receipt:r});
  g={...g,board:r.after,lines:r.total_lines,turn:g.turn+1,over:r.over};
 }
 assert(g.over);assert.equal(g.lines,expected.lines);assert.equal(g.turn,expected.turns);
 games.push({seed,turns:g.turn,lines:g.lines,receipts});
 writeFileSync(out,JSON.stringify({host,canister,games,paths,complete:false},null,2)+'\n');
 console.log(`seed ${seed}: ${g.turn} turns / ${g.lines} lines; every choice matches native`);
}
const after=await client.status();assert.deepEqual(Array.from(after.model),Array.from(before.model));
writeFileSync(out,JSON.stringify({host,canister,games,paths,complete:true,update_calls:0,server_state_unchanged:true},null,2)+'\n');
console.log('PASS: complete local games, native parity, independent board replay, one query per turn, <=41 tokens and <5B model instructions');
