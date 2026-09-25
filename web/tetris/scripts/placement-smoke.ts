// Local-only real-model smoke; uses an ephemeral signer and exports no secret keys.
import {writeFileSync,mkdirSync} from 'node:fs';
import assert from 'node:assert/strict';
import {connectPlacement} from '../src/placement-api';
import {clearLines,features,sequence} from '../src/game';
const host=process.env.PLACEMENT_HOST||'http://localhost:8011';
const canister=process.env.PLACEMENT_CANISTER;
if(!canister||!['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname))throw new Error('Explicit local canister required');
Object.defineProperty(globalThis,'sessionStorage',{value:{getItem:()=>null,setItem:()=>{}}});
const client=await connectPlacement(host,canister);
const before=await client.status();const results=[];const seed=184;
for(const mode of [0,1]){
 const game=await client.start(seed,mode,BigInt(Date.now())+BigInt(mode));const turns=[];
 for(let turn=0;turn<4;turn++){
 const begin=performance.now();const r=await client.step(game.id,turn);const response_ms=performance.now()-begin;
 assert.equal(r.piece,sequence(seed,4)[turn]);assert.equal(r.mode,mode);
 assert(r.candidates.length>0&&r.candidates.length<=12);
 for(const c of r.candidates){
 const b=Array.from({length:20},(_,y)=>r.before.slice(y*10,y*10+10));
 for(const [x,y] of c.cells){assert.equal(b[y][x],0);b[y][x]=r.piece+1;}
 const after=clearLines(b);assert.deepEqual(features(after.board,after.lines),{holes:c.holes,height:c.height,lines:c.lines});
 assert.equal(c.evaluation,10*c.lines-8*c.holes-c.height);
 if(c===r.candidates[r.selected])assert.deepEqual(after.board.flat(),r.after);
 }
 assert.equal(r.recommended,r.candidates.reduce((best,c,i,a)=>c.evaluation>a[best].evaluation?i:best,0));
 if(mode===1){assert.equal(r.selected,r.recommended);assert.equal(r.input_tokens,0);assert.equal(r.inference_instructions,'0');}
 else{assert(r.input_tokens>0&&r.input_tokens<=128);assert(BigInt(r.inference_instructions)>0n);}
 assert.deepEqual(await client.step(game.id,turn),r,'retry must return exact prior receipt');
 const verified=await client.verify(r.id);assert.deepEqual(verified.receipt,r);
 turns.push({receipt:r,response_ms,certificate_verified:true,retry_idempotent:true});
 }
 results.push({mode,game:game.id,turns});
}
const after=await client.status();assert.equal(before.model_turns_left-after.model_turns_left,4);
mkdirSync('artifacts/tetris-placement',{recursive:true});
writeFileSync('artifacts/tetris-placement/comparison-smoke.json',JSON.stringify({host,canister,seed,results,budget_consumed:4},null,2)+'\n');
console.log('PASS: 4 model + 4 heuristic turns, exact retries, independent board/features, 8 certificates, model-only budget consumption.');
