import assert from 'node:assert/strict';
import {writeFileSync,mkdirSync} from 'node:fs';
import {connectQuery} from '../src/placement-query-api';
import {clearLines,features} from '../src/game';
const host=process.env.QUERY_HOST||'http://localhost:8011';const canister=process.env.QUERY_CANISTER;
if(!canister||!['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname))throw new Error('Local only');
const nativeFetch=globalThis.fetch;const paths:string[]=[];
globalThis.fetch=async(...args)=>{const url=String(args[0] instanceof Request?args[0].url:args[0]);paths.push(new URL(url).pathname);assert(!url.match(/\/call(?:\?|$)/),'update calls forbidden');return nativeFetch(...args);};
const c=await connectQuery(host,canister);const before=await c.status();const results=[];
for(const mode of [0,1]){
 const count=paths.length;let g=c.start(184,mode);assert.equal(paths.length,count,'start must be browser-only');
 for(let t=0;t<3;t++){
 const begin=performance.now();const r=await c.step(g);
 for(const choice of r.candidates){const board=Array.from({length:20},(_,y)=>r.before.slice(y*10,y*10+10));for(const [x,y] of choice.cells){assert.equal(board[y][x],0);board[y][x]=r.piece+1;}const a=clearLines(board);assert.deepEqual(features(a.board,a.lines),{holes:choice.holes,height:choice.height,lines:choice.lines});if(choice===r.candidates[r.selected])assert.deepEqual(a.board.flat(),r.after);}
 for(const round of r.rounds){assert(round.input_tokens<=52);assert(BigInt(r.inference_instructions)<5_000_000_000n);assert(round.model_candidates.includes(round.selected));}
 if(mode===1){assert.equal(r.selected,r.recommended);assert.equal(r.query_count,0);}
 results.push({response_ms:performance.now()-begin,receipt:r});g={...g,board:r.after,lines:r.total_lines,turn:g.turn+1,over:r.over};
 }
}
const after=await c.status();assert.deepEqual(Array.from(after.model),Array.from(before.model));
mkdirSync('artifacts/tetris-placement-query',{recursive:true});writeFileSync('artifacts/tetris-placement-query/local-smoke.json',JSON.stringify({host,canister,paths,results,update_calls:0,state_unchanged:true},null,2)+'\n');
console.log('PASS: 3 model and 3 heuristic turns; no update calls; server state/budget unchanged; all model queries below 5B instructions; independent board replay.');
