import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {connectQuery} from '../src/placement-query-api';
const host='https://icp-api.io',canister='qojfj-6qaaa-aaaam-qjkaq-cai';
const nativeFetch=globalThis.fetch;const paths:string[]=[];
globalThis.fetch=async(...args)=>{const url=String(args[0] instanceof Request?args[0].url:args[0]);paths.push(new URL(url).pathname);assert(!/\/call(?:\?|$)/.test(url));return nativeFetch(...args);};
const expected=JSON.parse(readFileSync('artifacts/tetris-prompt-search/rollouts-cost.json','utf8')).find((g:any)=>g.seed===11&&g.policy==='cost/0');
const c=await connectQuery(host,canister);const before=await c.status();let g=c.start(11,0);const results=[];
for(const turn of expected.trace.slice(0,3)){
 const begin=performance.now();const r=await c.step(g);
 assert.equal(r.selected,turn.selected);assert.deepEqual(r.candidates,turn.options);assert.deepEqual(r.before,turn.board);
 assert(r.rules.endsWith('query-feature-cost-v1'));assert(r.prompt.endsWith('Min holes'));
 assert(r.input_tokens<=41);assert(BigInt(r.inference_instructions)<5_000_000_000n);
 assert.equal(r.query_count,1);assert.equal(r.rounds.length,1);
 results.push({response_ms:performance.now()-begin,receipt:r});g={...g,board:r.after,lines:r.total_lines,turn:g.turn+1,over:r.over};
}
const after=await c.status();assert.deepEqual(Array.from(after.model),Array.from(before.model));
writeFileSync('artifacts/tetris-prompt-search/production-query-check.json',JSON.stringify({host,canister,paths,results,update_calls:0,server_state_unchanged:true},null,2)+'\n');
console.log('PASS: 3 production model queries match native choices/candidates; <=41 tokens; <5B instructions; no update calls');
