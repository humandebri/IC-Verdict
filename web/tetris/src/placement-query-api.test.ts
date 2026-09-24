import {readFileSync} from 'node:fs';
import {describe,it,expect} from 'vitest';
import {choosePlacement,validateReply,type ModelReply,type Status} from './placement-query-api';
import {prepareTurn,renderPrompt} from './placement-local';
import type {Game} from './placement-api';

const model=Array(32).fill(7),status:Status={enabled:true,warmed:true,model,max_tokens:52};
const game:Game={id:0,seed:11,mode:0,board:Array(200).fill(0),turn:0,lines:0,over:false};
const reply=(ids:string[],selected='0'):ModelReply=>({ids,selected,model,probabilities:ids.map((_,i)=>i===Number(selected)?1:0),logits:ids.map(()=>0),confidence:1,input_tokens:41,measured_instructions:3_000_000_000n});
describe('browser-owned placement',()=>{
 it('matches production Rust candidate generation and prompt on recorded boards',()=>{
  const rows=JSON.parse(readFileSync(new URL('../../../artifacts/tetris-prompt-search/production-query-check.json',import.meta.url),'utf8')).results;
  for(const row of rows){const r=row.receipt;
   const turn=prepareTurn({...game,seed:r.seed,turn:r.turn,board:r.before,lines:r.total_lines-r.candidates[r.selected].lines});
   expect(turn.piece).toBe(r.piece);expect(turn.next).toBe(r.next);
   expect(turn.legal_count).toBe(r.legal_count);expect(turn.candidates).toEqual(r.candidates);
   expect(turn.recommended).toBe(r.recommended);expect(renderPrompt(turn.request)).toBe(r.prompt);
   const actual:ModelReply={ids:turn.request.options.map(o=>o.id),selected:String(r.selected),
    probabilities:r.scores,logits:r.logits,model:r.model,confidence:r.scores[r.selected],
    input_tokens:r.input_tokens,measured_instructions:BigInt(r.inference_instructions)};
   expect(validateReply(actual,turn.request,{...status,model:r.model})).toBe(r.selected);
  }
 });
 it('matches Rust candidates at early and middle turns across fixed seeds',()=>{
  const games=JSON.parse(readFileSync(new URL('../../../artifacts/tetris-prompt-search/rollouts-cost.json',import.meta.url),'utf8'));
  for(const run of games){
   for(const row of [...run.trace.slice(0,8),run.trace[Math.floor(run.trace.length/2)]]){
    const turn=prepareTurn({...game,seed:run.seed,turn:row.turn,board:row.board,lines:row.lines});
    expect(turn.piece).toBe(row.piece);expect(turn.candidates).toEqual(row.options);
   }
  }
 });
 it('sends only descriptions once and computes the next board locally',async()=>{
  let calls=0;const result=await choosePlacement(game,status,async request=>{
   calls++;expect(Object.keys(request).sort()).toEqual(['abstention','options','question','state','temperature']);
   expect(JSON.stringify(request)).not.toContain('board');return reply(request.options.map(o=>o.id),'1');
  });
  expect(calls).toBe(1);expect(result.selected).toBe(1);expect(result.query_count).toBe(1);
  expect(result.after).toHaveLength(200);expect(result.before).toEqual(game.board);
 });
 it('comparison mode makes no model query',async()=>{
  const r=await choosePlacement({...game,mode:1},status,async()=>{throw new Error('unexpected query');});
  expect(r.query_count).toBe(0);expect(r.selected).toBe(r.recommended);
 });
 it('rejects invalid identities, order, scores and model without applying a move',async()=>{
  const req=prepareTurn(game).request,ids=req.options.map(o=>o.id);
  for(const bad of [{...reply(ids),ids:ids.slice().reverse()},
   {...reply(ids),selected:'99'},{...reply(ids),probabilities:[NaN,...ids.slice(1).map(()=>0)]},
   {...reply(ids),probabilities:ids.map(()=>0)},
   {...reply(ids),model:Array(32).fill(8)}]){
   expect(()=>validateReply(bad,req,status)).toThrow('Invalid model response');
  }
  await expect(choosePlacement(game,status,async()=>{throw new Error('query failed');})).rejects.toThrow('query failed');
 });
});
