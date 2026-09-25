import {describe,it,expect,vi} from 'vitest';
import {benchmarkGame,heuristicAction,randomPolicy} from './benchmark';
import {RealtimeGame,actionsFor,decisionInput} from './realtime';
import {emptyBoard} from './game';

describe('comparable policies and query latency',()=>{
  it('uses a separate reproducible random stream with all five controls available',()=>{
    const a=randomPolicy(42),b=randomPolicy(42),values=Array.from({length:100},()=>a(5));
    expect(values).toEqual(Array.from({length:100},()=>b(5)));expect(new Set(values).size).toBe(5);
  });
  it('uses only controls allowed by the current state',()=>{
    const g=new RealtimeGame(42,0),a=g.active!;
    expect(actionsFor(decisionInput(g.board,a.piece,g.next,a.pose).legal_mask)).toContain(heuristicAction(g));
  });
  it('repeats baseline games with the same seed and controller',async()=>{
    const a=await benchmarkGame(42,'random',12),b=await benchmarkGame(42,'random',12);
    expect(a).toEqual(b);expect(a.queries).toBe(0);expect(a.turns.length).toBeGreaterThan(2);
  });
  it('charges slow model responses to falling pieces and reports deadline misses',async()=>{
    const result=await benchmarkGame(1,'model',2,async input=>({latency:20000,reply:{action:3,actions:actionsFor(input.legal_mask),scores:actionsFor(input.legal_mask).map(()=>.2),input_tokens:39,model:[],measured_instructions:1n,prompt:JSON.stringify(input)}}));
    expect(result.deadline_misses).toBeGreaterThan(0);expect(result.turns[0].outcome).toBe('late');
    expect(result.mean_latency_ms).toBe(20000);expect(result.max_tokens).toBe(39);
  });
  it('also charges time spent waiting for a failed request',async()=>{
    const clock=vi.spyOn(performance,'now').mockReturnValueOnce(0).mockReturnValueOnce(20000);
    try{
      const result=await benchmarkGame(1,'model',1,async()=>{throw new Error('network failure');});
      expect(result.errors).toBe(1);expect(result.deadline_misses).toBe(1);expect(result.turns[0].latency_ms).toBe(20000);
    }finally{clock.mockRestore();}
  });
});
