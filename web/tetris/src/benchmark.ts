import {RealtimeGame,FRAME_MS,actionsFor,landingsFor,type Action,type DecisionInput} from './realtime';
import type {Reply} from './api';

export type Policy='model'|'heuristic'|'random';
export type BenchAnswer={reply:Reply;latency:number};
export type BenchDecide=(input:DecisionInput)=>Promise<BenchAnswer>;
export function randomPolicy(seed:number){
  let state=(seed^0x9e3779b9)>>>0||1;
  return (count:number)=>{state^=state<<13;state^=state>>>17;state^=state<<5;return Math.floor((state>>>0)/4294967296*count);};
}

// The baseline sees the full board and searches clockwise-only legal routes.
export function heuristicAction(game:RealtimeGame):Action {
  const a=game.active!;
  const options=landingsFor(game.board,a.piece,a.pose);
  const ranked=options.map(o=>({o,path:o.path})).sort((a,b)=>
    b.o.score-a.o.score||a.path.length-b.path.length||a.o.pose.r-b.o.pose.r||a.o.pose.x-b.o.pose.x||a.o.pose.y-b.o.pose.y);
  const move=ranked[0]?.path[1];
  if(!move)return ranked.length?3:4;
  return move.r!==a.pose.r?2:move.x<a.pose.x?0:move.x>a.pose.x?1:3;
}
// Virtual frames; every policy has the same 1s dispatch interval. Baselines have zero RTT.
export async function benchmarkGame(seed:number,policy:Policy,maxPieces:number,decide?:BenchDecide){
  if(!Number.isInteger(maxPieces)||maxPieces<1||maxPieces>1000)throw new Error('Invalid piece limit');
  if(policy==='model'&&!decide)throw new Error('Model policy requires real inference');
  const game=new RealtimeGame(seed,0),random=randomPolicy(seed),turns:Record<string,unknown>[]=[];
  const frameLimit=Math.ceil(3600_000/FRAME_MS);
  const advanceLatency=(latency:number)=>{
    if(!Number.isFinite(latency)||latency<0)throw new Error('Invalid query latency');
    for(let i=0;i<Math.ceil(latency/FRAME_MS)&&!game.over&&game.frame<frameLimit;i++)game.tick();
  };
  let instructions=0n,maxTokens=0;
  while(!game.over&&game.count<maxPieces&&game.frame<frameLimit){
    const trace=game.beginQuery(game.frame*FRAME_MS);
    if(trace){
      const input=trace.input,record:Record<string,unknown>={piece:trace.piece,request:trace.request,input};
      try{
        let action:Action,latency=0;
        if(policy==='model'){
          const start=performance.now();let answer:BenchAnswer;
          try{answer=await decide!(input);}catch(error){
            latency=performance.now()-start;advanceLatency(latency);record.latency_ms=latency;throw error;
          }
          action=answer.reply.action;latency=answer.latency;advanceLatency(latency);
          instructions+=answer.reply.measured_instructions;maxTokens=Math.max(maxTokens,answer.reply.input_tokens);
          Object.assign(record,{scores:answer.reply.scores,actions:answer.reply.actions,tokens:answer.reply.input_tokens,instructions:answer.reply.measured_instructions.toString(),prompt:answer.reply.prompt});
        }else {const actions=actionsFor(input.legal_mask);action=policy==='heuristic'?heuristicAction(game):actions[random(actions.length)];}
        game.response(trace.request,action,latency);
        // Execute queued input before recording its outcome, with the same frame budget as UI.
        do {game.tick();}while(game.queued&&!game.over&&game.frame<frameLimit);
        Object.assign(record,{action,latency_ms:latency,outcome:trace.outcome});
      }catch(error){game.fail(trace.request);record.error=String(error);record.outcome=trace.outcome;}
      turns.push(record);
    }
    game.tick();
  }
  const metrics=game.metrics;
  return {seed,policy,lines:game.lines,score:game.score,pieces:game.count,game_over:game.over,
    truncated:!game.over,simulation_ms:game.frame*FRAME_MS,decisions:metrics.queries,
    queries:policy==='model'?metrics.queries:0,responses:metrics.responses,deadline_misses:metrics.late,blocked:metrics.blocked,executed:metrics.executed,
    errors:metrics.errors,mean_latency_ms:metrics.responses?metrics.latencyTotal/metrics.responses:0,
    max_latency_ms:metrics.latencyMax,max_tokens:maxTokens,instructions:instructions.toString(),turns};
}
