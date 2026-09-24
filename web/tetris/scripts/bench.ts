import {writeFile} from 'node:fs/promises';
import {connect,hex} from '../src/api';
import {benchmarkGame,type Policy} from '../src/benchmark';
import {RULES_VERSION} from '../src/realtime';

const host=process.env.IC_HOST||'http://localhost:8011',canister=process.env.CANISTER_ID;
if(!canister)throw new Error('CANISTER_ID is required');
if(process.env.IC_LOCAL!=='true'||!['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname))throw new Error('Accelerated benchmark requires an explicitly selected loopback canister');
const seedCount=Number(process.env.BENCH_SEEDS||10),maxPieces=Number(process.env.BENCH_PIECES||100);
if(!Number.isInteger(seedCount)||seedCount<1||seedCount>100)throw new Error('Invalid seed count');
const client=await connect(host,canister,true),initial=await client.status();
if(!initial.enabled||!initial.warmed||initial.max_tokens<52)throw new Error('Demo disabled, cold or query budget too small');
const model=hex(initial.model),games:Awaited<ReturnType<typeof benchmarkGame>>[]=[];
const policies:Policy[]=['model','heuristic','random'];
const report={rules:RULES_VERSION,host,canister,model,started:new Date().toISOString(),completed:'',max_pieces:maxPieces,
  clock:'virtual NTSC frames; measured local query RTT advances gravity (rounded up to a frame); one-second simulation dispatch interval; no rendering delay; maximum one RPC in flight',
  candidate_policy:'legal left/right/clockwise/down/wait; one input per decision; heuristic sees full board and searches all clockwise-only reachable landings; random uses an independent seeded stream; baselines have zero RTT',
  games,totals:{} as Record<string,unknown>};
const path=process.env.REPORT_PATH||'../../artifacts/tetris_query_v3_history_benchmark.json';
const save=()=>writeFile(path,JSON.stringify(report,null,2)+'\n');
for(let seed=1;seed<=seedCount;seed++)for(const policy of policies){
  const game=await benchmarkGame(seed,policy,maxPieces,async input=>{
    const start=performance.now(),reply=await client.decide(input),latency=performance.now()-start;
    if(hex(reply.model)!==model)throw new Error('Model changed');
    return {reply,latency};
  });
  games.push(game);await save();
  console.log(`seed=${seed} policy=${policy} lines=${game.lines} pieces=${game.pieces} late=${game.deadline_misses} errors=${game.errors} mean_ms=${game.mean_latency_ms.toFixed(1)}`);
}
for(const policy of policies){
  const runs=games.filter(g=>g.policy===policy),sum=(key:'lines'|'pieces'|'queries'|'errors'|'deadline_misses'|'blocked'|'simulation_ms')=>runs.reduce((n,g)=>n+g[key],0);
  const queries=sum('queries'),responses=runs.reduce((n,g)=>n+g.responses,0);
  report.totals[policy]={lines:sum('lines'),pieces:sum('pieces'),queries,errors:sum('errors'),deadline_misses:sum('deadline_misses'),blocked:sum('blocked'),simulation_ms:sum('simulation_ms'),
    truncated_runs:runs.filter(g=>g.truncated).length,mean_latency_ms:responses?runs.reduce((n,g)=>n+g.mean_latency_ms*g.responses,0)/responses:0,
    max_latency_ms:Math.max(...runs.map(g=>g.max_latency_ms)),max_tokens:Math.max(...runs.map(g=>g.max_tokens)),
    instructions:runs.reduce((n,g)=>n+BigInt(g.instructions),0n).toString()};
}
report.completed=new Date().toISOString();await save();
console.log(JSON.stringify(report.totals));
if(games.some(g=>g.errors))process.exitCode=1;
