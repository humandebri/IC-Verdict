import {Actor,HttpAgent} from '@icp-sdk/core/agent';
import {IDL} from '@icp-sdk/core/candid';
import type {Candidate,Game} from './placement-api';
import {MAX_TURNS,RULES,landed,placements,prepareTurn,renderPrompt} from './placement-local';
export type {Game,Pose} from './placement-api';

type Request=ReturnType<typeof prepareTurn>['request'];
export type ModelReply={ids:string[];selected:string;probabilities:number[];logits:number[];model:number[]|Uint8Array;confidence:number;input_tokens:number;measured_instructions:bigint};
export type Status={enabled:boolean;warmed:boolean;model:number[]|Uint8Array;max_tokens:number};
export type Decision={id:number;game:number;seed:number;mode:number;turn:number;piece:number;next:number;before:number[];after:number[];total_lines:number;over:boolean;legal_count:number;candidates:Candidate[];recommended:number;selected:number;logits:number[];scores:number[];model:number[];prompt:string;model_request:Request|null;input_tokens:number;inference_instructions:string;rules:string;model_candidates:number[];rounds:{model_candidates:number[];selected:number;input_tokens:number}[];query_count:number};
const errors=Object.fromEntries(['Unauthorized','Storage','Capacity','Numeric','TooLong','Transition','BindingMismatch','NonceGap','IdConflict','Expired','NotFound','Uncalibrated','Busy','NonceConsumed','OperationUsed','Budget','Stale','ReportOnly','LiveDisabled','OutcomeUnknown'].map(k=>[k,IDL.Null]));
const ErrorType=IDL.Variant({...errors,Invalid:IDL.Text,ModelUnavailable:IDL.Text,Denied:IDL.Text});
export const queryIdl=()=>IDL.Service({
 info:IDL.Func([],[IDL.Record({model:IDL.Vec(IDL.Nat8),warmed:IDL.Bool})],['query']),
 query_limits:IDL.Func([],[IDL.Record({max_tokens:IDL.Nat32})],['query']),
 decide_query:IDL.Func([IDL.Record({question:IDL.Text,state:IDL.Text,abstention:IDL.Bool,temperature:IDL.Float64,options:IDL.Vec(IDL.Record({id:IDL.Text,text:IDL.Text}))})],
  [IDL.Variant({Ok:IDL.Record({ids:IDL.Vec(IDL.Text),model:IDL.Vec(IDL.Nat8),probabilities:IDL.Vec(IDL.Float32),logits:IDL.Vec(IDL.Float32),selected:IDL.Text,input_tokens:IDL.Nat32,measured_instructions:IDL.Nat64,confidence:IDL.Float32}),Err:ErrorType})],['query']),
});
const same=(a:number[]|Uint8Array,b:number[]|Uint8Array)=>a.length===b.length&&Array.from(a).every((v,i)=>v===b[i]);
export function validateReply(reply:ModelReply,request:Request,status:Status):number{
 const expected=request.options.map(o=>o.id),selected=Number(reply.selected);
 if(!Number.isInteger(selected)||selected<0||selected>=expected.length||reply.selected!==expected[selected]
  ||reply.ids.length!==expected.length||reply.ids.some((id,i)=>id!==expected[i])
  ||reply.probabilities.length!==expected.length||reply.probabilities.some(v=>!Number.isFinite(v)||v<0||v>1)
  ||Math.abs(reply.probabilities.reduce((sum,v)=>sum+v,0)-1)>0.001
  ||reply.probabilities.some((v,i)=>v>reply.probabilities[selected]+0.000001||(v===reply.probabilities[selected]&&i<selected))
  ||reply.logits.length!==expected.length||reply.logits.some(v=>!Number.isFinite(v))
  ||reply.model.length!==32||!same(reply.model,status.model)
  ||!Number.isInteger(reply.input_tokens)||reply.input_tokens<1||reply.input_tokens>status.max_tokens
  ||reply.measured_instructions<0n||reply.measured_instructions>=5_000_000_000n
  ||!Number.isFinite(reply.confidence)||reply.confidence<0||reply.confidence>1
  ||Math.abs(reply.confidence-reply.probabilities[selected])>0.001)throw new Error('Invalid model response');
 return selected;
}
export async function choosePlacement(game:Game,status:Status,query:(request:Request)=>Promise<ModelReply>):Promise<Decision>{
 const turn=prepareTurn(game),{candidates,request}=turn;
 let selected=turn.recommended,reply:ModelReply|undefined;
 if(game.mode===0){
  if(!status.enabled)throw new Error('Model unavailable');
  reply=await query(request);
  selected=validateReply(reply,request,status);
 }
 const result=landed(game.board,turn.piece,candidates[selected].cells);
 const over=game.turn+1>=MAX_TURNS||placements(result.board,turn.next).length===0;
 const model_candidates=candidates.map((_,i)=>i);
 return {id:game.turn+1,game:0,seed:game.seed,mode:game.mode,turn:game.turn,piece:turn.piece,next:turn.next,
  before:game.board.slice(),after:result.board,total_lines:game.lines+result.lines,over,legal_count:turn.legal_count,
  candidates,recommended:turn.recommended,selected,logits:reply?Array.from(reply.logits):[],scores:reply?Array.from(reply.probabilities):[],
  model:Array.from(reply?.model??status.model),prompt:reply?renderPrompt(request):'',model_request:reply?request:null,
  input_tokens:reply?.input_tokens??0,inference_instructions:reply?.measured_instructions.toString()??'0',rules:RULES,
  model_candidates,rounds:reply?[{model_candidates,selected,input_tokens:reply.input_tokens}]:[],query_count:reply?1:0};
}
export async function connectQuery(host:string,canisterId:string){
 const local=['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname);
 const createActor=async()=>{
  if(!canisterId)throw new Error('Canister ID is not configured');
  const agent=await HttpAgent.create({host});
  if(local)await agent.fetchRootKey();
  return Actor.createActor(queryIdl,{agent,canisterId});
 };
 let cachedActor:ReturnType<typeof createActor>|undefined;
 const actor=()=>cachedActor??=createActor().catch(error=>{
  cachedActor=undefined;
  throw error;
 });
 const fetchStatus=async():Promise<Status>=>{
  const a=await actor();
  const [info,limits]=await Promise.all([a.info() as Promise<{model:Uint8Array;warmed:boolean}>,a.query_limits() as Promise<{max_tokens:number}>]);
 return {enabled:info.warmed&&limits.max_tokens>=41,warmed:info.warmed,model:info.model,max_tokens:limits.max_tokens};
 };
 let cachedStatus:Promise<Status>|undefined;
 const status=(refresh=false)=>{
  if(refresh)cachedStatus=undefined;
  return cachedStatus??=fetchStatus().then(value=>{
  if(!value.enabled)cachedStatus=undefined;
  return value;
  },error=>{cachedStatus=undefined;throw error;});
 };
 const query=async(request:Request):Promise<ModelReply>=>{
  const r=await (await actor()).decide_query(request) as {Ok?:ModelReply;Err?:unknown};
  if(!r.Ok)throw new Error(`Model query failed: ${JSON.stringify(r.Err)}`);
  return r.Ok;
 };
 return {local,status,start:(seed:number,mode:number):Game=>({id:0,seed,mode,board:Array(200).fill(0),turn:0,lines:0,over:false}),
  step:async(game:Game)=>{
   try{return await choosePlacement(game,game.mode===0?await status():{enabled:false,warmed:false,model:[],max_tokens:0},query);}
   catch(error){if(game.mode===0)cachedStatus=undefined;throw error;}
  }};
}
export type QueryClient=Awaited<ReturnType<typeof connectQuery>>;
