import { Actor, HttpAgent } from '@icp-sdk/core/agent';
import { IDL } from '@icp-sdk/core/candid';
import { actionsFor,CONTROL_TOKEN_LIMIT,type Action,type DecisionInput } from './realtime';
export type Reply = {action:Action;actions:Action[];scores:number[];input_tokens:number;model:Uint8Array|number[];measured_instructions:bigint;prompt:string};
export type Status = {enabled:boolean;warmed:boolean;max_tokens:number;model:Uint8Array|number[]};
const errors = Object.fromEntries(['Unauthorized','Storage','Capacity','Numeric','TooLong','Transition','BindingMismatch','NonceGap','IdConflict','Expired','NotFound','Uncalibrated','Busy','NonceConsumed','OperationUsed','Budget','Stale','ReportOnly','LiveDisabled','OutcomeUnknown'].map(k=>[k,IDL.Null]));
// Candid permits decoding a server variant into a supertype with additional fields.
const ErrorType=IDL.Variant({...errors,Invalid:IDL.Text,ModelUnavailable:IDL.Text,Denied:IDL.Text});
export const idlFactory=()=>IDL.Service({
  tetris_v3_status:IDL.Func([], [IDL.Record({enabled:IDL.Bool,warmed:IDL.Bool,max_tokens:IDL.Nat32,model:IDL.Vec(IDL.Nat8)})], ['query']),
  tetris_decide_v3_query:IDL.Func([IDL.Record({piece:IDL.Nat8,next:IDL.Nat8,x:IDL.Nat8,y:IDL.Nat8,rotation:IDL.Nat8,heights:IDL.Vec(IDL.Nat8),holes:IDL.Nat16,legal_mask:IDL.Nat8,previous:IDL.Opt(IDL.Record({x:IDL.Nat8,y:IDL.Nat8,rotation:IDL.Nat8,action:IDL.Nat8,outcome:IDL.Nat8})),history:IDL.Vec(IDL.Nat8)})],
    [IDL.Variant({Ok:IDL.Record({action:IDL.Nat8,actions:IDL.Vec(IDL.Nat8),scores:IDL.Vec(IDL.Float32),input_tokens:IDL.Nat32,model:IDL.Vec(IDL.Nat8),measured_instructions:IDL.Nat64,prompt:IDL.Text}),Err:ErrorType})],['query']),
});
export const hex=(bytes:Uint8Array|number[])=>Array.from(bytes).map(x=>x.toString(16).padStart(2,'0')).join('');
export async function connect(host:string,canisterId:string,local:boolean) {
  if(!canisterId) throw new Error('Set VITE_CANISTER_ID.');
  const url=new URL(host);
  if(local && !['localhost','127.0.0.1','[::1]'].includes(url.hostname)) throw new Error('The development root key can only be fetched from localhost.');
  const agent=await HttpAgent.create({host});
  if(local) await agent.fetchRootKey();
  const actor=Actor.createActor(idlFactory,{agent,canisterId});
  return {
    status:()=>actor.tetris_v3_status() as Promise<Status>,
    decide:async(input:DecisionInput):Promise<Reply>=>{
      const result=await actor.tetris_decide_v3_query(input) as {Ok?:Reply;Err?:Record<string,unknown>};
      if(!result.Ok) throw new Error(`Query failed: ${JSON.stringify(result.Err)}`);
      const r=result.Ok;
      r.actions=Array.from(r.actions);
      const expected=actionsFor(input.legal_mask);
      if(!Number.isInteger(r.action)||!expected.includes(r.action)||!Array.isArray(r.actions)||r.actions.length!==expected.length||r.actions.some((a,i)=>a!==expected[i])||r.scores.length!==expected.length||r.scores.some(x=>!Number.isFinite(x)||x<0||x>1)||r.model.length!==32||!Number.isInteger(r.input_tokens)||r.input_tokens<1||r.input_tokens>CONTROL_TOKEN_LIMIT) throw new Error('Invalid query response.');
      return r;
    },
  };
}
export type Client=Awaited<ReturnType<typeof connect>>;
