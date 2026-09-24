import {Actor, HttpAgent, Certificate, lookupResultToBuffer} from '@icp-sdk/core/agent';
import {Ed25519KeyIdentity} from '@icp-sdk/core/identity';
import {Principal} from '@icp-sdk/core/principal';
import {IDL} from '@icp-sdk/core/candid';
export type Pose={r:number;x:number;y:number};
export type Candidate={pose:Pose;cells:[number,number][];holes:number;height:number;lines:number;evaluation:number;path:Pose[]};
export type Game={id:number;seed:number;mode:number;board:number[];turn:number;lines:number;over:boolean};
export type Receipt={id:number;game:number;turn:number;piece:number;before:number[];after:number[];candidates:Candidate[];selected:number;recommended:number;legal_count:number;total_lines:number;over:boolean;mode:number;input_tokens:number;inference_instructions:string;prompt:string;rules:string;model:number[]};
export type Status={enabled:boolean;warmed:boolean;model_turns_left:number;games_left:number};
const result=IDL.Variant({Ok:IDL.Text,Err:IDL.Text});
export const placementIdl=()=>IDL.Service({
 tetris_demo_status:IDL.Func([],[IDL.Text],['query']),
 tetris_demo_start:IDL.Func([IDL.Nat32,IDL.Nat8,IDL.Nat64],[result],[]),
 tetris_demo_step:IDL.Func([IDL.Nat32,IDL.Nat32],[result],[]),
 tetris_demo_record:IDL.Func([IDL.Nat32],[IDL.Variant({Ok:IDL.Record({record:IDL.Vec(IDL.Nat8),index:IDL.Vec(IDL.Nat8),certificate:IDL.Vec(IDL.Nat8)}),Err:IDL.Text})],['query']),
});
const unwrap=<T>(r:{Ok?:T;Err?:string}):T=>{if(r.Ok===undefined)throw new Error(r.Err||'Missing response');return r.Ok;};
export const hex=(b:Uint8Array)=>Array.from(b,x=>x.toString(16).padStart(2,'0')).join('');
const digest=async(b:Uint8Array)=>new Uint8Array(await crypto.subtle.digest('SHA-256',new Uint8Array(b)));
export async function verifyMembership(id:number,record:Uint8Array,index:Uint8Array,root:Uint8Array){
 if(hex(await digest(index))!==hex(root))throw new Error('Certified index mismatch');
 if(!new TextDecoder().decode(index).split('\n').includes(`${id}:${hex(await digest(record))}`))throw new Error('Record hash mismatch');
 const r=JSON.parse(new TextDecoder().decode(record)) as Receipt;
 if(r.id!==id)throw new Error('Record ID mismatch');return r;
}
export async function connectPlacement(host:string,canisterId:string){
 const local=['localhost','127.0.0.1','[::1]'].includes(new URL(host).hostname);
 const saved=sessionStorage.getItem('placement-identity');
 const identity=saved?Ed25519KeyIdentity.fromJSON(saved):Ed25519KeyIdentity.generate();
 if(!saved)sessionStorage.setItem('placement-identity',JSON.stringify(identity.toJSON()));
 const agent=await HttpAgent.create({host,identity});if(local)await agent.fetchRootKey();
 const actor=Actor.createActor(placementIdl,{agent,canisterId});
 return {local,
 status:async()=>JSON.parse(await actor.tetris_demo_status() as string) as Status,
 start:async(seed:number,mode:number,nonce:bigint)=>JSON.parse(unwrap(await actor.tetris_demo_start(seed,mode,nonce) as {Ok?:string;Err?:string})) as Game,
 step:async(id:number,turn:number)=>JSON.parse(unwrap(await actor.tetris_demo_step(id,turn) as {Ok?:string;Err?:string})) as Receipt,
 verify:async(id:number)=>{
 const raw=unwrap(await actor.tetris_demo_record(id) as {Ok?:{record:Uint8Array;index:Uint8Array;certificate:Uint8Array};Err?:string});
 if(!agent.rootKey)throw new Error('Missing root key');
 const principal=Principal.fromText(canisterId);
 const cert=await Certificate.create({certificate:raw.certificate,rootKey:agent.rootKey,principal:{canisterId:principal},agent});
 const root=lookupResultToBuffer(cert.lookup_path(['canister',principal.toUint8Array(),'certified_data']));
 if(!root)throw new Error('Missing certified data');
 const receipt=await verifyMembership(id,raw.record,raw.index,root);
 return {receipt,canisterId,host,local,record:Array.from(raw.record),index:Array.from(raw.index),certificate:Array.from(raw.certificate)};
 }};
}
export type PlacementClient=Awaited<ReturnType<typeof connectPlacement>>;
