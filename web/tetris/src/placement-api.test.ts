import {describe,it,expect} from 'vitest';
import {verifyMembership,hex} from './placement-api';
const encode=(s:string)=>new TextEncoder().encode(s);
const digest=async(b:Uint8Array)=>new Uint8Array(await crypto.subtle.digest('SHA-256',new Uint8Array(b)));
describe('certified record membership (after certificate verification)',()=>{
 it('accepts exact record and rejects record/index/root/id tampering',async()=>{
 const record=encode('{"id":184}');const index=encode(`tetris-demo-v1\n184:${hex(await digest(record))}\n`);const root=await digest(index);
 expect((await verifyMembership(184,record,index,root)).id).toBe(184);
 await expect(verifyMembership(184,encode('{"id":185}'),index,root)).rejects.toThrow();
 await expect(verifyMembership(184,record,encode('modified'),root)).rejects.toThrow();
 await expect(verifyMembership(184,record,index,new Uint8Array(32))).rejects.toThrow();
 await expect(verifyMembership(185,record,index,root)).rejects.toThrow();
 });
});
