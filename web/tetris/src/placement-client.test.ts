import {beforeEach,expect,it,vi} from 'vitest';
const mock=vi.hoisted(()=>({info:vi.fn(),limits:vi.fn(),decide:vi.fn(),create:vi.fn()}));
vi.mock('@icp-sdk/core/agent',()=>({
 HttpAgent:{create:async()=>({})},
 Actor:{createActor:()=>{mock.create();return {info:mock.info,query_limits:mock.limits,decide_query:mock.decide};}},
}));
import {connectQuery} from './placement-query-api';

beforeEach(()=>{
 vi.clearAllMocks();
 mock.info.mockResolvedValue({warmed:true,model:Uint8Array.from(Array(32).fill(7))});
 mock.limits.mockResolvedValue({max_tokens:52});
 mock.decide.mockImplementation(async request=>({Ok:{ids:request.options.map((o:{id:string})=>o.id),selected:'0',
  probabilities:request.options.map((_:unknown,i:number)=>i===0?1:0),logits:request.options.map(()=>0),
  model:Uint8Array.from(Array(32).fill(7)),confidence:1,input_tokens:41,measured_instructions:3_000_000_000n}}));
});
it('comparison play performs no canister method call',async()=>{
 const client=await connectQuery('https://icp-api.io','aaaaa-aa');
 const result=await client.step(client.start(11,1));
 expect(result.query_count).toBe(0);expect(mock.info).not.toHaveBeenCalled();
 expect(mock.limits).not.toHaveBeenCalled();expect(mock.decide).not.toHaveBeenCalled();
});
it('model play sends descriptions in one generic query',async()=>{
 const client=await connectQuery('https://icp-api.io','aaaaa-aa');
 const result=await client.step(client.start(11,0));
 expect(mock.info).toHaveBeenCalledTimes(1);expect(mock.limits).toHaveBeenCalledTimes(1);
 expect(mock.decide).toHaveBeenCalledTimes(1);expect(result.query_count).toBe(1);
 const request=mock.decide.mock.calls[0][0];
 expect(Object.keys(request).sort()).toEqual(['abstention','options','question','state','temperature']);
 expect(JSON.stringify(request)).not.toContain('board');
});
it('rechecks status after the model warms up',async()=>{
 const client=await connectQuery('https://icp-api.io','aaaaa-aa');
 mock.info.mockResolvedValueOnce({warmed:false,model:Uint8Array.from(Array(32).fill(7))});
 expect((await client.status()).enabled).toBe(false);
 expect((await client.status()).enabled).toBe(true);
 expect(mock.info).toHaveBeenCalledTimes(2);
});
it('rechecks status after a transient query failure',async()=>{
 const client=await connectQuery('https://icp-api.io','aaaaa-aa');
 mock.info.mockRejectedValueOnce(new Error('temporary failure'));
 await expect(client.status()).rejects.toThrow('temporary failure');
 expect((await client.status()).enabled).toBe(true);
 expect(mock.info).toHaveBeenCalledTimes(2);
});
