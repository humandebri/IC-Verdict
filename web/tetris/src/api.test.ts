import {beforeEach,describe,it,expect,vi} from 'vitest';
const mock=vi.hoisted(()=>({fetchRootKey:vi.fn(),decide:vi.fn(),status:vi.fn()}));
vi.mock('@icp-sdk/core/agent',()=>({HttpAgent:{create:async()=>({fetchRootKey:mock.fetchRootKey})},Actor:{createActor:()=>({tetris_decide_v3_query:mock.decide,tetris_v3_status:mock.status})}}));
import {connect} from './api';
const input={piece:2,next:0,x:5,y:0,rotation:0,heights:Array(10).fill(0),holes:0,legal_mask:31,previous:[] as [],history:[]};
const reply=()=>({action:4,actions:[0,1,2,3,4],scores:[.1,.1,.1,.1,.6],input_tokens:52,model:Array(32).fill(7),measured_instructions:2n,prompt:'test'});
describe('anonymous query adapter',()=>{
  beforeEach(()=>{vi.clearAllMocks();mock.decide.mockResolvedValue({Ok:reply()});});
  it('requires a canister ID',async()=>{await expect(connect('http://localhost:8011','',true)).rejects.toThrow('CANISTER');});
  it('never replaces a mainnet root key',async()=>{
    await connect('https://icp-api.io','test',false);expect(mock.fetchRootKey).not.toHaveBeenCalled();
    await expect(connect('https://icp-api.io','test',true)).rejects.toThrow('localhost');
  });
  it('fetches only the explicitly selected loopback root',async()=>{await connect('http://localhost:8011','test',true);expect(mock.fetchRootKey).toHaveBeenCalledOnce();});
  it('propagates rejection without fallback',async()=>{const c=await connect('http://localhost:8011','test',true);mock.decide.mockResolvedValue({Err:{Capacity:null}});await expect(c.decide(input)).rejects.toThrow('Capacity');expect(mock.decide).toHaveBeenCalledOnce();});
  it('checks response bounds',async()=>{
    const c=await connect('http://localhost:8011','test',true);
    for(const patch of [{action:5},{action:-1},{action:.5},{actions:[4,3,2,1,0]},{scores:[NaN,.1,.2,.3,.1]},{scores:[.7,.3]},{model:[0]},{input_tokens:53}]){
      mock.decide.mockResolvedValue({Ok:{...reply(),...patch}});await expect(c.decide(input)).rejects.toThrow('Invalid query response');
    }
  });
  it('sends context and preserves the wait selection',async()=>{const c=await connect('http://localhost:8011','test',true);expect(await c.decide(input)).toEqual(reply());expect(mock.decide).toHaveBeenCalledWith(input);});
  it('validates sparse action IDs rather than interpreting action as an index',async()=>{
    const c=await connect('http://localhost:8011','test',true),short={...input,legal_mask:24};
    mock.decide.mockResolvedValue({Ok:{...reply(),action:4,actions:new Uint8Array([3,4]),scores:[.2,.8]}});
    expect((await c.decide(short)).action).toBe(4);await expect(c.decide(input)).rejects.toThrow('Invalid query response');
  });
});
