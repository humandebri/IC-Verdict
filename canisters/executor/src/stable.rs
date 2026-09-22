use bincode::Options;
use candid::Principal;
use ic_laya_core::{workflow::*,Digest,Error,Result};
use ic_stable_structures::{
    memory_manager::{MemoryId,MemoryManager,VirtualMemory},
    DefaultMemoryImpl,StableBTreeMap,StableCell,
};
use serde::{de::DeserializeOwned,Deserialize,Serialize};
use std::{cell::RefCell,collections::BTreeMap};

type Memory=VirtualMemory<DefaultMemoryImpl>;
const META:MemoryId=MemoryId::new(0);
const PLANS:MemoryId=MemoryId::new(1);
const OPERATIONS:MemoryId=MemoryId::new(2);
const GRANTS:MemoryId=MemoryId::new(3);
const REQUESTS:MemoryId=MemoryId::new(4);
const NONCES:MemoryId=MemoryId::new(5);
const FORMAT:u32=1;

#[derive(Clone,Serialize,Deserialize)]
struct Meta{format:u32,instance:Principal,owner:Principal,engine:Principal,mode:Mode,paused:bool,mock_ledgers:Vec<Principal>}

thread_local!{
    static MANAGER:RefCell<MemoryManager<DefaultMemoryImpl>>=RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
    static META_CELL:RefCell<StableCell<Vec<u8>,Memory>>=RefCell::new(StableCell::init(memory(META),Vec::new()));
    static PLAN_MAP:RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>=RefCell::new(StableBTreeMap::init(memory(PLANS)));
    static OPERATION_MAP:RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>=RefCell::new(StableBTreeMap::init(memory(OPERATIONS)));
    static GRANT_MAP:RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>=RefCell::new(StableBTreeMap::init(memory(GRANTS)));
    static REQUEST_MAP:RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>=RefCell::new(StableBTreeMap::init(memory(REQUESTS)));
    static NONCE_MAP:RefCell<StableBTreeMap<Principal,u64,Memory>>=RefCell::new(StableBTreeMap::init(memory(NONCES)));
}
fn memory(id:MemoryId)->Memory{MANAGER.with(|m|m.borrow().get(id))}
fn encode<T:Serialize>(value:&T)->Result<Vec<u8>>{bincode::DefaultOptions::new().with_fixint_encoding().serialize(value).map_err(|_|Error::Storage)}
fn decode<T:DeserializeOwned>(bytes:&[u8])->Result<T>{bincode::DefaultOptions::new().with_fixint_encoding().reject_trailing_bytes().deserialize(bytes).map_err(|_|Error::Storage)}
fn meta(s:&ExecutorState)->Meta{Meta{format:FORMAT,instance:s.instance,owner:s.owner,engine:s.engine,mode:s.mode,paused:s.paused,mock_ledgers:s.mock_ledgers.clone()}}

pub fn is_legacy_snapshot()->bool{
    if ic_cdk::stable::stable_size()==0{return false;}
    let mut magic=[0u8;8];ic_cdk::stable::stable_read(0,&mut magic);&magic==canister_common::SNAPSHOT_MAGIC
}

pub fn load()->Result<ExecutorState>{
    let m=META_CELL.with(|x|decode::<Meta>(x.borrow().get()))?;
    if m.format!=FORMAT{return Err(Error::Storage);}
    Ok(ExecutorState{instance:m.instance,owner:m.owner,engine:m.engine,mode:m.mode,paused:m.paused,
        plans:read_digest_map(&PLAN_MAP)?,operations:read_digest_map(&OPERATION_MAP)?,grants:read_digest_map(&GRANT_MAP)?,
        requests:read_digest_map(&REQUEST_MAP)?,next_nonce:NONCE_MAP.with(|x|x.borrow().iter().map(|e|(*e.key(),e.value())).collect()),mock_ledgers:m.mock_ledgers})
}

fn read_digest_map<T:DeserializeOwned>(map:&'static std::thread::LocalKey<RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>>)->Result<BTreeMap<Digest,T>>{
    map.with(|x|x.borrow().iter().map(|e|decode(&e.value()).map(|v|(*e.key(),v))).collect())
}

pub fn replace_all(s:&ExecutorState)->Result<()>{
    clear(&PLAN_MAP);clear(&OPERATION_MAP);clear(&GRANT_MAP);clear(&REQUEST_MAP);
    NONCE_MAP.with(|x|{let keys:Vec<_>=x.borrow().keys().collect();let mut x=x.borrow_mut();for k in keys{x.remove(&k);}});
    write_all(&PLAN_MAP,&s.plans)?;write_all(&OPERATION_MAP,&s.operations)?;write_all(&GRANT_MAP,&s.grants)?;write_all(&REQUEST_MAP,&s.requests)?;
    NONCE_MAP.with(|x|{let mut x=x.borrow_mut();for(k,v)in &s.next_nonce{x.insert(*k,*v);}});
    let bytes=encode(&meta(s))?;META_CELL.with(|x|{x.borrow_mut().set(bytes);});Ok(())
}

fn clear(map:&'static std::thread::LocalKey<RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>>){map.with(|x|{let keys:Vec<_>=x.borrow().keys().collect();let mut x=x.borrow_mut();for k in keys{x.remove(&k);}})}
fn write_all<T:Serialize>(map:&'static std::thread::LocalKey<RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>>,values:&BTreeMap<Digest,T>)->Result<()>{
    map.with(|x|{let mut x=x.borrow_mut();for(k,v)in values{x.insert(*k,encode(v)?);}Ok(())})
}

pub fn sync(old:&ExecutorState,new:&ExecutorState)->Result<()>{
    sync_map(&PLAN_MAP,&old.plans,&new.plans)?;sync_map(&OPERATION_MAP,&old.operations,&new.operations)?;
    sync_map(&GRANT_MAP,&old.grants,&new.grants)?;sync_map(&REQUEST_MAP,&old.requests,&new.requests)?;
    NONCE_MAP.with(|x|{let mut x=x.borrow_mut();for k in old.next_nonce.keys().filter(|k|!new.next_nonce.contains_key(*k)){x.remove(k);}for(k,v)in &new.next_nonce{if old.next_nonce.get(k)!=Some(v){x.insert(*k,*v);}}});
    if encode(&meta(old))?!=encode(&meta(new))?{let bytes=encode(&meta(new))?;META_CELL.with(|x|{x.borrow_mut().set(bytes);});}
    Ok(())
}
fn sync_map<T:Serialize+PartialEq>(map:&'static std::thread::LocalKey<RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>>,old:&BTreeMap<Digest,T>,new:&BTreeMap<Digest,T>)->Result<()>{
    map.with(|x|{let mut x=x.borrow_mut();for k in old.keys().filter(|k|!new.contains_key(*k)){x.remove(k);}for(k,v)in new{if old.get(k)!=Some(v){x.insert(*k,encode(v)?);}}Ok(())})
}
