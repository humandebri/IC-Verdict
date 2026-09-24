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
// Separate cell: existing metadata/snapshots keep their original encoding.
const ENGINE_CYCLES:MemoryId=MemoryId::new(6);
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
    static ENGINE_CYCLES_CELL:RefCell<StableCell<u128,Memory>>=RefCell::new(StableCell::init(memory(ENGINE_CYCLES),0));
}
pub fn engine_cycles()->u128{ENGINE_CYCLES_CELL.with(|x|*x.borrow().get())}
pub fn set_engine_cycles(cycles:u128){ENGINE_CYCLES_CELL.with(|x|{x.borrow_mut().set(cycles);});}
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
        requests:read_digest_map(&REQUEST_MAP)?,changes:Changes::default(),next_nonce:NONCE_MAP.with(|x|x.borrow().iter().map(|e|(*e.key(),e.value())).collect()),mock_ledgers:m.mock_ledgers})
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

/// Persist only keys changed by core, including transitions returning Err.
pub fn sync(s:&ExecutorState)->Result<()>{
    let c=&s.changes;
    write_changes(&PLAN_MAP,&s.plans,&c.plans)?;
    write_changes(&OPERATION_MAP,&s.operations,&c.operations)?;
    write_changes(&GRANT_MAP,&s.grants,&c.grants)?;
    write_changes(&REQUEST_MAP,&s.requests,&c.requests)?;
    NONCE_MAP.with(|x|{let mut x=x.borrow_mut();for k in &c.nonces{
        if let Some(v)=s.next_nonce.get(k){x.insert(*k,*v);}else{x.remove(k);}
    }});
    if c.meta{let bytes=encode(&meta(s))?;META_CELL.with(|x|{x.borrow_mut().set(bytes);});}
    Ok(())
}
fn write_changes<T:Serialize>(map:&'static std::thread::LocalKey<RefCell<StableBTreeMap<Digest,Vec<u8>,Memory>>>,
    values:&BTreeMap<Digest,T>,keys:&std::collections::BTreeSet<Digest>)->Result<()>{
    map.with(|x|{let mut x=x.borrow_mut();for k in keys{
        if let Some(v)=values.get(k){x.insert(*k,encode(v)?);}else{x.remove(k);}
    }Ok(())})
}

#[cfg(test)]
mod tests{
    use super::*;
    use ic_laya_core::demo::*;
    fn commit(s:&mut ExecutorState){sync(s).unwrap();s.changes=Changes::default();}
    fn same(s:&ExecutorState){assert_eq!(encode(s).unwrap(),encode(&load().unwrap()).unwrap());}
    #[test]
    fn adding_cycles_cell_preserves_existing_tables(){
        // Seed IDs 0..5 before ID 6 has ever been initialized, as in an old install.
        let (s,_,_,_)=setup().unwrap();replace_all(&s).unwrap();
        assert_eq!(engine_cycles(),0);same(&s);
        let amount=u128::from(u64::MAX)+1;
        set_engine_cycles(amount);
        let restored=StableCell::init(memory(ENGINE_CYCLES),0u128);
        assert_eq!(*restored.get(),amount);same(&s);
    }
    #[test]
    fn write_set_survives_async_transitions_errors_and_recovery(){
        let (mut s,mut engine,op,grant)=setup().unwrap();replace_all(&s).unwrap();s.changes=Changes::default();
        let id=s.submit(actor(2),0,op,grant,NOW).unwrap();commit(&mut s);same(&s);
        for _ in 0..3{
            let req=s.begin_evaluation(actor(2),id,NOW+1).unwrap();commit(&mut s);same(&s);
            let reply=engine.evaluate(s.instance,req.clone(),NOW+2,&FixtureTokenizer,&mut FixtureBackend::default());
            s.finish_evaluation(id,req.evaluation_id,reply,NOW+3).unwrap();commit(&mut s);same(&s);
        }
        s.set_mode(actor(1),Mode::Mock).unwrap();commit(&mut s);
        let cap=s.authorize(actor(2),id,NOW+4).unwrap();let cmd=s.prepare_dispatch(cap,NOW+4).unwrap();commit(&mut s);same(&s);
        s.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();commit(&mut s);same(&s);
        s.abandon_unknown(actor(1),id,"upgrade".into()).unwrap();commit(&mut s);
        s.recover_after_upgrade();commit(&mut s);same(&s);
        s.finish_ledger(&cmd,LedgerOutcome::Success("1".into())).unwrap();commit(&mut s);same(&s);s.check_invariants().unwrap();
    }
    #[test]
    fn registry_revisions_revocation_and_metadata_survive_reload(){
        let (fixture,_,op,grant)=setup().unwrap();
        let mut s=ExecutorState::new(fixture.instance,fixture.owner,fixture.engine);
        replace_all(&s).unwrap();
        let plan=fixture.plans.values().next().unwrap().clone();
        s.install_plan(actor(1),plan).unwrap();commit(&mut s);same(&s);
        s.install_operation(actor(1),fixture.operations[&op].clone()).unwrap();commit(&mut s);same(&s);
        s.install_grant(actor(1),fixture.grants[&grant].clone()).unwrap();commit(&mut s);same(&s);
        s.revise_operation(actor(1),op,"updated evidence".into()).unwrap();commit(&mut s);same(&s);
        let id=s.submit(actor(2),0,op,grant,NOW).unwrap();commit(&mut s);
        s.cancel(actor(2),id).unwrap();commit(&mut s);same(&s);
        s.revoke(actor(1),grant).unwrap();commit(&mut s);same(&s);
        s.set_paused(actor(1),true).unwrap();s.set_mode(actor(1),Mode::Shadow).unwrap();commit(&mut s);same(&s);
        assert!(s.install_operation(actor(2),fixture.operations[&op].clone()).is_err());
        assert!(!s.changes.meta && s.changes.operations.is_empty());same(&s);
    }
    #[test]
    fn one_changed_request_is_independent_of_history_size(){
        for count in [1,64,512]{
            let (mut s,_,op,grant)=setup().unwrap();
            let mut ids=Vec::new();for nonce in 0..count{ids.push(s.submit(actor(2),nonce,op,grant,NOW).unwrap());}
            replace_all(&s).unwrap();s.changes=Changes::default();
            let id=ids[0];assert!(s.begin_evaluation(actor(2),id,NOW+400_000_000_000).is_err());
            assert_eq!(s.changes.requests.len(),1);assert!(s.changes.grants.is_empty()&&s.changes.nonces.is_empty());
            commit(&mut s);same(&s);assert_eq!(s.requests[&id].status,Status::Stale);
        }
    }
    #[test]
    fn transport_retry_report_only_and_nonce_deletion_are_persisted(){
        let(mut s,mut engine,op,grant)=setup().unwrap();replace_all(&s).unwrap();s.changes=Changes::default();
        let id=s.submit(actor(2),0,op,grant,NOW).unwrap();commit(&mut s);
        let req=s.begin_evaluation(actor(2),id,NOW+1).unwrap();commit(&mut s);
        s.engine_transport_failed(id,req.evaluation_id).unwrap();commit(&mut s);same(&s);
        complete(&mut s,&mut engine,id).unwrap();commit(&mut s);
        let cap=s.authorize(actor(2),id,NOW+4).unwrap();assert!(matches!(s.prepare_dispatch(cap,NOW+4),Err(Error::ReportOnly)));commit(&mut s);same(&s);
        s.next_nonce.insert(actor(9),1);s.changes.nonces.insert(actor(9));commit(&mut s);
        s.release_caller(actor(1),actor(9)).unwrap();commit(&mut s);same(&s);
    }
}
