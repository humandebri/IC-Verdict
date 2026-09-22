//! Fixture-mode decision engine: schemas, calibration, evaluation and the ledger wire.
//!
//! The Laya checkpoint backend used to live here behind the `candle` feature. That
//! model was dropped, so the model-upload and warm-up endpoints went with it and this
//! canister now serves the synthetic fixture backend, which is what the integration
//! harness and the mock workflow exercise. The openJev backend is a separate canister
//! (`canisters/verdict-engine`).
use candid::{CandidType,Principal};
use ic_laya_core::{engine::EngineState,schema,*,demo::{FixtureBackend,FixtureTokenizer}};
use serde::{Serialize,Deserialize};
use std::cell::RefCell;

/// Supplies the getrandom 0.3 backend for wasm32-unknown-unknown. Must stay in the
/// canister (wasm link root) and unconditional, because `.cargo/config.toml` selects
/// that backend for the whole wasm32 target. Compiles to nothing on native targets.
#[cfg(target_arch = "wasm32")]
mod getrandom_ic;

#[derive(Clone,Serialize,Deserialize)]
struct Persistent {owner:Principal,engine:EngineState,fixture_mode:bool}
thread_local!{static STATE:RefCell<Option<Persistent>>=const{RefCell::new(None)};}
fn read<R>(f:impl FnOnce(&Persistent)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
fn mutate<R>(f:impl FnOnce(&mut Persistent)->R)->R{STATE.with(|x|{let mut state=x.borrow_mut();let state=state.as_mut().expect("initialized");let result=f(state);canister_common::persist_or_trap(state);result})}
fn owner()->Result<()>{read(|s|if ic_cdk::api::msg_caller()==s.owner{Ok(())}else{Err(Error::Unauthorized)})}
#[ic_cdk::init]
fn init(owner:Principal){
    if owner==Principal::anonymous() || owner==Principal::management_canister(){ic_cdk::trap("invalid owner");}
    let s=Persistent{owner,engine:EngineState::new([0;32]),fixture_mode:false};canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(canister_common::persist_or_trap);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){let s:Persistent=canister_common::restore().unwrap_or_else(|e|ic_cdk::trap(e.to_string()));STATE.with(|x|*x.borrow_mut()=Some(s));}
#[derive(CandidType,Serialize,Deserialize)]
pub struct EngineInfo {pub model:Digest,pub fixture:bool,pub schemas:u64,pub cached:u64,pub live_dispatch_supported:bool}
#[ic_cdk::query]
fn info()->EngineInfo{read(|s|EngineInfo{model:s.engine.active_model,fixture:s.fixture_mode,schemas:s.engine.schemas.len() as u64,cached:s.engine.cache.len() as u64,live_dispatch_supported:false})}
#[ic_cdk::update]
fn allow_caller(caller:Principal,per_minute:u32)->Result<()>{owner()?;mutate(|s|s.engine.allow_caller(caller,per_minute))}
#[ic_cdk::update]
fn enable_synthetic_fixture()->Result<Digest>{
    owner()?;let bundle=ic_laya_core::demo::fixture_bundle();
    mutate(|s|{s.fixture_mode=true;s.engine.active_model=bundle;Ok(bundle)})
}
/// Schemas are compiled against the fixture tokenizer, which is the only backend here.
/// A production backend canister registers its own schemas against its own tokenizer.
#[ic_cdk::update]
fn register_schema(schema:Schema,qtype_id:u32)->Result<CompiledSchema>{
    owner()?;
    if !read(|s|s.fixture_mode){return Err(Error::ModelUnavailable("enable_synthetic_fixture first".into()));}
    let compiled=schema::compile(schema,&FixtureTokenizer,qtype_id)?;
    mutate(|s|s.engine.register(compiled.clone()))?;Ok(compiled)
}
#[ic_cdk::update]
fn register_calibration(c:Calibration)->Result<()>{owner()?;mutate(|s|s.engine.register_calibration(c,ic_cdk::api::time()))}
#[ic_cdk::update]
fn evaluate(req:DecisionRequest)->Result<Receipt>{
    let caller=ic_cdk::api::msg_caller();let now=ic_cdk::api::time();
    // Reject outsiders before serializing the durable engine snapshot.
    if !read(|s|s.engine.callers.contains_key(&caller)){return Err(Error::Unauthorized);}
    if req.state.is_empty() || req.state.len()>MAX_STATE_BYTES{return Err(Error::TooLong);}
    // A reserve for administration/recovery; this is not a cycles-per-decision benchmark.
    if ic_cdk::api::canister_cycle_balance()<100_000_000_000 {return Err(Error::Capacity);}
    let start=ic_cdk::api::instruction_counter();
    mutate(|s|{
        let cached=s.engine.cache.contains_key(&(caller,req.evaluation_id));
        let key=(caller,req.evaluation_id);
        let mut result=if s.fixture_mode {
            s.engine.evaluate(caller,req,now,&FixtureTokenizer,&mut FixtureBackend::default())
        }else{
            Err(Error::ModelUnavailable("enable_synthetic_fixture first".into()))
        };
        if !cached {
            if let Ok(r)=&mut result{r.measured_instructions=ic_cdk::api::instruction_counter().saturating_sub(start);}
            if let Some(entry)=s.engine.cache.get_mut(&key){entry.result=result.clone();}
        }result
    })
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}
