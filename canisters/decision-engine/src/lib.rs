//! Default build starts unavailable. Synthetic fixture mode is an explicit admin action.
use candid::{CandidType,Principal};
use ic_laya_core::{engine::EngineState,schema,*,demo::{FixtureBackend,FixtureTokenizer}};
use serde::{Serialize,Deserialize};
use std::cell::RefCell;

/// Supplies the getrandom 0.3 backend that candle-core and tokenizers need on
/// wasm32-unknown-unknown. Must stay in the canister (wasm link root) and
/// unconditional, because `.cargo/config.toml` selects that backend for the whole
/// wasm32 target. Compiles to nothing on native targets.
#[cfg(target_arch = "wasm32")]
mod getrandom_ic;

#[derive(Clone,Serialize,Deserialize)]
struct Upload {manifest:Vec<u8>,tokenizer_length:u64,model_length:u64,received:u64,special:SpecialTokens}
#[derive(Clone,Serialize,Deserialize)]
struct Persistent {owner:Principal,engine:EngineState,fixture_mode:bool,upload:Option<Upload>}
thread_local!{static STATE:RefCell<Option<Persistent>>=const{RefCell::new(None)};}
#[cfg(feature="candle")]
thread_local!{
    static TOKENIZER:RefCell<Option<hf_tokenizer::HfTokenizer>>=const{RefCell::new(None)};
    static BUILDER:RefCell<Option<laya_candle::pack::Builder>>=const{RefCell::new(None)};
    static MODEL:RefCell<Option<laya_candle::LayaModel>>=const{RefCell::new(None)};
}
fn read<R>(f:impl FnOnce(&Persistent)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
fn mutate<R>(f:impl FnOnce(&mut Persistent)->R)->R{STATE.with(|x|{let mut state=x.borrow_mut();let state=state.as_mut().expect("initialized");let result=f(state);canister_common::persist_or_trap(state);result})}
fn owner()->Result<()>{read(|s|if ic_cdk::api::msg_caller()==s.owner{Ok(())}else{Err(Error::Unauthorized)})}
#[ic_cdk::init]
fn init(owner:Principal){
    if owner==Principal::anonymous() || owner==Principal::management_canister(){ic_cdk::trap("invalid owner");}
    let s=Persistent{owner,engine:EngineState::new([0;32]),fixture_mode:false,upload:None};canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(canister_common::persist_or_trap);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){let s:Persistent=canister_common::restore().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));STATE.with(|x|*x.borrow_mut()=Some(s));}
#[derive(CandidType,Serialize,Deserialize)]
pub struct EngineInfo {pub model:Digest,pub fixture:bool,pub schemas:u64,pub cached:u64,pub live_dispatch_supported:bool}
#[ic_cdk::query]
fn info()->EngineInfo{read(|s|EngineInfo{model:s.engine.active_model,fixture:s.fixture_mode,schemas:s.engine.schemas.len() as u64,cached:s.engine.cache.len() as u64,live_dispatch_supported:false})}
#[ic_cdk::update]
fn allow_caller(caller:Principal,per_minute:u32)->Result<()>{owner()?;mutate(|s|s.engine.allow_caller(caller,per_minute))}
#[ic_cdk::update]
fn enable_synthetic_fixture()->Result<Digest>{
    owner()?;let bundle=ic_laya_core::demo::fixture_bundle();
    mutate(|s|{if s.upload.is_some(){return Err(Error::Transition);}s.fixture_mode=true;s.engine.active_model=bundle;Ok(bundle)})
}
#[ic_cdk::update]
fn register_schema(schema:Schema,qtype_id:u32)->Result<CompiledSchema>{
    owner()?;
    if !read(|s|s.fixture_mode) {
        #[cfg(feature="candle")]
        {
            let upload=read(|s|s.upload.clone()).ok_or(Error::Transition)?;
            let manifest=laya_candle::pack::Manifest::parse(&upload.manifest)?;
            if manifest.primitive_to_qtype[schema.primitive.tag() as usize]!=qtype_id {
                return Err(Error::BindingMismatch);
            }
        }
    }
    let compiled=if read(|s|s.fixture_mode){schema::compile(schema,&FixtureTokenizer,qtype_id)?}else{
        #[cfg(feature="candle")]
        {TOKENIZER.with(|t|schema::compile(schema,t.borrow().as_ref().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?,qtype_id))?}
        #[cfg(not(feature="candle"))]
        {return Err(Error::ModelUnavailable("build decision-engine with --features candle".into()));}
    };
    mutate(|s|s.engine.register(compiled.clone()))?;Ok(compiled)
}
#[ic_cdk::update]
fn register_calibration(c:Calibration)->Result<()>{owner()?;mutate(|s|s.engine.register_calibration(c,ic_cdk::api::time()))}
/// MEASUREMENT ONLY. Reports where inference instructions are spent, phase by phase.
///
/// This exists because the acceptance targets are instruction budgets and nothing
/// reported a breakdown: `evaluate` returns one total for the whole call. It uses the
/// same validation and caller checks as `evaluate` but **bypasses the cache**, since
/// a cached result would report nothing.
///
/// It cannot move funds: it does not touch the executor, does not create a Receipt,
/// and is not reachable from any dispatch path. Gated on the candle feature because
/// the fixture backend has no real phases to report.
/// Wire form of `laya_candle::PhaseCost`. Kept local because deriving CandidType on
/// the crate type would make `laya-candle` depend on candid for a type that only
/// exists for measurement.
#[derive(Clone,CandidType,Serialize,Deserialize)]
pub struct PhaseCostDto { pub name:String, pub instructions:u64 }

#[cfg(feature="candle")]
#[ic_cdk::update]
fn measure_phases(req:DecisionRequest)->Result<Vec<PhaseCostDto>>{
    let caller=ic_cdk::api::msg_caller();let now=ic_cdk::api::time();
    if !read(|s|s.engine.callers.contains_key(&caller)){return Err(Error::Unauthorized);}
    if req.state.is_empty() || req.state.len()>MAX_STATE_BYTES{return Err(Error::TooLong);}
    if req.expires_at_ns<=now || req.expires_at_ns-now>600_000_000_000{return Err(Error::Expired);}
    // Validate against registered state exactly as evaluate would, so the measured
    // phases describe a request that would actually be accepted.
    let schema=read(|s|s.engine.schemas.get(&req.schema_hash).cloned()).ok_or(Error::NotFound)?;
    read(|s|{
        if req.model!=s.engine.active_model{return Err(Error::BindingMismatch);}
        if let Some(id)=req.calibration{
            let c=s.engine.calibrations.get(&id).ok_or(Error::Uncalibrated)?;
            c.validate(now)?;
            if c.model!=req.model || c.schema!=req.schema_hash || c.tokenizer!=schema.tokenizer_hash{return Err(Error::BindingMismatch);}
            if req.expires_at_ns>c.expires_at_ns{return Err(Error::Uncalibrated);}
        }Ok(())
    })?;
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let mut m=m.borrow_mut();
        let (t,m)=match (t.as_ref(),m.as_mut()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("checkpoint not warmed".into()))};
        let input=crate::schema::render(&schema,t,&req.state)?;
        m.infer_profiled_detailed(&input,&||ic_cdk::api::instruction_counter(),true)
            .map(|costs|costs.into_iter().map(|c|PhaseCostDto{name:c.name.to_string(),instructions:c.instructions}).collect())
    }))
}
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
            #[cfg(feature="candle")]
            {TOKENIZER.with(|t|MODEL.with(|m|{
                let t=t.borrow();let mut m=m.borrow_mut();
                match (t.as_ref(),m.as_mut()){
                    (Some(t),Some(m))=>s.engine.evaluate(caller,req,now,t,m),
                    _=>Err(Error::ModelUnavailable("checkpoint not warmed".into())),
                }
            }))}
            #[cfg(not(feature="candle"))]
            {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
        };
        if !cached {
            if let Ok(r)=&mut result{r.measured_instructions=ic_cdk::api::instruction_counter().saturating_sub(start);}
            if let Some(entry)=s.engine.cache.get_mut(&key){entry.result=result.clone();}
        }result
    })
}
#[ic_cdk::update]
fn begin_upload(manifest:Vec<u8>,tokenizer_length:u64,special:SpecialTokens)->Result<Digest>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let m=laya_candle::pack::Manifest::parse(&manifest)?;
        if m.config.mask_token_id!=special.mask{return Err(Error::BindingMismatch);}
        if tokenizer_length==0 || tokenizer_length>32*1024*1024{return Err(Error::TooLong);}
        let bundle=hash(&manifest);
        // No second active model is retained during replacement.
        MODEL.with(|x|*x.borrow_mut()=None);BUILDER.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=None);
        mutate(|s|{s.fixture_mode=false;s.upload=Some(Upload{manifest,tokenizer_length,model_length:m.total_bytes,received:0,special});Ok(bundle)})
    }
    #[cfg(not(feature="candle"))]
    {let _=(manifest,tokenizer_length,special);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[ic_cdk::update]
fn upload_chunk(offset:u64,bytes:Vec<u8>)->Result<u64>{
    owner()?;if bytes.is_empty() || bytes.len()>1024*1024{return Err(Error::TooLong);}
    mutate(|s|{
        let u=s.upload.as_mut().ok_or(Error::Transition)?;
        let end=offset.checked_add(bytes.len() as u64).ok_or(Error::TooLong)?;
        if end>u.model_length+u.tokenizer_length{return Err(Error::TooLong);}
        if offset<u.received {
            if end>u.received{return Err(Error::IdConflict);}let mut old=vec![0;bytes.len()];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+offset,&mut old);
            return if old==bytes{Ok(u.received)}else{Err(Error::IdConflict)};
        }
        if offset!=u.received{return Err(Error::NonceGap);}
        canister_common::grow_through(canister_common::BLOB_BASE+end)?;
        ic_cdk::stable::stable_write(canister_common::BLOB_BASE+offset,&bytes);u.received=end;Ok(end)
    })
}
#[ic_cdk::update]
fn start_warmup()->Result<u64>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let u=read(|s|s.upload.clone()).ok_or(Error::Transition)?;
        if u.received!=u.model_length+u.tokenizer_length{return Err(Error::Transition);}
        let builder=laya_candle::pack::Builder::new(&u.manifest)?;
        let mut bytes=vec![0;u.tokenizer_length as usize];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+u.model_length,&mut bytes);
        if hash(&bytes)!=builder.manifest.tokenizer_sha256{return Err(Error::BindingMismatch);}
        let tok=hf_tokenizer::HfTokenizer::from_bytes(&bytes,u.special)?;
        let count=builder.manifest.tensors.len() as u64;
        MODEL.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=Some(tok));BUILDER.with(|x|*x.borrow_mut()=Some(builder));Ok(count)
    }
    #[cfg(not(feature="candle"))]
    {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[ic_cdk::update]
fn warmup_next()->Result<bool>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let done=BUILDER.with(|b|{
            let mut b=b.borrow_mut();let builder=b.as_mut().ok_or(Error::Transition)?;
            if let Some(e)=builder.next_entry().cloned(){
                let mut bytes=vec![0;e.length as usize];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+e.offset,&mut bytes);builder.push(&bytes)?;
            }
            Ok::<bool,Error>(builder.next_entry().is_none())
        })?;
        if done {
            let builder=BUILDER.with(|b|b.borrow_mut().take()).ok_or(Error::Transition)?;let bundle=builder.bundle;let model=builder.finish()?;
            MODEL.with(|m|*m.borrow_mut()=Some(model));mutate(|s|s.engine.active_model=bundle);
        }Ok(done)
    }
    #[cfg(not(feature="candle"))]
    {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}
