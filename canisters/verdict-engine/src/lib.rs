//! openJev (GLiClass / ModernBERT-151M) decision engine canister.
//!
//! Scope: this canister exists to answer one question with evidence — does the
//! ported GLiClass forward actually run inside a canister, and what does one
//! question cost in instructions? Public inference updates accept attached cycles
//! at three times the configured execution tariff and measured instructions.
//! Ordinary inference queries remain free; replicated query execution is refused.
//!
//! The model side is `verdict-candle`; its encoder is `modernbert-candle`, the shared
//! ModernBERT implementation.
//!
//! Sizing: the production block-32 INT8 pack is about 162.5 MiB; no two-dimensional
//! F32 weight is accepted. Instructions, not memory, remain the binding limit:
//! see `MAX_INPUT_TOKENS` and the measurement note in `infer_tokens`.
use candid::{CandidType,Principal};
use candle_core::{Device,Module,Tensor};
use candle_core::quantized::{GgmlDType,QMatMul,QTensor};
use ic_laya_core::{hash,Digest,Error,Result,SpecialTokens};
use ic_laya_core::engine::{EngineState,InferenceBackend};
use ic_laya_core::schema::{self,TextTokenizer};
use serde::{Deserialize,Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
mod billing;
pub use billing::{CyclesPricing,ExecutionPricing};

#[cfg(target_arch = "wasm32")]
#[path = "../../decision-engine/src/getrandom_ic.rs"]
mod getrandom_ic;

/// Refuse inputs longer than this. It is a policy bound; the *budget* guard is
/// `estimated_cost`, which uses the measured cost model below.
pub const MAX_INPUT_TOKENS:u32=128;
/// Per-row INT8 estimate calibrated with fused CPU data movement
/// (docs/QUERY_OPTIMIZATION_V3.md). Admits 53 tokens with the encoding margin.
/// It is not a global upper bound: attention becomes quadratic at longer lengths.
/// Upgrades preserve an owner's stored fit; set_cost_model explicitly opts it in.
pub const COST_FIXED:u64=170_000_000;
pub const COST_PER_TOKEN:u64=90_000_000;
/// ICP's per-update instruction limit.
pub const UPDATE_BUDGET:u64=40_000_000_000;
/// ICP's per-query instruction limit (canister resource limits: 40B per update call,
/// 5B per query call). A query also costs no cycles and needs no consensus round; the
/// price is this lower ceiling and an uncertified reply.
///
/// The token ceiling follows from the same cost model as the update one
/// (`T <= (QUERY_BUDGET - COST_FIXED) / COST_PER_TOKEN`), with an encoding margin.
/// `query_limits()` reports the derived value; nothing hardcodes it.
pub const QUERY_BUDGET:u64=5_000_000_000;
/// Bound public query text before prompt construction and tokenization. The token
/// count guard runs afterward, so it cannot protect that preprocessing work.
const MAX_PUBLIC_QUERY_TEXT_BYTES:usize=16*1024;
const MAX_PUBLIC_QUERY_QUESTION_BYTES:usize=4*1024;
const MAX_PUBLIC_QUERY_OPTION_BYTES:usize=1024;
const MAX_PUBLIC_QUERY_ID_BYTES:usize=128;
/// Keep a margin for the reply encoding and the tokenizer, which the linear model
/// above does not cover. This is an estimate, not a bound for every possible input.
const BUDGET_MARGIN_PERMILLE:u64=1005;
/// One decision's option list plus the abstention slot must fit the checkpoint's
/// 25 logit slots.
pub const MAX_OPTIONS:usize=24;
/// The checkpoint's own abstention description (core/primitives.py).
pub const ABSTENTION_DESC:&str="insufficient evidence";

#[derive(Clone,Serialize,Deserialize)]
struct Upload{manifest:Vec<u8>,tokenizer_length:u64,model_length:u64,received:u64,special:SpecialTokens}
#[derive(Clone,Serialize,Deserialize)]
struct Persistent{
    owner:Principal,
    active_model:Digest,
    upload:Option<Upload>,
    callers:BTreeMap<Principal,u32>,
    #[serde(default="empty_workflow")] workflow:EngineState,
    max_input_tokens:u32,
    // Cost model used by the pre-flight budget guard. `#[serde(default)]` keeps an
    // older snapshot loadable across an upgrade.
    #[serde(default="default_cost_fixed")] cost_fixed:u64,
    #[serde(default="default_cost_per_token")] cost_per_token:u64,
    #[serde(default="default_budget")] budget:u64,
}
#[derive(Clone,Serialize,Deserialize)]
struct LegacyPersistent{owner:Principal,active_model:Digest,upload:Option<Upload>,callers:BTreeMap<Principal,u32>,max_input_tokens:u32,
    #[serde(default="default_cost_fixed")] cost_fixed:u64,#[serde(default="default_cost_per_token")] cost_per_token:u64,#[serde(default="default_budget")] budget:u64}
// An explicit wrapper avoids relying on serde defaults for bincode field additions.
#[derive(Clone,Serialize,Deserialize)]
struct DemoSnapshot { version:u32, state:Persistent, tetris_enabled:bool }
#[derive(Clone,Serialize,Deserialize)]
struct BillingSnapshot { version:u32, state:Persistent, tetris_enabled:bool, execution_pricing:Option<ExecutionPricing> }
#[derive(Clone,Serialize,Deserialize)]
struct GameSnapshot { version:u32, state:Persistent, tetris_enabled:bool, execution_pricing:Option<ExecutionPricing>, games:LegacyGameStore }
#[derive(Clone,Serialize,Deserialize)]
struct LegacyGameStore { games:BTreeMap<u32,LegacyGame>, records:BTreeMap<u32,Vec<u8>>, next_game:u32, next_record:u32, model_turns_left:u32 }
#[derive(Clone,Serialize,Deserialize)]
struct LegacyGame { id:u32, owner:Principal, nonce:u64, seed:u32, mode:u8, board:Vec<u8>, turn:u32, lines:u32, over:bool, last_record:Option<u32>, model:Digest }
#[derive(Clone,Serialize,Deserialize)]
struct ModelSnapshot { version:u32, state:Persistent, execution_pricing:Option<ExecutionPricing> }
fn save_snapshot(s:&Persistent) {
    canister_common::persist_or_trap(&ModelSnapshot{version:4,state:s.clone(),execution_pricing:billing::get()});
}
#[cfg(test)]
mod snapshot_migration_tests {
    use super::*;
    #[test]
    fn old_game_snapshot_decodes_without_copying_game_state_into_model_snapshot() {
        let state=Persistent{owner:Principal::from_slice(&[1]),active_model:[7;32],upload:None,
            callers:BTreeMap::new(),workflow:EngineState::new([7;32]),max_input_tokens:128,
            cost_fixed:COST_FIXED,cost_per_token:COST_PER_TOKEN,budget:UPDATE_BUDGET};
        let game=LegacyGame{id:1,owner:state.owner,nonce:3,seed:184,mode:0,board:vec![0;200],
            turn:2,lines:1,over:false,last_record:Some(1),model:[7;32]};
        let old=GameSnapshot{version:3,state:state.clone(),tetris_enabled:true,
            execution_pricing:None,games:LegacyGameStore{games:BTreeMap::from([(1,game)]),
                records:BTreeMap::from([(1,vec![1,2,3])]),next_game:2,next_record:2,model_turns_left:0}};
        let old_bytes=ic_laya_core::storage::encode(&old).unwrap();
        assert!(ic_laya_core::storage::decode::<ModelSnapshot>(&old_bytes).is_err());
        let restored:GameSnapshot=ic_laya_core::storage::decode(&old_bytes).unwrap();
        assert_eq!(restored.state.active_model,[7;32]);
        let new=ModelSnapshot{version:4,state:restored.state,execution_pricing:restored.execution_pricing};
        let new_bytes=ic_laya_core::storage::encode(&new).unwrap();
        assert!(new_bytes.len()<old_bytes.len());
        assert_eq!(ic_laya_core::storage::decode::<ModelSnapshot>(&new_bytes).unwrap().state.active_model,[7;32]);
    }
}
fn default_cost_fixed()->u64{COST_FIXED}
fn default_cost_per_token()->u64{COST_PER_TOKEN}
fn default_budget()->u64{UPDATE_BUDGET}
fn empty_workflow()->EngineState{EngineState::new([0;32])}
/// Projected instructions for one forward pass of `tokens` tokens, plus margin.
fn projected(s:&Persistent,tokens:usize)->u64{
    (s.cost_fixed.saturating_add(s.cost_per_token.saturating_mul(tokens as u64)))
        .saturating_mul(BUDGET_MARGIN_PERMILLE)/1000
}
/// Refuse before the call spends `budget`.
///
/// `Capacity`, not a new `Error` variant: the variant list is public Candid surface and
/// the caller already knows which method it called, so "over budget" needs no second
/// discriminator to be actionable.
fn guard_within(s:&Persistent,tokens:usize,budget:u64)->Result<()>{
    guard_cost(s.cost_fixed,s.cost_per_token,tokens,budget)
}
fn guard_cost(fixed:u64,per_token:u64,tokens:usize,budget:u64)->Result<()>{
    let cost=fixed.saturating_add(per_token.saturating_mul(tokens as u64))
        .saturating_mul(BUDGET_MARGIN_PERMILLE)/1000;
    if cost>budget {Err(Error::Capacity)}else{Ok(())}
}
/// Same pack after an upgrade: retain registered contracts and idempotency records.
fn activate_bundle(s:&mut Persistent,bundle:Digest){
    if s.workflow.active_model!=bundle {
        let callers=std::mem::take(&mut s.workflow.callers);
        s.workflow=EngineState::new(bundle);
        s.workflow.callers=callers;
    }
    s.active_model=bundle;
}
fn finish_warmup(s:&mut Persistent,builder:verdict_candle::pack::Builder)->Result<verdict_candle::VerdictModel>{
    let bundle=builder.bundle;
    let model=builder.finish()?;
    activate_bundle(s,bundle);
    Ok(model)
}
/// Update-path guard: the policy budget the owner may tighten with `set_cost_model`.
/// The update entry points pass `s.budget` into `infer_once`/`decide_once`, so this is
/// only the named form of that comparison; keeping it makes the difference between the
/// two budgets explicit at the call sites.
fn guard_budget(s:&Persistent,tokens:usize)->Result<()>{guard_within(s,tokens,s.budget)}
/// Largest `T` whose projected cost fits `budget`, capped by the policy bound.
///
/// Both divisions floor, so the advertised length is never one the guard would refuse.
/// Enforcement stays in `guard_within`: a cost model the owner corrects after measuring
/// raises the usable query length without touching this function.
fn max_tokens_within(s:&Persistent,budget:u64)->u32{
    if s.cost_per_token==0 {return 0;}
    let net=budget.saturating_mul(1000)/BUDGET_MARGIN_PERMILLE;
    let tokens=net.saturating_sub(s.cost_fixed)/s.cost_per_token;
    (tokens.min(u32::MAX as u64) as u32).min(s.max_input_tokens)
}
thread_local!{static STATE:RefCell<Option<Persistent>>=const{RefCell::new(None)};}
thread_local!{
    static TOKENIZER:RefCell<Option<hf_tokenizer::HfTokenizer>>=const{RefCell::new(None)};
    static BUILDER:RefCell<Option<verdict_candle::pack::Builder>>=const{RefCell::new(None)};
    static MODEL:RefCell<Option<verdict_candle::VerdictModel>>=const{RefCell::new(None)};
}

fn read<R>(f:impl FnOnce(&Persistent)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
// Update messages commit the heap atomically; only upgrades need a stable snapshot.
// This canister has no awaits. pre_upgrade serializes the bounded registries/cache.
fn mutate<R>(f:impl FnOnce(&mut Persistent)->R)->R{STATE.with(|x|{let mut state=x.borrow_mut();let state=state.as_mut().expect("initialized");f(state)})}
fn owner()->Result<()>{read(|s|if ic_cdk::api::msg_caller()==s.owner{Ok(())}else{Err(Error::Unauthorized)})}
fn admitted(caller:Principal)->Result<()>{
    read(|s|if caller==s.owner || (caller!=Principal::anonymous() && s.callers.contains_key(&caller)){Ok(())}else{Err(Error::Unauthorized)})
}

#[ic_cdk::init]
fn init(owner:Principal){
    if owner==Principal::anonymous() || owner==Principal::management_canister(){ic_cdk::trap("invalid owner");}
    let s=Persistent{owner,active_model:[0;32],upload:None,callers:BTreeMap::new(),workflow:EngineState::new([0;32]),max_input_tokens:MAX_INPUT_TOKENS,
        cost_fixed:COST_FIXED,cost_per_token:COST_PER_TOKEN,budget:UPDATE_BUDGET};
    save_snapshot(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(save_snapshot);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){
    // The heap model is never kept across an upgrade: the pack bytes and upload
    // metadata survive in stable memory, the Candle tensors do not.
    let model=canister_common::restore::<ModelSnapshot>();
    let latest=canister_common::restore::<GameSnapshot>();
    let current=canister_common::restore::<BillingSnapshot>();
    let s:Persistent=if let Ok(snapshot)=model {
        if snapshot.version!=4 {ic_cdk::trap("unsupported model snapshot");}
        if let Some(pricing)=snapshot.execution_pricing {pricing.validate().unwrap_or_else(|e|ic_cdk::trap(e.to_string()));}
        billing::set(snapshot.execution_pricing);
        snapshot.state
    } else if let Ok(snapshot)=latest {
        if snapshot.version!=3 {ic_cdk::trap("unsupported game snapshot");}
        if let Some(pricing)=snapshot.execution_pricing {pricing.validate().unwrap_or_else(|e|ic_cdk::trap(e.to_string()));}
        billing::set(snapshot.execution_pricing);
        snapshot.state
    } else if let Ok(snapshot)=current {
        if snapshot.version!=2 {ic_cdk::trap("unsupported billing snapshot");}
        if let Some(pricing)=snapshot.execution_pricing {pricing.validate().unwrap_or_else(|e|ic_cdk::trap(e.to_string()));}
        billing::set(snapshot.execution_pricing);
        snapshot.state
    } else if let Ok(snapshot)=canister_common::restore::<DemoSnapshot>() {
        if snapshot.version!=1 {ic_cdk::trap("unsupported demo snapshot");}
        snapshot.state
    } else {canister_common::restore().or_else(|_|->Result<Persistent>{
        let old:LegacyPersistent=canister_common::restore()?;let mut workflow=EngineState::new(old.active_model);
        for caller in old.callers.keys().copied(){workflow.allow_caller(caller,1000)?;}
        Ok(Persistent{owner:old.owner,active_model:old.active_model,upload:old.upload,callers:old.callers,workflow,
            max_input_tokens:old.max_input_tokens,cost_fixed:old.cost_fixed,cost_per_token:old.cost_per_token,budget:old.budget})
    }).unwrap_or_else(|e|ic_cdk::trap(e.to_string()))};
    MODEL.with(|x|*x.borrow_mut()=None);BUILDER.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=None);
    STATE.with(|x|*x.borrow_mut()=Some(s));
}

#[derive(CandidType,Serialize,Deserialize)]
pub struct EngineInfo{
    pub model:Digest,
    pub pack_format:String,
    pub model_bytes:u64,
    pub tensors:u64,
    pub warmed:bool,
    pub upload_complete:bool,
    pub max_input_tokens:u32,
    pub max_classes:u32,
    pub callers:u64,
    pub heap_bytes:u64,
    pub cost_fixed:u64,
    pub cost_per_token:u64,
    pub budget:u64,
}
#[ic_cdk::query]
fn info()->EngineInfo{
    read(|s|{
        let (tensors,warmed)=MODEL.with(|m|BUILDER.with(|b|{
            let n=b.borrow().as_ref().map(|x|x.manifest.tensors.len() as u64).unwrap_or(0);
            (n,m.borrow().is_some())
        }));
        EngineInfo{
            model:s.active_model,pack_format:verdict_candle::pack::FORMAT.into(),model_bytes:s.upload.as_ref().map(|u|u.model_length).unwrap_or(0),tensors,warmed,
            upload_complete:s.upload.as_ref().map(|u|u.received==u.model_length+u.tokenizer_length).unwrap_or(false),
            max_input_tokens:s.max_input_tokens,max_classes:verdict_candle::MAX_CLASSES as u32,
            callers:s.callers.len() as u64,heap_bytes:heap_bytes(),
            cost_fixed:s.cost_fixed,cost_per_token:s.cost_per_token,budget:s.budget,
        }
    })
}
#[ic_cdk::update]
fn allow_caller(caller:Principal)->Result<()>{
    owner()?;
    if caller==Principal::anonymous() || caller==Principal::management_canister(){return Err(Error::Invalid("caller".into()));}
    mutate(|s|{if !s.callers.contains_key(&caller) && s.callers.len()>=64{return Err(Error::Capacity);}s.callers.insert(caller,ic_cdk::api::time() as u32);s.workflow.allow_caller(caller,1000)?;Ok(())})
}
#[ic_cdk::update]
fn set_caller_quota(caller:Principal,per_minute:u32)->Result<()>{owner()?;mutate(|s|s.workflow.allow_caller(caller,per_minute))}

/// Configure the subnet's execution tariff; None suspends paid inference.
#[ic_cdk::update]
fn set_execution_pricing(pricing:Option<ExecutionPricing>)->Result<()>{
    owner()?;
    if let Some(p)=pricing {p.validate()?;p.fee(UPDATE_BUDGET)?;}
    billing::set(pricing);Ok(())
}
#[ic_cdk::query]
fn cycles_pricing()->Result<CyclesPricing>{billing::quote()}

#[ic_cdk::update]
fn register_schema(schema_value:ic_laya_core::Schema,qtype_id:u32)->Result<ic_laya_core::CompiledSchema>{
    owner()?;
    let compiled=TOKENIZER.with(|t|{
        let t=t.borrow();let t=t.as_ref().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
        schema::compile(schema_value,t,qtype_id)
    })?;
    mutate(|s|s.workflow.register(compiled.clone()))?;Ok(compiled)
}
#[ic_cdk::update]
fn register_calibration(calibration:ic_laya_core::Calibration)->Result<()>{owner()?;mutate(|s|s.workflow.register_calibration(calibration,ic_cdk::api::time()))}
#[ic_cdk::update]
fn set_max_input_tokens(n:u32)->Result<u32>{
    owner()?;
    if n==0 || n>1024{return Err(Error::Invalid("max_input_tokens".into()));}
    mutate(|s|{s.max_input_tokens=n;Ok(n)})
}

/// Heap as reported by the Wasm memory size.
///
/// `memory_size(0)` counts 64 KiB pages of the single wasm32 linear memory, which
/// is stack + heap + static data. It is an upper bound on heap, and it is the only
/// reading a canister can take of itself: there is no API for "heap used".
#[ic_cdk::query]
fn heap_bytes()->u64{
    #[cfg(target_arch="wasm32")]
    { (core::arch::wasm32::memory_size(0) as u64)*65536 }
    #[cfg(not(target_arch="wasm32"))]
    { 0 }
}
/// Projected cost of one forward pass, with the margin the guard applies.
#[ic_cdk::query]
fn estimated_cost(tokens:u32)->u64{read(|s|projected(s,tokens as usize))}
/// What the query path will accept, so a client never hardcodes the derived ceiling.
///
/// `max_tokens` is the answer to "how long an input can this canister score in a query
/// call"; it moves when the owner corrects the cost model with `set_cost_model`. The
/// cost model itself is returned alongside it: a recorded measurement is only readable
/// later if it says which model produced the ceiling, and the same ceiling means
/// different things for an average fit and an upper-bound fit.
///
/// **The ceiling is exact for the installed model, not for the hardware.** The guard
/// compares a linear projection against the budget, so it refuses before the replica
/// does exactly when the installed `cost_fixed`/`cost_per_token` upper-bound the inputs
/// in question. For F32 that holds with the default 0.5% margin (measured spread 0.3%);
/// for a data-dependent kernel such as int8 the measured cost varies by ~20%, so the
/// owner must install a fit from the *worst observed* cost, not the average one.
#[derive(CandidType,Serialize,Deserialize)]
pub struct QueryLimits{
    pub budget:u64,
    pub margin_permille:u64,
    pub max_tokens:u32,
    pub max_input_tokens:u32,
    pub cost_fixed:u64,
    pub cost_per_token:u64,
}
#[ic_cdk::query]
fn query_limits()->QueryLimits{
    read(|s|QueryLimits{budget:QUERY_BUDGET,margin_permille:BUDGET_MARGIN_PERMILLE,
        max_tokens:max_tokens_within(s,QUERY_BUDGET),max_input_tokens:s.max_input_tokens,
        cost_fixed:s.cost_fixed,cost_per_token:s.cost_per_token})
}
#[ic_cdk::update]
fn set_cost_model(cost_fixed:u64,cost_per_token:u64,budget:u64)->Result<()>{
    owner()?;
    if cost_fixed==0 || cost_per_token==0 || budget==0 || budget>UPDATE_BUDGET {return Err(Error::Invalid("cost model".into()));}
    mutate(|s|{s.cost_fixed=cost_fixed;s.cost_per_token=cost_per_token;s.budget=budget;Ok(())})
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BenchReply{pub m:u32,pub n:u32,pub k:u32,pub iterations:u32,
    pub macs:u64,pub instructions:u64,pub per_iteration:u64,pub instructions_per_mac:f64}
fn bench_matrix(rows:usize,cols:usize)->Result<Tensor>{
    let n=rows.checked_mul(cols).ok_or(Error::TooLong)?;
    let data:Vec<f32>=(0..n).map(|i|(((i%251) as f32)-125.0)*0.0007).collect();
    Tensor::from_vec(data,(rows,cols),&Device::Cpu).map_err(|e|Error::ModelUnavailable(e.to_string()))
}
/// MEASUREMENT ONLY. Times one `matmul` shape inside the canister.
///
/// This exists to separate "the gemm kernel is slow" from "the code around it is
/// slow": the phase profile says 91% of a real forward pass is four dense matmuls
/// and gemm runs them at 2.501 instructions/MAC. The quantised kernel in
/// `crates/verdict-simd` reaches 0.780 (docs/VERDICT_ENGINE.md 5.1). Owner-only, never
/// reachable from an inference path, no state written.
#[ic_cdk::update]
fn bench_matmul(m:u32,n:u32,k:u32,iterations:u32)->Result<BenchReply>{
    owner()?;
    if m==0||n==0||k==0||m>512||n>4096||k>4096||iterations==0||iterations>64{return Err(Error::Invalid("bench shape".into()));}
    let a=bench_matrix(m as usize,k as usize)?;
    let b=bench_matrix(k as usize,n as usize)?;
    // Warm-up outside the measurement: allocation and lazy init are not the cost we
    // are trying to attribute.
    let _=a.matmul(&b).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let before=ic_cdk::api::instruction_counter();
    for _ in 0..iterations {
        let y=a.matmul(&b).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if y.dims()!=&[m as usize,n as usize][..]{return Err(Error::Numeric);}
    }
    let instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let macs=(m as u64)*(n as u64)*(k as u64)*(iterations as u64);
    Ok(BenchReply{m,n,k,iterations,macs,instructions,per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0{instructions as f64/macs as f64}else{0.0}})
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct QuantBenchReply{pub m:u32,pub n:u32,pub k:u32,pub iterations:u32,pub dtype:String,
    pub quantize_instructions:u64,pub instructions:u64,pub per_iteration:u64,
    pub instructions_per_mac:f64,pub max_abs_diff_vs_f32:f32}
/// MEASUREMENT ONLY. Times `candle`'s quantized matmul on a shape the model uses.
///
/// Why this exists: a hand-written f32x4 kernel could not be made to beat gemm (five
/// attempts, 12.5 instructions/MAC against gemm's 2.65 — docs/VERDICT_ENGINE.md
/// 5.1.3). `candle` already ships wasm SIMD quantized kernels
/// (`quantized/simd128.rs`) that `k_quants.rs` selects under
/// `#[cfg(target_feature = "simd128")]`, so quantization needs no new kernel at all.
/// This endpoint measures whether that path is actually cheaper, and how far the
/// result drifts from the f32 product on the same data.
#[ic_cdk::update]
fn bench_qmatmul(m:u32,n:u32,k:u32,iterations:u32,dtype:String)->Result<QuantBenchReply>{
    owner()?;
    if m==0||n==0||k==0||m>512||n>4096||k>4096||iterations==0||iterations>64{return Err(Error::Invalid("bench shape".into()));}
    let ggml=match dtype.as_str(){
        "q8_0"=>GgmlDType::Q8_0,"q4_0"=>GgmlDType::Q4_0,"q4_1"=>GgmlDType::Q4_1,
        "q5_0"=>GgmlDType::Q5_0,"q5_1"=>GgmlDType::Q5_1,"f16"=>GgmlDType::F16,
        _=>return Err(Error::Invalid("dtype".into())),
    };
    let (m,n,k)=(m as usize,n as usize,k as usize);
    let w=bench_matrix(n,k)?;
    let x=bench_matrix(m,k)?;
    let before=ic_cdk::api::instruction_counter();
    let qt=QTensor::quantize(&w,ggml).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let quantize_instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let qm=QMatMul::from_qtensor(qt).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let wt=w.t().and_then(|t|t.contiguous()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let reference=x.matmul(&wt).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let _=qm.forward(&x).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let before=ic_cdk::api::instruction_counter();
    for _ in 0..iterations {
        let y=qm.forward(&x).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if y.dims()!=&[m,n][..]{return Err(Error::Numeric);}
    }
    let instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let y=qm.forward(&x).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let got=y.flatten_all().and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let want=reference.flatten_all().and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let max_abs_diff_vs_f32=got.iter().zip(want.iter()).fold(0.0f32,|acc,(g,w)|acc.max((g-w).abs()));
    let macs=(m*n*k*iterations as usize) as f64;
    Ok(QuantBenchReply{m:m as u32,n:n as u32,k:k as u32,iterations,dtype,
        quantize_instructions,instructions,per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0.0{instructions as f64/macs}else{0.0},max_abs_diff_vs_f32})
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct Int8BenchReply{pub m:u32,pub n:u32,pub k:u32,pub iterations:u32,
    pub simd_used:bool,pub instructions:u64,pub per_iteration:u64,pub instructions_per_mac:f64,
    pub quantize_weights_instructions:u64,pub quantize_activations_instructions:u64,
    pub max_abs_diff_vs_f32:f32,pub max_rel_diff_vs_f32:f32}
/// MEASUREMENT ONLY. Times the int8 kernel against the f32 product on the same data.
///
/// The weight matrix is quantised once (that is a load-time cost in production) and the
/// activations once per call (that is a real per-call cost), and both are reported
/// separately so the numbers can be read against the f32 path's 2.5 instructions/MAC.
#[ic_cdk::update]
fn bench_int8(m:u32,n:u32,k:u32,iterations:u32)->Result<Int8BenchReply>{
    owner()?;
    if m==0||n==0||k==0||m>512||n>4096||k>4096||iterations==0||iterations>64||!k.is_multiple_of(8){return Err(Error::Invalid("bench shape".into()));}
    let (m,n,k)=(m as usize,n as usize,k as usize);
    // Weight as [n, k] (the checkpoint layout) and activations as [m, k].
    let w=bench_matrix(n,k)?.flatten_all().and_then(|x|x.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let x=bench_matrix(m,k)?.flatten_all().and_then(|x|x.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let before=ic_cdk::api::instruction_counter();
    let (wq,wsx)=verdict_simd::quantize_rows_i8(&w,n,k);
    let quantize_weights_instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let before=ic_cdk::api::instruction_counter();
    let (xq,xsx)=verdict_simd::quantize_acts_i16(&x,m,k);
    let quantize_activations_instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let mut out=vec![0f32;m*n];
    let simd_used=verdict_simd::matmul_i8(&xq,&wq,&xsx,&wsx,m,k,n,&mut out);
    // Check the actual Wasm SIMD path against the quantized scalar reference,
    // outside the timing region. F32 error alone cannot validate a new kernel.
    let mut exact=vec![0f32;m*n];
    verdict_simd::matmul_i8_scalar(&xq,&wq,&xsx,&wsx,m,k,n,&mut exact);
    if out.iter().zip(&exact).any(|(a,b)|a.to_bits()!=b.to_bits()){return Err(Error::Numeric);}
    drop(exact);
    // Reference: f32 product with the same layout ([n,k] weights transposed conceptually).
    let mut want=vec![0f32;m*n];
    for i in 0..m { for j in 0..n {
        let mut acc=0f32;
        for p in 0..k { acc+=x[i*k+p]*w[j*k+p]; }
        want[i*n+j]=acc;
    }}
    let (mut max_abs,mut max_rel)=(0f32,0f32);
    for (g,wv) in out.iter().zip(want.iter()) {
        max_abs=max_abs.max((g-wv).abs());
        let d=if wv.abs()>1e-3 {(g-wv).abs()/wv.abs()}else{0.0};
        max_rel=max_rel.max(d);
    }
    let before=ic_cdk::api::instruction_counter();
    for _ in 0..iterations { let _=verdict_simd::matmul_i8(&xq,&wq,&xsx,&wsx,m,k,n,&mut out); }
    let instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let macs=(m*n*k*iterations as usize) as f64;
    Ok(Int8BenchReply{m:m as u32,n:n as u32,k:k as u32,iterations,simd_used,instructions,
        per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0.0{instructions as f64/macs}else{0.0},
        quantize_weights_instructions,quantize_activations_instructions,max_abs_diff_vs_f32:max_abs,max_rel_diff_vs_f32:max_rel})
}

/// Owner-only block32 benchmark. tile=0 measures the production kernel; 2/4/8
/// measure candidates without changing the inference path or its cost model.
#[ic_cdk::update]
fn bench_int8_block32(m:u32,n:u32,k:u32,iterations:u32,tile:u32)->Result<BenchReply>{
    owner()?;
    if m==0||n==0||k==0||m>128||n>4096||k>4096||!k.is_multiple_of(32)
        ||iterations==0||iterations>4||![0,2,4,8].contains(&tile){return Err(Error::Invalid("bench shape/tile".into()));}
    let (m,n,k)=(m as usize,n as usize,k as usize);
    let w=bench_matrix(n,k)?.flatten_all().and_then(|x|x.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let x=bench_matrix(m,k)?.flatten_all().and_then(|x|x.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let(wq,ws)=verdict_simd::quantize_blocks_i8(&w,n,k,32);
    let(aq,asx)=verdict_simd::quantize_acts_i16(&x,m,k);
    let mut want=vec![0f32;m*n];let _=verdict_simd::matmul_i8_blocked(&aq,&wq,&asx,&ws,m,k,n,32,&mut want);
    let mut out=vec![0f32;m*n];
    let run=|out:&mut[f32]|if tile==0{verdict_simd::matmul_i8_blocked(&aq,&wq,&asx,&ws,m,k,n,32,out)}
        else{verdict_simd::matmul_i8_block32_candidate(&aq,&wq,&asx,&ws,m,k,n,out,tile as usize)};
    let _=run(&mut out);
    if out!=want{return Err(Error::Numeric);}
    let before=ic_cdk::api::instruction_counter();for _ in 0..iterations{let _=run(&mut out);}
    let instructions=ic_cdk::api::instruction_counter()-before;
    let macs=(m*n*k*iterations as usize) as u64;
    Ok(BenchReply{m:m as u32,n:n as u32,k:k as u32,iterations,macs,instructions,per_iteration:instructions/iterations as u64,
        instructions_per_mac:instructions as f64/macs as f64})
}

/// Measure only the QK/softmax/AV core. Candidate differences are checked against
/// the dense masked reference; no candidate can affect normal inference.
#[ic_cdk::update]
fn bench_local_attention(tokens:u32,distance:u32,candidate:bool)->Result<BenchReply>{
    owner()?;
    if tokens==0||tokens>128||distance>128{return Err(Error::TooLong);}
    let (t,d,h)=(tokens as usize,64usize,12usize);
    let q=bench_matrix(h*t,d)?.reshape((h,t,d)).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let flat=q.flatten_all().and_then(|x|x.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let mask:Vec<f32>=(0..t).flat_map(|i|(0..t).map(move|j|if i.abs_diff(j)>distance as usize{f32::NEG_INFINITY}else{0.})).collect();
    let mask=Tensor::from_vec(mask,(1,t,t),&Device::Cpu).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let dense=||->candle_core::Result<Vec<f32>>{
        let scores=(q.matmul(&q.transpose(1,2)?.contiguous()?)?*(1.0/(d as f64).sqrt()))?.broadcast_add(&mask)?;
        let mut scores=scores.flatten_all()?.to_vec1::<f32>()?;
        verdict_simd::softmax_rows_inplace(&mut scores,h*t,t);
        Tensor::from_vec(scores,(h,t,t),&Device::Cpu)?.matmul(&q)?.flatten_all()?.to_vec1::<f32>()
    };
    let want=dense().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let run=||->Result<Vec<f32>>{if candidate{Ok(verdict_simd::local_attention_candidate(&flat,&flat,&flat,h,t,d,distance as usize))}
        else{dense().map_err(|e|Error::ModelUnavailable(e.to_string()))}};
    let got=run()?;
    if got.iter().zip(&want).any(|(a,b)|!a.is_finite()||(a-b).abs()>1e-4){return Err(Error::Numeric);}
    let before=ic_cdk::api::instruction_counter();let got=run()?;
    let instructions=ic_cdk::api::instruction_counter()-before;
    std::hint::black_box(got);
    Ok(BenchReply{m:tokens,n:distance,k:d as u32,iterations:1,macs:(2*h*t*t*d) as u64,instructions,per_iteration:instructions,
        instructions_per_mac:instructions as f64/(2*h*t*t*d) as f64})
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct F16BenchReply{pub m:u32,pub n:u32,pub k:u32,pub iterations:u32,
    pub convert_instructions:u64,pub instructions:u64,pub per_iteration:u64,pub instructions_per_mac:f64,
    pub max_abs_diff_vs_f32:f32}
/// MEASUREMENT ONLY. Times an f16 matmul with `candle`'s own kernels against the f32
/// product. Wasm has no f16 arithmetic, so this is expected to lose on instructions and
/// only win on weight size; the point is to record the measurement rather than assume.
#[ic_cdk::update]
fn bench_f16(m:u32,n:u32,k:u32,iterations:u32)->Result<F16BenchReply>{
    owner()?;
    if m==0||n==0||k==0||m>512||n>4096||k>4096||iterations==0||iterations>64{return Err(Error::Invalid("bench shape".into()));}
    let (m,n,k)=(m as usize,n as usize,k as usize);
    let w=bench_matrix(n,k)?;
    let x=bench_matrix(m,k)?;
    let before=ic_cdk::api::instruction_counter();
    let w16=w.to_dtype(candle_core::DType::F16).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let x16=x.to_dtype(candle_core::DType::F16).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let convert_instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let reference=x.matmul(&w.t().and_then(|t|t.contiguous()).map_err(|e|Error::ModelUnavailable(e.to_string()))?).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let _=x16.matmul(&w16.t().and_then(|t|t.contiguous()).map_err(|e|Error::ModelUnavailable(e.to_string()))?).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let before=ic_cdk::api::instruction_counter();
    for _ in 0..iterations {
        let y=x16.matmul(&w16.t().and_then(|t|t.contiguous()).map_err(|e|Error::ModelUnavailable(e.to_string()))?).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if y.dims()!=&[m,n][..]{return Err(Error::Numeric);}
    }
    let instructions=ic_cdk::api::instruction_counter().saturating_sub(before);
    let y=x16.matmul(&w16.t().and_then(|t|t.contiguous()).map_err(|e|Error::ModelUnavailable(e.to_string()))?).map_err(|e|Error::ModelUnavailable(e.to_string()))?
        .to_dtype(candle_core::DType::F32).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let got=y.flatten_all().and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let want=reference.flatten_all().and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let max_abs_diff_vs_f32=got.iter().zip(want.iter()).fold(0.0f32,|a,(g,w)|a.max((g-w).abs()));
    let macs=(m*n*k*iterations as usize) as f64;
    Ok(F16BenchReply{m:m as u32,n:n as u32,k:k as u32,iterations,convert_instructions,instructions,
        per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0.0{instructions as f64/macs}else{0.0},max_abs_diff_vs_f32})
}

/// Start a pack upload: validate the manifest, bind the CLS id and reset the builder.
/// The blob is written by `upload_chunk` and consumed by `warmup_next`.
#[ic_cdk::update]
fn begin_upload(manifest:Vec<u8>,tokenizer_length:u64,special:SpecialTokens)->Result<Digest>{
    owner()?;
    let m=verdict_candle::pack::Manifest::parse(&manifest)?;
    if m.config.cls_token_id!=special.cls {return Err(Error::BindingMismatch);}
    if tokenizer_length==0 || tokenizer_length>32*1024*1024{return Err(Error::TooLong);}
    let bundle=hash(&manifest);
    MODEL.with(|x|*x.borrow_mut()=None);BUILDER.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=None);
    mutate(|s|{s.upload=Some(Upload{manifest,tokenizer_length,model_length:m.total_bytes,received:0,special});Ok(bundle)})
}
#[ic_cdk::update]
fn upload_chunk(offset:u64,bytes:Vec<u8>)->Result<u64>{
    owner()?;
    if bytes.is_empty() || bytes.len()>1024*1024{return Err(Error::TooLong);}
    mutate(|s|{
        let u=s.upload.as_mut().ok_or(Error::Transition)?;
        let end=offset.checked_add(bytes.len() as u64).ok_or(Error::TooLong)?;
        if end>u.model_length+u.tokenizer_length{return Err(Error::TooLong);}
        if offset<u.received {
            if end>u.received{return Err(Error::IdConflict);}
            let mut old=vec![0;bytes.len()];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+offset,&mut old);
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
    let u=read(|s|s.upload.clone()).ok_or(Error::Transition)?;
    if u.received!=u.model_length+u.tokenizer_length{return Err(Error::Transition);}
    let builder=verdict_candle::pack::Builder::new(&u.manifest)?;
    let mut bytes=vec![0;u.tokenizer_length as usize];
    ic_cdk::stable::stable_read(canister_common::BLOB_BASE+u.model_length,&mut bytes);
    if hash(&bytes)!=builder.manifest.tokenizer_sha256{return Err(Error::BindingMismatch);}
    let tok=hf_tokenizer::HfTokenizer::from_bytes(&bytes,u.special)?;
    let count=builder.manifest.tensors.len() as u64;
    MODEL.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=Some(tok));BUILDER.with(|x|*x.borrow_mut()=Some(builder));Ok(count)
}
#[ic_cdk::update]
fn warmup_next()->Result<bool>{
    owner()?;
    let done=BUILDER.with(|b|{
        let mut b=b.borrow_mut();let builder=b.as_mut().ok_or(Error::Transition)?;
        if let Some(e)=builder.next_entry().cloned(){
            let mut bytes=vec![0;e.length as usize];
            ic_cdk::stable::stable_read(canister_common::BLOB_BASE+e.offset,&mut bytes);
            builder.push(&bytes)?;
        }
        Ok::<bool,Error>(builder.next_entry().is_none())
    })?;
    if done {
        let builder=BUILDER.with(|b|b.borrow_mut().take()).ok_or(Error::Transition)?;
        let model=mutate(|s|finish_warmup(s,builder))?;
        MODEL.with(|m|*m.borrow_mut()=Some(model));
    }
    Ok(done)
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct InferReply{
    pub model:Digest,
    pub class_positions:Vec<u32>,
    pub logits:Vec<f32>,
    pub input_tokens:u32,
    pub measured_instructions:u64,
}
/// Shared body of `infer_tokens` (update) and `infer_tokens_query` (query).
///
/// Only the budget differs, so the two entry points cannot drift apart. `logits` takes
/// `&self`, which is what lets the query path score an input without mutating the
/// cached model.
fn infer_once(s:&Persistent,input_ids:Vec<u32>,budget:u64)->Result<InferReply>{
    if input_ids.is_empty() || input_ids.len()>s.max_input_tokens as usize {return Err(Error::TooLong);}
    guard_within(s,input_ids.len(),budget)?;
    MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
        let before=ic_cdk::api::instruction_counter();
        let logits=model.logits(&input_ids)?;
        let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
        Ok(InferReply{model:model.bundle_id(),class_positions:model.class_positions(&input_ids),logits,input_tokens:input_ids.len() as u32,measured_instructions:measured})
    })
}
/// Raw token ids in, one logit per `<<LABEL>>` token out.
///
/// The instruction count is measured around the forward pass only, so it excludes
/// Candid decoding of the arguments and the reply encoding. `measured_instructions`
/// is the number the 40B update-call limit applies to.
#[ic_cdk::update(manual_reply = true)]
fn infer_tokens(input_ids:Vec<u32>)->PhantomData<Result<InferReply>>{
    billing::paid(||read(|s|infer_once(s,input_ids,s.budget)))
}
/// `infer_tokens` over a query call: the same forward pass under the 5B query budget.
///
/// What it buys: no consensus round, no cycles, lower latency, and a 3 MiB reply cap
/// instead of 2 MiB. What it costs: the ceiling falls to roughly fourteen tokens
/// (`query_limits()` reports the derived value), the reply is **not certified**, and
/// the same caller allowlist is enforced by code the caller cannot verify. Use it for
/// interactive short-input scoring, never as the authority for moving funds.
///
/// The model must already be warm: a query cannot load the pack, because `stable_write`
/// is illegal in a query, and loading it would exceed the 5B limit anyway.
#[ic_cdk::query]
fn infer_tokens_query(input_ids:Vec<u32>)->Result<InferReply>{
    billing::query_only()?;
    let caller=ic_cdk::api::msg_caller();admitted(caller)?;
    read(|s|infer_once(s,input_ids,QUERY_BUDGET))
}

/// Workflow-compatible, durable one-question evaluation using the resident checkpoint.
#[ic_cdk::update(manual_reply = true)]
fn evaluate(req:ic_laya_core::DecisionRequest)->PhantomData<Result<ic_laya_core::Receipt>>{
    billing::paid(||evaluate_once(req))
}
fn evaluate_once(req:ic_laya_core::DecisionRequest)->Result<ic_laya_core::Receipt>{
    let caller=ic_cdk::api::msg_caller();let now=ic_cdk::api::time();admitted(caller)?;
    let (max_input_tokens,cost_fixed,cost_per_token,budget)=read(|s|(s.max_input_tokens,s.cost_fixed,s.cost_per_token,s.budget));
    mutate(|s|s.workflow.evaluate_with(caller,req,now,|compiled,_,state|{
        let labels:Vec<String>=compiled.schema.options.iter().map(|o|o.text.clone()).collect();
        let special=TOKENIZER.with(|t|t.borrow().as_ref().map(|t|t.special_tokens()))
            .ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
        let prompt=verdict_candle::render_prompt_checked(&compiled.schema.instructions,state,&labels,&special)?;
        TOKENIZER.with(|t|MODEL.with(|m|{
            let t=t.borrow();let m=m.borrow();
            let (t,model)=match(t.as_ref(),m.as_ref()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("warm-up required".into()))};
            let mut input=vec![model.config.cls_token_id];input.extend(t.encode_piece(&prompt)?);input.push(model.config.sep_token_id);
            if input.len()>max_input_tokens as usize||input.len()>verdict_candle::MAX_SEQUENCE{return Err(Error::TooLong);}
            guard_cost(cost_fixed,cost_per_token,input.len(),budget)?;
            let before=ic_cdk::api::instruction_counter();let logits=model.logits(&input)?;
            let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
            Ok((logits,input.len() as u32,model.kind(),measured))
        }))
    }))
}

/// Wire form of `verdict_candle::PhaseCost`. Kept local so the canister's Candid
/// surface does not depend on how the model crate serialises its internals.
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct PhaseCostDto{pub name:String,pub instructions:u64}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct ProfileReply{
    pub model:Digest,
    pub input_tokens:u32,
    pub measured_instructions:u64,
    /// Execution order. `instructions` is the delta since the previous boundary, so
    /// the entries sum to `measured_instructions`.
    pub phases:Vec<PhaseCostDto>,
}
/// `infer_tokens` with a per-phase instruction breakdown, for measurement only.
///
/// `detailed` marks every encoder sub-phase, which multiplies the entry count by the
/// layer count; callers aggregate by name. Owner-only: it exists to answer where the
/// instructions go, not to serve decisions, and it returns strictly more than
/// `infer_tokens` does.
#[ic_cdk::update]
fn infer_profiled(input_ids:Vec<u32>,detailed:bool)->Result<ProfileReply>{
    owner()?;
    let limit=read(|s|s.max_input_tokens);
    if input_ids.is_empty() || input_ids.len()>limit as usize {return Err(Error::TooLong);}
    MODEL.with(|m|{
        let mut m=m.borrow_mut();let model=m.as_mut().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
        let mut phases:Vec<PhaseCostDto>=Vec::new();
        let start=ic_cdk::api::instruction_counter();
        let mut last=start;
        let logits={
            let mut record=|name:&'static str|{
                let now=ic_cdk::api::instruction_counter();
                phases.push(PhaseCostDto{name:name.to_string(),instructions:now.saturating_sub(last)});
                last=now;
            };
            model.logits_profiled(&input_ids,&mut record,detailed)?
        };
        let measured=last.saturating_sub(start);
        if logits.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}
        Ok(ProfileReply{model:model.bundle_id(),input_tokens:input_ids.len() as u32,
            measured_instructions:measured,phases})
    })
}

#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct OptionSpec{pub id:String,pub text:String}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct DecideRequest{
    pub state:String,
    pub question:String,
    pub options:Vec<OptionSpec>,
    /// Append the checkpoint's own abstention slot after the given options.
    pub abstention:bool,
    /// Divided into the logits before the softmax. Use the temperature recorded in
    /// the calibration artifact; 1.0 returns the raw model distribution.
    pub temperature:f64,
}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct DecideReply{
    pub model:Digest,
    pub ids:Vec<String>,
    pub logits:Vec<f32>,
    pub probabilities:Vec<f32>,
    pub selected:String,
    pub confidence:f32,
    pub input_tokens:u32,
    pub measured_instructions:u64,
}
fn guard_public_query_text(req:&DecideRequest)->Result<()> {
    if req.question.len()>MAX_PUBLIC_QUERY_QUESTION_BYTES {return Err(Error::TooLong);}
    let mut total=req.state.len().saturating_add(req.question.len());
    if total>MAX_PUBLIC_QUERY_TEXT_BYTES {return Err(Error::TooLong);}
    for option in &req.options {
        if option.id.len()>MAX_PUBLIC_QUERY_ID_BYTES || option.text.len()>MAX_PUBLIC_QUERY_OPTION_BYTES {
            return Err(Error::TooLong);
        }
        total=total.saturating_add(option.id.len()).saturating_add(option.text.len());
        if total>MAX_PUBLIC_QUERY_TEXT_BYTES {return Err(Error::TooLong);}
    }
    Ok(())
}
/// Shared body of `decide` (update) and `decide_query` (query).
///
/// Every prompt, policy and shape check lives here, so the two entry points cannot end
/// up enforcing different rules. Only the budget differs.
fn decide_once(req:DecideRequest,budget:u64)->Result<DecideReply>{
    if req.state.is_empty() || req.state.len()>ic_laya_core::MAX_STATE_BYTES {return Err(Error::TooLong);}
    if req.options.is_empty() || req.options.len()>MAX_OPTIONS {return Err(Error::TooLong);}
    if !req.temperature.is_finite() || req.temperature<=0.0 || req.temperature>100.0 {return Err(Error::Numeric);}
    for o in &req.options {if o.id.is_empty() || o.text.is_empty(){return Err(Error::Invalid("option".into()));}}
    let limits=read(|s|s.max_input_tokens);
    let mut ids:Vec<String>=req.options.iter().map(|o|o.id.clone()).collect();
    let mut labels:Vec<String>=req.options.iter().map(|o|o.text.clone()).collect();
    if req.abstention {ids.push("__insufficient_evidence__".into());labels.push(ABSTENTION_DESC.into());}
    if ids.len()>verdict_candle::MAX_CLASSES{return Err(Error::TooLong);}
    {let mut seen=std::collections::BTreeSet::new();for id in &ids{if !seen.insert(id.clone()){return Err(Error::Invalid("duplicate option id".into()));}}}
    // The tokenizer's own added tokens include `<<LABEL>>`/`<<SEP>>`, and it emits the
    // special id wherever that literal appears. An injected state or option text would
    // therefore add a class slot with no id (measured: 2 ids, 3 logits), and the argmax
    // could index past the id list. `ic-laya-core`'s schema path rejects these literals.
    let special=TOKENIZER.with(|t|t.borrow().as_ref().map(|t|t.special_tokens()))
        .ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
    let prompt=verdict_candle::render_prompt_checked(&req.question,&req.state,&labels,&special)?;
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let m=m.borrow();
        let (t,model)=match (t.as_ref(),m.as_ref()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("warm-up required".into()))};
        let mut input=vec![model.config.cls_token_id];
        input.extend(t.encode_piece(&prompt)?);
        input.push(model.config.sep_token_id);
        if input.len()>limits as usize || input.len()>verdict_candle::MAX_SEQUENCE{return Err(Error::TooLong);}
        read(|s|guard_within(s,input.len(),budget))?;
        let before=ic_cdk::api::instruction_counter();
        let logits=model.logits(&input)?;
        if logits.len()!=ids.len(){return Err(Error::BindingMismatch);}
        let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
        let scaled:Vec<f32>=logits.iter().map(|x|(*x as f64/req.temperature) as f32).collect();
        let probs=softmax(&scaled)?;
        let best=probs.iter().enumerate().fold((0usize,f32::NEG_INFINITY),|a,(i,p)|if *p>a.1{(i,*p)}else{a});
        Ok(DecideReply{
            model:model.bundle_id(),ids:ids.clone(),logits,probabilities:probs.clone(),
            selected:ids[best.0].clone(),confidence:best.1,input_tokens:input.len() as u32,measured_instructions:measured,
        })
    }))
}
/// One typed decision: state + question + options in, calibrated distribution out.
///
/// The prompt is the checkpoint's own contract
/// (`<<LABEL>>desc...<<SEP>>Question: ...\n\nContext:\n...`, wrapped as
/// `[CLS] ... [SEP]`), so the tokenizer and the head see exactly what the
/// reference engine produces.
#[ic_cdk::update(manual_reply = true)]
fn decide(req:DecideRequest)->PhantomData<Result<DecideReply>>{
    billing::paid(||decide_once(req,read(|s|s.budget)))
}
/// `decide` over a query call.
///
/// Public to anonymous callers. The same prompt validation and tokenizer apply under
/// the 5B query budget; requests beyond the installed cost guard return `Capacity`.
/// Replicated execution is refused, and callers can read `query_limits` first.
#[ic_cdk::query]
fn decide_query(req:DecideRequest)->Result<DecideReply>{
    billing::query_only()?;
    guard_public_query_text(&req)?;
    decide_once(req,QUERY_BUDGET)
}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchQuestion{pub id:String,pub question:String,pub options:Vec<OptionSpec>,pub abstention:bool}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchRequest{pub state:String,pub questions:Vec<BatchQuestion>,pub temperature:f64}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct QuestionResult{pub id:String,pub ids:Vec<String>,pub logits:Vec<f32>,pub probabilities:Vec<f32>,pub selected:String,pub confidence:f32}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchReply{pub model:Digest,pub questions:Vec<QuestionResult>,pub input_tokens:u32,pub measured_instructions:u64}
type BatchLayout=(Vec<String>,Vec<String>,Vec<usize>);
fn batch_layout(questions:&[BatchQuestion])->Result<BatchLayout>{
    let mut question_ids=std::collections::BTreeSet::new();
    let (mut labels,mut ids,mut counts)=(Vec::new(),Vec::new(),Vec::new());
    for q in questions {
        if q.id.trim().is_empty() || !question_ids.insert(&q.id){return Err(Error::Invalid("question id".into()));}
        let mut seen=std::collections::BTreeSet::new();
        if q.options.is_empty() || q.options.len()>MAX_OPTIONS{return Err(Error::TooLong);}
        let mut n=0usize;
        for o in &q.options {
            if o.id.is_empty()||o.text.is_empty(){return Err(Error::Invalid("option".into()));}
            if !seen.insert(o.id.as_str()){return Err(Error::Invalid("duplicate option id".into()));}
            labels.push(o.text.clone());ids.push(o.id.clone());n+=1;
        }
        if q.abstention {if !seen.insert("__insufficient_evidence__"){return Err(Error::Invalid("duplicate option id".into()));}labels.push(ABSTENTION_DESC.into());ids.push("__insufficient_evidence__".into());n+=1;}
        counts.push(n);
    }
    if ids.len()>verdict_candle::MAX_CLASSES{return Err(Error::TooLong);}
    Ok((labels,ids,counts))
}
fn batch_results(questions:&[BatchQuestion],ids:&[String],counts:&[usize],logits:&[f32],temperature:f64)->Result<Vec<QuestionResult>>{
        let mut out=Vec::new();let mut offset=0usize;
        for (q,count) in questions.iter().zip(counts.iter()) {
            let end=offset+count;
            let qlogits=logits[offset..end].to_vec();
            let scaled:Vec<f32>=qlogits.iter().map(|x|(*x as f64/temperature) as f32).collect();
            let qprobs=softmax(&scaled)?;
            let qids=ids[offset..end].to_vec();
            let best=qprobs.iter().enumerate().fold((0usize,f32::NEG_INFINITY),|a,(i,p)|if *p>a.1{(i,*p)}else{a});
            out.push(QuestionResult{id:q.id.clone(),ids:qids.clone(),logits:qlogits,probabilities:qprobs,
                selected:qids[best.0].clone(),confidence:best.1});
            offset=end;
        }
    Ok(out)
}
/// Several typed questions over one state, in a single forward pass.
///
/// Measured reason (docs/VERDICT_ENGINE.md 5.1.6): sending the same state three times
/// costs 63.1e9 instructions for three decisions; one pass with all labels costs
/// 34.2e9 (-45.8%), or 26.8e9 with short label text. The trade is that all labels are
/// scored against one `[CLS]` representation, so the per-question distributions shift
/// (argmax was unchanged in the sample, logits moved by up to 1.7) and a batched layout
/// needs its own calibration.
#[ic_cdk::update(manual_reply = true)]
fn decide_batch(req:BatchRequest)->PhantomData<Result<BatchReply>>{
    billing::paid(||decide_batch_once(req))
}
fn decide_batch_once(req:BatchRequest)->Result<BatchReply>{
    if req.state.is_empty() || req.state.len()>ic_laya_core::MAX_STATE_BYTES{return Err(Error::TooLong);}
    if req.questions.is_empty() || req.questions.len()>8{return Err(Error::TooLong);}
    if !req.temperature.is_finite()||req.temperature<=0.0||req.temperature>100.0{return Err(Error::Numeric);}
    let limits=read(|s|s.max_input_tokens);
    let (labels,ids,counts)=batch_layout(&req.questions)?;
    let question=req.questions.iter().map(|q|q.question.clone()).collect::<Vec<_>>().join(" | ");
    // The tokenizer's own added tokens include `<<LABEL>>`/`<<SEP>>`, and it emits the
    // special id wherever that literal appears. An injected state or option text would
    // therefore add a class slot with no id (measured: 2 ids, 3 logits), and the argmax
    // could index past the id list. `ic-laya-core`'s schema path rejects these literals.
    let special=TOKENIZER.with(|t|t.borrow().as_ref().map(|t|t.special_tokens()))
        .ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
    let prompt=verdict_candle::render_prompt_checked(&question,&req.state,&labels,&special)?;
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let mut m=m.borrow_mut();
        let (t,model)=match (t.as_ref(),m.as_mut()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("warm-up required".into()))};
        let mut input=vec![model.config.cls_token_id];
        input.extend(t.encode_piece(&prompt)?);
        input.push(model.config.sep_token_id);
        if input.len()>limits as usize || input.len()>verdict_candle::MAX_SEQUENCE{return Err(Error::TooLong);}
        read(|s|guard_budget(s,input.len()))?;
        let before=ic_cdk::api::instruction_counter();
        let logits=model.logits(&input)?;
        if logits.len()!=ids.len(){return Err(Error::BindingMismatch);}
        let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
        let out=batch_results(&req.questions,&ids,&counts,&logits,req.temperature)?;
        Ok(BatchReply{model:model.bundle_id(),questions:out,input_tokens:input.len() as u32,measured_instructions:measured})
    }))
}

/// Softmax over the option logits.
///
/// Returns an error instead of a fabricated distribution. The previous version returned
/// an all-zero vector when the sum was 0 or NaN, which `decide` then reported as "first
/// option selected, confidence 0.0": a wrong answer with no signal.
fn softmax(xs:&[f32])->Result<Vec<f32>>{
    let m=xs.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
    if !m.is_finite(){return Err(Error::Numeric);}
    let e:Vec<f32>=xs.iter().map(|x|(x-m).exp()).collect();
    let s:f32=e.iter().sum();
    if !s.is_finite()||s<=0.0{return Err(Error::Numeric);}
    Ok(e.iter().map(|x|x/s).collect())
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn public_query_rejects_oversized_text_before_tokenization(){
        let mut req=DecideRequest{state:"Min holes".into(),question:String::new(),
            options:vec![OptionSpec{id:"0".into(),text:"min holes".into()}],
            abstention:false,temperature:1.0};
        assert!(guard_public_query_text(&req).is_ok());
        req.question="q".repeat(MAX_PUBLIC_QUERY_QUESTION_BYTES+1);
        assert!(matches!(guard_public_query_text(&req),Err(Error::TooLong)));
        req.question.clear();req.options[0].text="x".repeat(MAX_PUBLIC_QUERY_OPTION_BYTES+1);
        assert!(matches!(guard_public_query_text(&req),Err(Error::TooLong)));
        req.options[0].text="min holes".into();req.options[0].id="i".repeat(MAX_PUBLIC_QUERY_ID_BYTES+1);
        assert!(matches!(guard_public_query_text(&req),Err(Error::TooLong)));
        req.options[0].id="0".into();req.state=" ".repeat(MAX_PUBLIC_QUERY_TEXT_BYTES);
        assert!(matches!(guard_public_query_text(&req),Err(Error::TooLong)));
    }
    fn question(id:&str)->BatchQuestion{
        BatchQuestion{id:id.into(),question:"choose".into(),options:vec![OptionSpec{id:"yes".into(),text:"yes".into()}],abstention:true}
    }
    #[test]
    fn batch_allows_shared_ids_but_normalizes_each_question(){
        let qs=vec![question("one"),question("two")];
        let (_,ids,counts)=batch_layout(&qs).unwrap();
        let got=batch_results(&qs,&ids,&counts,&[0.,0.,100.,100.],1.).unwrap();
        assert_eq!(got.len(),2);
        for q in got{assert_eq!(q.probabilities,vec![0.5,0.5]);assert_eq!(q.confidence,0.5);assert_eq!(q.selected,"yes");}
    }
    #[test]
    fn batch_rejects_ambiguous_ids(){
        assert!(batch_layout(&[question("" )]).is_err());
        assert!(batch_layout(&[question("one"),question("one")]).is_err());
        let mut q=question("one");q.options.push(q.options[0].clone());
        assert!(batch_layout(&[q]).is_err());
        let mut q=question("one");q.options[0].id="__insufficient_evidence__".into();
        assert!(batch_layout(&[q]).is_err());
    }
    #[test]
    fn rewarm_preserves_contracts_and_idempotency_even_after_snapshot_restore(){
        let mut s=state(COST_FIXED,COST_PER_TOKEN);
        let (mut executor,mut engine,op,grant)=ic_laya_core::demo::setup().unwrap();
        let id=executor.submit(ic_laya_core::demo::actor(2),0,op,grant,ic_laya_core::demo::NOW).unwrap();
        ic_laya_core::demo::complete(&mut executor,&mut engine,id).unwrap();
        s.active_model=engine.active_model;s.workflow=engine;
        let before=ic_laya_core::storage::encode(&s).unwrap();
        let mut restored:Persistent=ic_laya_core::storage::decode(&before).unwrap();
        let bundle=restored.active_model;activate_bundle(&mut restored,bundle);
        assert_eq!(before,ic_laya_core::storage::encode(&restored).unwrap());
        let callers=ic_laya_core::storage::encode(&restored.workflow.callers).unwrap();
        activate_bundle(&mut restored,[9;32]);
        assert!(restored.workflow.schemas.is_empty()&&restored.workflow.calibrations.is_empty()&&restored.workflow.cache.is_empty());
        assert_eq!(callers,ic_laya_core::storage::encode(&restored.workflow.callers).unwrap());
    }
    #[test]
    fn incomplete_warmup_does_not_reset_registered_state(){
        let mut s=state(COST_FIXED,COST_PER_TOKEN);
        s.workflow=ic_laya_core::demo::setup().unwrap().1;s.active_model=s.workflow.active_model;
        let before=ic_laya_core::storage::encode(&s).unwrap();
        let builder=verdict_candle::pack::Builder::new(include_bytes!("../../../fixtures/verdict-tiny/manifest.json")).unwrap();
        assert!(matches!(finish_warmup(&mut s,builder),Err(Error::Transition)));
        assert_eq!(before,ic_laya_core::storage::encode(&s).unwrap());
    }
    /// The budget guard is pure arithmetic, so it is testable off-wasm. Nothing here
    /// calls `ic0`: `instruction_counter` and the caller APIs are only read from the
    /// entry points, which these tests exercise through the guard functions.
    fn state(cost_fixed:u64,cost_per_token:u64)->Persistent{
        Persistent{owner:Principal::from_slice(&[1]),active_model:[0;32],upload:None,
            callers:BTreeMap::new(),workflow:EngineState::new([0;32]),max_input_tokens:MAX_INPUT_TOKENS,
            cost_fixed,cost_per_token,budget:UPDATE_BUDGET}
    }
    #[test]
    fn the_query_budget_is_the_protocol_limit(){
        assert_eq!(QUERY_BUDGET,5_000_000_000);
        assert_eq!(UPDATE_BUDGET,40_000_000_000);
    }
    #[test]
    fn legacy_snapshot_defaults_the_workflow_registry(){
        let old=LegacyPersistent{owner:Principal::from_slice(&[1]),active_model:[7;32],upload:None,callers:BTreeMap::new(),max_input_tokens:128,cost_fixed:COST_FIXED,cost_per_token:COST_PER_TOKEN,budget:UPDATE_BUDGET};
        let bytes=ic_laya_core::storage::encode(&old).unwrap();
        let restored=ic_laya_core::storage::decode::<Persistent>(&bytes).or_else(|_|->Result<Persistent>{
            let old:LegacyPersistent=ic_laya_core::storage::decode(&bytes)?;
            Ok(Persistent{owner:old.owner,active_model:old.active_model,upload:old.upload,callers:old.callers,workflow:EngineState::new(old.active_model),max_input_tokens:old.max_input_tokens,cost_fixed:old.cost_fixed,cost_per_token:old.cost_per_token,budget:old.budget})
        }).unwrap();
        assert_eq!(restored.workflow.active_model,[7;32]);
    }
    #[test]
    fn the_default_model_advertises_fifty_two_query_tokens(){
        let s=state(COST_FIXED,COST_PER_TOKEN);
        assert_eq!(max_tokens_within(&s,QUERY_BUDGET),53);
        assert!(guard_within(&s,53,QUERY_BUDGET).is_ok());
        assert!(matches!(guard_within(&s,54,QUERY_BUDGET),Err(Error::Capacity)));
        assert_eq!(max_tokens_within(&s,UPDATE_BUDGET),128);
    }
    #[test]
    fn stored_pre_optimization_fit_keeps_its_existing_query_policy(){
        let s=state(48_746_986,121_028_580);
        let bytes=ic_laya_core::storage::encode(&s).unwrap();
        let restored:Persistent=ic_laya_core::storage::decode(&bytes).unwrap();
        assert_eq!(max_tokens_within(&restored,QUERY_BUDGET),40);
        assert_eq!(restored.cost_fixed,48_746_986);
        assert_eq!(restored.cost_per_token,121_028_580);
    }
    #[test]
    fn the_measured_slope_raises_the_query_ceiling(){
        // 3.068e8/token is the *fitted marginal* slope of the two recorded points
        // (T=6 and T=120, docs/VERDICT_ENGINE.md 5.1.1/5.3). `I(120)/120 = 3.081e8`
        // is an average and still carries the fixed cost, so it is not the slope.
        let s=state(254_400_000,306_804_548);
        assert_eq!(max_tokens_within(&s,QUERY_BUDGET),15);
    }
    #[test]
    fn the_measured_fit_relaxes_the_update_ceiling_to_the_policy_bound(){
        // Installing the two-point fit is the documented way to stop refusing inputs the
        // hardware can afford. The update ceiling then lands on the *policy* bound, not on
        // a budget computation: 129 fits 40B, and MAX_INPUT_TOKENS caps it at 128.
        let s=state(159_525_640,306_804_548);
        assert_eq!(max_tokens_within(&s,UPDATE_BUDGET),MAX_INPUT_TOKENS);
        assert!(guard_within(&s,MAX_INPUT_TOKENS as usize,UPDATE_BUDGET).is_ok());
    }
    #[test]
    fn the_update_guard_still_pins_the_measured_ceiling(){
        let s=state(254_400_000,327_600_000);
        assert!(guard_budget(&s,120).is_ok());
        assert!(matches!(guard_budget(&s,126),Err(Error::Capacity)));
        // 121..128 fit the budget under the fitted model and are refused by the default
        // one; the default is the conservative choice, not the accurate one.
        let fitted=state(159_525_640,306_804_548);
        assert!(matches!(guard_budget(&s,121),Err(Error::Capacity)));
        assert!(guard_budget(&fitted,121).is_ok());
        assert!(guard_budget(&fitted,128).is_ok());
    }
    #[test]
    fn the_advertised_length_is_never_refused_by_the_guard(){
        for per_token in [COST_PER_TOKEN,306_804_548,121_000_000]{
            for budget in [QUERY_BUDGET,UPDATE_BUDGET]{
                let s=state(COST_FIXED,per_token);
                let t=max_tokens_within(&s,budget) as usize;
                assert!(guard_within(&s,t,budget).is_ok(),"T={t} per_token={per_token} budget={budget}");
            }
        }
    }
}
