//! openJev (GLiClass / ModernBERT-151M) decision engine canister.
//!
//! Scope: this canister exists to answer one question with evidence — does the
//! ported GLiClass forward actually run inside a canister, and what does one
//! question cost in instructions? It therefore has no ledger, no workflow and no
//! fund movement: it loads a pack, runs a forward pass and reports the measured
//! instruction count with the logits.
//!
//! The model side is `verdict-candle`; the encoder it uses is the same
//! `laya-candle` ModernBERT code path the Laya backend uses, so the two backends
//! differ only in their heads.
//!
//! Sizing: F32 weights are 151M x 4 B = 605 MiB, which fits the 4 GiB wasm heap
//! with room for activations. Instructions, not memory, are the binding limit:
//! see `MAX_INPUT_TOKENS` and the measurement note in `infer_tokens`.
use candid::{CandidType,Principal};
use candle_core::{Device,Module,Tensor};
use candle_core::quantized::{GgmlDType,QMatMul,QTensor};
use ic_laya_core::{hash,Digest,Error,Result,SpecialTokens};
use ic_laya_core::engine::InferenceBackend;
use ic_laya_core::schema::TextTokenizer;
use serde::{Deserialize,Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[cfg(target_arch = "wasm32")]
#[path = "../../decision-engine/src/getrandom_ic.rs"]
mod getrandom_ic;

/// Refuse inputs longer than this. It is a policy bound; the *budget* guard is
/// `estimated_cost`, which uses the measured cost model below.
pub const MAX_INPUT_TOKENS:u32=128;
/// Measured on a local replica against the real checkpoint, F32, wasm
/// (docs/VERDICT_ENGINE.md 5.1): `I(T) ≈ COST_FIXED + COST_PER_TOKEN × T`, with the
/// real prompt sweep pinning the ceiling between 120 (39.57e9) and 126 (rejected).
/// The guard must refuse *before* spending the budget, so the model is a constant
/// rather than a measurement; `set_cost_model` lets the owner correct it.
pub const COST_FIXED:u64=254_400_000;
pub const COST_PER_TOKEN:u64=327_600_000;
/// ICP's per-update instruction limit.
pub const UPDATE_BUDGET:u64=40_000_000_000;
/// Keep a margin for the reply encoding and the tokenizer, which the linear model
/// above does not cover. Fitted so the guard reproduces the measurement: T=120
/// (39.57e9 measured, 39.77e9 projected) is accepted and T=126 (41.74e9) is refused,
/// which is exactly where the replica rejected the call.
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
    max_input_tokens:u32,
    // Cost model used by the pre-flight budget guard. `#[serde(default)]` keeps an
    // older snapshot loadable across an upgrade.
    #[serde(default="default_cost_fixed")] cost_fixed:u64,
    #[serde(default="default_cost_per_token")] cost_per_token:u64,
    #[serde(default="default_budget")] budget:u64,
}
fn default_cost_fixed()->u64{COST_FIXED}
fn default_cost_per_token()->u64{COST_PER_TOKEN}
fn default_budget()->u64{UPDATE_BUDGET}
/// Projected instructions for one forward pass of `tokens` tokens, plus margin.
fn projected(s:&Persistent,tokens:usize)->u64{
    (s.cost_fixed.saturating_add(s.cost_per_token.saturating_mul(tokens as u64)))
        .saturating_mul(BUDGET_MARGIN_PERMILLE)/1000
}
fn guard_budget(s:&Persistent,tokens:usize)->Result<()>{
    if projected(s,tokens)>s.budget {return Err(Error::Capacity);}
    Ok(())
}
thread_local!{static STATE:RefCell<Option<Persistent>>=const{RefCell::new(None)};}
thread_local!{
    static TOKENIZER:RefCell<Option<hf_tokenizer::HfTokenizer>>=const{RefCell::new(None)};
    static BUILDER:RefCell<Option<verdict_candle::pack::Builder>>=const{RefCell::new(None)};
    static MODEL:RefCell<Option<verdict_candle::VerdictModel>>=const{RefCell::new(None)};
}

fn read<R>(f:impl FnOnce(&Persistent)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
fn mutate<R>(f:impl FnOnce(&mut Persistent)->R)->R{STATE.with(|x|{let mut state=x.borrow_mut();let state=state.as_mut().expect("initialized");let result=f(state);canister_common::persist_or_trap(state);result})}
fn owner()->Result<()>{read(|s|if ic_cdk::api::msg_caller()==s.owner{Ok(())}else{Err(Error::Unauthorized)})}
fn admitted(caller:Principal)->Result<()>{
    read(|s|if caller==s.owner || (caller!=Principal::anonymous() && s.callers.contains_key(&caller)){Ok(())}else{Err(Error::Unauthorized)})
}

#[ic_cdk::init]
fn init(owner:Principal){
    if owner==Principal::anonymous() || owner==Principal::management_canister(){ic_cdk::trap("invalid owner");}
    let s=Persistent{owner,active_model:[0;32],upload:None,callers:BTreeMap::new(),max_input_tokens:MAX_INPUT_TOKENS,
        cost_fixed:COST_FIXED,cost_per_token:COST_PER_TOKEN,budget:UPDATE_BUDGET};
    canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(canister_common::persist_or_trap);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){
    // The heap model is never kept across an upgrade: the pack bytes and upload
    // metadata survive in stable memory, the Candle tensors do not.
    let s:Persistent=canister_common::restore().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));
    MODEL.with(|x|*x.borrow_mut()=None);BUILDER.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=None);
    STATE.with(|x|*x.borrow_mut()=Some(s));
}

#[derive(CandidType,Serialize,Deserialize)]
pub struct EngineInfo{
    pub model:Digest,
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
            model:s.active_model,tensors,warmed,
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
    mutate(|s|{if !s.callers.contains_key(&caller) && s.callers.len()>=64{return Err(Error::Capacity);}s.callers.insert(caller,ic_cdk::api::time() as u32);Ok(())})
}
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
/// running at ~2.7 instructions/MAC, while a hand-written f32x4 kernel should reach
/// 0.5-1.0. Owner-only, never reachable from an inference path, no state written.
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
    Ok(QuantBenchReply{m:m as u32,n:n as u32,k:k as u32,iterations:iterations as u32,dtype,
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
    if m==0||n==0||k==0||m>512||n>4096||k>4096||iterations==0||iterations>64||k%8!=0{return Err(Error::Invalid("bench shape".into()));}
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
    Ok(Int8BenchReply{m:m as u32,n:n as u32,k:k as u32,iterations:iterations as u32,simd_used,instructions,
        per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0.0{instructions as f64/macs}else{0.0},
        quantize_weights_instructions,quantize_activations_instructions,max_abs_diff_vs_f32:max_abs,max_rel_diff_vs_f32:max_rel})
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
    Ok(F16BenchReply{m:m as u32,n:n as u32,k:k as u32,iterations:iterations as u32,convert_instructions,instructions,
        per_iteration:instructions/iterations as u64,
        instructions_per_mac:if macs>0.0{instructions as f64/macs}else{0.0},max_abs_diff_vs_f32})
}

/// MEASUREMENT ONLY. What the instruction counter charges for one wasm SIMD operator.
///
/// The kernel work is planned against `instr_per_mac`, so knowing the per-operator
/// weights separates "the backend is slow" from "there is nothing left to remove".
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
        let bundle=builder.bundle;let model=builder.finish()?;
        MODEL.with(|m|*m.borrow_mut()=Some(model));mutate(|s|s.active_model=bundle);
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
/// Raw token ids in, one logit per `<<LABEL>>` token out.
///
/// The instruction count is measured around the forward pass only, so it excludes
/// Candid decoding of the arguments and the reply encoding. `measured_instructions`
/// is the number the 40B update-call limit applies to.
#[ic_cdk::update]
fn infer_tokens(input_ids:Vec<u32>)->Result<InferReply>{
    let caller=ic_cdk::api::msg_caller();admitted(caller)?;
    read(|s|{
        if input_ids.is_empty() || input_ids.len()>s.max_input_tokens as usize {return Err(Error::TooLong);}
        guard_budget(s,input_ids.len())?;
        MODEL.with(|m|{
            let mut m=m.borrow_mut();let model=m.as_mut().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?;
            let before=ic_cdk::api::instruction_counter();
            let logits=model.logits(&input_ids)?;
            let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
            Ok(InferReply{model:model.bundle_id(),class_positions:model.class_positions(&input_ids),logits,input_tokens:input_ids.len() as u32,measured_instructions:measured})
        })
    })
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
/// One typed decision: state + question + options in, calibrated distribution out.
///
/// The prompt is the checkpoint's own contract
/// (`<<LABEL>>desc...<<SEP>>Question: ...\n\nContext:\n...`, wrapped as
/// `[CLS] ... [SEP]`), so the tokenizer and the head see exactly what the
/// reference engine produces.
#[ic_cdk::update]
fn decide(req:DecideRequest)->Result<DecideReply>{
    let caller=ic_cdk::api::msg_caller();admitted(caller)?;
    if req.state.is_empty() || req.state.len()>ic_laya_core::MAX_STATE_BYTES {return Err(Error::TooLong);}
    if req.options.is_empty() || req.options.len()>MAX_OPTIONS {return Err(Error::TooLong);}
    if !req.temperature.is_finite() || req.temperature<=0.0 || req.temperature>100.0 {return Err(Error::Numeric);}
    for o in &req.options {if o.id.is_empty() || o.text.is_empty(){return Err(Error::Invalid("option".into()));}}
    let limits=read(|s|s.max_input_tokens);
    let mut ids:Vec<String>=req.options.iter().map(|o|o.id.clone()).collect();
    let mut labels:Vec<String>=req.options.iter().map(|o|o.text.clone()).collect();
    if req.abstention {ids.push("__insufficient_evidence__".into());labels.push(ABSTENTION_DESC.into());}
    if ids.len()>verdict_candle::MAX_CLASSES{return Err(Error::TooLong);}
    let prompt=verdict_candle::render_prompt(&req.question,&req.state,&labels);
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let mut m=m.borrow_mut();
        let (t,model)=match (t.as_ref(),m.as_mut()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("warm-up required".into()))};
        let mut input=vec![model.config.cls_token_id];
        input.extend(t.encode_piece(&prompt)?);
        input.push(model.config.sep_token_id);
        if input.len()>limits as usize || input.len()>verdict_candle::MAX_SEQUENCE{return Err(Error::TooLong);}
        guard_budget(&read(|s|s.clone()),input.len())?;
        let before=ic_cdk::api::instruction_counter();
        let logits=model.logits(&input)?;
        let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
        let scaled:Vec<f32>=logits.iter().map(|x|(*x as f64/req.temperature) as f32).collect();
        let probs=softmax(&scaled);
        let best=probs.iter().enumerate().fold((0usize,f32::NEG_INFINITY),|a,(i,p)|if *p>a.1{(i,*p)}else{a});
        Ok(DecideReply{
            model:model.bundle_id(),ids:ids.clone(),logits,probabilities:probs.clone(),
            selected:ids[best.0].clone(),confidence:best.1,input_tokens:input.len() as u32,measured_instructions:measured,
        })
    }))
}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchQuestion{pub id:String,pub question:String,pub options:Vec<OptionSpec>,pub abstention:bool}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchRequest{pub state:String,pub questions:Vec<BatchQuestion>,pub temperature:f64}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct QuestionResult{pub id:String,pub ids:Vec<String>,pub logits:Vec<f32>,pub probabilities:Vec<f32>,pub selected:String,pub confidence:f32}
#[derive(CandidType,Serialize,Deserialize,Clone)]
pub struct BatchReply{pub model:Digest,pub questions:Vec<QuestionResult>,pub input_tokens:u32,pub measured_instructions:u64}
/// Several typed questions over one state, in a single forward pass.
///
/// Measured reason (docs/VERDICT_ENGINE.md 5.1.6): sending the same state three times
/// costs 63.1e9 instructions for three decisions; one pass with all labels costs
/// 34.2e9 (-45.8%), or 26.8e9 with short label text. The trade is that all labels are
/// scored against one `[CLS]` representation, so the per-question distributions shift
/// (argmax was unchanged in the sample, logits moved by up to 1.7) and a batched layout
/// needs its own calibration.
#[ic_cdk::update]
fn decide_batch(req:BatchRequest)->Result<BatchReply>{
    let caller=ic_cdk::api::msg_caller();admitted(caller)?;
    if req.state.is_empty() || req.state.len()>ic_laya_core::MAX_STATE_BYTES{return Err(Error::TooLong);}
    if req.questions.is_empty() || req.questions.len()>8{return Err(Error::TooLong);}
    if !req.temperature.is_finite()||req.temperature<=0.0||req.temperature>100.0{return Err(Error::Numeric);}
    let limits=read(|s|s.max_input_tokens);
    let (mut labels,mut ids,mut counts)=(Vec::new(),Vec::new(),Vec::new());
    for q in &req.questions {
        if q.options.is_empty() || q.options.len()>MAX_OPTIONS{return Err(Error::TooLong);}
        let mut n=0usize;
        for o in &q.options {
            if o.id.is_empty()||o.text.is_empty(){return Err(Error::Invalid("option".into()));}
            labels.push(o.text.clone());ids.push(o.id.clone());n+=1;
        }
        if q.abstention {labels.push(ABSTENTION_DESC.into());ids.push("__insufficient_evidence__".into());n+=1;}
        counts.push(n);
    }
    if ids.len()>verdict_candle::MAX_CLASSES{return Err(Error::TooLong);}
    let question=req.questions.iter().map(|q|q.question.clone()).collect::<Vec<_>>().join(" | ");
    let prompt=verdict_candle::render_prompt(&question,&req.state,&labels);
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let mut m=m.borrow_mut();
        let (t,model)=match (t.as_ref(),m.as_mut()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("warm-up required".into()))};
        let mut input=vec![model.config.cls_token_id];
        input.extend(t.encode_piece(&prompt)?);
        input.push(model.config.sep_token_id);
        if input.len()>limits as usize || input.len()>verdict_candle::MAX_SEQUENCE{return Err(Error::TooLong);}
        guard_budget(&read(|s|s.clone()),input.len())?;
        let before=ic_cdk::api::instruction_counter();
        let logits=model.logits(&input)?;
        let measured=ic_cdk::api::instruction_counter().saturating_sub(before);
        let scaled:Vec<f32>=logits.iter().map(|x|(*x as f64/req.temperature) as f32).collect();
        let probabilities=softmax(&scaled);
        let mut out=Vec::new();let mut offset=0usize;
        for (q,count) in req.questions.iter().zip(counts.iter()) {
            let end=offset+count;
            let qlogits=logits[offset..end].to_vec();
            let qprobs=probabilities[offset..end].to_vec();
            let qids=ids[offset..end].to_vec();
            let best=qprobs.iter().enumerate().fold((0usize,f32::NEG_INFINITY),|a,(i,p)|if *p>a.1{(i,*p)}else{a});
            out.push(QuestionResult{id:q.id.clone(),ids:qids.clone(),logits:qlogits,probabilities:qprobs,
                selected:qids[best.0].clone(),confidence:best.1});
            offset=end;
        }
        Ok(BatchReply{model:model.bundle_id(),questions:out,input_tokens:input.len() as u32,measured_instructions:measured})
    }))
}

fn softmax(xs:&[f32])->Vec<f32>{
    let m=xs.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
    let e:Vec<f32>=xs.iter().map(|x|(x-m).exp()).collect();
    let s:f32=e.iter().sum();
    if !(s>0.0) {return vec![0.0;xs.len()];}
    e.iter().map(|x|x/s).collect()
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}
