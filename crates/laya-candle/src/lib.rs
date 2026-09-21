//! Batch-one, unpadded F32 ModernBERT + option-marker decision head.
//! This is a source implementation, NOT a verified port of the 421M checkpoint.
//! Canonical tensor mapping and original-model parity remain release gates.
#![forbid(unsafe_code)]
pub mod pack;
use candle_core::{DType,Device,Tensor,D};
use ic_laya_core::{engine::InferenceBackend,BackendKind,Digest,Error,Result,TokenInput,MAX_TOKENS};
use serde::{Deserialize,Serialize};
use std::collections::BTreeMap;

type CResult<T> = candle_core::Result<T>;
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ModelConfig {
    pub vocab_size:usize,pub hidden_size:usize,pub layers:usize,pub attention_heads:usize,
    pub intermediate_size:usize,pub norm_eps:f64,pub global_every:usize,pub local_attention:usize,
    pub global_rope_theta:f64,pub local_rope_theta:f64,pub first_layer_attention_norm:bool,
    pub decision_layers:usize,pub decision_heads:usize,pub decision_ff:usize,pub decision_norm_eps:f64,
    pub decision_norm_first:bool,pub decision_activation:Activation,
    pub scorer_norm_eps:f64,pub qtypes:usize,pub mask_token_id:u32,
}
#[derive(Debug,Clone,Copy,Serialize,Deserialize)]
pub enum Activation { Relu, Gelu }
impl ModelConfig {
    pub fn validate(&self)->Result<()> {
        let h=self.hidden_size;
        if h==0 || h>2048 || self.vocab_size<5 || self.vocab_size>300_000 || self.layers==0 || self.layers>64
            || self.attention_heads==0 || h%self.attention_heads!=0 || (h/self.attention_heads)%2!=0
            || self.decision_heads==0 || h%self.decision_heads!=0 || self.decision_layers==0 || self.decision_layers>4
            || self.global_every==0 || self.local_attention==0 || self.intermediate_size==0 || self.intermediate_size>16384
            || self.decision_ff==0 || self.decision_ff>16384 || self.qtypes!=3 || self.mask_token_id as usize>=self.vocab_size {
            return Err(Error::Invalid("unsupported architecture config".into()));
        }
        for v in [self.norm_eps,self.decision_norm_eps,self.scorer_norm_eps]{if !v.is_finite() || v<=0.0 || v>1.0{return Err(Error::Numeric);}}
        for v in [self.global_rope_theta,self.local_rope_theta]{if !v.is_finite() || v<=1.0 || v>1e12{return Err(Error::Numeric);}}Ok(())
    }
}
/// A weight matrix quantised to int8 with one scale per output row.
///
/// Kept in the `[out, in]` layout, which is what the checkpoint uses and what the
/// int8 kernel wants: `i32x4.dot_i16x8_s` accumulates along the contiguous `in` axis.
#[derive(Clone)]
pub struct QuantWeight { pub w:Vec<i8>,pub scales:Vec<f32>,pub out_features:usize,pub in_features:usize }
#[derive(Clone)]
pub struct Linear { weight:Option<Tensor>,bias:Option<Tensor>,transposed:bool,quant:Option<QuantWeight> }
impl Linear {
    /// `weight` is `[out, in]`, the layout checkpoints use, and every forward pass
    /// pays a transpose plus a full materialised copy of it. Correct, but the copy
    /// is independent of the token count, so it dominates short inputs.
    pub fn new(weight:Tensor,bias:Option<Tensor>)->Self{Self{weight:Some(weight),bias,transposed:false,quant:None}}
    /// `weight` is already `[in, out]` (transposed once at load time): no per-call copy.
    pub fn new_transposed(weight:Tensor,bias:Option<Tensor>)->Self{Self{weight:Some(weight),bias,transposed:true,quant:None}}
    /// int8 weights: the f32 tensor is dropped, so this also cuts the resident weight
    /// memory by roughly four.
    pub fn new_quantized(quant:QuantWeight,bias:Option<Tensor>)->Self{Self{weight:None,bias,transposed:false,quant:Some(quant)}}
    pub fn forward(&self,x:&Tensor)->CResult<Tensor>{
        if let Some(q)=&self.quant{
            let dims=x.dims().to_vec();
            let k=*dims.last().ok_or(candle_core::Error::Msg("empty input".into()))?;
            if k!=q.in_features{return Err(candle_core::Error::Msg("quantised input width".into()));}
            let m=dims[..dims.len()-1].iter().product::<usize>().max(1);
            let flat=x.flatten_all()?.to_vec1::<f32>()?;
            let (aq,asx)=verdict_simd::quantize_acts_i16(&flat,m,k);
            let mut out=vec![0f32;m*q.out_features];
            verdict_simd::matmul_i8(&aq,&q.w,&asx,&q.scales,m,k,q.out_features,&mut out);
            let mut shape=dims[..dims.len()-1].to_vec();shape.push(q.out_features);
            let y=Tensor::from_vec(out,shape,x.device())?;
            return match &self.bias{Some(b)=>y.broadcast_add(b),None=>Ok(y)};
        }
        let weight=self.weight.as_ref().ok_or_else(||candle_core::Error::Msg("no weight".into()))?;
        let y=if self.transposed{x.matmul(weight)?}else{x.matmul(&weight.t()?.contiguous()?)?};
        match &self.bias{Some(b)=>y.broadcast_add(b),None=>Ok(y)}
    }
}
#[derive(Clone)]
pub struct Norm { weight:Tensor,bias:Option<Tensor>,eps:f64 }
impl Norm {
    pub fn new(weight:Tensor,bias:Option<Tensor>,eps:f64)->Self{Self{weight,bias,eps}}
    pub fn forward(&self,x:&Tensor)->CResult<Tensor>{
        let mean=x.mean_keepdim(D::Minus1)?;
        let centered=x.broadcast_sub(&mean)?;
        let var=centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y=centered.broadcast_div(&(var+self.eps)?.sqrt()?)?.broadcast_mul(&self.weight)?;
        match &self.bias{Some(b)=>y.broadcast_add(b),None=>Ok(y)}
    }
}
#[derive(Clone)]
pub struct Attention { qkv:Linear,out:Linear,heads:usize }
/// Softmax over the last dimension, computed in place on one buffer.
///
/// candle's chain (`max_keepdim`, `broadcast_sub`, `exp`, `sum_keepdim`,
/// `broadcast_div`) materialises four intermediate tensors and walks them with per-op
/// dispatch; that measured 220 instruction units per element (5.7% of a decision). The
/// arithmetic here is the same sequence with the same `f32::exp`, so the values are
/// unchanged; only the intermediate allocations are gone. See
/// `docs/VERDICT_ENGINE.md` 5.1.13.
fn softmax_last_fast(x:&Tensor)->CResult<Tensor>{
    let dims=x.dims().to_vec();
    let cols=match dims.last(){Some(c)=>*c,None=>return Err(candle_core::Error::Msg("softmax on empty".into()))};
    let rows:usize=dims[..dims.len()-1].iter().product::<usize>().max(1);
    let mut data=x.flatten_all()?.to_vec1::<f32>()?;
    verdict_simd::softmax_rows_inplace(&mut data,rows,cols);
    Tensor::from_vec(data,dims,x.device()).map_err(|e|candle_core::Error::Msg(e.to_string()))
}
fn softmax_last(x:&Tensor)->CResult<Tensor>{softmax_last_fast(x)}
/// One table per distinct `theta` among `layers` (at most two per encoder), built once.
pub fn rope_tables(layers:&[EncoderLayer],tokens:usize,hidden:usize,mark:&mut dyn FnMut(&'static str))->CResult<Vec<(f64,RopeTable)>>{
    let mut out:Vec<(f64,RopeTable)>=Vec::new();
    for layer in layers {
        let theta=layer.theta();
        let head_dim=layer.head_dim(hidden);
        if !out.iter().any(|(t,_)|*t==theta){let table=rope_table(tokens,head_dim,theta,mark)?;out.push((theta,table));}
    }
    Ok(out)
}
pub fn table_for<'a>(tables:&'a [(f64,RopeTable)],theta:f64)->&'a RopeTable{
    &tables.iter().find(|(t,_)|*t==theta).expect("rope table for layer theta").1
}

/// Rotary tables for one `(tokens, head_dim, theta)` combination.
///
/// Building them measured 6.1e9 instructions per decision (29.6% of the whole budget)
/// when it happened inside every layer and once more for `k`: `powf`/`cos`/`sin` are
/// libm calls and there are `tokens * head_dim / 2` of them. The tables depend only on
/// the sequence length and `theta`, so the encoder builds each distinct one once per
/// forward pass. `docs/VERDICT_ENGINE.md` 5.1.9.
#[derive(Clone)]
pub struct RopeTable{cos:Tensor,sin:Tensor}
fn rope_table(t:usize,d:usize,theta:f64,mark:&mut dyn FnMut(&'static str))->CResult<RopeTable>{
    let half=d/2;
    // `theta^(2i/d)` depends only on `i`; keeping the division form (rather than
    // multiplying by a reciprocal) keeps the values bit-identical to the previous
    // implementation, so parity is unchanged.
    let denom:Vec<f64>=(0..half).map(|i|theta.powf((2*i) as f64/d as f64)).collect();
    let mut cos=Vec::with_capacity(t*half);let mut sin=Vec::with_capacity(t*half);
    for pos in 0..t {let p=pos as f64;for i in 0..half {let a=p/denom[i];cos.push(a.cos() as f32);sin.push(a.sin() as f32);}}
    mark("rope.table");
    let cos=Tensor::from_vec(cos,(1,t,half),&Device::Cpu)?;let sin=Tensor::from_vec(sin,(1,t,half),&Device::Cpu)?;
    Ok(RopeTable{cos,sin})
}
fn rope_apply(x:&Tensor,table:&RopeTable,mark:&mut dyn FnMut(&'static str))->CResult<Tensor>{
    let (_,_t,d)=x.dims3()?;let half=d/2;
    let a=x.narrow(2,0,half)?;let b=x.narrow(2,half,half)?;
    let left=(a.broadcast_mul(&table.cos)?-b.broadcast_mul(&table.sin)?)?;
    let right=(b.broadcast_mul(&table.cos)?+a.broadcast_mul(&table.sin)?)?;
    mark("rope.apply");
    Tensor::cat(&[&left,&right],2)
}
impl Attention {
    pub fn heads(&self)->usize{self.heads}
    pub fn new(qkv:Linear,out:Linear,heads:usize)->Self{Self{qkv,out,heads}}
    pub fn forward(&self,x:&Tensor,table:Option<&RopeTable>,max_distance:Option<usize>)->CResult<Tensor>{
        self.forward_marked(x,table,max_distance,&mut |_|{})
    }
    /// Same computation, splitting the two halves that have different fixes.
    ///
    /// `attn.pre` is the projection plus RoPE plus the head reshape/transpose: dense
    /// matmul and pure data movement. `attn.core` is scores, softmax, masking and the
    /// value matmul: element-wise work over a `t x t` matrix. Marking them apart is
    /// what tells a kernel problem from an algorithm problem — they are improved by
    /// different changes, and one "attention" number cannot separate them.
    pub fn forward_marked(&self,x:&Tensor,table:Option<&RopeTable>,max_distance:Option<usize>,
                          mark:&mut dyn FnMut(&'static str))->CResult<Tensor>{
        let (t,h)=x.dims2()?;let d=h/self.heads;let y=self.qkv.forward(x)?;
        let mut q=y.narrow(1,0,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let mut k=y.narrow(1,h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let v=y.narrow(1,2*h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        if let Some(table)=table {q=rope_apply(&q,table,mark)?;k=rope_apply(&k,table,mark)?;}
        mark("attn.pre");
        let mut scores=(q.contiguous()?.matmul(&k.transpose(1,2)?.contiguous()?)?*(1.0/(d as f64).sqrt()))?;
        mark("attn.scores");
        if let Some(distance)=max_distance {
            let mask:Vec<f32>=(0..t).flat_map(|i|(0..t).map(move|j|if i.abs_diff(j)>distance{f32::NEG_INFINITY}else{0.0})).collect();
            scores=scores.broadcast_add(&Tensor::from_vec(mask,(1,t,t),x.device())?)?;
        }
        mark("attn.mask");
        let weights=softmax_last(&scores)?;
        mark("attn.softmax");
        let merged=weights.matmul(&v)?.transpose(0,1)?.contiguous()?.reshape((t,h))?;
        mark("attn.core");
        self.out.forward(&merged)
    }
}
#[derive(Clone)]
pub struct EncoderLayer { attention_norm:Option<Norm>,attention:Attention,mlp_norm:Norm,wi:Linear,wo:Linear,theta:f64,distance:Option<usize> }
impl EncoderLayer {
    pub fn new(attention_norm:Option<Norm>,attention:Attention,mlp_norm:Norm,wi:Linear,wo:Linear,theta:f64,distance:Option<usize>)->Self{
        Self{attention_norm,attention,mlp_norm,wi,wo,theta,distance}
    }
    pub fn forward(&self,x:&Tensor,table:&RopeTable)->CResult<Tensor>{self.forward_marked(x,table,&mut |_|{})}
    pub fn theta(&self)->f64{self.theta}
    /// Rotary tables are indexed by the *per-head* dimension, not the hidden size.
    pub fn head_dim(&self,hidden:usize)->usize{hidden/self.attention.heads()}
    /// Same computation, with a marker after each sub-phase.
    ///
    /// The split matters because the two halves have different fixes: the attention
    /// projections and the MLP are dense matmuls (quantisation territory), whereas
    /// softmax and the gated activation are element-wise and would need different
    /// treatment.
    pub fn forward_marked(&self,x:&Tensor,table:&RopeTable,mark:&mut dyn FnMut(&'static str))->CResult<Tensor>{
        let normalized=match &self.attention_norm{Some(n)=>n.forward(x)?,None=>x.clone()};
        mark("layer.attn_norm");
        let attended=self.attention.forward_marked(&normalized,Some(table),self.distance,mark)?;
        mark("layer.attn");
        let x=(x+attended)?;
        mark("layer.attn_resid");
        let y=self.wi.forward(&self.mlp_norm.forward(&x)?)?;let half=y.dim(1)?/2;
        mark("layer.mlp_up");
        let gate=(y.narrow(1,0,half)?.gelu_erf()?*y.narrow(1,half,half)?)?;
        mark("layer.mlp_act");
        let out=(&x+self.wo.forward(&gate)?)?;
        mark("layer.mlp_down");
        Ok(out)
    }
}
#[derive(Clone)]
struct DecisionLayer { norm1:Norm,norm2:Norm,attention:Attention,linear1:Linear,linear2:Linear,norm_first:bool,activation:Activation }
impl DecisionLayer {
    fn ff(&self,x:&Tensor)->CResult<Tensor>{let x=self.linear1.forward(x)?;self.linear2.forward(&match self.activation{Activation::Relu=>x.relu()?,Activation::Gelu=>x.gelu_erf()?})}
    fn forward(&self,x:&Tensor)->CResult<Tensor>{
        if self.norm_first {
            let x=(x+self.attention.forward(&self.norm1.forward(x)?,None,None)?)?;
            &x+self.ff(&self.norm2.forward(&x)?)?
        }else{
            let x=self.norm1.forward(&(x+self.attention.forward(x,None,None)?)?)?;
            self.norm2.forward(&(&x+self.ff(&x)?)?)
        }
    }
}

pub fn expected_tensors(c:&ModelConfig)->Result<BTreeMap<String,Vec<usize>>>{
    c.validate()?;let mut m=BTreeMap::new();let h=c.hidden_size;
    m.insert("embeddings.weight".into(),vec![c.vocab_size,h]);m.insert("embeddings.norm.weight".into(),vec![h]);m.insert("final_norm.weight".into(),vec![h]);
    for i in 0..c.layers {
        let p=format!("encoder.{i}");
        if i>0 || c.first_layer_attention_norm{m.insert(format!("{p}.attn_norm.weight"),vec![h]);}
        for (name,shape) in [("qkv.weight",vec![3*h,h]),("out.weight",vec![h,h]),("mlp_norm.weight",vec![h]),("wi.weight",vec![2*c.intermediate_size,h]),("wo.weight",vec![h,c.intermediate_size])] {m.insert(format!("{p}.{name}"),shape);}
    }
    m.insert("qtype.weight".into(),vec![c.qtypes,h]);
    for i in 0..c.decision_layers {
        let p=format!("decision.{i}");
        for (name,shape) in [("qkv.weight",vec![3*h,h]),("qkv.bias",vec![3*h]),("out.weight",vec![h,h]),("out.bias",vec![h]),("norm1.weight",vec![h]),("norm1.bias",vec![h]),("norm2.weight",vec![h]),("norm2.bias",vec![h]),("linear1.weight",vec![c.decision_ff,h]),("linear1.bias",vec![c.decision_ff]),("linear2.weight",vec![h,c.decision_ff]),("linear2.bias",vec![h])] {m.insert(format!("{p}.{name}"),shape);}
    }
    for (name,shape) in [("norm.weight",vec![h]),("norm.bias",vec![h]),("dense.weight",vec![h,h]),("dense.bias",vec![h]),("out.weight",vec![1,h]),("out.bias",vec![1])] {m.insert(format!("scorer.{name}"),shape);}
    Ok(m)
}

pub struct LayaModel {
    pub config:ModelConfig,pub bundle:Digest,pub backend_kind:BackendKind,
    embedding:Tensor,embedding_norm:Norm,layers:Vec<EncoderLayer>,final_norm:Norm,qtype:Tensor,
    decision:Vec<DecisionLayer>,scorer_norm:Norm,scorer_dense:Linear,scorer_out:Linear,
}
fn tensor(m:&BTreeMap<String,Tensor>,name:&str)->Result<Tensor>{m.get(name).cloned().ok_or_else(||Error::Invalid(format!("missing tensor: {name}")))}
fn linear(m:&BTreeMap<String,Tensor>,p:&str,bias:bool)->Result<Linear>{
    let weight=tensor(m,&format!("{p}.weight"))?;
    let b=if bias{Some(tensor(m,&format!("{p}.bias"))?)}else{None};
    Ok(Linear::new(weight,b))
}
fn norm(m:&BTreeMap<String,Tensor>,p:&str,bias:bool,eps:f64)->Result<Norm>{Ok(Norm{weight:tensor(m,&format!("{p}.weight"))?,bias:if bias{Some(tensor(m,&format!("{p}.bias"))?)}else{None},eps})}
impl LayaModel {
    pub fn from_tensors(c:ModelConfig,bundle:Digest,kind:BackendKind,m:BTreeMap<String,Tensor>)->Result<Self>{
        let expected=expected_tensors(&c)?;
        if expected.len()!=m.len(){return Err(Error::Invalid("unexpected tensor set".into()));}
        for (name,shape) in &expected {let t=m.get(name).ok_or_else(||Error::Invalid(format!("missing {name}")))?;if t.dims()!=shape.as_slice() || t.dtype()!=DType::F32{return Err(Error::Invalid(format!("shape/dtype: {name}")));}}
        let mut layers=Vec::new();
        for i in 0..c.layers {let p=format!("encoder.{i}");let local=i%c.global_every!=0;
            layers.push(EncoderLayer{attention_norm:if i>0||c.first_layer_attention_norm{Some(norm(&m,&format!("{p}.attn_norm"),false,c.norm_eps)?)}else{None},
                attention:Attention{qkv:linear(&m,&format!("{p}.qkv"),false)?,out:linear(&m,&format!("{p}.out"),false)?,heads:c.attention_heads},mlp_norm:norm(&m,&format!("{p}.mlp_norm"),false,c.norm_eps)?,
                wi:linear(&m,&format!("{p}.wi"),false)?,wo:linear(&m,&format!("{p}.wo"),false)?,theta:if local{c.local_rope_theta}else{c.global_rope_theta},distance:if local{Some(c.local_attention/2)}else{None}});
        }
        let mut decision=Vec::new();
        for i in 0..c.decision_layers {let p=format!("decision.{i}");decision.push(DecisionLayer{norm1:norm(&m,&format!("{p}.norm1"),true,c.decision_norm_eps)?,norm2:norm(&m,&format!("{p}.norm2"),true,c.decision_norm_eps)?,
            attention:Attention{qkv:linear(&m,&format!("{p}.qkv"),true)?,out:linear(&m,&format!("{p}.out"),true)?,heads:c.decision_heads},linear1:linear(&m,&format!("{p}.linear1"),true)?,linear2:linear(&m,&format!("{p}.linear2"),true)?,norm_first:c.decision_norm_first,activation:c.decision_activation});}
        Ok(Self{embedding:tensor(&m,"embeddings.weight")?,embedding_norm:norm(&m,"embeddings.norm",false,c.norm_eps)?,final_norm:norm(&m,"final_norm",false,c.norm_eps)?,qtype:tensor(&m,"qtype.weight")?,
            scorer_norm:norm(&m,"scorer.norm",true,c.scorer_norm_eps)?,scorer_dense:linear(&m,"scorer.dense",true)?,scorer_out:linear(&m,"scorer.out",true)?,config:c,bundle,backend_kind:kind,layers,decision})
    }
    fn forward_tensor(&self,input:&TokenInput)->CResult<Tensor>{
        let dev=&Device::Cpu;let ids=Tensor::from_vec(input.input_ids.clone(),input.input_ids.len(),dev)?;
        let mut x=self.embedding_norm.forward(&self.embedding.index_select(&ids,0)?)?;
        let tables=rope_tables(&self.layers,x.dim(0)?,x.dim(1)?,&mut |_|{})?;
        for layer in &self.layers{let table=table_for(&tables,layer.theta()).clone();x=layer.forward(&x,&table)?;}
        x=self.final_norm.forward(&x)?;
        let qt=self.qtype.narrow(0,input.qtype_id as usize,1)?;x=x.broadcast_add(&qt)?;
        for layer in &self.decision{x=layer.forward(&x)?;}
        let markers=Tensor::from_vec(input.markers.clone(),input.markers.len(),dev)?;
        let selected=x.index_select(&markers,0)?;
        self.scorer_out.forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&selected)?)?.gelu_erf()?)?.squeeze(1)
    }

    /// `forward_tensor` with a callback at each phase boundary.
    ///
    /// Measurement only: `InferenceBackend::infer` does not use this, so the
    /// production path carries no instrumentation. The callback is a plain
    /// function pointer and this crate does not depend on `ic-cdk`, so it works
    /// on native (where the counter is a stub returning 0) and inside a canister
    /// (where it reads `instruction_counter`) without a second implementation.
    fn forward_profiled(&self,input:&TokenInput,mark:&mut dyn FnMut(&'static str),detailed:bool)->CResult<Tensor>{
        let dev=&Device::Cpu;let ids=Tensor::from_vec(input.input_ids.clone(),input.input_ids.len(),dev)?;
        let mut x=self.embedding_norm.forward(&self.embedding.index_select(&ids,0)?)?;
        mark("embedding");
        // Only the detailed profile exposes the rope phases; the coarse one keeps the
        // documented phase list.
        let tokens=x.dim(0)?;let hidden=x.dim(1)?;
        let tables=if detailed {rope_tables(&self.layers,tokens,hidden,mark)?}
                    else {rope_tables(&self.layers,tokens,hidden,&mut |_|{})?};
        if detailed {
            for layer in &self.layers{let table=table_for(&tables,layer.theta()).clone();x=layer.forward_marked(&x,&table,mark)?;}
        } else {
            for layer in &self.layers{let table=table_for(&tables,layer.theta()).clone();x=layer.forward(&x,&table)?;}
        }
        mark("encoder");
        x=self.final_norm.forward(&x)?;
        let qt=self.qtype.narrow(0,input.qtype_id as usize,1)?;x=x.broadcast_add(&qt)?;
        mark("final_norm");
        for layer in &self.decision{x=layer.forward(&x)?;}
        mark("decision");
        let markers=Tensor::from_vec(input.markers.clone(),input.markers.len(),dev)?;
        let selected=x.index_select(&markers,0)?;
        let out=self.scorer_out.forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&selected)?)?.gelu_erf()?)?.squeeze(1);
        mark("scorer");
        out
    }

    /// Per-phase instruction cost for one question.
    ///
    /// Returns one entry per phase in execution order. `instructions` is the
    /// delta since the previous boundary, so the entries sum to the total for
    /// this call (the `decode` phase covers `.to_vec1` and the finite check).
    pub fn infer_profiled(&mut self,input:&TokenInput,counter:&dyn Fn()->u64)->Result<Vec<PhaseCost>>{
        self.infer_profiled_detailed(input,counter,false)
    }

    /// As `infer_profiled`, but also marks the sub-phases of every encoder layer.
    /// The per-layer entries repeat, so callers should aggregate by name.
    pub fn infer_profiled_detailed(&mut self,input:&TokenInput,counter:&dyn Fn()->u64,detailed:bool)->Result<Vec<PhaseCost>>{
        if input.input_ids.is_empty() || input.input_ids.len()>MAX_TOKENS || !(2..=7).contains(&input.markers.len()) || input.qtype_id as usize>=self.config.qtypes
            || input.input_ids.iter().any(|&v|v as usize>=self.config.vocab_size){return Err(Error::Invalid("token input".into()));}
        let mut prev=None;
        for &p in &input.markers {if input.input_ids.get(p as usize)!=Some(&self.config.mask_token_id) || prev.is_some_and(|v|p<=v){return Err(Error::Invalid("option markers".into()));}prev=Some(p);}
        let mut costs:Vec<PhaseCost>=Vec::new();
        let mut last=counter();
        {
            let mut record=|name:&'static str|{let now=counter();costs.push(PhaseCost{name,instructions:now.saturating_sub(last)});last=now;};
            let tensor=self.forward_profiled(input,&mut record,detailed).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
            let values=tensor.to_vec1::<f32>().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
            record("decode");
            if values.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}
        }
        Ok(costs)
    }
}

/// One phase of a profiled inference. Measurement only; never part of a Receipt.
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct PhaseCost { pub name:&'static str, pub instructions:u64 }
impl InferenceBackend for LayaModel {
    fn bundle_id(&self)->Digest{self.bundle}
    fn kind(&self)->BackendKind{self.backend_kind}
    fn infer(&mut self,input:&TokenInput)->Result<Vec<f32>>{
        if input.input_ids.is_empty() || input.input_ids.len()>MAX_TOKENS || !(2..=7).contains(&input.markers.len()) || input.qtype_id as usize>=self.config.qtypes
            || input.input_ids.iter().any(|&v|v as usize>=self.config.vocab_size){return Err(Error::Invalid("token input".into()));}
        let mut prev=None;
        for &p in &input.markers {if input.input_ids.get(p as usize)!=Some(&self.config.mask_token_id) || prev.is_some_and(|v|p<=v){return Err(Error::Invalid("option markers".into()));}prev=Some(p);}
        let v=self.forward_tensor(input).and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if v.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}Ok(v)
    }
}

/// Shared ModernBERT encoder surface.
///
/// `verdict-candle` (the openJev/GLiClass backend) reuses this instead of a
/// second encoder implementation, so both backends run identical encoder
/// kernels and differ only in their heads.
pub mod encoder {
    use super::{CResult,EncoderLayer,Norm,RopeTable,Tensor};
    /// ModernBERT = token embeddings + pre-norm layers + final norm.
    pub struct ModernBert { pub embedding:Tensor,pub embedding_norm:Norm,pub layers:Vec<EncoderLayer>,pub final_norm:Norm }
    impl ModernBert {
        pub fn new(embedding:Tensor,embedding_norm:Norm,layers:Vec<EncoderLayer>,final_norm:Norm)->Self{
            Self{embedding,embedding_norm,layers,final_norm}
        }
        /// One rotary table per distinct `theta` among the layers (at most two), built
        /// once per forward pass. Measured: building them per layer and per `q`/`k` cost
        /// 6.1e9 instructions per decision, 29.6% of the total
        /// (`docs/VERDICT_ENGINE.md` 5.1.9).
        pub fn rope_tables(&self,tokens:usize,dim:usize,mark:&mut dyn FnMut(&'static str))->CResult<Vec<(f64,RopeTable)>>{
            super::rope_tables(&self.layers,tokens,dim,mark)
        }
        pub fn table_for<'a>(tables:&'a [(f64,RopeTable)],theta:f64)->&'a RopeTable{super::table_for(tables,theta)}
        /// `input_ids` is a 1-D token id tensor; the result is `[tokens, hidden]`.
        pub fn forward(&self,input_ids:&Tensor)->CResult<Tensor>{
            let mut x=self.embedding_norm.forward(&self.embedding.index_select(input_ids,0)?)?;
            let tables=self.rope_tables(x.dim(0)?,x.dim(1)?,&mut |_|{})?;
            for layer in &self.layers{let table=Self::table_for(&tables,layer.theta()).clone();x=layer.forward(&x,&table)?;}
            self.final_norm.forward(&x)
        }
    }
}
