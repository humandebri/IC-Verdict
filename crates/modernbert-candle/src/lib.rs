//! Batch-one, unpadded F32 ModernBERT encoder and the layers it is built from.
//!
//! This crate holds the encoder shared by the decision backends: attention with rotary
//! embeddings, the pre-norm layer, and the int8 quantised `Linear` used by the openJev
//! backend. The Laya decision head and its pack format were removed when that model was
//! dropped (see docs/VERDICT_ENGINE.md).
#![forbid(unsafe_code)]
use candle_core::{Device,Tensor,D};
use serde::{Deserialize,Serialize};

type CResult<T> = candle_core::Result<T>;
#[derive(Debug,Clone,Copy,Serialize,Deserialize)]
pub enum Activation { Relu, Gelu }
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
            let _=verdict_simd::matmul_i8(&aq,&q.w,&asx,&q.scales,m,k,q.out_features,&mut out);
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
