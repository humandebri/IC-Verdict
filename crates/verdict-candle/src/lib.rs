//! openJev / GLiClass "uni-encoder" decision backend.
//!
//! This is a source port of the reference implementation, not a verified port of
//! the checkpoint by construction: `tests/golden.rs` compares whole logits against
//! the author's own recorded decisions, which is the release gate.
//!
//! The head follows `gliclass` 0.1.20 exactly:
//!   * `GLiClassBaseModel._extract_class_features_first` with `embed_class_token=true`
//!     and `class_token_pooling="first"`: class `k` is the hidden state of the k-th
//!     `<<LABEL>>` token in the sequence, in order of appearance.
//!   * `extract_text_features=false`: the text side is the whole sequence, and
//!     `pooling_strategy="first"` takes position 0, i.e. the `[CLS]` token.
//!   * `text_projector` and `classes_projector` are Linear->GELU(erf)->Linear with bias.
//!   * `scorer_type="simple"` is a dot product `text_rep · class_rep`.
//!   * `normalize_features=false`, so the checkpoint's scalar `logit_scale` is NOT
//!     applied. It is deliberately absent from the pack and from the expected set.
//!
//! The encoder is `laya_candle::encoder::ModernBert` (HF ModernBERT-base semantics:
//! pre-norm with layer 0 attention norm omitted, GeGLU with `act(first)*second`,
//! NeoX-style RoPE on the full head dimension, sliding window of
//! `local_attention/2` on every layer that is not global).
#![forbid(unsafe_code)]
pub mod pack;
use candle_core::{DType,Device,Tensor,D};
use ic_laya_core::{engine::InferenceBackend,BackendKind,Digest,Error,Result,TokenInput};
use laya_candle::encoder::ModernBert;
use laya_candle::{Activation,Attention,EncoderLayer,Linear,Norm};
use serde::{Deserialize,Serialize};
use std::collections::BTreeMap;

type CResult<T> = candle_core::Result<T>;

/// Prompt contract of the checkpoint, from its own `core/formatting.py`:
/// `<<LABEL>>desc1<<LABEL>>desc2<<SEP>>Question: ...\n\nContext:\n...`, wrapped by
/// the tokenizer's template as `[CLS] ... [SEP]`.
pub const LABEL_MARKER:&str="<<LABEL>>";
pub const SEP_MARKER:&str="<<SEP>>";
/// `[CLS]` is added by the template, so the text side of the head is this position.
pub const CLS_POSITION:usize=0;
/// Hard architectural bound for one forward pass. The canister profile applies a
/// tighter cap at its own API boundary; this one only stops absurd inputs.
pub const MAX_SEQUENCE:usize=1024;

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct VerdictConfig {
    pub vocab_size:usize,pub hidden_size:usize,pub layers:usize,pub attention_heads:usize,
    pub intermediate_size:usize,pub norm_eps:f64,pub global_every:usize,pub local_attention:usize,
    pub global_rope_theta:f64,pub local_rope_theta:f64,pub first_layer_attention_norm:bool,
    pub cls_token_id:u32,pub class_token_id:u32,pub max_classes:usize,
    pub projector_activation:Activation,
}
impl VerdictConfig {
    pub fn validate(&self)->Result<()> {
        let h=self.hidden_size;
        if h==0 || h>2048 || self.vocab_size<5 || self.vocab_size>300_000 || self.layers==0 || self.layers>64
            || self.attention_heads==0 || h%self.attention_heads!=0 || (h/self.attention_heads)%2!=0
            || self.intermediate_size==0 || self.intermediate_size>16384
            || self.global_every==0 || self.local_attention==0
            || self.max_classes==0 || self.max_classes>64
            || self.cls_token_id as usize>=self.vocab_size || self.class_token_id as usize>=self.vocab_size
            || self.cls_token_id==self.class_token_id {
            return Err(Error::Invalid("unsupported verdict config".into()));
        }
        if !self.norm_eps.is_finite() || self.norm_eps<=0.0 || self.norm_eps>1.0 {return Err(Error::Numeric);}
        for v in [self.global_rope_theta,self.local_rope_theta]{if !v.is_finite() || v<=1.0 || v>1e12{return Err(Error::Numeric);}}
        Ok(())
    }
    /// GLiClass's `max_num_classes` fixes the number of logit slots. A request that
    /// needs more classes is refused rather than silently truncated.
    pub fn check_class_count(&self,count:usize)->Result<()> {
        if count==0 {return Err(Error::Invalid("no class tokens".into()));}
        if count>self.max_classes {return Err(Error::TooLong);}
        Ok(())
    }
}

/// Canonical tensor names. The encoder half is byte-identical to the Laya pack
/// naming, so one exporter can serve both backends.
pub fn expected_tensors(c:&VerdictConfig)->Result<BTreeMap<String,Vec<usize>>> {
    c.validate()?;let mut m=BTreeMap::new();let h=c.hidden_size;
    m.insert("embeddings.weight".into(),vec![c.vocab_size,h]);
    m.insert("embeddings.norm.weight".into(),vec![h]);
    m.insert("final_norm.weight".into(),vec![h]);
    for i in 0..c.layers {
        let p=format!("encoder.{i}");
        if i>0 || c.first_layer_attention_norm {m.insert(format!("{p}.attn_norm.weight"),vec![h]);}
        for (name,shape) in [("qkv.weight",vec![3*h,h]),("out.weight",vec![h,h]),("mlp_norm.weight",vec![h]),
                             ("wi.weight",vec![2*c.intermediate_size,h]),("wo.weight",vec![h,c.intermediate_size])] {
            m.insert(format!("{p}.{name}"),shape);
        }
    }
    for p in ["text_projector","classes_projector"] {
        for (name,shape) in [("linear_1.weight",vec![h,h]),("linear_1.bias",vec![h]),("linear_2.weight",vec![h,h]),("linear_2.bias",vec![h])] {
            m.insert(format!("{p}.{name}"),shape);
        }
    }
    Ok(m)
}

fn tensor(m:&BTreeMap<String,Tensor>,name:&str)->Result<Tensor>{m.get(name).cloned().ok_or_else(||Error::Invalid(format!("missing tensor: {name}")))}
fn linear(m:&BTreeMap<String,Tensor>,p:&str)->Result<Linear>{
    Ok(Linear::new(tensor(m,&format!("{p}.weight"))?,Some(tensor(m,&format!("{p}.bias"))?)))
}
fn norm(m:&BTreeMap<String,Tensor>,p:&str,eps:f64)->Result<Norm>{Ok(Norm::new(tensor(m,&format!("{p}.weight"))?,None,eps))}

pub struct VerdictModel {
    pub config:VerdictConfig,pub bundle:Digest,pub backend_kind:BackendKind,
    encoder:ModernBert,text_1:Linear,text_2:Linear,class_1:Linear,class_2:Linear,
}
fn activate(x:&Tensor,kind:Activation)->CResult<Tensor>{match kind{Activation::Relu=>x.relu(),Activation::Gelu=>x.gelu_erf()}}
impl VerdictModel {
    pub fn from_tensors(c:VerdictConfig,bundle:Digest,kind:BackendKind,m:BTreeMap<String,Tensor>)->Result<Self>{
        let expected=expected_tensors(&c)?;
        if expected.len()!=m.len(){return Err(Error::Invalid("unexpected tensor set".into()));}
        for (name,shape) in &expected {
            let t=m.get(name).ok_or_else(||Error::Invalid(format!("missing {name}")))?;
            if t.dims()!=shape.as_slice() || t.dtype()!=DType::F32{return Err(Error::Invalid(format!("shape/dtype: {name}")));}
        }
        let mut layers=Vec::new();
        for i in 0..c.layers {
            let p=format!("encoder.{i}");let local=i%c.global_every!=0;
            let attention_norm=if i>0 || c.first_layer_attention_norm {Some(norm(&m,&format!("{p}.attn_norm"),c.norm_eps)?)} else {None};
            layers.push(EncoderLayer::new(
                attention_norm,
                Attention::new(linear(&m,&format!("{p}.qkv"))?,linear(&m,&format!("{p}.out"))?,c.attention_heads),
                norm(&m,&format!("{p}.mlp_norm"),c.norm_eps)?,
                linear(&m,&format!("{p}.wi"))?,linear(&m,&format!("{p}.wo"))?,
                if local{c.local_rope_theta}else{c.global_rope_theta},
                if local{Some(c.local_attention/2)}else{None},
            ));
        }
        let encoder=ModernBert::new(tensor(&m,"embeddings.weight")?,norm(&m,"embeddings.norm",c.norm_eps)?,layers,norm(&m,"final_norm",c.norm_eps)?);
        Ok(Self{text_1:linear(&m,"text_projector.linear_1")?,text_2:linear(&m,"text_projector.linear_2")?,
            class_1:linear(&m,"classes_projector.linear_1")?,class_2:linear(&m,"classes_projector.linear_2")?,
            encoder,config:c,bundle,backend_kind:kind})
    }
    /// Positions of the `<<LABEL>>` tokens, in order. These are the class slots.
    pub fn class_positions(&self,ids:&[u32])->Vec<u32>{
        ids.iter().enumerate().filter(|(_,id)|**id==self.config.class_token_id).map(|(i,_)|i as u32).collect()
    }
    fn hidden(&self,ids:&[u32])->Result<Tensor>{
        if ids.is_empty() || ids.len()>MAX_SEQUENCE {return Err(Error::Invalid("token input".into()));}
        if ids.iter().any(|&v|v as usize>=self.config.vocab_size){return Err(Error::Invalid("token id".into()));}
        let t=Tensor::from_vec(ids.to_vec(),ids.len(),&Device::Cpu).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        self.encoder.forward(&t).map_err(|e|Error::ModelUnavailable(e.to_string()))
    }
    /// Encoder output `[tokens, hidden]`, exposed for parity tooling.
    pub fn encode(&self,ids:&[u32])->Result<Vec<f32>>{
        let h=self.hidden(ids)?;let (t,d)=h.dims2().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        h.flatten_all().and_then(|x|x.to_vec1::<f32>()).map(|v|{debug_assert_eq!(v.len(),t*d);v}).map_err(|e|Error::ModelUnavailable(e.to_string()))
    }
    fn projector(one:&Linear,two:&Linear,act:Activation,x:&Tensor)->CResult<Tensor>{
        two.forward(&activate(&one.forward(x)?,act)?)
    }
    /// Option logits, one per `<<LABEL>>` token, in order of appearance.
    ///
    /// `scorer_type="simple"` is a dot product between the projected `[CLS]` state
    /// and each projected class state. No softmax, no temperature and no
    /// `logit_scale` here: those belong to the calibration layer.
    pub fn logits(&self,ids:&[u32])->Result<Vec<f32>>{
        let positions=self.class_positions(ids);
        self.config.check_class_count(positions.len())?;
        let h=self.hidden(ids)?;
        let tlen=h.dim(0).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if tlen!=ids.len(){return Err(Error::Invalid("encoder output shape".into()));}
        let cls=h.narrow(0,CLS_POSITION,1).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let text=Self::projector(&self.text_1,&self.text_2,self.config.projector_activation,&cls)
            .map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let idx=Tensor::from_vec(positions.clone(),positions.len(),&Device::Cpu).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let selected=h.index_select(&idx,0).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let classes=Self::projector(&self.class_1,&self.class_2,self.config.projector_activation,&selected)
            .map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let logits=classes.broadcast_mul(&text).and_then(|x|x.sum(D::Minus1)).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let v=logits.to_vec1::<f32>().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if v.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}
        Ok(v)
    }
}

impl InferenceBackend for VerdictModel {
    fn bundle_id(&self)->Digest{self.bundle}
    fn kind(&self)->BackendKind{self.backend_kind}
    /// `markers` are ignored: GLiClass derives its class slots from the token ids
    /// themselves, so a caller cannot claim a class position that is not a
    /// `<<LABEL>>` token. The returned vector therefore has one entry per label.
    fn infer(&mut self,input:&TokenInput)->Result<Vec<f32>>{self.logits(&input.input_ids)}
}

/// Reference-side prompt renderer, mirroring the checkpoint's `core/formatting.py`.
/// Kept here so the canister and the parity tool cannot drift from each other.
pub fn render_prompt(question:&str,context:&str,labels:&[String])->String{
    let prefix: String = labels.iter().map(|l|format!("{LABEL_MARKER}{l}")).collect();
    let text=if question.is_empty(){context.to_string()}else{format!("Question: {question}\n\nContext:\n{context}")};
    format!("{prefix}{SEP_MARKER}{text}")
}
