//! Canonical INT8 production pack for the openJev/GLiClass backend.
//!
//! Every two-dimensional model parameter is quantised offline with one scale per row.
//! Block-32 entries are rejected rather than silently selecting a slower kernel. One-dimensional
//! normalization weights and projector biases remain little-endian f32 auxiliaries.
use crate::{expected_tensors,is_matrix_weight,QuantMap,VerdictConfig,VerdictModel};
use candle_core::{Device,DType,Tensor};
use ic_laya_core::{hash,BackendKind,Digest,Error,Result};
use serde::{Deserialize,Serialize};
use std::collections::{BTreeMap,BTreeSet};

pub const FORMAT:&str="ic-verdict-int8-pack-v1";

#[derive(Debug,Clone,Copy,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="snake_case")]
pub enum Encoding { I8RowSymmetric,F32Le }

#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct TensorEntry {pub name:String,pub shape:Vec<usize>,pub encoding:Encoding,pub offset:u64,pub length:u64,pub sha256:Digest}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Manifest {
    pub format:String,pub source_repo:String,pub source_revision:String,pub test_only:bool,
    pub tokenizer_sha256:Digest,pub config:VerdictConfig,
    pub total_bytes:u64,pub tensors:Vec<TensorEntry>,
}
fn numel(shape:&[usize])->Result<u64>{shape.iter().try_fold(1u64,|a,&b|a.checked_mul(b as u64).ok_or(Error::TooLong))}
fn expected_encoding(name:&str)->Encoding{if is_matrix_weight(name){Encoding::I8RowSymmetric}else{Encoding::F32Le}}
fn expected_length(name:&str,shape:&[usize])->Result<u64>{
    let n=numel(shape)?;
    match expected_encoding(name){Encoding::I8RowSymmetric=>{
        let rows=match shape{[r,_]=>*r as u64,_=>return Err(Error::Invalid("matrix shape".into()))};
        n.checked_add(rows.checked_mul(4).ok_or(Error::TooLong)?).ok_or(Error::TooLong)
    },Encoding::F32Le=>n.checked_mul(4).ok_or(Error::TooLong)}
}
impl Manifest {
    pub fn parse(raw:&[u8])->Result<Self>{
        if raw.len()>2*1024*1024{return Err(Error::TooLong);}
        let m:Self=serde_json::from_slice(raw).map_err(|e|Error::Invalid(e.to_string()))?;m.validate()?;Ok(m)
    }
    pub fn validate(&self)->Result<()> {
        if self.format!=FORMAT || self.source_repo.is_empty() || self.source_revision.is_empty() || self.total_bytes==0 || self.total_bytes>2*1024*1024*1024{return Err(Error::Invalid("pack metadata".into()));}
        if !self.test_only&&(self.source_revision.len()!=40||!self.source_revision.bytes().all(|x|x.is_ascii_hexdigit())){return Err(Error::Invalid("a real pack requires an immutable source commit".into()));}
        let expected=expected_tensors(&self.config)?;if expected.len()!=self.tensors.len(){return Err(Error::Invalid("tensor count".into()));}
        let mut pos=0u64;let mut names=BTreeSet::new();
        for e in &self.tensors {
            if !names.insert(e.name.clone())||expected.get(&e.name)!=Some(&e.shape)||e.offset!=pos{return Err(Error::Invalid("tensor name/shape/offset".into()));}
            if e.encoding!=expected_encoding(&e.name){return Err(Error::Invalid(format!("wrong encoding: {}",e.name)));}
            let len=expected_length(&e.name,&e.shape)?;
            if e.length!=len||len>256*1024*1024{return Err(Error::TooLong);}pos=pos.checked_add(len).ok_or(Error::TooLong)?;
        }
        if pos!=self.total_bytes{return Err(Error::Invalid("pack length".into()));}Ok(())
    }
}
pub struct Builder {pub manifest:Manifest,pub bundle:Digest,pub next:usize,tensors:BTreeMap<String,Tensor>,quant:QuantMap}
impl Builder {
    pub fn new(raw:&[u8])->Result<Self>{Ok(Self{manifest:Manifest::parse(raw)?,bundle:hash(raw),next:0,tensors:BTreeMap::new(),quant:BTreeMap::new()})}
    pub fn next_entry(&self)->Option<&TensorEntry>{self.manifest.tensors.get(self.next)}
    pub fn push(&mut self,bytes:&[u8])->Result<()> {
        let e=self.next_entry().ok_or(Error::Transition)?.clone();
        if bytes.len() as u64!=e.length||hash(bytes)!=e.sha256{return Err(Error::Invalid("tensor integrity".into()));}
        match e.encoding {
            Encoding::I8RowSymmetric=>{
                let (rows,cols)=match e.shape.as_slice(){[r,c]=>(*r,*c),_=>return Err(Error::Invalid("quantised weight shape".into()))};
                let data_len=rows.checked_mul(cols).ok_or(Error::TooLong)?;
                let w=bytes[..data_len].iter().map(|&v|v as i8).collect::<Vec<_>>();
                if w.contains(&i8::MIN){return Err(Error::Numeric);}
                let scales=bytes[data_len..].chunks_exact(4).map(|b|f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect::<Vec<_>>();
                if scales.len()!=rows||scales.iter().any(|v|!v.is_finite()||*v<=0.0){return Err(Error::Numeric);}
                self.quant.insert(e.name,modernbert_candle::QuantWeight{w,scales,out_features:rows,in_features:cols,block_size:cols});
            }
            Encoding::F32Le=>{
                if bytes.chunks_exact(4).any(|b|!f32::from_le_bytes([b[0],b[1],b[2],b[3]]).is_finite()){return Err(Error::Numeric);}
                let tensor=Tensor::from_raw_buffer(bytes,DType::F32,&e.shape,&Device::Cpu).map_err(|x|Error::ModelUnavailable(x.to_string()))?;
                self.tensors.insert(e.name,tensor);
            }
        }
        self.next+=1;Ok(())
    }
    pub fn finish(self)->Result<VerdictModel>{
        if self.next!=self.manifest.tensors.len(){return Err(Error::Transition);}
        let kind=if self.manifest.test_only{BackendKind::SyntheticFixture}else{BackendKind::Checkpoint};
        VerdictModel::from_quantized(self.manifest.config,self.bundle,kind,self.tensors,self.quant)
    }
}
#[cfg(not(target_arch="wasm32"))]
pub fn load_directory(dir:&std::path::Path)->Result<VerdictModel>{
    use std::io::{Read,Seek,SeekFrom};
    let raw=std::fs::read(dir.join("manifest.json")).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let mut b=Builder::new(&raw)?;
    let mut f=std::fs::File::open(dir.join("model.bin")).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    if f.metadata().map_err(|e|Error::ModelUnavailable(e.to_string()))?.len()!=b.manifest.total_bytes{return Err(Error::Invalid("file length".into()));}
    while let Some(entry)=b.next_entry().cloned(){
        let mut bytes=vec![0u8;entry.length as usize];f.seek(SeekFrom::Start(entry.offset)).and_then(|_|f.read_exact(&mut bytes)).map_err(|e|Error::ModelUnavailable(e.to_string()))?;b.push(&bytes)?;
    }b.finish()
}
