use ic_laya_core::hash;
use modernbert_candle::Activation;
use safetensors::{Dtype,SafeTensors};
use serde_json::Value;
use std::collections::{BTreeMap,BTreeSet};
use std::io::Write;
use std::path::PathBuf;
use verdict_candle::{expected_tensors,is_matrix_weight,pack::{Encoding,Manifest,TensorEntry,FORMAT},VerdictConfig};

struct Args{source:PathBuf,config:PathBuf,tokenizer:PathBuf,out:PathBuf,repo:String,revision:String,test:bool,random:bool}
fn args()->Result<Args,String>{
    let mut it=std::env::args().skip(1);let mut m=BTreeMap::new();let mut test=false;let mut random=false;
    while let Some(k)=it.next(){match k.as_str(){"--test"=>test=true,"--random"=>random=true,_=>{m.insert(k,it.next().ok_or("missing argument value")?);}}}
    let take=|k:&str|m.get(k).cloned().ok_or_else(||format!("{k} is required"));
    Ok(Args{source:take("--safetensors")?.into(),config:take("--config")?.into(),tokenizer:take("--tokenizer")?.into(),out:take("--out")?.into(),
        repo:m.get("--source-repo").cloned().unwrap_or_else(||"heman10x/rlcd-modernbert-151m".into()),revision:m.get("--source-revision").cloned().unwrap_or_default(),test,random})
}
fn usize_at(v:&Value,path:&[&str])->Result<usize,String>{let mut x=v;for k in path{x=&x[*k];}x.as_u64().map(|n|n as usize).ok_or_else(||format!("missing integer {}",path.join(".")))}
fn f64_or(v:&Value,path:&[&str],default:f64)->f64{let mut x=v;for k in path{x=&x[*k];}x.as_f64().unwrap_or(default)}
fn config(v:&Value)->Result<VerdictConfig,String>{
    let enc=&v["encoder_config"];
    let activation=match v["projector_hidden_act"].as_str().unwrap_or("gelu"){"gelu"=>Activation::Gelu,"relu"=>Activation::Relu,x=>return Err(format!("unsupported projector_hidden_act {x}"))};
    if v["normalize_features"].as_bool().unwrap_or(false){return Err("normalize_features=true is unsupported".into());}
    let global_every=usize_at(v,&["encoder_config","global_attn_every_n_layers"]).unwrap_or(3);
    if let Some(types)=enc["layer_types"].as_array(){for (i,t) in types.iter().enumerate(){let want=if i%global_every==0{"full_attention"}else{"sliding_attention"};if t.as_str()!=Some(want){return Err("layer_types disagrees with global_attn_every_n_layers".into());}}}
    let c=VerdictConfig{vocab_size:usize_at(v,&["encoder_config","vocab_size"])?,hidden_size:usize_at(v,&["encoder_config","hidden_size"])?,
        layers:usize_at(v,&["encoder_config","num_hidden_layers"])?,attention_heads:usize_at(v,&["encoder_config","num_attention_heads"])?,
        intermediate_size:usize_at(v,&["encoder_config","intermediate_size"])?,norm_eps:f64_or(v,&["encoder_config","norm_eps"],f64_or(v,&["encoder_config","layer_norm_eps"],1e-5)),
        global_every,local_attention:usize_at(v,&["encoder_config","local_attention"]).unwrap_or(128),
        global_rope_theta:f64_or(v,&["encoder_config","rope_parameters","full_attention","rope_theta"],f64_or(v,&["encoder_config","global_rope_theta"],160000.0)),
        local_rope_theta:f64_or(v,&["encoder_config","rope_parameters","sliding_attention","rope_theta"],f64_or(v,&["encoder_config","local_rope_theta"],10000.0)),
        first_layer_attention_norm:false,cls_token_id:usize_at(v,&["encoder_config","cls_token_id"]).unwrap_or(50281) as u32,
        sep_token_id:usize_at(v,&["encoder_config","sep_token_id"]).unwrap_or(50282) as u32,class_token_id:usize_at(v,&["class_token_index"])? as u32,
        max_classes:usize_at(v,&["max_num_classes"]).unwrap_or(25),projector_activation:activation};
    c.validate().map_err(|e|e.to_string())?;Ok(c)
}
fn canonical(name:&str)->Option<String>{
    if name=="model.logit_scale"{return None;}
    if let Some(rest)=name.strip_prefix("model.encoder_model."){
        return match rest{"embeddings.tok_embeddings.weight"=>Some("embeddings.weight".into()),"embeddings.norm.weight"=>Some("embeddings.norm.weight".into()),"final_norm.weight"=>Some("final_norm.weight".into()),_=>{
            let mut p=rest.splitn(3,'.');if p.next()!=Some("layers"){return None;}let i=p.next()?;let tail=p.next()?;
            let mapped=match tail{"attn.Wqkv.weight"=>"qkv.weight","attn.Wo.weight"=>"out.weight","attn_norm.weight"=>"attn_norm.weight","mlp.Wi.weight"=>"wi.weight","mlp.Wo.weight"=>"wo.weight","mlp_norm.weight"=>"mlp_norm.weight",_=>return None};Some(format!("encoder.{i}.{mapped}"))}};
    }
    if let Some(rest)=name.strip_prefix("model."){if rest.starts_with("text_projector.")||rest.starts_with("classes_projector."){return Some(rest.into());}}
    None
}
fn floats(raw:&[u8])->Result<Vec<f32>,String>{
    if !raw.len().is_multiple_of(4){return Err("unaligned f32 payload".into());}
    raw.chunks_exact(4).map(|b|{let v=f32::from_le_bytes([b[0],b[1],b[2],b[3]]);if v.is_finite(){Ok(v)}else{Err("non-finite checkpoint value".into())}}).collect()
}
fn random_floats(n:usize)->Vec<f32>{let mut state=0x5eedu64;(0..n).map(|_|{state=state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);((((state>>33)&0xffff_ffff)as f32)/(0x7fff_ffffu32 as f32)-1.0)*0.4}).collect()}
fn payload(name:&str,shape:&[usize],values:&[f32])->(Encoding,Vec<u8>){
    if is_matrix_weight(name){let rows=shape[0];let cols=shape[1];let(q,s)=verdict_simd::quantize_rows_i8(values,rows,cols);let mut out=Vec::with_capacity(q.len()+4*s.len());out.extend(q.iter().map(|v|*v as u8));for v in s{out.extend_from_slice(&v.to_le_bytes());}(Encoding::I8RowSymmetric,out)}
    else{let mut out=Vec::with_capacity(values.len()*4);for v in values{out.extend_from_slice(&v.to_le_bytes());}(Encoding::F32Le,out)}
}
fn run()->Result<(),String>{
    let a=args()?;if a.random&&!a.test{return Err("--random requires --test".into());}
    let raw_cfg:Value=serde_json::from_slice(&std::fs::read(&a.config).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;let mut cfg=config(&raw_cfg)?;
    let source=if a.random{None}else{Some(std::fs::read(&a.source).map_err(|e|e.to_string())?)};
    let st=source.as_ref().map(|b|SafeTensors::deserialize(b).map_err(|e|e.to_string())).transpose()?;
    if let Some(st)=&st{cfg.first_layer_attention_norm=st.tensor("model.encoder_model.layers.0.attn_norm.weight").is_ok();}
    let expected=expected_tensors(&cfg).map_err(|e|e.to_string())?;
    let mut found=BTreeMap::<String,String>::new();let mut unmapped=BTreeSet::<String>::new();
    if let Some(st)=&st{for(name,view)in st.iter(){if name=="model.logit_scale"{continue;}
        if let Some(c)=canonical(name){if view.dtype()!=Dtype::F32{return Err(format!("{name}: expected F32"));}found.insert(c,name.into());}else{unmapped.insert(name.into());}}if !unmapped.is_empty(){return Err(format!("unmapped checkpoint tensors: {:?}",unmapped.iter().take(5).collect::<Vec<_>>()));}}
    if !a.random&&(found.len()!=expected.len()||expected.keys().any(|k|!found.contains_key(k))){return Err("checkpoint tensor set mismatch".into());}
    std::fs::create_dir_all(&a.out).map_err(|e|e.to_string())?;let mut file=std::fs::File::create(a.out.join("model.bin")).map_err(|e|e.to_string())?;
    let mut entries=Vec::new();let mut offset=0u64;
    for(name,shape)in &expected{let n=shape.iter().product::<usize>();let values=if a.random{random_floats(n)}else{let view=st.as_ref().unwrap().tensor(&found[name]).map_err(|e|e.to_string())?;if view.shape()!=shape{return Err(format!("{name}: shape mismatch"));}floats(view.data())?};if values.len()!=n{return Err(format!("{name}: length mismatch"));}let(encoding,bytes)=payload(name,shape,&values);file.write_all(&bytes).map_err(|e|e.to_string())?;entries.push(TensorEntry{name:name.clone(),shape:shape.clone(),encoding,offset,length:bytes.len()as u64,sha256:hash(&bytes)});offset+=bytes.len()as u64;}
    let revision=if a.revision.is_empty()&&a.test{"0".repeat(40)}else{a.revision};if !a.test&&(revision.len()!=40||!revision.bytes().all(|b|b.is_ascii_hexdigit())){return Err("real pack requires a 40-hex source revision".into());}
    let tokenizer=std::fs::read(&a.tokenizer).map_err(|e|e.to_string())?;let manifest=Manifest{format:FORMAT.into(),source_repo:a.repo,source_revision:revision,test_only:a.test,tokenizer_sha256:hash(&tokenizer),config:cfg,total_bytes:offset,tensors:entries};manifest.validate().map_err(|e|e.to_string())?;
    std::fs::write(a.out.join("manifest.json"),serde_json::to_vec_pretty(&manifest).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;println!("wrote {} tensors, {} bytes ({:.1} MiB), format={FORMAT}",manifest.tensors.len(),offset,offset as f64/1048576.0);Ok(())
}
fn main(){if let Err(e)=run(){eprintln!("error: {e}");std::process::exit(1)}}
