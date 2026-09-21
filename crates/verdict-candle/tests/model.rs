//! Kernel-level checks for the GLiClass backend.
//!
//! These use a tiny deterministic config, so they prove the wiring, the head
//! arithmetic and the pack path — not checkpoint quality. Checkpoint parity is
//! `tests/golden.rs` (ignored by default: it needs the 605 MiB pack) and the
//! `verdict-infer check` run recorded in docs/VERDICT_ENGINE.md.
use candle_core::{DType,Device,Tensor};
use ic_laya_core::{hash,BackendKind,SpecialTokens};
use std::collections::BTreeMap;
use verdict_candle::{expected_tensors,pack,VerdictConfig,VerdictModel};

fn cfg()->VerdictConfig{
    VerdictConfig{
        vocab_size:64,hidden_size:16,layers:2,attention_heads:2,intermediate_size:24,
        norm_eps:1e-5,global_every:2,local_attention:4,
        global_rope_theta:160000.0,local_rope_theta:10000.0,first_layer_attention_norm:false,
        cls_token_id:1,sep_token_id:2,class_token_id:3,max_classes:4,
        projector_activation:modernbert_candle::Activation::Gelu,
    }
}

/// Deterministic pseudo-random weights: no `rand` dependency, and the same values
/// on every platform so a failure is reproducible.
struct Lcg(u64);
impl Lcg{
    fn next_f32(&mut self)->f32{
        self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v=((self.0>>33) as f32)/(u32::MAX as f32/2.0);
        v-1.0
    }
}

fn weights(c:&VerdictConfig)->BTreeMap<String,Tensor>{
    let mut rng=Lcg(0x5eed);
    let mut m=BTreeMap::new();
    for (name,shape) in expected_tensors(c).expect("expected"){
        let n:usize=shape.iter().product();
        let data:Vec<f32>=(0..n).map(|_|rng.next_f32()*0.4).collect();
        m.insert(name,Tensor::from_slice(&data,shape.clone(),&Device::Cpu).expect("tensor"));
    }
    m
}

fn model()->(VerdictModel,BTreeMap<String,Tensor>){
    let c=cfg();let w=weights(&c);
    let m=VerdictModel::from_tensors(c,[7u8;32],BackendKind::SyntheticFixture,w.clone()).expect("model");
    (m,w)
}

/// The profiled path must return exactly what the production path returns.
///
/// `logits_profiled` duplicates the forward pass so it can mark phase boundaries.
/// That duplication is the risk: if the two ever drift, every instruction number
/// measured through it would describe a model that is not the one serving
/// decisions. Comparing the outputs is what keeps the instrumentation honest.
#[test]
fn profiled_forward_returns_the_same_logits(){
    let (m,_)=model();
    let ids=vec![1,3,10,11,12,13,3,14,2];
    let plain=m.logits(&ids).expect("logits");
    let mut names:Vec<&'static str>=Vec::new();
    let profiled=m.logits_profiled(&ids,&mut |name|names.push(name),false).expect("profiled");
    assert_eq!(plain,profiled);
    // Coarse markers only: one per phase, and the encoder is marked exactly once
    // even though it contains every layer.
    assert_eq!(names,vec!["embedding","encoder","text_projector","class_projector","dot_decode"]);
}

/// Marking every encoder sub-phase must not change the result either, and the
/// detailed markers must be a superset that repeats per layer.
#[test]
fn detailed_profiling_only_adds_repeated_markers(){
    let (m,_)=model();
    let ids=vec![1,3,11,3,12,2];
    let plain=m.logits(&ids).expect("logits");
    let mut names:Vec<&'static str>=Vec::new();
    let profiled=m.logits_profiled(&ids,&mut |name|names.push(name),true).expect("profiled");
    assert_eq!(plain,profiled);
    assert!(names.contains(&"layer.attn"),"expected per-layer markers");
    let layers=names.iter().filter(|n|**n=="layer.attn").count();
    assert_eq!(layers,cfg().layers,"one marker per encoder layer");
}

/// The profiled path must refuse the same inputs the production path refuses.
#[test]
fn profiled_forward_keeps_the_same_rejections(){
    let (m,_)=model();
    let mut ignore=|_name:&'static str|{};
    assert!(m.logits_profiled(&[1,10,11,2],&mut ignore,false).is_err(),"no class token");
    assert!(m.logits_profiled(&[1,999_999,3,2],&mut ignore,false).is_err(),"out of vocab");
    assert!(m.logits_profiled(&[],&mut ignore,false).is_err(),"empty");
}

#[test]
fn logits_are_one_per_class_token_in_order(){
    let (m,_)=model();
    // [CLS] = 1, class tokens = 3, ordinary ids = 10.
    let ids=vec![1,3,10,11,3,12,3,2];
    let positions=m.class_positions(&ids);
    assert_eq!(positions,vec![1,4,6]);
    let logits=m.logits(&ids).expect("logits");
    assert_eq!(logits.len(),3);
    assert!(logits.iter().all(|x|x.is_finite()));
}

/// The head scores exactly the `<<LABEL>>` positions, in sequence order: a
/// duplicated marker yields two slots, and no other id contributes a slot.
#[test]
fn every_label_marker_contributes_one_slot_in_order(){
    let (m,_)=model();
    assert_eq!(m.class_positions(&[1,3,10,3,3,2]),vec![1,3,4]);
    assert_eq!(m.class_positions(&[1,10,11,2]),Vec::<u32>::new());
    assert!(m.logits(&[1,10,11,2]).is_err(),"no class token must be refused, not scored as zero slots");
}

#[test]
fn head_matches_an_independent_recomputation_from_the_encoder_output(){
    let (m,w)=model();
    let ids=vec![1,3,10,11,12,13,3,14,2];
    let logits=m.logits(&ids).expect("logits");
    // Recompute the head from the exposed encoder output and the same weights, with
    // the same int8 quantisation the model applies (the model's dense weights are
    // int8; see docs/VERDICT_ENGINE.md 5.1.6). Without this the comparison would be
    // testing "int8 vs f32" instead of the head wiring.
    // With `--features int8` the model quantises weights *and* activations per row, so
    // the reference has to do the same or the comparison measures quantisation error
    // rather than the wiring. Without the feature both are exact copies.
    #[cfg(not(feature = "int8"))]
    let dequant_acts=|t:&Tensor|->Tensor{t.clone()};
    #[cfg(feature = "int8")]
    let dequant_acts=|t:&Tensor|->Tensor{
        let dims=t.dims().to_vec();
        let k=*dims.last().expect("nonempty");
        let rows=dims[..dims.len()-1].iter().product::<usize>().max(1);
        let flat=t.flatten_all().expect("flat").to_vec1::<f32>().expect("vec");
        let (q,scales)=verdict_simd::quantize_acts_i16(&flat,rows,k);
        let out:Vec<f32>=q.iter().enumerate().map(|(i,&v)|(v as f32)*scales[i/k]).collect();
        Tensor::from_slice(&out,dims.as_slice(),&Device::Cpu).expect("dequant acts")
    };
    #[cfg(not(feature = "int8"))]
    let dequant=|t:&Tensor|->Tensor{t.clone()};
    #[cfg(feature = "int8")]
    let dequant=|t:&Tensor|->Tensor{
        let (rows,cols)=t.dims2().expect("2d");
        let flat=t.flatten_all().expect("flat").to_vec1::<f32>().expect("vec");
        let (q,scales)=verdict_simd::quantize_rows_i8(&flat,rows,cols);
        let out:Vec<f32>=q.iter().enumerate().map(|(i,&v)|(v as f32)*scales[i/cols]).collect();
        Tensor::from_slice(&out,(rows,cols),&Device::Cpu).expect("dequant")
    };
    let hidden=m.encode(&ids).expect("encode");
    let h=hidden.len()/ids.len();
    let cls=Tensor::from_slice(&hidden[0..h],(1,h),&Device::Cpu).expect("cls");
    let proj=|p:&str,x:&Tensor|->Tensor{
        let l1=dequant_acts(x).matmul(&dequant(&w[&format!("{p}.linear_1.weight")]).t().expect("t").contiguous().expect("c")).expect("l1")
            .broadcast_add(&w[&format!("{p}.linear_1.bias")]).expect("b1");
        let a=l1.gelu_erf().expect("act");
        dequant_acts(&a).matmul(&dequant(&w[&format!("{p}.linear_2.weight")]).t().expect("t").contiguous().expect("c")).expect("l2")
            .broadcast_add(&w[&format!("{p}.linear_2.bias")]).expect("b2")
    };
    let text=proj("text_projector",&cls);
    let positions=m.class_positions(&ids);
    let rows:Vec<f32>=positions.iter().flat_map(|&p|{let s=(p as usize)*h;hidden[s..s+h].to_vec()}).collect();
    let classes=Tensor::from_slice(&rows,(positions.len(),h),&Device::Cpu).expect("rows");
    let classes=proj("classes_projector",&classes);
    let expect=classes.broadcast_mul(&text).expect("mul").sum(candle_core::D::Minus1).expect("sum")
        .to_vec1::<f32>().expect("vec");
    for (a,b) in logits.iter().zip(expect.iter()){assert!((a-b).abs()<1e-4,"{a} vs {b}");}
}

#[test]
fn refuses_more_classes_than_the_checkpoint_scores(){
    let (m,_)=model();
    let ids=vec![1,3,3,3,3,3,2]; // five class tokens, max_classes = 4
    assert!(m.logits(&ids).is_err());
}

#[test]
fn refuses_non_finite_and_out_of_vocab_ids(){
    let (m,_)=model();
    assert!(m.logits(&[]).is_err());
    assert!(m.logits(&[1,9999,2]).is_err());
}

#[test]
fn same_input_gives_the_same_logits(){
    let (m,_)=model();
    let ids=vec![1,3,10,3,11,2];
    assert_eq!(m.logits(&ids).expect("a"),m.logits(&ids).expect("b"));
}

/// The canister path loads tensors one at a time through `pack::Builder`, so the
/// pack layout is part of the model contract, not a transport detail.
#[test]
fn pack_round_trip_reproduces_the_model(){
    let c=cfg();let w=weights(&c);
    let expected=expected_tensors(&c).expect("expected");
    let (tokenizer,tok_hash)=(b"TEST-ONLY-TOKENIZER".to_vec(),hash(b"TEST-ONLY-TOKENIZER"));
    let mut tensors=Vec::new();let mut offset=0u64;
    for (name,shape) in &expected {
        let bytes=w[name].flatten_all().expect("flat").to_vec1::<f32>().expect("vec")
            .iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>();
        tensors.push(pack::TensorEntry{name:name.clone(),shape:shape.clone(),offset,length:bytes.len() as u64,sha256:hash(&bytes)});
        offset+=bytes.len() as u64;
    }
    let manifest=pack::Manifest{
        format:pack::FORMAT.into(),source_repo:"local-test".into(),source_revision:"test".into(),test_only:true,
        tokenizer_sha256:tok_hash,config:c.clone(),total_bytes:offset,tensors,
    };
    let raw=serde_json::to_vec(&manifest).expect("manifest");
    let mut builder=pack::Builder::new(&raw).expect("builder");
    while let Some(entry)=builder.next_entry().cloned(){
        let bytes=w[&entry.name].flatten_all().expect("flat").to_vec1::<f32>().expect("vec")
            .iter().flat_map(|x|x.to_le_bytes()).collect::<Vec<u8>>();
        builder.push(&bytes).expect("push");
    }
    let loaded=builder.finish().expect("finish");
    let (direct,_)=model();
    let ids=vec![1,3,10,11,3,12,2];
    assert_eq!(loaded.logits(&ids).expect("loaded"),direct.logits(&ids).expect("direct"));
    assert_eq!(hash(&tokenizer),tok_hash);
}

/// A pack that claims a tensor the model does not expect, or a wrong shape, must
/// be refused before any tensor bytes are accepted.
#[test]
fn pack_rejects_tensor_set_mismatch(){
    let c=cfg();let expected=expected_tensors(&c).expect("expected");
    let mut tensors=Vec::new();let mut offset=0u64;
    for (name,shape) in &expected {
        let n:usize=shape.iter().product();
        let length=(n*4) as u64;
        let shape=if name=="embeddings.weight"{vec![shape[0],shape[1]+1]}else{shape.clone()};
        tensors.push(pack::TensorEntry{name:name.clone(),shape,offset,length,sha256:[0u8;32]});
        offset+=length;
    }
    let manifest=pack::Manifest{
        format:pack::FORMAT.into(),source_repo:"local-test".into(),source_revision:"test".into(),test_only:true,
        tokenizer_sha256:[0u8;32],config:c,total_bytes:offset,tensors,
    };
    let raw=serde_json::to_vec(&manifest).expect("manifest");
    assert!(pack::Builder::new(&raw).is_err());
    let _=DType::F32;
}

/// The tokenizer emits its added-token literals wherever they appear, so a state or
/// option text carrying `<<LABEL>>` changes the number of scored classes. `decide` on
/// the canister used to return two ids against three logits for such an input; the
/// checked renderer rejects it up front.
#[test]
fn reserved_token_literals_are_rejected_in_prompt_inputs() {
    let special = SpecialTokens {
        cls: 1, sep: 2, mask: 3, pad: 0,
        literals: vec!["[PAD]".into(), "[CLS]".into(), "[SEP]".into(), "[MASK]".into(),
                       "<<LABEL>>".into(), "<<SEP>>".into()],
    };
    let clean = vec!["alpha".to_string(), "beta".to_string()];
    assert!(verdict_candle::render_prompt_checked("Which one?", "a clean state", &clean, &special).is_ok());

    let injections = [
        ("state", "Which one?", "state <<LABEL>> injected", clean.clone()),
        ("label", "Which one?", "clean", vec!["<<SEP>>beta".to_string()]),
        ("question", "Which one? [MASK]", "clean", clean.clone()),
    ];
    for (field, question, state, labels) in injections {
        let got = verdict_candle::render_prompt_checked(question, state, &labels, &special);
        assert!(got.is_err(), "{field} must be rejected");
    }
}
