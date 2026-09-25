//! Local JSONL probe: frozen weights, no training or network. One request per line.
use std::io::{self,BufRead,Write};
use std::path::Path;
use hf_tokenizer::HfTokenizer;
use ic_laya_core::{schema::TextTokenizer,SpecialTokens};
use serde_json::{Value,json};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<String>=std::env::args().collect();
    let root=args.get(1).map(String::as_str).unwrap_or(".");
    let bytes=std::fs::read(Path::new(root).join("models/verdict-151m/tokenizer.json"))?;
    let tok=HfTokenizer::from_bytes(&bytes,SpecialTokens{cls:50281,sep:50282,mask:50284,pad:50283,literals:vec!["[CLS]".into(),"[SEP]".into(),"[MASK]".into(),"[PAD]".into()]}).map_err(|e|format!("{e:?}"))?;
    let mut model=None;
    for line in io::stdin().lock().lines() {
        let v:Value=serde_json::from_str(&line?)?;
        if let Some(text)=v["raw"].as_str(){
            let ids=tok.encode_piece(text).map_err(|e|format!("{e:?}"))?;
            println!("{}",json!({"tokens":ids.len(),"ids":ids}));io::stdout().flush()?;continue;
        }
        let labels:Vec<String>=serde_json::from_value(v["labels"].clone())?;
        let prompt=verdict_candle::render_prompt("",v["text"].as_str().ok_or("missing text")?,&labels);
        let mut ids=vec![50281];ids.extend(tok.encode_piece(&prompt).map_err(|e|format!("{e:?}"))?);ids.push(50282);
        let mut out=json!({"tokens":ids.len(),"prompt":prompt,"ids":ids});
        if v["infer"].as_bool()==Some(true) {
            if ids.len()>52 {return Err("refusing inference over 52 tokens".into());}
            if model.is_none(){model=Some(verdict_candle::pack::load_directory(&Path::new(root).join("models/verdict-pack")).map_err(|e|format!("{e:?}"))?);}
            let logits=model.as_ref().unwrap().logits(&ids).map_err(|e|format!("{e:?}"))?;
            let max=logits.iter().copied().fold(f32::NEG_INFINITY,f32::max);
            let exp:Vec<f32>=logits.iter().map(|x|(x-max).exp()).collect();let sum:f32=exp.iter().sum();
            let scores:Vec<f32>=exp.iter().map(|x|x/sum).collect();
            let selected=scores.iter().enumerate().fold(0,|best,(i,s)|if *s>scores[best]{i}else{best});
            out["scores"]=json!(scores);out["selected"]=json!(selected);
        }
        println!("{out}");io::stdout().flush()?;
    }
    Ok(())
}
