//! Native inspection and parity tool for the openJev (GLiClass) pack.
//!
//! Not deployed and not compiled into any canister. It exists because the only
//! external evidence that this port is correct is the checkpoint author's own
//! recorded decisions: `check` re-runs those inputs through the Rust kernels and
//! compares the argmax and the calibrated top probability.
//!
//! Usage:
//!   verdict-infer ids    --pack DIR --ids 50281,50368,123,50282
//!   verdict-infer check  --pack DIR --tokenizer FILE --cases FILE.jsonl \
//!                        --predictions FILE.jsonl [--temp F] [--limit N]
//!                        [--min-cases N] [--min-argmax-ratio F] [--abstain-id ID] [--quiet]
//!   verdict-infer tokens --pack DIR --tokenizer FILE --case-json JSON \
//!                        [--tokens N] [--index I]
use hf_tokenizer::HfTokenizer;
use ic_laya_core::{schema::TextTokenizer,SpecialTokens};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;
use verdict_candle::{pack,render_prompt};

/// ModernBERT special ids as they appear in the shipped tokenizer.
const CLS:u32=50281;
const SEP:u32=50282;
const MASK:u32=50284;
const PAD:u32=50283;

fn usage()->&'static str{
    "usage:\n  verdict-infer ids   --pack DIR --ids 50281,50368,123,50282\n  verdict-infer check --pack DIR --tokenizer FILE --cases FILE.jsonl --predictions FILE.jsonl [--temp F] [--limit N] [--min-cases N] [--min-argmax-ratio F] [--abstain-id ID] [--quiet]\n  verdict-infer tokens --pack DIR --tokenizer FILE --case-json JSON [--tokens N] [--index I]"
}

/// Neutral filler for `tokens --tokens N`.
///
/// Ordinary words, deliberately not `[PAD]`: the sweep measures what a longer
/// *sequence* costs, and a pad id would change the sequence the canister sees.
/// They are inserted before the trailing `[SEP]`, so the class markers keep their
/// positions and the sequence still ends the way every measured forward pass does.
const NEUTRAL_FILLER:&[u32]=&[1234,5678,9012];

/// Grows `ids` to exactly `want` by inserting filler before the final token.
///
/// The filler block is inserted contiguously and truncated at the end, so the result
/// is the original sequence with a single insertion, never a rewritten tail. The
/// final id (`[SEP]`) is left in place.
fn pad_to(ids:&mut Vec<u32>,want:usize,filler:&[u32]){
    if ids.len()>=want || filler.is_empty() {return;}
    let insert=want-ids.len();
    let mut block=Vec::with_capacity(insert);
    while block.len()<insert {block.extend_from_slice(filler);}
    block.truncate(insert);
    let at=ids.len()-1;
    ids.splice(at..at,block);
}

/// One case's token sequence, built exactly as `check` builds it.
/// The sequence order matters and is not obvious: `[CLS]`, the rendered prompt,
/// then the trailing `[SEP]`. The trailing separator is *appended* here rather
/// than rendered into the prompt text, so `--tokens N` pads before it and never
/// after it — the last token stays `[SEP]`, as in every measured forward pass.
fn case_tokens(tok:&HfTokenizer,case:&Value)->Option<(Vec<u32>,Vec<String>)>{
    let question=case["question"].as_str().unwrap_or("");
    let text=case["text"].as_str().unwrap_or("");
    let candidates=case["candidates"].as_array()?;
    let labels:Vec<String>=candidates.iter().map(|c|c["description"].as_str().unwrap_or("").to_string()).collect();
    let prompt=render_prompt(question,text,&labels);
    let mut ids=vec![CLS];
    ids.extend(tok.encode_piece(&prompt).ok()?);
    ids.push(SEP);
    Some((ids,labels))
}

/// Raw dense-matmul throughput for the shapes this model actually uses.
///
/// Native, so the numbers are not ICP instructions: they are the *relative* cost of
/// one gemm against another, which is what tells a kernel problem from surrounding
/// overhead. The instruction counts come from the canister (`infer_profiled`); this
/// mode exists to say whether a phase's cost is in its matmuls or around them.
fn gemm_mode(opt:&dyn Fn(&str)->Option<String>)->Result<(),String>{
    let t:usize=opt("--tokens").unwrap_or_else(||"120".into()).parse().map_err(|_|"bad --tokens".to_string())?;
    let hidden:usize=opt("--hidden").unwrap_or_else(||"768".into()).parse().map_err(|_|"bad --hidden".to_string())?;
    let out:usize=opt("--out-features").unwrap_or_else(||"2304".into()).parse().map_err(|_|"bad --out-features".to_string())?;
    let repeat:usize=opt("--repeat").unwrap_or_else(||"20".into()).parse().unwrap_or(1);
    let device=candle_core::Device::Cpu;
    // Deterministic fill: this measures time, and a reproducible input makes a re-run
    // comparable. An RNG would add a dependency for no measurement value.
    let fill=|n:usize|->Vec<f32>{(0..n).map(|i|((i%1000) as f32)*0.001-0.5).collect()};
    let x=candle_core::Tensor::from_vec(fill(t*hidden),(t,hidden),&device).map_err(|e|format!("input: {e}"))?;
    let w=candle_core::Tensor::from_vec(fill(hidden*out),(hidden,out),&device).map_err(|e|format!("weights: {e}"))?;
    // Warm-up call: the first matmul pays for any lazily built kernel tables.
    let _=x.matmul(&w).map_err(|e|format!("gemm: {e}"))?;
    let started=std::time::Instant::now();
    for _ in 0..repeat.max(1) {
        let y=x.matmul(&w).map_err(|e|format!("gemm: {e}"))?;
        std::hint::black_box(y.flatten_all().and_then(|v|v.to_vec1::<f32>()).unwrap_or_default());
    }
    let ms=started.elapsed().as_secs_f64()*1000.0/repeat.max(1) as f64;
    let macs=(t*hidden*out) as f64;
    println!("gemm m={t} n={out} k={hidden} macs={macs:.0} ms={ms:.2} gmac/s={:.2}",
        macs/(ms/1000.0)/1e9);
    Ok(())
}

fn softmax(xs:&[f32])->Vec<f32>{
    let m=xs.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
    let e:Vec<f32>=xs.iter().map(|x|(x-m).exp()).collect();
    let s:f32=e.iter().sum();
    e.iter().map(|x|x/s).collect()
}

fn quality_pass(cases:usize,matched:usize,missing:usize,_abstention_escape:usize,min_cases:usize,min_ratio:f64)->bool{
    cases>0 && cases>=min_cases && missing==0 && matched as f64/cases as f64>=min_ratio
}


fn main(){
    let args:Vec<String>=std::env::args().skip(1).collect();
    if args.len()<2 {eprintln!("{}",usage());std::process::exit(2);}
    let opt=|name:&str|->Option<String>{args.iter().position(|a|a==name).and_then(|i|args.get(i+1)).cloned()};
    let flag=|name:&str|->bool{args.iter().any(|a|a==name)};
    let die=|m:String|->!{eprintln!("error: {m}");std::process::exit(1)};
    // `gemm` benchmarks candle's matmul on a shape; it needs no weights, so the pack
    // is loaded only for the modes that actually read it.
    if args[0].as_str()=="gemm" { if let Err(e)=gemm_mode(&opt){die(e);} return; }
    let pack_dir=opt("--pack").unwrap_or_else(||die("--pack is required".into()));
    let model=pack::load_directory(Path::new(&pack_dir)).unwrap_or_else(|e|die(format!("pack: {e}")));
    match args[0].as_str() {
        "ids"=>{
            let raw=opt("--ids").unwrap_or_else(||die("--ids is required".into()));
            let ids:Vec<u32>=raw.split(',').filter(|s|!s.is_empty()).map(|s|s.trim().parse::<u32>().unwrap_or_else(|_|die(format!("bad id {s}")))).collect();
            let positions=model.class_positions(&ids);
            let repeat:usize=opt("--repeat").map(|v|v.parse().unwrap_or(1)).unwrap_or(1);
            let started=std::time::Instant::now();
            let mut logits=Vec::new();
            for _ in 0..repeat.max(1) {
                logits=model.logits(&ids).unwrap_or_else(|e|die(format!("infer: {e}")));
            }
            if repeat>1 {
                let ms=started.elapsed().as_secs_f64()*1000.0/repeat as f64;
                eprintln!("forward {ms:.1} ms/call over {repeat} calls (pack load excluded)");
            }
            let probs=softmax(&logits);
            println!("tokens={} class_positions={:?}",ids.len(),positions);
            for (i,(l,p)) in logits.iter().zip(probs.iter()).enumerate(){
                println!("class {i} pos {} logit {l:.6} p {p:.6}",positions[i]);
            }
        }
        "check"=>{
            let tokenizer=opt("--tokenizer").unwrap_or_else(||die("--tokenizer is required".into()));
            let cases=opt("--cases").unwrap_or_else(||die("--cases is required".into()));
            let predictions=opt("--predictions").unwrap_or_else(||die("--predictions is required".into()));
            let temp:f32=opt("--temp").unwrap_or_else(||"1.4265148639678955".into()).parse().unwrap_or_else(|_|die("bad --temp".into()));
            let limit:usize=opt("--limit").unwrap_or_else(||"1000000".into()).parse().unwrap_or_else(|_|die("bad --limit".into()));
            let offset:usize=opt("--offset").unwrap_or_else(||"0".into()).parse().unwrap_or_else(|_|die("bad --offset".into()));
            // Agreement is reported, not a deployment gate unless explicitly requested.
            let min_argmax_ratio:f64=opt("--min-argmax-ratio").unwrap_or_else(||"0.0".into()).parse().unwrap_or_else(|_|die("bad --min-argmax-ratio".into()));
            if !(0.0..=1.0).contains(&min_argmax_ratio){die("--min-argmax-ratio must be in 0..=1".into());}
            let min_cases:usize=opt("--min-cases").unwrap_or_else(||"1000".into()).parse().unwrap_or_else(|_|die("bad --min-cases".into()));
            let abstain_id=opt("--abstain-id").unwrap_or_else(||"__insufficient_evidence__".into());
            let quiet=flag("--quiet");
            let bytes=std::fs::read(&tokenizer).unwrap_or_else(|e|die(format!("tokenizer: {e}")));
            let special=SpecialTokens{cls:CLS,sep:SEP,mask:MASK,pad:PAD,
                literals:vec!["[CLS]".into(),"[SEP]".into(),"[MASK]".into(),"[PAD]".into()]};
            let tok=HfTokenizer::from_bytes(&bytes,special).unwrap_or_else(|e|die(format!("tokenizer: {e}")));
            let mut truth:BTreeMap<String,(String,f32)>=BTreeMap::new();
            for line in std::fs::read_to_string(&predictions).unwrap_or_else(|e|die(format!("predictions: {e}"))).lines(){
                if line.trim().is_empty(){continue;}
                let v:Value=serde_json::from_str(line).unwrap_or_else(|e|die(format!("predictions json: {e}")));
                if let (Some(id),Some(pid))=(v["id"].as_str(),v["predicted_id"].as_str()){
                    if truth.insert(id.into(),(pid.into(),v["confidence"].as_f64().unwrap_or(0.0) as f32)).is_some(){die(format!("duplicate prediction id: {id}"));}
                }
            }
            let (mut n,mut argmax_ok,mut unsafe_escape,mut max_dev,mut sum_dev)=(0usize,0usize,0usize,0f32,0f64);
            let (mut missing,mut longest)=(0usize,0usize);
            let mut seen=std::collections::BTreeSet::new();
            for line in std::fs::read_to_string(&cases).unwrap_or_else(|e|die(format!("cases: {e}"))).lines().skip(offset){
                if line.trim().is_empty(){continue;}
                if n>=limit {break;}
                let v:Value=serde_json::from_str(line).unwrap_or_else(|e|die(format!("cases json: {e}")));
                let id=match v["id"].as_str(){Some(x)=>x.to_string(),None=>die("case id missing".into())};
                if !seen.insert(id.clone()){die(format!("duplicate case id: {id}"));}
                let (want,want_p)=match truth.get(&id){Some(x)=>x.clone(),None=>{missing+=1;continue;}};
                let candidates=match v["candidates"].as_array(){Some(c)=>c,None=>{die(format!("{id}: no candidates"))}};
                let ids_c:Vec<String>=candidates.iter().map(|c|c["id"].as_str().unwrap_or("").to_string()).collect();
                let (input,labels)=match case_tokens(&tok,&v){Some(x)=>x,None=>die(format!("{id}: tokenize"))};
                longest=longest.max(input.len());
                let logits=model.logits(&input).unwrap_or_else(|e|die(format!("{id}: infer: {e}")));
                let scaled:Vec<f32>=logits.iter().map(|x|x/temp).collect();
                let probs=softmax(&scaled);
                let best=probs.iter().enumerate().fold((0usize,f32::NEG_INFINITY),|a,(i,p)|if *p>a.1{(i,*p)}else{a});
                let got=&ids_c[best.0];
                let ok=*got==want;
                if ok {argmax_ok+=1;}
                if want==abstain_id && *got!=abstain_id {unsafe_escape+=1;}
                let dev=(best.1-want_p).abs();
                max_dev=max_dev.max(dev);
                sum_dev+=dev as f64;
                if !ok && !quiet {
                    eprintln!("MISMATCH {id}: want {want} ({want_p:.6}) got {got} ({:.6}) tokens={} labels={}",best.1,input.len(),labels.len());
                }
                n+=1;
                if n%100==0 {
                    eprintln!("progress cases={n} argmax_match={argmax_ok} max_dev_so_far={max_dev:.6}");
                }
            }
            let mean=if n>0{sum_dev/n as f64}else{0.0};
            println!("cases={n} skipped_no_truth={missing} argmax_match={argmax_ok}/{n} ({:.2}%) longest_input_tokens={longest} temp={temp}",
                if n>0{100.0*argmax_ok as f64/n as f64}else{0.0});
            println!("top-probability |p-gold|: max={max_dev:.6} mean={mean:.6}");
            println!("quality-gate min_cases={min_cases} min_argmax_ratio={min_argmax_ratio:.4} unsafe_abstention_escape={unsafe_escape}");
            if !quality_pass(n,argmax_ok,missing,unsafe_escape,min_cases,min_argmax_ratio){std::process::exit(1);}
        }
        // Emits one real case's token ids so a token-length sweep can drive the
        // canister with whatever length the caller asks for, without reimplementing
        // the prompt contract or the tokenizer outside Rust.
        "tokens"=>{
            let tokenizer=opt("--tokenizer").unwrap_or_else(||die("--tokenizer is required".into()));
            let case=opt("--case-json").unwrap_or_else(||die("--case-json is required".into()));
            let bytes=std::fs::read(&tokenizer).unwrap_or_else(|e|die(format!("tokenizer: {e}")));
            let special=SpecialTokens{cls:CLS,sep:SEP,mask:MASK,pad:PAD,
                literals:vec!["[CLS]".into(),"[SEP]".into(),"[MASK]".into(),"[PAD]".into()]};
            let tok=HfTokenizer::from_bytes(&bytes,special).unwrap_or_else(|e|die(format!("tokenizer: {e}")));
            let v:Value=serde_json::from_str(&case).unwrap_or_else(|e|die(format!("case json: {e}")));
            let (mut ids,labels)=case_tokens(&tok,&v).unwrap_or_else(||die("tokenize failed".into()));
            let natural=ids.len();
            if let Some(want)=opt("--tokens") {
                let want:usize=want.parse().unwrap_or_else(|_|die("bad --tokens".into()));
                if want<natural {
                    die(format!("--tokens {want} truncates the real case ({natural} tokens); this tool only pads, so the measured call keeps the whole prompt"));
                }
                // Neutral filler, not [PAD]: the count is what is being measured, and
                // padding ids would change the sequence the canister validates.
                pad_to(&mut ids,want,NEUTRAL_FILLER);
            }
            let positions=model.class_positions(&ids);
            println!("tokens={} natural={} class_positions={:?} labels={}",
                ids.len(),natural,positions,labels.len());
            println!("ids={}",ids.iter().map(|i|i.to_string()).collect::<Vec<_>>().join(","));
        }
        other=>{eprintln!("unknown mode {other}\n{}",usage());std::process::exit(2);}
    }
}

#[cfg(test)]
mod tests{
    use super::quality_pass;
    #[test]
    fn quality_reports_drift_and_enforces_only_explicit_ratio(){
        assert!(quality_pass(1000,995,0,0,1000,0.995));
        assert!(!quality_pass(1000,994,0,0,1000,0.995));
        assert!(quality_pass(1000,997,0,1,1000,0.995));
        assert!(quality_pass(1000,994,0,3,1000,0.0));
        assert!(!quality_pass(999,999,0,0,1000,0.995));
        assert!(!quality_pass(1000,1000,1,0,1000,0.995));
        assert!(!quality_pass(0,0,0,0,0,0.));
    }
}
