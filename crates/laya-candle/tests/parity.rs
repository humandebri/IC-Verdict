use ic_laya_core::{engine::InferenceBackend,TokenInput,BackendKind,Error};
use std::path::PathBuf;
fn dir(name:&str)->PathBuf{PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name)}
/// Tolerance for the only numeric comparison `cargo test` runs by default.
///
/// The previous value was `1e-3 + 1e-3*|b|`. The fixture logits are ~6e-3, so that
/// absolute term was 13-17% of the value and 33-43x the largest *within-case class
/// spread* (2.3e-5..3.0e-5). A tolerance above the class spread cannot distinguish a
/// correct encoder from one whose axes are wrong: interleaved RoPE, a
/// `local_attention`-wide window, swapped GeGLU operands or `1/sqrt(hidden)` scaling
/// all shift the logits by 1e-6..2e-4, i.e. they passed.
///
/// Measured on arm64 (2026-09-21) against both fixtures: max |Rust - fixture| =
/// 1.118e-8. `5e-7` leaves ~45x headroom for a different SIMD kernel on x86_64 CI
/// while staying ~47x below the smallest class spread, and the assertion in the test
/// keeps it that way if the fixtures are ever regenerated.
const ATOL:f32=5e-7;
const RTOL:f32=1e-5;
fn spread(expected:&[f32])->f32{
    let max=expected.iter().copied().fold(f32::NEG_INFINITY,f32::max);
    let min=expected.iter().copied().fold(f32::INFINITY,f32::min);
    max-min
}
#[test]
fn synthetic_pytorch_logits_match_candle(){
    for folder in ["tiny-prenorm","tiny-postnorm"]{
        let d=dir(folder);let mut model=laya_candle::pack::load_directory(&d).unwrap();
        assert_eq!(model.kind(),BackendKind::SyntheticFixture);
        let cases:serde_json::Value=serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
        for case in cases.as_array().unwrap(){
            let input:TokenInput=serde_json::from_value(case["input"].clone()).unwrap();
            let expected:Vec<f32>=serde_json::from_value(case["expected_logits"].clone()).unwrap();
            let output=model.infer(&input).unwrap();assert_eq!(expected.len(),output.len());
            // Fail loudly if the fixture is regenerated with a smaller spread: a
            // tolerance this wide would make the rest of this test unable to detect
            // the axis errors it exists for.
            let spread=spread(&expected);
            assert!(expected.iter().all(|b|ATOL+RTOL*b.abs()<spread/10.0),
                "{folder}: tolerance {:e}+{:e}*|b| is not discriminating (class spread {spread:e})",ATOL,RTOL);
            for (a,b) in output.iter().zip(expected.iter()) {
                assert!((a-b).abs()<=ATOL+RTOL*b.abs(),"{folder}: {a} versus {b}");}
        }
    }
}
#[test]
fn corrupted_tensor_rejected(){
    let d=dir("tiny-prenorm");let raw=std::fs::read(d.join("manifest.json")).unwrap();let mut builder=laya_candle::pack::Builder::new(&raw).unwrap();
    let e=builder.next_entry().unwrap().clone();let all=std::fs::read(d.join("model.bin")).unwrap();
    let mut bytes=all[e.offset as usize..(e.offset+e.length) as usize].to_vec();bytes[0]^=1;assert!(builder.push(&bytes).is_err());
}
#[test]
fn rejects_missing_option_markers(){let mut model=laya_candle::pack::load_directory(&dir("tiny-prenorm")).unwrap();let bad=TokenInput{input_ids:vec![1,5,6,2],markers:vec![1,2],qtype_id:0};assert!(model.infer(&bad).is_err());}
#[test]
fn rejects_unknown_qtype(){let mut model=laya_candle::pack::load_directory(&dir("tiny-prenorm")).unwrap();let bad=TokenInput{input_ids:vec![1,3,3,2],markers:vec![1,2],qtype_id:3};assert!(matches!(model.infer(&bad),Err(Error::Invalid(_))));}

#[test]
fn profiled_inference_matches_plain_inference(){
    // The profiler must be observation-only: identical logits, and the phase
    // costs must account for the whole call.
    use std::cell::Cell;
    for folder in ["tiny-prenorm","tiny-postnorm"]{
        let d=dir(folder);
        let mut plain=laya_candle::pack::load_directory(&d).unwrap();
        let mut profiled=laya_candle::pack::load_directory(&d).unwrap();
        let cases:serde_json::Value=serde_json::from_slice(&std::fs::read(d.join("cases.json")).unwrap()).unwrap();
        for case in cases.as_array().unwrap(){
            let input:ic_laya_core::TokenInput=serde_json::from_value(case["input"].clone()).unwrap();
            let expected=plain.infer(&input).unwrap();
            // A counter that advances by a fixed amount per read, so phase deltas
            // are non-zero and their sum is the total.
            let tick=Cell::new(0u64);
            let counter=||{tick.set(tick.get()+10);tick.get()};
            let phases=profiled.infer_profiled(&input,&counter).unwrap();
            let actual=profiled.infer(&input).unwrap();
            assert_eq!(expected,actual,"{folder}: profiled model diverged from plain inference");
            let names:Vec<&str>=phases.iter().map(|p|p.name).collect();
            assert_eq!(names,vec!["embedding","encoder","final_norm","decision","scorer","decode"],"{folder}");
            let total:u64=phases.iter().map(|p|p.instructions).sum();
            assert_eq!(total,60,"{folder}: phases must sum to the measured span");
            assert!(phases.iter().all(|p|p.instructions>0),"{folder}: every phase must consume instructions");
        }
    }
}
