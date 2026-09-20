#[cfg(not(target_arch="wasm32"))]
fn main()->Result<(),Box<dyn std::error::Error>>{
    use ic_laya_core::{engine::InferenceBackend,TokenInput};
    let args:Vec<String>=std::env::args().collect();
    if args.len()!=3{return Err("usage: laya-infer PACK_DIRECTORY TOKEN_INPUT_JSON".into());}
    let mut model=laya_candle::pack::load_directory(std::path::Path::new(&args[1]))?;
    let input:TokenInput=serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let logits=model.infer(&input)?;let mass=ic_laya_core::math::from_logits(&logits,1.0)?.into_vec();
    println!("{}",serde_json::to_string_pretty(&serde_json::json!({"backend":format!("{:?}",model.kind()),"bundle":ic_laya_core::hex(&model.bundle_id()),"raw_logits":logits,"uncalibrated_mass_ppm":mass}))?);Ok(())
}
#[cfg(target_arch="wasm32")]
fn main(){}
