//! Upload a canonical verdict pack to a canister over the IC HTTP interface.
//!
//! Why this exists: `docs/archive/PERFORMANCE_MEASUREMENTS.md` records that a 1.57 GiB pack
//! cannot be pushed through `icp canister call`, because Candid arguments have to
//! be escaped into argv (and this CLI version does not decode `--args-format
//! hex`/`bin` for a multi-argument method). An agent speaking the ingress protocol
//! directly has no such limit: one 1 MiB chunk per update call, the canister's own
//! 1 MiB cap, and nothing else.
//!
//! Usage:
//!   verdict-upload --url http://127.0.0.1:8000 --canister <principal> \
//!       --pem identity.pem --pack DIR --tokenizer FILE \
//!       --cls 50281 --sep 50282 --mask 50284 --pad 50283 [--chunk 1048576]
//!
//! The identity must be the canister's owner: `begin_upload`, `upload_chunk` and
//! the warm-up calls are owner-only.
use candid::{Decode, Encode, Principal};
use ic_agent::identity::BasicIdentity;
use ic_agent::Agent;
use ic_laya_core::SpecialTokens;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

fn validate_pack(manifest_raw:&[u8],model_path:&std::path::Path,tokenizer:&[u8])->Result<verdict_candle::pack::Manifest,String>{
    let manifest=verdict_candle::pack::Manifest::parse(manifest_raw).map_err(|e|format!("manifest: {e}"))?;
    if ic_laya_core::hash(tokenizer)!=manifest.tokenizer_sha256{return Err("tokenizer hash does not match manifest".into());}
    let mut file=std::fs::File::open(model_path).map_err(|e|format!("model.bin: {e}"))?;
    if file.metadata().map_err(|e|e.to_string())?.len()!=manifest.total_bytes{return Err("model.bin length does not match manifest".into());}
    for entry in &manifest.tensors{
        let mut bytes=vec![0;entry.length as usize];file.seek(SeekFrom::Start(entry.offset)).and_then(|_|file.read_exact(&mut bytes)).map_err(|e|format!("{}: {e}",entry.name))?;
        if ic_laya_core::hash(&bytes)!=entry.sha256{return Err(format!("{}: sha256 mismatch",entry.name));}
    }
    Ok(manifest)
}

struct Args {
    url: String,
    canister: String,
    pem: PathBuf,
    pack: Option<PathBuf>,
    tokenizer: Option<PathBuf>,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    chunk: usize,
    print_principal: bool,
    no_upload: bool,
    warm_only: bool,
    allow_caller: Option<String>,
    infer: Option<String>,
    query_infer: Option<String>,
    query_limits: bool,
    set_cost_model: Option<String>,
    profile: Option<String>,
    profile_detailed: bool,
    decide: bool,
    bench: Option<String>,
    quant: Option<String>,
    int8: Option<String>,
    f16: Option<String>,
    decide_batch: bool,
    decide_many: u32,
}

#[derive(candid::Deserialize, candid::CandidType)]
struct OptionSpec { id: String, text: String }
/// Mirror of the canister's `DecideRequest`.
#[derive(candid::Deserialize, candid::CandidType)]
struct DecideRequest { state: String, question: String, options: Vec<OptionSpec>, abstention: bool, temperature: f64 }
/// Mirror of the canister's `DecideReply`.
#[derive(candid::Deserialize, candid::CandidType)]
struct DecideReply {
    model: Vec<u8>, ids: Vec<String>, logits: Vec<f32>, probabilities: Vec<f32>,
    selected: String, confidence: f32, input_tokens: u32, measured_instructions: u64,
}
#[derive(candid::Deserialize, candid::CandidType)]
struct BatchQuestion { id: String, question: String, options: Vec<OptionSpec>, abstention: bool }
#[derive(candid::Deserialize, candid::CandidType)]
struct BatchRequest { state: String, questions: Vec<BatchQuestion>, temperature: f64 }
#[derive(candid::Deserialize, candid::CandidType)]
struct QuestionResult {
    id: String, ids: Vec<String>, logits: Vec<f32>, probabilities: Vec<f32>,
    selected: String, confidence: f32,
}
#[derive(candid::Deserialize, candid::CandidType)]
struct BatchReply {
    model: Vec<u8>, questions: Vec<QuestionResult>, input_tokens: u32, measured_instructions: u64,
}
/// Mirror of the canister's `F16BenchReply`.
#[derive(candid::Deserialize, candid::CandidType)]
struct F16BenchReply {
    m: u32, n: u32, k: u32, iterations: u32, convert_instructions: u64, instructions: u64,
    per_iteration: u64, instructions_per_mac: f64, max_abs_diff_vs_f32: f32,
}
/// Mirror of the canister's `Int8BenchReply`.
#[derive(candid::Deserialize, candid::CandidType)]
struct Int8BenchReply {
    m: u32, n: u32, k: u32, iterations: u32, simd_used: bool, instructions: u64,
    per_iteration: u64, instructions_per_mac: f64, quantize_weights_instructions: u64,
    quantize_activations_instructions: u64, max_abs_diff_vs_f32: f32, max_rel_diff_vs_f32: f32,
}
/// Mirror of the canister's `QuantBenchReply`.
#[derive(candid::Deserialize, candid::CandidType)]
struct QuantBenchReply {
    m: u32, n: u32, k: u32, iterations: u32, dtype: String, quantize_instructions: u64,
    instructions: u64, per_iteration: u64, instructions_per_mac: f64, max_abs_diff_vs_f32: f32,
}
/// Mirror of the canister's `BenchReply`.
#[derive(candid::Deserialize, candid::CandidType)]
struct BenchReply {
    m: u32, n: u32, k: u32, iterations: u32, macs: u64, instructions: u64,
    per_iteration: u64, instructions_per_mac: f64,
}
/// Mirror of the canister's `InferReply`. Candid hashes field names, so the
/// struct does not have to come from the canister crate.
#[derive(candid::Deserialize, candid::CandidType)]
struct InferReply {
    model: Vec<u8>,
    class_positions: Vec<u32>,
    logits: Vec<f32>,
    input_tokens: u32,
    measured_instructions: u64,
}

/// Mirror of the canister's `ProfileReply` / `PhaseCostDto`.
#[derive(candid::Deserialize, candid::CandidType)]
struct PhaseCost {
    name: String,
    instructions: u64,
}

#[derive(candid::Deserialize, candid::CandidType)]
struct ProfileReply {
    model: Vec<u8>,
    input_tokens: u32,
    measured_instructions: u64,
    phases: Vec<PhaseCost>,
}

/// Mirror of the canister's `QueryLimits`: what the 5B query path accepts, and the cost
/// model that produced that ceiling (so a recorded measurement stays readable).
#[derive(candid::Deserialize, candid::CandidType)]
struct QueryLimits {
    budget: u64,
    margin_permille: u64,
    max_tokens: u32,
    max_input_tokens: u32,
    cost_fixed: u64,
    cost_per_token: u64,
}

#[derive(candid::Deserialize, candid::CandidType)]
struct EngineBudget { budget: u64 }

/// Read `query_limits` over a query call.
async fn read_query_limits(agent: &Agent, canister: &Principal) -> Result<QueryLimits, String> {
    let raw = agent
        .query(canister, "query_limits")
        .call()
        .await
        .map_err(|e| format!("query_limits: {e}"))?;
    Decode!(&raw, QueryLimits).map_err(|e| format!("query_limits reply: {e}"))
}

async fn read_info(agent: &Agent, canister: &Principal) -> Result<EngineBudget, String> {
    let raw = agent.query(canister, "info").call().await.map_err(|e| format!("info: {e}"))?;
    Decode!(&raw, EngineBudget).map_err(|e| format!("info reply: {e}"))
}

fn parse_args() -> Result<Args, String> {
    let mut url = "http://127.0.0.1:8000".to_string();
    let mut canister = None;
    let mut pem = None;
    let mut pack = None;
    let mut tokenizer = None;
    let (mut cls, mut sep, mut mask, mut pad) = (50281u32, 50282u32, 50284u32, 50283u32);
    let mut chunk = 1024 * 1024usize;
    let mut print_principal = false;
    let mut no_upload = false;
    let mut warm_only = false;
    let mut allow_caller = None;
    let mut infer = None;
    let mut query_infer = None;
    let mut query_limits = false;
    let mut set_cost_model = None;
    let mut decide = false;
    let mut bench = None;
    let mut quant = None;
    let mut int8 = None;
    let mut f16 = None;
    let mut decide_batch = false;
    let mut decide_many = 0u32;
    let mut profile = None;
    let mut profile_detailed = false;
    let mut rest = std::env::args().skip(1);
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--url" => url = value()?,
            "--canister" => canister = Some(value()?),
            "--pem" => pem = Some(PathBuf::from(value()?)),
            "--pack" => pack = Some(PathBuf::from(value()?)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value()?)),
            "--cls" => cls = value()?.parse().map_err(|e| format!("--cls: {e}"))?,
            "--sep" => sep = value()?.parse().map_err(|e| format!("--sep: {e}"))?,
            "--mask" => mask = value()?.parse().map_err(|e| format!("--mask: {e}"))?,
            "--pad" => pad = value()?.parse().map_err(|e| format!("--pad: {e}"))?,
            "--chunk" => chunk = value()?.parse().map_err(|e| format!("--chunk: {e}"))?,
            "--print-principal" => print_principal = true,
            "--no-upload" => no_upload = true,
            "--warm-only" => warm_only = true,
            "--allow-caller" => allow_caller = Some(value()?),
            "--infer" => infer = Some(value()?),
            "--query-infer" => query_infer = Some(value()?),
            "--query-limits" => query_limits = true,
            "--set-cost-model" => set_cost_model = Some(value()?),
            "--decide" => decide = true,
            "--bench" => bench = Some(value()?),
            "--quant" => quant = Some(value()?),
            "--int8" => int8 = Some(value()?),
            "--f16" => f16 = Some(value()?),
            "--decide-batch" => decide_batch = true,
            "--decide-many" => decide_many = value()?.parse().unwrap_or(0),
            "--profile" => profile = Some(value()?),
            "--profile-detailed" => profile_detailed = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        url,
        canister: canister.unwrap_or_default(),
        pem: pem.ok_or("--pem is required")?,
        pack,
        tokenizer,
        cls,
        sep,
        mask,
        pad,
        chunk: chunk.clamp(1024, 1024 * 1024),
        print_principal,
        no_upload,
        warm_only,
        allow_caller,
        infer,
        query_infer,
        query_limits,
        set_cost_model,
        profile,
        profile_detailed,
        decide,
        bench,
        quant,
        int8,
        f16,
        decide_batch,
        decide_many,
    })
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("FAILED: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let identity = BasicIdentity::from_pem_file(&args.pem).map_err(|e| format!("pem: {e}"))?;
    if args.print_principal {
        use ic_agent::Identity;
        println!("{}", identity.sender().map_err(|e| format!("sender: {e}"))?);
        return Ok(());
    }
    if (args.pack.is_some() || args.tokenizer.is_some()) && (args.pack.is_none() || args.tokenizer.is_none()) {
        return Err("--pack and --tokenizer must be given together".into());
    }
    if args.no_upload && (args.pack.is_some() || args.tokenizer.is_some()) {
        return Err("--no-upload does not take --pack/--tokenizer".into());
    }
    // `--no-upload` reuses a canister that is already warm, so the pack is only
    // read when there is something to push.
    let mut pack_inputs = None;
    if !args.no_upload && !args.warm_only {
        let manifest = std::fs::read(args.pack.clone().ok_or("--pack is required")?.join("manifest.json"))
            .map_err(|e| format!("manifest: {e}"))?;
        let model_path=args.pack.clone().ok_or("--pack is required")?.join("model.bin");
        let model_length=std::fs::metadata(&model_path).map_err(|e|format!("model.bin: {e}"))?.len();
        let tokenizer = std::fs::read(args.tokenizer.clone().ok_or("--tokenizer is required")?)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let meta=validate_pack(&manifest,&model_path,&tokenizer)?;
        println!(
            "pack: manifest={} model={} tokenizer={} test_only={}",
            manifest.len(),
            model_length,
            tokenizer.len(),
            meta.test_only
        );
        pack_inputs = Some((manifest, model_path, model_length, tokenizer));
    }

    let agent = Agent::builder()
        .with_url(&args.url)
        .with_identity(identity)
        .build()
        .map_err(|e| format!("agent: {e}"))?;
    if args.url.starts_with("http://") {
        agent.fetch_root_key().await.map_err(|e| format!("root key: {e}"))?;
    }
    let canister = Principal::from_text(&args.canister).map_err(|e| format!("canister principal: {e}"))?;

    let special = SpecialTokens {
        cls: args.cls,
        sep: args.sep,
        mask: args.mask,
        pad: args.pad,
        literals: vec!["[CLS]".into(), "[SEP]".into(), "[MASK]".into(), "[PAD]".into()],
    };

    if let Some((manifest, model_path, model_length, tokenizer)) = &pack_inputs {
        let reply = agent
            .update(&canister, "begin_upload")
            .with_arg(Encode!(&manifest, &(tokenizer.len() as u64), &special).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("begin_upload: {e}"))?;
        Decode!(&reply, Result<Vec<u8>, ic_laya_core::Error>)
            .map_err(|e|format!("begin_upload reply: {e}"))?
            .map_err(|e|format!("begin_upload rejected: {e:?}"))?;
        println!("begin_upload ok");

        let total=*model_length as usize+tokenizer.len();
        let mut model=std::fs::File::open(model_path).map_err(|e|format!("model.bin: {e}"))?;
        let started = Instant::now();
        let mut offset = 0usize;
        let mut next_report = 0usize;
        while offset < total {
            let end = (offset + args.chunk).min(total);
            let mut chunk=Vec::with_capacity(end-offset);
            if offset<*model_length as usize {
                let model_end=end.min(*model_length as usize);
                model.seek(SeekFrom::Start(offset as u64)).map_err(|e|format!("model seek: {e}"))?;
                chunk.resize(model_end-offset,0);
                model.read_exact(&mut chunk).map_err(|e|format!("model read: {e}"))?;
            }
            if end>*model_length as usize {
                let begin=offset.max(*model_length as usize)-*model_length as usize;
                let finish=end-*model_length as usize;
                chunk.extend_from_slice(&tokenizer[begin..finish]);
            }
            let reply = agent
                .update(&canister, "upload_chunk")
                .with_arg(
                    Encode!(&(offset as u64), &chunk).map_err(|e| e.to_string())?,
                )
                .call_and_wait()
                .await
                .map_err(|e| format!("upload_chunk at {offset}: {e}"))?;
            let confirmed = candid::Decode!(&reply, Result<u64, ic_laya_core::Error>)
                .map_err(|e| format!("upload_chunk reply: {e}"))?
                .map_err(|e| format!("canister rejected chunk at {offset}: {e:?}"))?;
            offset = confirmed as usize;
            if offset >= next_report || offset == total {
                let mib = total as f64 / 1048576.0;
                println!(
                    "  {:.1} / {:.1} MiB ({:.1}s)",
                    offset as f64 / 1048576.0,
                    mib,
                    started.elapsed().as_secs_f64()
                );
                next_report = offset + 32 * 1024 * 1024;
            }
        }
        println!("uploaded {offset} bytes in {:.1}s", started.elapsed().as_secs_f64());

        let reply = agent
            .update(&canister, "start_warmup")
            .with_arg(Encode!().map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("start_warmup: {e}"))?;
        let total = candid::Decode!(&reply, Result<u64, ic_laya_core::Error>)
            .map_err(|e| format!("start_warmup reply: {e}"))?
            .map_err(|e| format!("start_warmup rejected: {e:?}"))?;
        println!("warm-up: {total} tensors");
        for index in 0..total {
            let reply = agent
                .update(&canister, "warmup_next")
                .with_arg(Encode!().map_err(|e| e.to_string())?)
                .call_and_wait()
                .await
                .map_err(|e| format!("warmup_next {index}: {e}"))?;
            let done = candid::Decode!(&reply, Result<bool, ic_laya_core::Error>)
                .map_err(|e| format!("warmup_next reply: {e}"))?
                .map_err(|e| format!("warmup_next rejected at {index}: {e:?}"))?;
            if done {
                println!("  model resident after {} of {total} tensors", index + 1);
                break;
            }
            if index % 25 == 24 {
                println!("  {} / {total}", index + 1);
            }
        }
        println!("warm-up complete");
    }

    if args.warm_only {
        println!("warm-only: using pack already stored by the canister");
        let reply=agent.update(&canister,"start_warmup").with_arg(Encode!().map_err(|e|e.to_string())?)
            .call_and_wait().await.map_err(|e|format!("start_warmup: {e}"))?;
        let total=Decode!(&reply,Result<u64,ic_laya_core::Error>).map_err(|e|format!("start_warmup reply: {e}"))?
            .map_err(|e|format!("start_warmup rejected: {e:?}"))?;
        for _ in 0..total {
            let reply=agent.update(&canister,"warmup_next").with_arg(Encode!().map_err(|e|e.to_string())?)
                .call_and_wait().await.map_err(|e|format!("warmup_next: {e}"))?;
            if Decode!(&reply,Result<bool,ic_laya_core::Error>).map_err(|e|format!("warmup_next reply: {e}"))?
                .map_err(|e|format!("warmup_next rejected: {e:?}"))? {break;}
        }
    }

    if let Some(caller) = &args.allow_caller {
        let principal = Principal::from_text(caller).map_err(|e| format!("caller principal: {e}"))?;
        let reply=agent
            .update(&canister, "allow_caller")
            .with_arg(Encode!(&principal).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("allow_caller: {e}"))?;
        Decode!(&reply,Result<(),ic_laya_core::Error>).map_err(|e|format!("allow_caller reply: {e}"))?
            .map_err(|e|format!("allow_caller rejected: {e:?}"))?;
        println!("allowed caller {caller}");
    }

    // Owner-only. The budget guard's slope is a measurement, not a constant: after
    // measuring a build (int8, a faster kernel) the owner installs the new fit and the
    // derived query ceiling moves with it. Read `query-limits` afterwards to confirm.
    if let Some(spec) = &args.set_cost_model {
        let parts: Vec<u64> = spec
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u64>().map_err(|e| format!("cost model {part}: {e}")))
            .collect::<Result<_, _>>()?;
        if parts.len() != 3 {
            return Err("--set-cost-model needs fixed,per_token,budget".into());
        }
        let reply=agent
            .update(&canister, "set_cost_model")
            .with_arg(Encode!(&parts[0], &parts[1], &parts[2]).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("set_cost_model: {e}"))?;
        Decode!(&reply,Result<(),ic_laya_core::Error>).map_err(|e|format!("set_cost_model reply: {e}"))?
            .map_err(|e|format!("set_cost_model rejected: {e:?}"))?;
        // Assert instead of announcing: a canister-side `Err(Invalid(...))` still comes back
        // as a successful transport call, so the only proof that the model changed is
        // reading it back. `budget` is the field that cannot coincide by accident.
        let observed = read_query_limits(&agent, &canister).await?;
        let info=read_info(&agent,&canister).await?;
        if info.budget != parts[2] || observed.cost_fixed != parts[0]
            || observed.cost_per_token != parts[1]
        {
            return Err(format!(
                "set_cost_model did not take effect: asked fixed={} per_token={} budget={}, \
                 canister reports fixed={} per_token={} budget={}",
                parts[0], parts[1], parts[2],
                observed.cost_fixed, observed.cost_per_token, info.budget));
        }
        println!(
            "cost model set to fixed={} per_token={} budget={} (max query tokens now {})",
            parts[0], parts[1], parts[2], observed.max_tokens);
    }

    if let Some(ids) = &args.infer {
        let tokens: Vec<u32> = ids
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().map_err(|e| format!("id {part}: {e}")))
            .collect::<Result<_, _>>()?;
        let reply = agent
            .update(&canister, "infer_tokens")
            .with_arg(Encode!(&tokens).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("infer_tokens: {e}"))?;
        let reply = Decode!(&reply, Result<InferReply, ic_laya_core::Error>)
            .map_err(|e| format!("infer_tokens reply: {e}"))?;
        let reply = match reply {
            Ok(reply) => reply,
            Err(error) => {
                // `Capacity` is the pre-flight budget guard refusing before spending
                // anything; a machine-readable line keeps `measure_verdict.py` from
                // classifying that by matching English text.
                println!("REJECTED {error:?}");
                return Err(format!("infer_tokens rejected: {error:?}"));
            }
        };
        println!(
            "tokens={} class_positions={:?}",
            reply.input_tokens, reply.class_positions
        );
        for (index, logit) in reply.logits.iter().enumerate() {
            println!("  class {index} logit {logit:.6}");
        }
        println!("MEASURED_INSTRUCTIONS {} tokens={}", reply.measured_instructions, reply.input_tokens);
    }

    // The query path. Both classifiers are printed before failing so callers (and
    // `tools/measure_verdict.py`) never have to parse English error text to tell
    // "the canister's budget guard refused this" from "the replica cut the call off".
    if let Some(ids) = &args.query_infer {
        let tokens: Vec<u32> = ids
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().map_err(|e| format!("id {part}: {e}")))
            .collect::<Result<_, _>>()?;
        let raw = match agent
            .query(&canister, "infer_tokens_query")
            .with_arg(Encode!(&tokens).map_err(|e| e.to_string())?)
            .call()
            .await
        {
            Ok(raw) => raw,
            Err(error) => {
                println!("TRAPPED {error}");
                return Err(format!("infer_tokens_query: {error}"));
            }
        };
        let reply = Decode!(&raw, Result<InferReply, ic_laya_core::Error>)
            .map_err(|e| format!("infer_tokens_query reply: {e}"))?;
        let reply = match reply {
            Ok(reply) => reply,
            Err(error) => {
                println!("REJECTED {error:?}");
                return Err(format!("infer_tokens_query rejected: {error:?}"));
            }
        };
        println!(
            "QUERY tokens={} class_positions={:?}",
            reply.input_tokens, reply.class_positions
        );
        for (index, logit) in reply.logits.iter().enumerate() {
            println!("  class {index} logit {logit:.6}");
        }
        println!("MEASURED_INSTRUCTIONS {} tokens={}", reply.measured_instructions, reply.input_tokens);
    }

    if args.query_limits {
        let limits = read_query_limits(&agent, &canister).await?;
        println!(
            "QUERY_BUDGET {} MAX_TOKENS {} MARGIN {} MAX_INPUT {} COST_FIXED {} PER_TOKEN {}",
            limits.budget, limits.max_tokens, limits.margin_permille, limits.max_input_tokens,
            limits.cost_fixed, limits.cost_per_token
        );
    }

    if let Some(ids) = &args.profile {
        let tokens: Vec<u32> = ids
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| part.trim().parse::<u32>().map_err(|e| format!("id {part}: {e}")))
            .collect::<Result<_, _>>()?;
        let reply = agent
            .update(&canister, "infer_profiled")
            .with_arg(Encode!(&tokens, &args.profile_detailed).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("infer_profiled: {e}"))?;
        let reply = Decode!(&reply, Result<ProfileReply, ic_laya_core::Error>)
            .map_err(|e| format!("infer_profiled reply: {e}"))?
            .map_err(|e| format!("infer_profiled rejected: {e:?}"))?;
        // Repeated names (the detailed per-layer markers) are aggregated so the
        // output stays readable and the shares are computed over one total.
        let mut totals: Vec<(String, u64, usize)> = Vec::new();
        for phase in &reply.phases {
            match totals.iter_mut().find(|(name, _, _)| name == &phase.name) {
                Some(entry) => {
                    entry.1 += phase.instructions;
                    entry.2 += 1;
                }
                None => totals.push((phase.name.clone(), phase.instructions, 1)),
            }
        }
        println!("PROFILE tokens={} measured={}", reply.input_tokens, reply.measured_instructions);
        for (name, instructions, count) in &totals {
            let share = if reply.measured_instructions == 0 {
                0.0
            } else {
                100.0 * *instructions as f64 / reply.measured_instructions as f64
            };
            let repeats = if *count > 1 { format!(" x{count}") } else { String::new() };
            println!("  PHASE {name:<16} {instructions:>14} {share:5.1}%{repeats}");
        }
    }

    if let Some(shape) = &args.bench {
        let parts: Vec<u64> = shape.split(',').map(|v| v.trim().parse().unwrap_or(0)).collect();
        if parts.len() != 4 { return Err("--bench wants m,n,k,iterations".into()); }
        let reply = agent
            .update(&canister, "bench_matmul")
            .with_arg(Encode!(&(parts[0] as u32), &(parts[1] as u32), &(parts[2] as u32), &(parts[3] as u32))
                .map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("bench_matmul: {e}"))?;
        let reply = Decode!(&reply, Result<BenchReply, ic_laya_core::Error>)
            .map_err(|e| format!("bench reply: {e}"))?
            .map_err(|e| format!("bench rejected: {e:?}"))?;
        println!(
            "BENCH m={} n={} k={} iters={} macs={} instructions={} per_iteration={} instr_per_mac={:.3}",
            reply.m, reply.n, reply.k, reply.iterations, reply.macs, reply.instructions,
            reply.per_iteration, reply.instructions_per_mac
        );
    }

    if let Some(spec) = &args.quant {
        // --quant m,n,k,iterations,q8_0
        let parts: Vec<&str> = spec.split(',').map(|v| v.trim()).collect();
        if parts.len() != 5 { return Err("--quant wants m,n,k,iterations,dtype".into()); }
        let dtype = parts[4].to_string();
        let reply = agent
            .update(&canister, "bench_qmatmul")
            .with_arg(Encode!(&parts[0].parse::<u32>().unwrap_or(0), &parts[1].parse::<u32>().unwrap_or(0),
                              &parts[2].parse::<u32>().unwrap_or(0), &parts[3].parse::<u32>().unwrap_or(0), &dtype)
                .map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("bench_qmatmul: {e}"))?;
        let reply = Decode!(&reply, Result<QuantBenchReply, ic_laya_core::Error>)
            .map_err(|e| format!("bench_qmatmul reply: {e}"))?
            .map_err(|e| format!("bench_qmatmul rejected: {e:?}"))?;
        println!(
            "QUANT dtype={} m={} n={} k={} iters={} instr_per_mac={:.3} quantize_instr={} max_abs_diff_vs_f32={:.4}",
            reply.dtype, reply.m, reply.n, reply.k, reply.iterations, reply.instructions_per_mac,
            reply.quantize_instructions, reply.max_abs_diff_vs_f32
        );
    }

    if let Some(spec) = &args.int8 {
        let parts: Vec<&str> = spec.split(',').map(|v| v.trim()).collect();
        if parts.len() != 4 { return Err("--int8 wants m,n,k,iterations".into()); }
        let reply = agent
            .update(&canister, "bench_int8")
            .with_arg(Encode!(&parts[0].parse::<u32>().unwrap_or(0), &parts[1].parse::<u32>().unwrap_or(0),
                              &parts[2].parse::<u32>().unwrap_or(0), &parts[3].parse::<u32>().unwrap_or(0))
                .map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("bench_int8: {e}"))?;
        let reply = Decode!(&reply, Result<Int8BenchReply, ic_laya_core::Error>)
            .map_err(|e| format!("bench_int8 reply: {e}"))?
            .map_err(|e| format!("bench_int8 rejected: {e:?}"))?;
        println!(
            "INT8 m={} n={} k={} iters={} simd={} instr_per_mac={:.3} quant_w={} quant_a={} max_abs={:.4} max_rel={:.4}",
            reply.m, reply.n, reply.k, reply.iterations, reply.simd_used, reply.instructions_per_mac,
            reply.quantize_weights_instructions, reply.quantize_activations_instructions,
            reply.max_abs_diff_vs_f32, reply.max_rel_diff_vs_f32
        );
    }

    if let Some(spec) = &args.f16 {
        let parts: Vec<&str> = spec.split(',').map(|v| v.trim()).collect();
        if parts.len() != 4 { return Err("--f16 wants m,n,k,iterations".into()); }
        let reply = agent
            .update(&canister, "bench_f16")
            .with_arg(Encode!(&parts[0].parse::<u32>().unwrap_or(0), &parts[1].parse::<u32>().unwrap_or(0),
                              &parts[2].parse::<u32>().unwrap_or(0), &parts[3].parse::<u32>().unwrap_or(0))
                .map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("bench_f16: {e}"))?;
        let reply = Decode!(&reply, Result<F16BenchReply, ic_laya_core::Error>)
            .map_err(|e| format!("bench_f16 reply: {e}"))?
            .map_err(|e| format!("bench_f16 rejected: {e:?}"))?;
        println!("F16 m={} n={} k={} iters={} instr_per_mac={:.3} convert={} max_abs={:.4}",
            reply.m, reply.n, reply.k, reply.iterations, reply.instructions_per_mac,
            reply.convert_instructions, reply.max_abs_diff_vs_f32);
    }

    // Three questions over one state: one batched pass vs the same three sent
    // separately. This is the measured design lever (-45.8% at 22 layers).
    if args.decide_batch || args.decide_many > 0 {
        let state = "I lost my wallet yesterday and need to stop my debit card immediately.".to_string();
        let questions = vec![
            ("dept", "Which team should handle this?", "billing", "charges invoices refunds", "technical", "bugs outages errors"),
            ("refund", "Does the customer request a refund?", "yes", "asks for a refund", "no", "no refund is requested"),
            ("urgency", "How urgent is this request?", "low", "not urgent", "high", "blocking or deadline"),
        ];
        if args.decide_many > 0 {
            let mut total = 0u64;
            for (id, q, a, ad, b, bd) in &questions {
                let request = DecideRequest {
                    state: state.clone(), question: (*q).to_string(),
                    options: vec![
                        OptionSpec { id: (*a).into(), text: (*ad).into() },
                        OptionSpec { id: (*b).into(), text: (*bd).into() },
                    ],
                    abstention: true, temperature: 1.4265148639678955,
                };
                let reply = agent.update(&canister, "decide")
                    .with_arg(Encode!(&request).map_err(|e| e.to_string())?)
                    .call_and_wait().await.map_err(|e| format!("decide {id}: {e}"))?;
                let reply = Decode!(&reply, Result<DecideReply, ic_laya_core::Error>)
                    .map_err(|e| format!("decide reply: {e}"))?.map_err(|e| format!("decide rejected: {e:?}"))?;
                println!("  separate {id}: {} {:.4} tokens={} instr={}", reply.selected, reply.confidence,
                    reply.input_tokens, reply.measured_instructions);
                total += reply.measured_instructions;
            }
            println!("SEPARATE_TOTAL {total}");
        }
        if args.decide_batch {
            let request = BatchRequest {
                state: state.clone(),
                questions: questions.iter().map(|(id,q,a,ad,b,bd)| BatchQuestion {
                    id: (*id).into(), question: (*q).into(), abstention: true,
                    options: vec![
                        OptionSpec { id: (*a).into(), text: (*ad).into() },
                        OptionSpec { id: (*b).into(), text: (*bd).into() },
                    ],
                }).collect(),
                temperature: 1.4265148639678955,
            };
            let reply = agent.update(&canister, "decide_batch")
                .with_arg(Encode!(&request).map_err(|e| e.to_string())?)
                .call_and_wait().await.map_err(|e| format!("decide_batch: {e}"))?;
            let reply = Decode!(&reply, Result<BatchReply, ic_laya_core::Error>)
                .map_err(|e| format!("decide_batch reply: {e}"))?.map_err(|e| format!("decide_batch rejected: {e:?}"))?;
            for q in &reply.questions {
                println!("  batched  {}: {} {:.4}", q.id, q.selected, q.confidence);
            }
            println!("BATCHED tokens={} instr={}", reply.input_tokens, reply.measured_instructions);
        }
    }

    if args.decide {
        // A short, real-weights decision: the point is that the typed-decision path
        // (tokenize -> render -> forward -> softmax) completes inside the budget.
        let request = DecideRequest {
            state: "I lost my wallet yesterday and need to stop my debit card immediately.".into(),
            question: "What is the primary customer inquiry?".into(),
            options: vec![
                OptionSpec { id: "card_lost".into(), text: "Reporting a lost or stolen card".into() },
                OptionSpec { id: "dispute_charge".into(), text: "Disputing an unrecognized charge".into() },
            ],
            abstention: true,
            temperature: 1.4265148639678955,
        };
        let reply = agent
            .update(&canister, "decide")
            .with_arg(Encode!(&request).map_err(|e| e.to_string())?)
            .call_and_wait()
            .await
            .map_err(|e| format!("decide: {e}"))?;
        let reply = Decode!(&reply, Result<DecideReply, ic_laya_core::Error>)
            .map_err(|e| format!("decide reply: {e}"))?
            .map_err(|e| format!("decide rejected: {e:?}"))?;
        println!("DECIDE selected={} confidence={:.6} tokens={}", reply.selected, reply.confidence, reply.input_tokens);
        for (index, id) in reply.ids.iter().enumerate() {
            let logit = reply.logits.get(index).copied().unwrap_or(f32::NAN);
            let probability = reply.probabilities.get(index).copied().unwrap_or(f32::NAN);
            println!("  {id:28} logit {logit:9.4} p {probability:.6}");
        }
        println!("MEASURED_INSTRUCTIONS {} tokens={}", reply.measured_instructions, reply.input_tokens);
    }
    Ok(())
}
