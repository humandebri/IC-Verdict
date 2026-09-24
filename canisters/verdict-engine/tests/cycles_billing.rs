//! Isolated real-Wasm tests; never connects to mainnet or an existing replica.
//! Build with tools/build_one.sh verdict-engine, then run with --ignored.
use candid::{CandidType, Decode, Encode, Principal};
use ic_laya_core::{Error, Result, SpecialTokens};
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use std::{fs, path::PathBuf};
use verdict_engine::{CyclesPricing, ExecutionPricing, InferReply};

#[derive(CandidType, Deserialize)]
struct LegacyPlacementQuery {
    board: Vec<u8>, seed: u32, turn: u32, lines: u32, mode: u8,
}

#[test]
#[ignore = "requires built Wasm and a local POCKET_IC_BIN"]
fn cycles_payment_queries_and_upgrade() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let wasm = fs::read(root.join("build/verdict-engine.wasm")).unwrap();
    let pic = PocketIcBuilder::new().with_application_subnet().build();
    let owner = Principal::self_authenticating(b"cycles-test-owner");
    let outsider = Principal::self_authenticating(b"cycles-test-outsider");
    let canister = pic.create_canister();
    let proxy = pic.create_canister();
    for id in [canister, proxy] { pic.add_cycles(id, 100_000_000_000_000); }
    pic.install_canister(proxy, wat::parse_str(include_str!("cycles_proxy.wat")).unwrap(), vec![], None);
    let old = std::env::var("VERDICT_OLD_WASM").ok().map(|path| fs::read(path).unwrap());
    pic.install_canister(canister, old.clone().unwrap_or_else(|| wasm.clone()), Encode!(&owner).unwrap(), None);
    let call = |sender, method: &str, args| pic.update_call(canister, sender, method, args).unwrap();
    let query = |sender, method: &str, args| pic.query_call(canister, sender, method, args).unwrap();
    let ok = |method: &str, args| { Decode!(&call(owner, method, args), Result<()>).unwrap().unwrap(); };
    ok("allow_caller", Encode!(&outsider).unwrap());
    ok("set_cost_model", Encode!(&1u64, &1u64, &40_000_000_000u64).unwrap());

    // Seed real model bytes in the existing blob region before either upgrade.
    let fixture = root.join("fixtures/verdict-tiny");
    let manifest = fs::read(fixture.join("manifest.json")).unwrap();
    let tokenizer = fs::read(fixture.join("tokenizer.json")).unwrap();
    let special = SpecialTokens { cls: 1, sep: 2, mask: 4, pad: 0,
        literals: vec!["[CLS]".into(), "[SEP]".into(), "[MASK]".into(), "[PAD]".into()] };
    Decode!(&call(owner, "begin_upload", Encode!(&manifest, &(tokenizer.len() as u64), &special).unwrap()), Result<[u8;32]>).unwrap().unwrap();
    let mut data = fs::read(fixture.join("model.bin")).unwrap();
    data.extend(tokenizer);
    Decode!(&call(owner, "upload_chunk", Encode!(&0u64, &data).unwrap()), Result<u64>).unwrap().unwrap();
    let warm = || {
        Decode!(&call(owner, "start_warmup", Encode!().unwrap()), Result<u64>).unwrap().unwrap();
        for _ in 0..200 {
            if Decode!(&call(owner, "warmup_next", Encode!().unwrap()), Result<bool>).unwrap().unwrap() { return; }
        }
        panic!("warmup did not finish");
    };
    warm();
    if old.is_some() {
        if std::env::var_os("CHECK_COMPAT_LEGACY_QUERY").is_some() {
            let prior:serde_json::Value=serde_json::from_slice(&fs::read(root.join("artifacts/tetris-prompt-search/production-query-check.json")).unwrap()).unwrap();
            let mut first=None;
            for row in prior["results"].as_array().unwrap() {
                let recorded=&row["receipt"];
                let selected=recorded["selected"].as_u64().unwrap() as usize;
                let placement = LegacyPlacementQuery { board: serde_json::from_value(recorded["before"].clone()).unwrap(),
                    seed: recorded["seed"].as_u64().unwrap() as u32, turn: recorded["turn"].as_u64().unwrap() as u32,
                    lines: (recorded["total_lines"].as_u64().unwrap()-recorded["candidates"][selected]["lines"].as_u64().unwrap()) as u32,
                    mode: 1 };
                let result = Decode!(&query(owner, "tetris_placement_query", Encode!(&placement).unwrap()), std::result::Result<String,String>).unwrap().unwrap();
                let parsed:serde_json::Value=serde_json::from_str(&result).unwrap();
                assert_eq!(parsed["candidates"], recorded["candidates"]);
                assert_eq!(parsed["legal_count"], recorded["legal_count"]);
                assert_eq!(parsed["piece"], recorded["piece"]);
                assert_eq!(parsed["next"], recorded["next"]);
                assert_eq!(parsed["rules"], recorded["rules"]);
                if first.is_none(){first=Some(placement);}
            }
            let placement = LegacyPlacementQuery { mode: 0, ..first.unwrap() };
            let result = Decode!(&query(Principal::anonymous(), "tetris_placement_query", Encode!(&placement).unwrap()), std::result::Result<String,String>).unwrap().unwrap();
            assert!(result.contains("\"input_tokens\""));
        }
        pic.upgrade_canister(canister, wasm.clone(), Encode!().unwrap(), None).unwrap();
        assert!(Decode!(&query(owner, "cycles_pricing", Encode!().unwrap()), Result<CyclesPricing>).unwrap().is_err());
        warm();
        println!("PASS old snapshot upgrade preserves owner, allowlist and model bytes");
    }

    let pricing = ExecutionPricing { base_cycles: 5_000_000,
        instruction_cycles_numerator: 1, instruction_cycles_denominator: 1 };
    assert_eq!(Decode!(&call(outsider, "set_execution_pricing", Encode!(&Some(pricing)).unwrap()), Result<()>).unwrap(), Err(Error::Unauthorized));
    assert!(Decode!(&call(owner, "set_execution_pricing", Encode!(&Some(ExecutionPricing {
        instruction_cycles_denominator: 0, ..pricing })).unwrap()), Result<()>).unwrap().is_err());
    ok("set_execution_pricing", Encode!(&Some(pricing)).unwrap());
    let quote = Decode!(&query(owner, "cycles_pricing", Encode!().unwrap()), Result<CyclesPricing>).unwrap().unwrap();
    assert_eq!(quote.multiplier, 3);
    assert_eq!(quote.required_attachment, 120_015_000_000);
    let deposit = quote.required_attachment as u64;
    let ids = vec![1u32, 3, 11, 3, 12, 2];
    let args = Encode!(&ids).unwrap();
    let forward = |method: &str, args: Vec<u8>, cycles: u64| {
        let mut request = Vec::new();
        request.extend((canister.as_slice().len() as u32).to_le_bytes());
        request.extend((method.len() as u32).to_le_bytes());
        request.extend(cycles.to_le_bytes());
        request.extend(canister.as_slice()); request.extend(method.as_bytes()); request.extend(args);
        pic.update_call(proxy, owner, "forward", request).unwrap()
    };
    let refund = || u128::from_le_bytes(pic.query_call(proxy, owner, "refund", vec![]).unwrap().try_into().unwrap());
    // Neither owner nor the existing allowlist bypasses the charge.
    for who in [owner, outsider, Principal::anonymous()] {
        assert_eq!(Decode!(&call(who, "infer_tokens", args.clone()), Result<InferReply>).unwrap().err(), Some(Error::Budget));
    }
    assert_eq!(Decode!(&forward("infer_tokens", args.clone(), deposit - 1), Result<InferReply>).unwrap().err(), Some(Error::Budget));
    assert_eq!(refund(), u128::from(deposit - 1));
    // Proxy was never allowlisted, but can purchase inference.
    let result = Decode!(&forward("infer_tokens", args.clone(), deposit), Result<InferReply>).unwrap().unwrap();
    let charged = u128::from(deposit) - refund();
    assert!(charged >= 3 * (5_000_000 + u128::from(result.measured_instructions)));
    assert!(charged < u128::from(deposit));
    assert_eq!(charged % 3, 0);
    println!("PASS paid inference: charged={charged}, returned={}", refund());
    // A completed error also pays only for the work actually performed.
    assert_eq!(Decode!(&forward("infer_tokens", Encode!(&Vec::<u32>::new()).unwrap(), deposit), Result<InferReply>).unwrap().err(), Some(Error::TooLong));
    assert!(refund() > 0 && refund() < u128::from(deposit));
    println!("PASS underpayment rejected with full refund; validation errors settle measured cost");

    let before = pic.cycle_balance(canister);
    Decode!(&query(outsider, "infer_tokens_query", args.clone()), Result<InferReply>).unwrap().unwrap();
    assert_eq!(pic.cycle_balance(canister), before);
    assert!(matches!(Decode!(&call(owner, "infer_tokens_query", args.clone()), Result<InferReply>).unwrap(), Err(Error::Denied(_))));
    let decision = verdict_engine::DecideRequest { state: "evidence".into(), question: "choose".into(),
        options: vec![verdict_engine::OptionSpec { id: "yes".into(), text: "yes".into() }],
        abstention: true, temperature: 1.0 };
    let decide_args = Encode!(&decision).unwrap();
    Decode!(&forward("decide", decide_args.clone(), deposit), Result<verdict_engine::DecideReply>).unwrap().unwrap();
    Decode!(&query(Principal::anonymous(), "decide_query", decide_args.clone()), Result<verdict_engine::DecideReply>).unwrap().unwrap();
    let mut oversized=decision.clone();oversized.question="q".repeat(4097);
    assert_eq!(Decode!(&query(Principal::anonymous(), "decide_query", Encode!(&oversized).unwrap()),
        Result<verdict_engine::DecideReply>).unwrap().err(),Some(Error::TooLong));
    assert!(matches!(Decode!(&call(owner, "decide_query", decide_args), Result<verdict_engine::DecideReply>).unwrap(), Err(Error::Denied(_))));
    let batch = verdict_engine::BatchRequest { state: "evidence".into(), temperature: 1.0,
        questions: vec![verdict_engine::BatchQuestion { id: "one".into(), question: "choose".into(),
            options: decision.options.clone(), abstention: true }] };
    Decode!(&forward("decide_batch", Encode!(&batch).unwrap(), deposit), Result<verdict_engine::BatchReply>).unwrap().unwrap();

    // The durable workflow keeps its authorization/quota contract, and a cached
    // receipt pays for the cache read rather than its historical inference count.
    let schema = ic_laya_core::Schema { id: "paid".into(), version: 1,
        primitive: ic_laya_core::Primitive::Noul, instructions: "choose".into(),
        options: vec![ic_laya_core::OptionDef { id: "false".into(), text: "false".into() },
            ic_laya_core::OptionDef { id: "true".into(), text: "true".into() }] };
    let compiled = Decode!(&call(owner, "register_schema", Encode!(&schema, &1u32).unwrap()), Result<ic_laya_core::CompiledSchema>).unwrap().unwrap();
    let req = ic_laya_core::DecisionRequest { evaluation_id: [77;32],
        model: ic_laya_core::hash(&manifest), schema_hash: compiled.schema_hash,
        calibration: None, binding: None, state: "evidence".into(),
        expires_at_ns: pic.get_time().as_nanos_since_unix_epoch() + 300_000_000_000 };
    let eval_args = Encode!(&req).unwrap();
    assert_eq!(Decode!(&forward("evaluate", eval_args.clone(), deposit), Result<ic_laya_core::Receipt>).unwrap(), Err(Error::Unauthorized));
    ok("allow_caller", Encode!(&proxy).unwrap());
    ok("set_caller_quota", Encode!(&proxy, &1u32).unwrap());
    let receipt = Decode!(&forward("evaluate", eval_args.clone(), deposit), Result<ic_laya_core::Receipt>).unwrap().unwrap();
    let first_fee = u128::from(deposit) - refund();
    let cached = Decode!(&forward("evaluate", eval_args.clone(), deposit), Result<ic_laya_core::Receipt>).unwrap().unwrap();
    assert_eq!(cached, receipt);
    assert!(u128::from(deposit) - refund() < first_fee);
    assert_eq!(Decode!(&forward("evaluate", eval_args.clone(), 0), Result<ic_laya_core::Receipt>).unwrap(), Err(Error::Budget));
    println!("PASS evaluate authorization and cached-receipt settlement");

    // Exercise the actual workflow caller, not just a hand-written forwarding WAT.
    use ic_laya_core::workflow::{Plan, RequiredSignal, Rule, RequestRecord, Status as WorkflowStatus};
    let executor = pic.create_canister();
    pic.add_cycles(executor, 100_000_000_000_000);
    let executor_wasm = fs::read(root.join("build/executor.wasm")).unwrap();
    let old_executor = std::env::var("EXECUTOR_OLD_WASM").ok().map(|p| fs::read(p).unwrap());
    pic.install_canister(executor, old_executor.clone().unwrap_or_else(|| executor_wasm.clone()),
        Encode!(&owner, &canister).unwrap(), None);
    let ex_call = |sender, method: &str, args| pic.update_call(executor, sender, method, args).unwrap();
    let ex_ok = |method: &str, args| { Decode!(&ex_call(owner, method, args), Result<()>).unwrap().unwrap(); };
    let now = pic.get_time().as_nanos_since_unix_epoch();
    let calibration = ic_laya_core::Calibration { id: [66;32], model: req.model,
        schema: compiled.schema_hash, tokenizer: compiled.tokenizer_hash, temperature: 1.0,
        expires_at_ns: now + 600_000_000_000, holdout_hash: [65;32], sample_count: 1, test_only: true };
    ok("register_calibration", Encode!(&calibration).unwrap());
    ok("allow_caller", Encode!(&executor).unwrap());
    let (fixture, _, op_id, grant_id) = ic_laya_core::demo::setup().unwrap();
    let mut grant = fixture.grants[&grant_id].clone();
    grant.delegate = outsider; grant.expires_at_ns = now + 600_000_000_000;
    let plan = Plan { id: grant.plan, version: 1, model: req.model,
        signals: vec![RequiredSignal { schema: compiled.clone(), calibration: calibration.id,
            rule: Rule::NoulTrue { minimum_ppm: 0 } }] };
    let mut operation = fixture.operations[&op_id].clone(); operation.evidence = "evidence".into();
    ex_ok("register_plan", Encode!(&plan).unwrap());
    ex_ok("register_operation", Encode!(&operation).unwrap());
    ex_ok("register_grant", Encode!(&grant).unwrap());
    let workflow = Decode!(&ex_call(outsider, "submit", Encode!(&op_id, &grant_id, &0u64).unwrap()), Result<[u8;32]>).unwrap().unwrap();
    if old_executor.is_some() {
        pic.upgrade_canister(executor, executor_wasm.clone(), Encode!().unwrap(), None).unwrap();
    }
    let attachment = || Decode!(&pic.query_call(executor, owner, "engine_cycles", Encode!().unwrap()).unwrap(), u128).unwrap();
    assert_eq!(attachment(), 0, "old/free engines keep the zero-cycle default");
    assert_eq!(Decode!(&ex_call(outsider, "set_engine_cycles", Encode!(&u128::from(deposit)).unwrap()), Result<()>).unwrap(), Err(Error::Unauthorized));
    ex_ok("set_engine_cycles", Encode!(&u128::MAX).unwrap());
    assert_eq!(Decode!(&ex_call(outsider, "advance", Encode!(&workflow).unwrap()), Result<WorkflowStatus>).unwrap(), Err(Error::Budget));
    let record = Decode!(&pic.query_call(executor, outsider, "get_request", Encode!(&workflow).unwrap()).unwrap(), Result<RequestRecord>).unwrap().unwrap();
    assert_eq!(record.status, WorkflowStatus::Received);
    assert!(record.pending.is_none(), "unfunded attachment must not consume an attempt");
    ex_ok("set_engine_cycles", Encode!(&u128::from(deposit)).unwrap());
    pic.upgrade_canister(executor, executor_wasm, Encode!().unwrap(), None).unwrap();
    assert_eq!(attachment(), u128::from(deposit));
    // Executor upgrades intentionally pause workflows until the owner resumes.
    ex_ok("pause", Encode!(&false).unwrap());
    let result = Decode!(&ex_call(outsider, "advance", Encode!(&workflow).unwrap()), Result<WorkflowStatus>).unwrap().unwrap();
    assert_eq!(result, WorkflowStatus::ReadyToDispatch);
    let record = Decode!(&pic.query_call(executor, outsider, "get_request", Encode!(&workflow).unwrap()).unwrap(), Result<RequestRecord>).unwrap().unwrap();
    assert_eq!(record.receipts.len(), 1);
    println!("PASS executor paid evaluation, owner-only configuration, insufficient funds and upgrade persistence");

    // The public generic query still refuses replicated execution.
    assert!(matches!(Decode!(&forward("decide_query", Encode!(&decision).unwrap(), deposit), Result<verdict_engine::DecideReply>).unwrap(), Err(Error::Denied(_))));
    assert_eq!(refund(), u128::from(deposit));
    println!("PASS public ordinary query; replicated inference query denied");

    pic.upgrade_canister(canister, wasm, Encode!().unwrap(), None).unwrap();
    let restored = Decode!(&query(owner, "cycles_pricing", Encode!().unwrap()), Result<CyclesPricing>).unwrap().unwrap();
    assert_eq!(restored.execution, pricing);
    warm();
    Decode!(&query(outsider, "infer_tokens_query", args), Result<InferReply>).unwrap().unwrap();
    assert_eq!(Decode!(&forward("evaluate", eval_args, deposit), Result<ic_laya_core::Receipt>).unwrap().unwrap(), receipt);
    ok("set_execution_pricing", Encode!(&None::<ExecutionPricing>).unwrap());
    assert!(Decode!(&forward("infer_tokens", Encode!(&ids).unwrap(), deposit), Result<InferReply>).unwrap().is_err());
    assert_eq!(refund(), u128::from(deposit));
    println!("PASS billing configuration persists across upgrade; suspension returns all cycles");
}
