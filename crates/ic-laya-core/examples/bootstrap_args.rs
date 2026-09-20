//! Emit Candid-encoded setup calls for LOCAL scripted/mock integration only.
use candid::{utils::ArgumentEncoder,Principal};
use ic_laya_core::{demo,workflow::Mode,Canonical,Calibration};
use serde_json::{json,Value};
fn step<T:ArgumentEncoder>(canister:&str,method:&str,args:T)->Value {
    let bytes=candid::encode_args(args).expect("encode setup arguments");
    let raw=bytes.iter().map(|x|format!("{x:02x}")).collect::<String>();
    json!({"canister":canister,"method":method,"argument_hex":raw})
}
fn main()->std::result::Result<(),Box<dyn std::error::Error>> {
    let a:Vec<String>=std::env::args().skip(1).collect();
    if a.len()!=5{return Err("usage: bootstrap_args OWNER ENGINE EXECUTOR MOCK_LEDGER NOW_NS".into());}
    let owner=Principal::from_text(&a[0])?;let engine=Principal::from_text(&a[1])?;
    let executor=Principal::from_text(&a[2])?;let ledger=Principal::from_text(&a[3])?;let now:u64=a[4].parse()?;
    if owner==Principal::anonymous(){return Err("anonymous owner".into());}
    let (ex,en,operation_id,grant_id)=demo::setup()?;
    let mut operation=ex.operations[&operation_id].clone();operation.proposal.ledger=ledger;operation.proposal.to.owner=owner;
    let mut grant=ex.grants[&grant_id].clone();grant.delegate=owner;grant.ledger=ledger;
    grant.recipients=vec![operation.proposal.to.clone()];grant.expires_at_ns=now.checked_add(600_000_000_000).ok_or("time overflow")?;
    let plan=ex.plans[&grant.plan].clone();
    let mut steps=vec![step("decision_engine","enable_synthetic_fixture",()),step("decision_engine","allow_caller",(executor,30u32))];
    for required in &plan.signals {
        steps.push(step("decision_engine","register_schema",(required.schema.schema.clone(),required.schema.qtype_id)));
        let mut c:Calibration=en.calibrations[&required.calibration].clone();c.expires_at_ns=grant.expires_at_ns;
        steps.push(step("decision_engine","register_calibration",(c,)));
    }
    steps.push(step("executor","register_plan",(plan,)));
    steps.push(step("executor","register_operation",(operation,)));
    steps.push(step("executor","register_grant",(grant,)));
    steps.push(step("executor","register_mock_ledger",(ledger,)));
    steps.push(step("executor","set_mode",(Mode::Mock,)));
    steps.push(step("executor","submit",(operation_id,grant_id,0u64)));
    let mut hash=Canonical::new("ic-laya/workflow/v1");hash.bytes(executor.as_slice()).bytes(owner.as_slice()).u64(0);
    let id=hash.finish();
    for _ in 0..3 {steps.push(step("executor","advance",(id,)));}
    steps.push(step("executor","dispatch",(id,)));
    steps.push(step("executor","get_request_verified",(id,)));
    println!("{}",serde_json::to_string_pretty(&json!({"test_only":true,"engine":engine.to_text(),"request_id":id,"steps":steps}))?);
    Ok(())
}
