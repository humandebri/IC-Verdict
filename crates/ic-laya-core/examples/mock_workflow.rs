use ic_laya_core::{demo::*,workflow::*,*};
fn main()->Result<()> {
    let (mut executor,mut engine,op,grant)=setup()?;
    let id=executor.submit(actor(2),0,op,grant,NOW)?;
    complete(&mut executor,&mut engine,id)?;
    let r=executor.get(actor(2),id)?;
    println!("TEST FIXTURE ONLY: these are scripted logits, not Laya predictions.");
    println!("{}",serde_json::to_string_pretty(&r.receipts).map_err(|e|Error::Invalid(e.to_string()))?);
    executor.set_mode(actor(1),Mode::Mock)?;
    let cap=executor.authorize(actor(2),id,NOW+4)?;
    let first=executor.prepare_dispatch(cap,NOW+4)?;
    executor.finish_ledger(&first,LedgerOutcome::Unknown("simulated lost reply".into()))?;
    executor.check_invariants()?;
    let retry=executor.retry_unknown(actor(2),id,NOW+5)?;
    assert_eq!(first.transfer,retry.transfer);
    executor.finish_ledger(&retry,LedgerOutcome::Duplicate("0".into()))?;
    executor.check_invariants()?;
    println!("Final mock status: {:?}",executor.get(actor(2),id)?.status);
    println!("Reserved={}, spent={}",executor.grants[&grant].total.reserved,executor.grants[&grant].total.spent);
    Ok(())
}
