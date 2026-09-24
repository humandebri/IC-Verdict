//! Public API recovery using the real executor, free fixture engine and mock ledger.
use candid::{CandidType, Decode, Encode};
use ic_laya_core::{demo::*, workflow::*, *};
use pocket_ic::PocketIcBuilder;
use serde::Deserialize;
use std::{fs, path::PathBuf, time::Duration};

#[derive(CandidType, Deserialize)]
enum Fault { CommitThenCallbackTrap }

#[test]
#[ignore = "requires executor, decision-engine, mock-ledger Wasm and POCKET_IC_BIN"]
fn parked_transfer_reconciles_after_upgrade_expiry_and_revocation() {
    let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pic=PocketIcBuilder::new().with_application_subnet().build();
    let owner=actor(1); let delegate=actor(2);
    let engine=pic.create_canister(); let executor=pic.create_canister(); let ledger=pic.create_canister();
    let executor_wasm=fs::read(root.join("build/executor.wasm")).unwrap();
    for (id,name,args) in [(engine,"decision-engine",Encode!(&owner).unwrap()),
        (executor,"executor",Encode!(&owner,&engine).unwrap()),
        (ledger,"mock-ledger",Encode!(&owner).unwrap())] {
        pic.add_cycles(id,100_000_000_000_000);
        pic.install_canister(id,fs::read(root.join(format!("build/{name}.wasm"))).unwrap(),args,None);
    }
    let call=|id,who,method:&str,args|pic.update_call(id,who,method,args).unwrap();
    let ok=|id,method:&str,args|Decode!(&call(id,owner,method,args),Result<()>).unwrap().unwrap();
    Decode!(&call(engine,owner,"enable_synthetic_fixture",Encode!().unwrap()),Result<Digest>).unwrap().unwrap();
    ok(engine,"allow_caller",Encode!(&executor,&30u32).unwrap());
    let (fixture, fixture_engine, op, grant_id)=setup().unwrap();
    let now=pic.get_time().as_nanos_since_unix_epoch();
    let plan=fixture.plans.values().next().unwrap().clone();
    for signal in &plan.signals {
        let schema=&signal.schema.schema;
        Decode!(&call(engine,owner,"register_schema",Encode!(schema,&(schema.primitive.tag() as u32)).unwrap()),Result<CompiledSchema>).unwrap().unwrap();
        let mut calibration=fixture_engine.calibrations[&signal.calibration].clone();
        calibration.expires_at_ns=now+600_000_000_000;
        ok(engine,"register_calibration",Encode!(&calibration).unwrap());
    }
    let mut operation=fixture.operations[&op].clone(); operation.proposal.ledger=ledger;
    let mut grant=fixture.grants[&grant_id].clone(); grant.ledger=ledger; grant.expires_at_ns=now+600_000_000_000;
    ok(executor,"register_plan",Encode!(&plan).unwrap());
    ok(executor,"register_operation",Encode!(&operation).unwrap());
    ok(executor,"register_grant",Encode!(&grant).unwrap());
    ok(executor,"register_mock_ledger",Encode!(&ledger).unwrap());
    ok(executor,"set_mode",Encode!(&Mode::Mock).unwrap());
    let id=Decode!(&call(executor,delegate,"submit",Encode!(&op,&grant_id,&0u64).unwrap()),Result<Digest>).unwrap().unwrap();
    assert_eq!(Decode!(&call(executor,delegate,"reconcile_mock",Encode!(&id).unwrap()),Result<Status>).unwrap(),Err(Error::Transition));
    for _ in 0..3 { Decode!(&call(executor,delegate,"advance",Encode!(&id).unwrap()),Result<Status>).unwrap().unwrap(); }
    ok(ledger,"set_fault",Encode!(&Fault::CommitThenCallbackTrap).unwrap());
    assert!(matches!(Decode!(&call(executor,delegate,"dispatch",Encode!(&id).unwrap()),Result<Status>).unwrap().unwrap(),Status::OutcomeUnknown(_)));
    let record=||Decode!(&pic.query_call(executor,owner,"get_request",Encode!(&id).unwrap()).unwrap(),Result<RequestRecord>).unwrap().unwrap();
    let reservation=record().reservation.unwrap();
    Decode!(&call(executor,owner,"abandon_unknown",Encode!(&id,&"upgrade".to_string()).unwrap()),Result<Status>).unwrap().unwrap();
    ok(executor,"revoke_grant",Encode!(&grant_id).unwrap());
    pic.upgrade_canister(executor,executor_wasm,Encode!().unwrap(),None).unwrap();
    pic.advance_time(Duration::from_secs(601));
    assert_eq!(record().reservation,Some(reservation));
    assert_eq!(Decode!(&call(executor,actor(99),"reconcile_mock",Encode!(&id).unwrap()),Result<Status>).unwrap(),Err(Error::Unauthorized));
    // Fault fixtures preserve the ledger's stable snapshot; restore the real
    // ledger afterwards to prove only its matching evidence settles the funds.
    for (lookup, expected) in [(None, Error::OutcomeUnknown),
        (Some(canister_common::Lookup{frozen_hash:[0;32],block_index:"0".into()}),Error::BindingMismatch)] {
        let reply=Encode!(&lookup).unwrap();
        let data=reply.iter().map(|b|format!("\\{b:02x}")).collect::<String>();
        let stub=wat::parse_str(format!(r#"(module
            (import "ic0" "msg_reply_data_append" (func $append (param i32 i32)))
            (import "ic0" "msg_reply" (func $reply))
            (memory 1) (data (i32.const 0) "{data}")
            (func (export "canister_update ic_laya_lookup")
                (call $append (i32.const 0) (i32.const {})) (call $reply)))"#,reply.len())).unwrap();
        pic.upgrade_canister(ledger,stub,Encode!().unwrap(),None).unwrap();
        let before=record();
        assert_eq!(Decode!(&call(executor,delegate,"reconcile_mock",Encode!(&id).unwrap()),Result<Status>).unwrap(),Err(expected));
        assert_eq!(record(),before);
    }
    pic.upgrade_canister(ledger,wat::parse_str("(module (func (export \"canister_update ic_laya_lookup\") unreachable))").unwrap(),Encode!().unwrap(),None).unwrap();
    let before=record();
    assert_eq!(Decode!(&call(executor,delegate,"reconcile_mock",Encode!(&id).unwrap()),Result<Status>).unwrap(),Err(Error::OutcomeUnknown));
    assert_eq!(record(),before);
    pic.upgrade_canister(ledger,fs::read(root.join("build/mock-ledger.wasm")).unwrap(),Encode!().unwrap(),None).unwrap();
    for who in [delegate,owner,delegate] {
        assert_eq!(Decode!(&call(executor,who,"reconcile_mock",Encode!(&id).unwrap()),Result<Status>).unwrap().unwrap(),Status::Succeeded("0".into()));
    }
    assert!(record().reservation.is_none());
    assert_eq!(record().ledger_attempt,1);
    assert_eq!(Decode!(&pic.query_call(ledger,owner,"committed_transfers",Encode!().unwrap()).unwrap(),u64).unwrap(),1);
    ok(executor,"audit",Encode!().unwrap());
}
