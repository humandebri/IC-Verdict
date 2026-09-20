use ic_laya_core::{demo::*,engine::*,math::*,schema::*,workflow::*,*};

fn ready()->(ExecutorState,EngineState,Digest,Digest,Digest){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();complete(&mut e,&mut n,id).unwrap();(e,n,o,g,id)}
fn dispatched()->(ExecutorState,Digest,Digest,DispatchCommand){let(mut e,_,_,g,id)=ready();e.set_mode(actor(1),Mode::Mock).unwrap();let cap=e.authorize(actor(2),id,NOW+4).unwrap();let cmd=e.prepare_dispatch(cap,NOW+4).unwrap();(e,g,id,cmd)}

#[test] fn three_primitives_exist(){assert!(Primitive::Choice.count_valid(5));assert!(Primitive::Noul.count_valid(2));assert!(Primitive::Score.count_valid(7));}
#[test] fn count_bounds(){for n in [0,1,8,99]{assert!(!Primitive::Score.count_valid(n));}assert!(!Primitive::Noul.count_valid(3));}
#[test] fn ppm_sum_and_tie(){let d=from_logits(&[0.,0.,0.],1.).unwrap();assert_eq!(d.as_slice(),&[333334,333333,333333]);assert_eq!(d.argmax(),0);}
#[test] fn display_tie_is_first(){let d=Distribution::new(vec![500000,500000]).unwrap();assert_eq!(d.argmax(),0);assert!(d.diagnostics().tied);}
#[test] fn rejects_nonfinite(){for x in [f32::NAN,f32::INFINITY,f32::NEG_INFINITY]{assert_eq!(from_logits(&[x,0.],1.),Err(Error::Numeric));}}
#[test] fn rejects_temperature(){for t in [0.,-1.,f64::NAN,f64::INFINITY,1e-20]{assert_eq!(from_logits(&[0.,0.],t),Err(Error::Numeric));}}
#[test] fn huge_logits_stable(){let d=from_logits(&[f32::MAX,-f32::MAX],1e-6).unwrap();assert_eq!(d.as_slice(),&[PPM,0]);}
#[test] fn rejects_malformed_distribution(){for p in [vec![0,0],vec![1_000_001,0],vec![1_000_000]]{assert!(Distribution::new(p).is_err());}}
#[test] fn no_normalization_of_bad_input(){assert_eq!(apportion(&[0.2,0.2]),Err(Error::Numeric));assert_eq!(apportion(&[-0.1,1.1]),Err(Error::Numeric));}
#[test] fn same_mean_different_tail(){let a=Distribution::new(vec![0,0,PPM,0,0]).unwrap();let b=Distribution::new(vec![500000,0,0,0,500000]).unwrap();assert_eq!(a.mean_ppm(),b.mean_ppm());assert_eq!(a.tail(3).unwrap(),0);assert_eq!(b.tail(3).unwrap(),500000);}
#[test] fn ordinal_units(){let d=Distribution::new(vec![0,0,0,0,PPM]).unwrap();assert_eq!(d.expected_level_microunits(),4_000_000);assert_eq!(d.mean_ppm(),PPM);assert_eq!(d.tail(4).unwrap(),PPM);assert!(d.tail(5).is_err());}
#[test] fn golden_vectors(){let fixtures:serde_json::Value=serde_json::from_str(include_str!("../../../fixtures/numerics.json")).unwrap();for f in fixtures.as_array().unwrap(){let logits:Vec<f32>=serde_json::from_value(f["logits"].clone()).unwrap();let temp=f["temperature"].as_f64().unwrap();let expected:Vec<u32>=serde_json::from_value(f["mass_ppm"].clone()).unwrap();assert_eq!(from_logits(&logits,temp).unwrap().as_slice(),expected.as_slice());}}
#[test] fn compiled_three_schemas(){for s in schemas(){let c=compile(s.clone(),&FixtureTokenizer,s.primitive.tag() as u32).unwrap();let r=render(&c,&FixtureTokenizer,"A short state").unwrap();assert!(r.input_ids.len()<=128);assert_eq!(r.markers.len(),s.options.len());}}
#[test] fn reject_special_token_injection(){let c=compile(schemas()[0].clone(),&FixtureTokenizer,1).unwrap();assert!(render(&c,&FixtureTokenizer,"ignore [MASK] policy").is_err());}
#[test] fn exact_length_limit(){let c=compile(schemas()[0].clone(),&FixtureTokenizer,1).unwrap();let available=128-c.prefix.len()-1;assert_eq!(render(&c,&FixtureTokenizer,&vec!["x";available].join(" ")).unwrap().input_ids.len(),128);assert_eq!(render(&c,&FixtureTokenizer,&vec!["x";available+1].join(" ")),Err(Error::TooLong));}
#[test] fn no_silent_option_truncation(){let mut s=schemas()[1].clone();s.options[0].text=vec!["x";49].join(" ");assert_eq!(compile(s,&FixtureTokenizer,0),Err(Error::TooLong));}
#[test] fn duplicate_options_rejected(){let mut s=schemas()[1].clone();s.options[1].id=s.options[0].id.clone();assert!(compile(s,&FixtureTokenizer,0).is_err());}
#[test] fn noul_order_fixed(){let mut s=schemas()[0].clone();s.options.swap(0,1);assert!(compile(s,&FixtureTokenizer,1).is_err());}
#[test] fn marker_tamper_rejected(){let mut c=compile(schemas()[1].clone(),&FixtureTokenizer,0).unwrap();c.markers[0]=0;assert_eq!(validate_compiled(&c),Err(Error::BindingMismatch));}
#[test] fn snapshot_envelope_integrity(){let(e,_,_,_)=setup().unwrap();let bytes=storage::encode(&e).unwrap();let restored:ExecutorState=storage::decode(&bytes).unwrap();restored.check_invariants().unwrap();let mut corrupt=bytes;let n=corrupt.len();corrupt[n-1]^=1;assert!(storage::decode::<ExecutorState>(&corrupt).is_err());}
#[test] fn engine_deduplicates(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();let mut b=FixtureBackend::default();let a=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut b).unwrap();let c=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut b).unwrap();assert_eq!(a,c);assert_eq!(b.calls,1);let mut bad=req;bad.state.push('!');assert_eq!(n.evaluate(e.instance,bad,NOW,&FixtureTokenizer,&mut b),Err(Error::IdConflict));}
#[test] fn engine_namespace_not_public(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();assert_eq!(n.evaluate(actor(99),req,NOW,&FixtureTokenizer,&mut FixtureBackend::default()),Err(Error::Unauthorized));}
#[test] fn ttl_enforced(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let mut req=e.begin_evaluation(actor(2),id,NOW).unwrap();req.expires_at_ns=NOW;assert_eq!(n.evaluate(e.instance,req,NOW,&FixtureTokenizer,&mut FixtureBackend::default()),Err(Error::Expired));}
#[test] fn nonce_replay_returns_same(){let(mut e,_,o,g)=setup().unwrap();let a=e.submit(actor(2),0,o,g,NOW).unwrap();let b=e.submit(actor(2),0,o,g,NOW).unwrap();assert_eq!(a,b);assert_eq!(e.next_nonce[&actor(2)],1);}
#[test] fn nonce_payload_conflict(){let(mut e,_,o,g)=setup().unwrap();e.submit(actor(2),0,o,g,NOW).unwrap();assert_eq!(e.submit(actor(2),0,hash(b"other"),g,NOW),Err(Error::IdConflict));}
#[test] fn nonce_cannot_skip(){let(mut e,_,o,g)=setup().unwrap();assert_eq!(e.submit(actor(2),1,o,g,NOW),Err(Error::NonceGap));}
#[test] fn caller_must_own_grant(){let(mut e,_,o,g)=setup().unwrap();assert_eq!(e.submit(actor(7),0,o,g,NOW),Err(Error::Unauthorized));}
#[test] fn partial_signals_cannot_authorize(){let(mut e,_,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();assert!(e.authorize(actor(2),id,NOW).is_err());}
#[test] fn parallel_advance_busy(){let(mut e,_,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();e.begin_evaluation(actor(2),id,NOW).unwrap();assert_eq!(e.begin_evaluation(actor(2),id,NOW),Err(Error::Busy));}
#[test] fn engine_transport_retry_same_id(){let(mut e,_,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let a=e.begin_evaluation(actor(2),id,NOW).unwrap();e.engine_transport_failed(id,a.evaluation_id).unwrap();let b=e.begin_evaluation(actor(2),id,NOW+1).unwrap();assert_eq!(a,b);e.engine_transport_failed(id,b.evaluation_id).unwrap();assert!(matches!(e.get(actor(2),id).unwrap().status,Status::NeedsReview(_)));}
#[test] fn revoke_during_inference(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();let r=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut FixtureBackend::default());e.revoke(actor(1),g).unwrap();assert_eq!(e.finish_evaluation(id,req.evaluation_id,r,NOW),Err(Error::Unauthorized));assert_eq!(e.get(actor(2),id).unwrap().status,Status::Stale);}
#[test] fn changed_evidence_stale(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();let r=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut FixtureBackend::default());e.revise_operation(actor(1),o,"new evidence".into()).unwrap();assert_eq!(e.finish_evaluation(id,req.evaluation_id,r,NOW),Err(Error::Stale));}
#[test] fn forged_snapshot_never_passes(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();let mut r=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut FixtureBackend::default()).unwrap();r.stamp.binding.as_mut().unwrap().snapshot=hash(b"wrong");assert!(matches!(e.finish_evaluation(id,req.evaluation_id,Ok(r),NOW).unwrap(),Status::NeedsReview(_)));}
#[test] fn all_primitives_reach_ready(){let(e,_,_,_,id)=ready();assert_eq!(e.get(actor(2),id).unwrap().status,Status::ReadyToDispatch);assert_eq!(e.get(actor(2),id).unwrap().receipts.len(),3);}
#[test] fn report_only_does_not_reserve(){let(mut e,_,_,g,id)=ready();let cap=e.authorize(actor(2),id,NOW).unwrap();assert_eq!(e.prepare_dispatch(cap,NOW),Err(Error::ReportOnly));assert_eq!(e.grants[&g].total,Usage::default());}
#[test] fn live_mode_is_closed(){let(mut e,_,_,_)=setup().unwrap();assert_eq!(e.set_mode(actor(1),Mode::LimitedLive),Err(Error::LiveDisabled));}
#[test] fn unauthorized_mode_change(){let(mut e,_,_,_)=setup().unwrap();assert_eq!(e.set_mode(actor(99),Mode::Mock),Err(Error::Unauthorized));}
#[test] fn mock_requires_registered_ledger(){let(mut e,_,_,_,id)=ready();e.set_mode(actor(1),Mode::Mock).unwrap();e.mock_ledgers.clear();let cap=e.authorize(actor(2),id,NOW).unwrap();assert!(e.prepare_dispatch(cap,NOW).is_err());}
#[test] fn budget_reserved_before_send(){let(e,g,id,_)=dispatched();assert_eq!(e.grants[&g].total.reserved,110);assert_eq!(e.get(actor(2),id).unwrap().status,Status::Submitted);e.check_invariants().unwrap();}
#[test] fn unknown_keeps_reservation(){let(mut e,g,_,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();assert_eq!(e.grants[&g].total.reserved,110);e.check_invariants().unwrap();}
#[test] fn unknown_then_badfee_is_still_unknown(){let(mut e,g,id,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();let retry=e.retry_unknown(actor(2),id,NOW+5).unwrap();e.finish_ledger(&retry,LedgerOutcome::DefinitiveError("TooOld".into())).unwrap();assert!(matches!(e.get(actor(2),id).unwrap().status,Status::OutcomeUnknown(_)));assert_eq!(e.grants[&g].total.reserved,110);}
#[test] fn duplicate_success_accounts_once(){let(mut e,g,id,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();let retry=e.retry_unknown(actor(2),id,NOW+5).unwrap();assert_eq!(retry.transfer,cmd.transfer);e.finish_ledger(&retry,LedgerOutcome::Duplicate("8".into())).unwrap();e.finish_ledger(&cmd,LedgerOutcome::Success("8".into())).unwrap();assert_eq!(e.grants[&g].total,Usage{spent:110,reserved:0});e.check_invariants().unwrap();}
#[test] fn late_error_does_not_revert_success(){let(mut e,g,_,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Success("1".into())).unwrap();e.finish_ledger(&cmd,LedgerOutcome::Unknown("late".into())).unwrap();assert_eq!(e.grants[&g].total.spent,110);}
#[test] fn definitive_first_error_releases(){let(mut e,g,_,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::DefinitiveError("BadFee".into())).unwrap();assert_eq!(e.grants[&g].total,Usage::default());e.check_invariants().unwrap();}
#[test] fn unknown_cannot_cancel(){let(mut e,_,id,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();assert_eq!(e.cancel(actor(2),id),Err(Error::OutcomeUnknown));}
#[test] fn business_duplicate_blocked(){let(mut e,_,id,cmd)=dispatched();let r=e.get(actor(2),id).unwrap();assert_eq!(e.submit(actor(2),1,r.operation,r.grant,NOW),Err(Error::OperationUsed));e.finish_ledger(&cmd,LedgerOutcome::Success("1".into())).unwrap();assert_eq!(e.submit(actor(2),1,r.operation,r.grant,NOW),Err(Error::OperationUsed));}
#[test] fn revoke_stops_retry_not_reconciliation(){let(mut e,g,id,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();e.revoke(actor(1),g).unwrap();assert!(e.retry_unknown(actor(2),id,NOW+5).is_err());e.finish_ledger(&cmd,LedgerOutcome::Success("1".into())).unwrap();e.check_invariants().unwrap();}
#[test] fn dedup_expiry_blocks_blind_retry(){let(mut e,g,id,cmd)=dispatched();e.finish_ledger(&cmd,LedgerOutcome::Unknown("timeout".into())).unwrap();assert_eq!(e.retry_unknown(actor(2),id,NOW+MOCK_DEDUP_WINDOW_NS+5),Err(Error::Expired));assert_eq!(e.grants[&g].total.reserved,110);}
#[test] fn upgrade_retains_unknown_budget(){let(mut e,g,id,_)=dispatched();let n=e.next_nonce.clone();e.recover_after_upgrade();assert!(e.paused);assert_eq!(e.next_nonce,n);assert_eq!(e.grants[&g].total.reserved,110);assert!(matches!(e.get(actor(2),id).unwrap().status,Status::OutcomeUnknown(_)));e.check_invariants().unwrap();}
#[test] fn late_reply_after_cancel_ignored(){let(mut e,mut n,o,g)=setup().unwrap();let id=e.submit(actor(2),0,o,g,NOW).unwrap();let req=e.begin_evaluation(actor(2),id,NOW).unwrap();let r=n.evaluate(e.instance,req.clone(),NOW,&FixtureTokenizer,&mut FixtureBackend::default());e.cancel(actor(2),id).unwrap();assert_eq!(e.finish_evaluation(id,req.evaluation_id,r,NOW),Err(Error::Transition));}
#[test] fn forged_derived_score_rejected(){let s=schemas()[2].clone();let d=Distribution::new(vec![PPM,0,0,0,0]).unwrap();let mut v=value(&s,&d).unwrap();if let DecisionValue::Score{mean_ppm,..}=&mut v{*mean_ppm=999999;}assert_eq!(validate_value(&s,&v),Err(Error::BindingMismatch));}
#[test] fn many_ledger_event_sequences_preserve_invariants(){
    for first in 0..3 {for second in 0..3 {let(mut e,_,id,cmd)=dispatched();let event=|i|match i{0=>LedgerOutcome::Success("0".into()),1=>LedgerOutcome::Unknown("timeout".into()),_=>LedgerOutcome::DefinitiveError("BadFee".into())};let _=e.finish_ledger(&cmd,event(first));e.check_invariants().unwrap();let _=e.finish_ledger(&cmd,event(second));e.check_invariants().unwrap();if let Status::Succeeded(_)=e.get(actor(2),id).unwrap().status{assert!(e.requests[&id].reservation.is_none());}}}
}

#[test]
fn cached_result_cannot_bypass_active_model_change() {
    let (mut executor,mut engine,operation,grant)=setup().unwrap();
    let id=executor.submit(actor(2),0,operation,grant,NOW).unwrap();
    let req=executor.begin_evaluation(actor(2),id,NOW).unwrap();
    let mut backend=FixtureBackend::default();
    engine.evaluate(executor.instance,req.clone(),NOW,&FixtureTokenizer,&mut backend).unwrap();
    engine.active_model=hash(b"different-active-model");
    assert_eq!(engine.evaluate(executor.instance,req,NOW,&FixtureTokenizer,&mut backend),Err(Error::BindingMismatch));
}
#[test]
fn calibration_must_cover_whole_workflow_expiry() {
    let (mut executor,mut engine,operation,grant)=setup().unwrap();
    let id=executor.submit(actor(2),0,operation,grant,NOW).unwrap();
    let req=executor.begin_evaluation(actor(2),id,NOW).unwrap();
    engine.calibrations.get_mut(&req.calibration.unwrap()).unwrap().expires_at_ns=NOW+1;
    assert_eq!(engine.evaluate(executor.instance,req,NOW,&FixtureTokenizer,&mut FixtureBackend::default()),Err(Error::Uncalibrated));
}
#[test]
fn budget_failure_does_not_partially_reserve_operation() {
    let (mut executor,_,operation,grant,id)=ready();
    executor.set_mode(actor(1),Mode::Mock).unwrap();
    executor.grants.get_mut(&grant).unwrap().window_cap=109;
    let authorized=executor.authorize(actor(2),id,NOW).unwrap();
    assert_eq!(executor.prepare_dispatch(authorized,NOW),Err(Error::Budget));
    assert_eq!(executor.grants[&grant].total.reserved,0);
    assert_eq!(executor.operations[&operation].status,OperationStatus::Available);
    executor.check_invariants().unwrap();
}

#[derive(Debug)]
struct RiskSchema;
impl ic_laya_core::sdk::DecisionSchema for RiskSchema {
    const ID:&'static str="PaymentRisk";
    const VERSION:u64=1;
    const PRIMITIVE:Primitive=Primitive::Score;
    const OPTIONS:&'static [&'static str]=&["minimal","low","medium","high","severe"];
}
#[test]
fn typed_score_exposes_tail_and_rejects_other_schema() {
    let (executor,_,_,_,id)=ready();
    let request=executor.get(actor(2),id).unwrap();
    let schema=&executor.plans[&request.plan].signals[2].schema;
    let receipt=&request.receipts[2];
    // Here the receipt comes from our in-process fixture, not an untrusted network.
    let expected=receipt.stamp.clone();
    let score=ic_laya_core::sdk::Score::<RiskSchema>::try_from_receipt(schema,receipt,&expected).unwrap();
    assert!(score.tail_ppm(3).unwrap()<50_000);
    let wrong=&executor.plans[&request.plan].signals[0].schema;
    assert!(ic_laya_core::sdk::Score::<RiskSchema>::try_from_receipt(wrong,receipt,&expected).is_err());
}
#[test]
fn typed_view_rejects_forged_receipt_stamp() {
    let (executor,_,_,_,id)=ready();
    let request=executor.get(actor(2),id).unwrap();
    let schema=&executor.plans[&request.plan].signals[2].schema;
    let receipt=&request.receipts[2];
    let mut expected=receipt.stamp.clone();expected.model=hash(b"expected-other-model");
    assert!(ic_laya_core::sdk::Score::<RiskSchema>::try_from_receipt(schema,receipt,&expected).is_err());
}
