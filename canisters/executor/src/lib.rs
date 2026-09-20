//! Durable orchestration adapter. No endpoint accepts a model score or capability.
//! LimitedLive is deliberately rejected by core in this implementation release.
use candid::Principal;
use canister_common::{Lookup,TransferError,TransferResult,to_icrc};
use ic_laya_core::{workflow::*,*};
use std::cell::RefCell;
thread_local!{static STATE:RefCell<Option<ExecutorState>>=const{RefCell::new(None)};}
fn read<R>(f:impl FnOnce(&ExecutorState)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
fn admitted()->Result<()> {
    let caller=ic_cdk::api::msg_caller();
    read(|s|if caller!=Principal::anonymous() && (caller==s.owner || s.grants.values().any(|g|g.delegate==caller)){Ok(())}else{Err(Error::Unauthorized)})
}
fn mutate<R>(f:impl FnOnce(&mut ExecutorState)->R)->R{STATE.with(|x|{let mut s=x.borrow_mut();let s=s.as_mut().expect("initialized");let r=f(s);if let Err(e)=s.check_invariants(){ic_cdk::trap(&format!("invariant: {e}"));}canister_common::persist_or_trap(s);r})}
#[ic_cdk::init]
fn init(owner:Principal,engine:Principal){
    if [owner,engine].iter().any(|&p|p==Principal::anonymous()||p==Principal::management_canister()){ic_cdk::trap("invalid principal");}
    let s=ExecutorState::new(ic_cdk::api::canister_self(),owner,engine);canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(|s|{
    if s.requests.values().any(|r|matches!(r.status,Status::Submitted|Status::OutcomeUnknown(_))){ic_cdk::trap("unresolved transfer: reconcile before ordinary upgrade");}
    canister_common::persist_or_trap(s);
});}
#[ic_cdk::post_upgrade]
fn post_upgrade(){let mut s:ExecutorState=canister_common::restore().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));s.recover_after_upgrade();s.check_invariants().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));}
#[ic_cdk::update]
fn register_plan(plan:Plan)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.install_plan(caller,plan))}
#[ic_cdk::update]
fn register_operation(operation:Operation)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.install_operation(caller,operation))}
#[ic_cdk::update]
fn revise_evidence(operation:Digest,evidence:String)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.revise_operation(caller,operation,evidence))}
#[ic_cdk::update]
fn register_grant(grant:Grant)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.install_grant(caller,grant))}
#[ic_cdk::update]
fn revoke_grant(grant:Digest)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.revoke(caller,grant))}
#[ic_cdk::update]
fn set_mode(mode:Mode)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.set_mode(caller,mode))}
#[ic_cdk::update]
fn pause(value:bool)->Result<()>{let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;mutate(|s|s.set_paused(caller,value))}
#[ic_cdk::update]
async fn register_mock_ledger(ledger:Principal)->Result<()> {
    let caller=ic_cdk::api::msg_caller();read(|s|s.assert_owner(caller))?;
    let response=ic_cdk::call::Call::bounded_wait(ledger,"ic_laya_mock_profile").change_timeout(10).await.map_err(|_|Error::Denied("mock handshake failed".into()))?;
    let marker:String=response.candid().map_err(|_|Error::BindingMismatch)?;
    if marker!=canister_common::MOCK_MAGIC{return Err(Error::Denied("not an IC-Laya mock ledger".into()));}
    mutate(|s|{s.assert_owner(caller)?;if !s.mock_ledgers.contains(&ledger){if s.mock_ledgers.len()>=8{return Err(Error::Capacity);}s.mock_ledgers.push(ledger);}Ok(())})
}
#[ic_cdk::update]
fn submit(operation:Digest,grant:Digest,client_nonce:u64)->Result<Digest>{admitted()?;let caller=ic_cdk::api::msg_caller();mutate(|s|s.submit(caller,client_nonce,operation,grant,ic_cdk::api::time()))}
#[ic_cdk::query]
fn get_request(id:Digest)->Result<RequestRecord>{read(|s|s.get(ic_cdk::api::msg_caller(),id))}
#[ic_cdk::update]
fn get_request_verified(id:Digest)->Result<RequestRecord>{read(|s|s.get(ic_cdk::api::msg_caller(),id))}
#[ic_cdk::update]
fn cancel(id:Digest)->Result<()>{admitted()?;let caller=ic_cdk::api::msg_caller();mutate(|s|s.cancel(caller,id))}
/// One engine question per step. A client may advance a registered plan up to three times.
#[ic_cdk::update]
async fn advance(id:Digest)->Result<Status>{
    admitted()?;
    let caller=ic_cdk::api::msg_caller();
    let req=mutate(|s|s.begin_evaluation(caller,id,ic_cdk::api::time()))?;
    let eval_id=req.evaluation_id;let engine=read(|s|s.engine);
    // All metadata is stable before this await; no RefCell borrow crosses it.
    let response=ic_cdk::call::Call::bounded_wait(engine,"evaluate").with_arg(req).change_timeout(30).await;
    match response {
        Ok(response)=>{
            let decoded:std::result::Result<Result<Receipt>,_>=response.candid();
            match decoded {
                Ok(result)=>mutate(|s|s.finish_evaluation(id,eval_id,result,ic_cdk::api::time())),
                Err(_)=>{mutate(|s|s.engine_transport_failed(id,eval_id))?;Err(Error::ModelUnavailable("engine response could not be decoded; bounded retry allowed".into()))},
            }
        },
        Err(_)=>{mutate(|s|s.engine_transport_failed(id,eval_id))?;Err(Error::ModelUnavailable("engine call outcome unknown; same evaluation ID retained".into()))},
    }
}
async fn send(cmd:DispatchCommand)->Result<Status>{
    let arg=to_icrc(&cmd.transfer);
    let response=ic_cdk::call::Call::bounded_wait(cmd.transfer.proposal.ledger,"icrc1_transfer").with_arg(arg).change_timeout(15).await;
    let outcome=match response {
        Ok(r)=>match r.candid::<TransferResult>() {
            Ok(Ok(index))=>LedgerOutcome::Success(index.0.to_string()),
            Ok(Err(TransferError::Duplicate{duplicate_of}))=>LedgerOutcome::Duplicate(duplicate_of.0.to_string()),
            Ok(Err(error))=>LedgerOutcome::DefinitiveError(format!("{error:?}")),
            Err(_)=>LedgerOutcome::Unknown("undecodable ledger response".into()),
        },
        // Conservative: no transport failure is used as a proof of non-execution.
        Err(_)=>LedgerOutcome::Unknown("ledger call failed or timed out".into()),
    };
    mutate(|s|s.finish_ledger(&cmd,outcome))
}
#[ic_cdk::update]
async fn dispatch(id:Digest)->Result<Status>{
    let caller=ic_cdk::api::msg_caller();let now=ic_cdk::api::time();
    let authorized=read(|s|s.authorize(caller,id,now))?;
    let prepared=mutate(|s|s.prepare_dispatch(authorized,now));
    match prepared {Ok(cmd)=>send(cmd).await,Err(Error::ReportOnly)=>Ok(Status::Reported),Err(e)=>Err(e)}
}
#[ic_cdk::update]
async fn retry_mock_transfer(id:Digest)->Result<Status>{admitted()?;let caller=ic_cdk::api::msg_caller();let cmd=mutate(|s|s.retry_unknown(caller,id,ic_cdk::api::time()))?;send(cmd).await}
#[ic_cdk::update]
async fn reconcile_mock(id:Digest)->Result<Status>{
    let caller=ic_cdk::api::msg_caller();let r=read(|s|s.get(caller,id))?;
    if !matches!(r.status,Status::Submitted|Status::OutcomeUnknown(_)){return Err(Error::Transition);}
    let transfer=r.frozen.ok_or(Error::Storage)?;
    if !read(|s|s.mock_ledgers.contains(&transfer.proposal.ledger)){return Err(Error::Unauthorized);}
    let cmd=DispatchCommand{request:id,attempt:r.ledger_attempt,transfer};
    let response=ic_cdk::call::Call::bounded_wait(cmd.transfer.proposal.ledger,"ic_laya_lookup").with_arg(to_icrc(&cmd.transfer)).change_timeout(10).await.map_err(|_|Error::OutcomeUnknown)?;
    let lookup:Option<Lookup>=response.candid().map_err(|_|Error::OutcomeUnknown)?;
    match lookup {
        Some(found) if found.frozen_hash==cmd.transfer.digest()=>mutate(|s|s.finish_ledger(&cmd,LedgerOutcome::Success(found.block_index))),
        Some(_)=>Err(Error::BindingMismatch),
        None=>mutate(|s|s.finish_ledger(&cmd,LedgerOutcome::Unknown("not found is not proof of non-execution".into()))),
    }
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}
