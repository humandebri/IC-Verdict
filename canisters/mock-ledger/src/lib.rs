//! TEST ONLY. An ICRC-1 transfer test double, not an asset ledger.
use candid::{CandidType,Nat,Principal};
use canister_common::*;
use ic_laya_core::{workflow::MOCK_DEDUP_WINDOW_NS,*};
use serde::{Serialize,Deserialize};
use std::{cell::RefCell,collections::BTreeMap};
#[derive(Debug,Clone,Copy,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum Fault { None, BadFee, TooOld, CommitThenCallbackTrap }
#[derive(Clone,Serialize,Deserialize)]
struct State {owner:Principal,fault:Fault,next:u64,records:BTreeMap<(Principal,Digest),Lookup>,balances:BTreeMap<Principal,u128>}
thread_local!{static STATE:RefCell<Option<State>>=const{RefCell::new(None)};}
fn read<R>(f:impl FnOnce(&State)->R)->R{STATE.with(|s|f(s.borrow().as_ref().expect("initialized")))}
fn mutate<R>(f:impl FnOnce(&mut State)->R)->R{STATE.with(|s|{let mut s=s.borrow_mut();let s=s.as_mut().expect("initialized");let result=f(s);persist_or_trap(s);result})}
#[ic_cdk::init]
fn init(owner:Principal){if owner==Principal::anonymous(){ic_cdk::trap("invalid owner");}let s=State{owner,fault:Fault::None,next:0,records:BTreeMap::new(),balances:BTreeMap::new()};persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(persist_or_trap);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){let s:State=restore().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));STATE.with(|x|*x.borrow_mut()=Some(s));}
#[ic_cdk::update]
fn ic_laya_mock_profile()->String{MOCK_MAGIC.into()}
#[ic_cdk::update]
fn set_fault(fault:Fault)->Result<()>{let caller=ic_cdk::api::msg_caller();mutate(|s|{if caller!=s.owner{return Err(Error::Unauthorized);}s.fault=fault;Ok(())})}
#[ic_cdk::query]
fn committed_transfers()->u64{read(|s|s.next)}
#[ic_cdk::update]
fn ic_laya_lookup(arg:TransferArg)->Option<Lookup>{
    let caller=ic_cdk::api::msg_caller();let f=from_icrc(&arg,ic_cdk::api::canister_self()).ok()?;
    read(|s|s.records.get(&(caller,f.digest())).cloned())
}
#[ic_cdk::update]
fn ack_barrier(){if ic_cdk::api::msg_caller()!=ic_cdk::api::canister_self(){ic_cdk::trap("self call only");}}
#[ic_cdk::update]
async fn icrc1_transfer(arg:TransferArg)->TransferResult {
    let caller=ic_cdk::api::msg_caller();
    if caller==Principal::anonymous(){return Err(TransferError::GenericError{error_code:Nat::from(1u8),message:"anonymous".into()});}
    let f=match from_icrc(&arg,ic_cdk::api::canister_self()){Ok(f)=>f,Err(e)=>return Err(TransferError::GenericError{error_code:Nat::from(2u8),message:e.to_string()})};
    let now=ic_cdk::api::time();let fingerprint=f.digest();
    let outcome:std::result::Result<(u64,bool),TransferError>=mutate(|s|{
        if let Some(old)=s.records.get(&(caller,fingerprint)){return Err(TransferError::Duplicate{duplicate_of:Nat::from(old.block_index.parse::<u64>().expect("mock index"))});}
        if s.fault==Fault::BadFee{return Err(TransferError::BadFee{expected_fee:Nat::from(10u8)});}
        if s.fault==Fault::TooOld || now.saturating_sub(f.created_at_time_ns)>MOCK_DEDUP_WINDOW_NS{return Err(TransferError::TooOld);}
        if f.created_at_time_ns>now.saturating_add(1_000_000_000){return Err(TransferError::CreatedInFuture{ledger_time:now});}
        if f.proposal.fee!=10{return Err(TransferError::BadFee{expected_fee:Nat::from(10u8)});}
        if s.records.len()>=2048 || (!s.balances.contains_key(&caller)&&s.balances.len()>=64){return Err(TransferError::TemporarilyUnavailable);}
        let balance=s.balances.get(&caller).copied().unwrap_or(1_000_000_000_000);
        let charge=f.proposal.charge().map_err(|_|TransferError::InsufficientFunds{balance:Nat::from(balance)})?;
        if charge>balance{return Err(TransferError::InsufficientFunds{balance:Nat::from(balance)});}
        let index=s.next;s.next+=1;s.balances.insert(caller,balance-charge);
        s.records.insert((caller,fingerprint),Lookup{frozen_hash:fingerprint,block_index:index.to_string()});
        Ok((index,s.fault==Fault::CommitThenCallbackTrap))
    });
    match outcome {
        Err(e)=>Err(e),
        Ok((index,trap_after_commit))=>{
            if trap_after_commit {
                // The self-call creates a message boundary AFTER the mock commit.
                // Trapping only in its callback does not roll the earlier transfer back.
                let _=ic_cdk::call::Call::unbounded_wait(ic_cdk::api::canister_self(),"ack_barrier").await;
                ic_cdk::trap("TEST: reply lost after committed mock transfer");
            }
            Ok(Nat::from(index))
        }
    }
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}
