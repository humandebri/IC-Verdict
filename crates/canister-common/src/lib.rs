#![forbid(unsafe_code)]
use candid::{CandidType,Nat,Principal};
use ic_laya_core::{workflow::FrozenTransfer,*};
use serde::{Serialize,Deserialize,de::DeserializeOwned};
pub const BLOB_BASE:u64=64*1024*1024;
pub const MOCK_MAGIC:&str="IC_LAYA_MOCK_NO_REAL_ASSETS_V1";
pub fn grow_through(end:u64)->Result<()> {
    let pages=end.checked_add(65535).ok_or(Error::Capacity)?/65536;
    let have=ic_cdk::stable::stable_size();
    if pages>have {ic_cdk::stable::stable_grow(pages-have).map_err(|_|Error::Capacity)?;}Ok(())
}
pub fn save<T:Serialize>(state:&T)->Result<()> {
    let bytes=ic_laya_core::storage::encode(state)?;
    if bytes.len() as u64+8>=BLOB_BASE{return Err(Error::Capacity);}
    grow_through(bytes.len() as u64+8)?;
    ic_cdk::stable::stable_write(8,&bytes);
    ic_cdk::stable::stable_write(0,&(bytes.len() as u64).to_be_bytes());Ok(())
}
pub fn restore<T:DeserializeOwned>()->Result<T>{
    if ic_cdk::stable::stable_size()==0{return Err(Error::Storage);}
    let mut n=[0u8;8];ic_cdk::stable::stable_read(0,&mut n);let n=u64::from_be_bytes(n);
    if n>ic_laya_core::storage::MAX_SNAPSHOT as u64+48 || n+8>ic_cdk::stable::stable_size()*65536{return Err(Error::Storage);}
    let mut bytes=vec![0u8;n as usize];ic_cdk::stable::stable_read(8,&mut bytes);ic_laya_core::storage::decode(&bytes)
}
pub fn persist_or_trap<T:Serialize>(state:&T){if let Err(e)=save(state){ic_cdk::trap(&format!("snapshot commit failed: {e}"));}}

#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct IcrcAccount {pub owner:Principal,pub subaccount:Option<Vec<u8>>}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct TransferArg {pub from_subaccount:Option<Vec<u8>>,pub to:IcrcAccount,pub amount:Nat,pub fee:Option<Nat>,pub memo:Option<Vec<u8>>,pub created_at_time:Option<u64>}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum TransferError {
    BadFee{expected_fee:Nat},BadBurn{min_burn_amount:Nat},InsufficientFunds{balance:Nat},TooOld,
    CreatedInFuture{ledger_time:u64},TemporarilyUnavailable,Duplicate{duplicate_of:Nat},GenericError{error_code:Nat,message:String},
}
pub type TransferResult=std::result::Result<Nat,TransferError>;
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Lookup {pub frozen_hash:Digest,pub block_index:String}
pub fn to_icrc(f:&FrozenTransfer)->TransferArg {
    TransferArg{from_subaccount:Some(f.proposal.from_subaccount.to_vec()),to:IcrcAccount{owner:f.proposal.to.owner,subaccount:Some(f.proposal.to.subaccount.to_vec())},
        amount:Nat::from(f.proposal.amount),fee:Some(Nat::from(f.proposal.fee)),memo:Some(f.memo.to_vec()),created_at_time:Some(f.created_at_time_ns)}
}
pub fn nat_u128(n:&Nat)->Result<u128>{n.0.to_string().parse().map_err(|_|Error::Invalid("Nat out of u128 range".into()))}
fn sub(v:&Option<Vec<u8>>)->Result<[u8;32]>{match v{None=>Ok([0;32]),Some(x)=>x.as_slice().try_into().map_err(|_|Error::Invalid("subaccount length".into()))}}
pub fn from_icrc(a:&TransferArg,ledger:Principal)->Result<FrozenTransfer>{
    let proposal=TransferProposal{ledger,from_subaccount:sub(&a.from_subaccount)?,to:Account{owner:a.to.owner,subaccount:sub(&a.to.subaccount)?},amount:nat_u128(&a.amount)?,fee:nat_u128(a.fee.as_ref().ok_or_else(||Error::Invalid("explicit fee required".into()))?)?};
    proposal.validate()?;
    Ok(FrozenTransfer{proposal,memo:a.memo.as_ref().ok_or_else(||Error::Invalid("memo required".into()))?.as_slice().try_into().map_err(|_|Error::Invalid("memo length".into()))?,created_at_time_ns:a.created_at_time.ok_or_else(||Error::Invalid("timestamp required".into()))?})
}
