#![forbid(unsafe_code)]
use candid::{CandidType,Nat,Principal};
use ic_laya_core::{workflow::FrozenTransfer,*};
use serde::{Serialize,Deserialize,de::DeserializeOwned};
pub const BLOB_BASE:u64=64*1024*1024;
/// Snapshot slots alternate so a torn commit can never destroy the last good copy: the
/// body is written to the inactive slot and only then does the 16-byte header point at
/// it. Each slot holds the self-describing block from `storage::encode`, so a partial
/// body write is detected by its own length and checksum.
pub const SNAPSHOT_SLOT_A:u64=1<<20;
pub const SNAPSHOT_SLOT_B:u64=33<<20;
pub const SNAPSHOT_MAGIC:&[u8;8]=b"ICSLOT01";

/// `MAGIC(8) | generation(4 BE) | active_slot(4 BE)`.
fn header(generation:u32,slot:u32)->[u8;16]{
    let mut h=[0u8;16];h[..8].copy_from_slice(SNAPSHOT_MAGIC);
    h[8..12].copy_from_slice(&generation.to_be_bytes());
    h[12..16].copy_from_slice(&slot.to_be_bytes());h
}
fn parse_header(raw:&[u8;16])->Option<(u32,u32)>{
    if &raw[..8]!=SNAPSHOT_MAGIC{return None;}
    let generation=u32::from_be_bytes(raw[8..12].try_into().ok()?);
    let slot=u32::from_be_bytes(raw[12..16].try_into().ok()?);
    if slot>1{return None;}
    Some((generation,slot))
}
fn slot_offset(slot:u32)->u64{if slot==0{SNAPSHOT_SLOT_A}else{SNAPSHOT_SLOT_B}}
pub const MOCK_MAGIC:&str="IC_LAYA_MOCK_NO_REAL_ASSETS_V1";
pub fn grow_through(end:u64)->Result<()> {
    let pages=end.checked_add(65535).ok_or(Error::Capacity)?/65536;
    let have=ic_cdk::stable::stable_size();
    if pages>have {ic_cdk::stable::stable_grow(pages-have).map_err(|_|Error::Capacity)?;}Ok(())
}
/// Write `state` into the inactive slot, then commit it by pointing the header at that
/// slot. A trap or an out-of-cycles abort between the two leaves the previous snapshot
/// intact and reachable.
pub fn save<T:Serialize>(state:&T)->Result<()> {
    let bytes=ic_laya_core::storage::encode(state)?;
    if bytes.len() as u64>SNAPSHOT_SLOT_B-SNAPSHOT_SLOT_A||SNAPSHOT_SLOT_B+bytes.len() as u64>=BLOB_BASE{return Err(Error::Capacity);}
    let (generation,active)=read_header().unwrap_or((0,1));
    let target=1-active;
    let at=slot_offset(target);
    grow_through(at+bytes.len() as u64)?;
    ic_cdk::stable::stable_write(at,&bytes);
    ic_cdk::stable::stable_write(0,&header(generation.wrapping_add(1),target));Ok(())
}
fn read_header()->Option<(u32,u32)>{
    if ic_cdk::stable::stable_size()==0{return None;}
    let mut raw=[0u8;16];ic_cdk::stable::stable_read(0,&mut raw);parse_header(&raw)
}
fn read_slot<T:DeserializeOwned>(slot:u32)->Result<T>{
    let at=slot_offset(slot);
    // A slot holds the self-describing block `storage::encode` produces:
    // MAGIC(8) | len(8) | sha256(32) | payload. Read the block's own length; the slot
    // itself carries no separate length field.
    let mut head=[0u8;16];ic_cdk::stable::stable_read(at,&mut head);
    if &head[..8]!=ic_laya_core::storage::MAGIC{return Err(Error::Storage);}
    let n=u64::from_be_bytes(head[8..16].try_into().map_err(|_|Error::Storage)?);
    if n>ic_laya_core::storage::MAX_SNAPSHOT as u64{return Err(Error::Storage);}
    let total=ic_laya_core::storage::BLOCK_HEADER as u64+n;
    if at+total>ic_cdk::stable::stable_size()*65536{return Err(Error::Storage);}
    let mut bytes=vec![0u8;total as usize];ic_cdk::stable::stable_read(at,&mut bytes);
    ic_laya_core::storage::decode(&bytes)
}
/// Read the committed slot, falling back to the other one. Both slots are validated by
/// their own length and checksum, so a torn body fails here rather than decoding into
/// garbage.
pub fn restore<T:DeserializeOwned>()->Result<T>{
    if ic_cdk::stable::stable_size()==0{return Err(Error::Storage);}
    let (_,active)=read_header().ok_or(Error::Storage)?;
    match read_slot::<T>(active){Ok(v)=>Ok(v),Err(first)=>read_slot::<T>(1-active).map_err(|_|first)}
}
pub fn persist_or_trap<T:Serialize>(state:&T){if let Err(e)=save(state){ic_cdk::trap(format!("snapshot commit failed: {e}"));}}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trips_and_rejects_foreign_bytes() {
        let h=header(7,1);
        assert_eq!(parse_header(&h),Some((7,1)));
        assert_eq!(parse_header(&header(0,0)),Some((0,0)));
        let mut wrong=h;wrong[0]^=1;
        assert_eq!(parse_header(&wrong),None,"bad magic must be rejected");
        let mut bad_slot=h;bad_slot[15]=9;
        assert_eq!(parse_header(&bad_slot),None,"an unknown slot must be rejected");
    }

    #[test]
    fn slots_are_disjoint_and_fit_below_the_blob_area() {
        let max=ic_laya_core::storage::MAX_SNAPSHOT as u64+48;
        assert!(SNAPSHOT_SLOT_A+max<=SNAPSHOT_SLOT_B,"slot A may not reach slot B");
        assert!(SNAPSHOT_SLOT_B+max<=BLOB_BASE,"slot B may not reach the upload area");
    }
}
