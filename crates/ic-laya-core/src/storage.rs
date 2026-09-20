//! Versioned, checksummed snapshots. The canister adapter owns commit/rollback.
use crate::*;
use serde::{Serialize,de::DeserializeOwned};
use bincode::Options;
const MAGIC:&[u8;8]=b"ICLAYA01";
pub const MAX_SNAPSHOT:usize=16*1024*1024;
pub fn encode<T:Serialize>(value:&T)->Result<Vec<u8>> {
    let bytes=bincode::DefaultOptions::new().with_fixint_encoding().serialize(value).map_err(|_|Error::Storage)?;
    if bytes.len()>MAX_SNAPSHOT{return Err(Error::Capacity);}
    let mut out=Vec::with_capacity(48+bytes.len());out.extend_from_slice(MAGIC);out.extend_from_slice(&(bytes.len() as u64).to_be_bytes());out.extend_from_slice(&hash(&bytes));out.extend(bytes);Ok(out)
}
pub fn decode<T:DeserializeOwned>(bytes:&[u8])->Result<T>{
    if bytes.len()<48 || &bytes[..8]!=MAGIC{return Err(Error::Storage);}
    let len=u64::from_be_bytes(bytes[8..16].try_into().map_err(|_|Error::Storage)?) as usize;
    if len>MAX_SNAPSHOT || bytes.len()!=48+len || bytes[16..48]!=hash(&bytes[48..]){return Err(Error::Storage);}
    bincode::DefaultOptions::new().with_fixint_encoding().with_limit(MAX_SNAPSHOT as u64).reject_trailing_bytes().deserialize(&bytes[48..]).map_err(|_|Error::Storage)
}
