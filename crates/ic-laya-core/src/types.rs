use candid::{CandidType, Principal};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fmt;

pub const PPM: u32 = 1_000_000;
pub const MAX_TOKENS: usize = 128;
pub const MAX_PREFIX: usize = 64;
pub const MAX_STATE_BYTES: usize = 16 * 1024;
pub const MAX_SLOTS: usize = 3;
pub const PROFILE: &str = "Compact128-v1";
pub type Digest = [u8; 32];
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType, thiserror::Error)]
pub enum Error {
    #[error("unauthorized")] Unauthorized,
    #[error("invalid input: {0}")] Invalid(String),
    #[error("input exceeds a registered limit")] TooLong,
    #[error("unknown schema/model/operation")] NotFound,
    #[error("version or binding mismatch")] BindingMismatch,
    #[error("expired")] Expired,
    #[error("duplicate ID with different content")] IdConflict,
    #[error("quota or bounded storage exhausted")] Capacity,
    #[error("model unavailable: {0}")] ModelUnavailable(String),
    #[error("invalid numeric output")] Numeric,
    #[error("uncalibrated or expired calibration")] Uncalibrated,
    #[error("workflow busy")] Busy,
    #[error("invalid state transition")] Transition,
    #[error("nonce already consumed")] NonceConsumed,
    #[error("unexpected nonce")] NonceGap,
    #[error("trusted business operation already reserved or consumed")] OperationUsed,
    #[error("hard policy rejected request: {0}")] Denied(String),
    #[error("budget exhausted or arithmetic overflow")] Budget,
    #[error("state changed during inference")] Stale,
    #[error("report-only; no transaction dispatched")] ReportOnly,
    #[error("live dispatch is not implemented/enabled in this release")] LiveDisabled,
    #[error("ledger outcome is unknown; reservation retained")] OutcomeUnknown,
    #[error("storage corruption or incompatible version")] Storage,
}

pub fn hash(bytes: &[u8]) -> Digest { Sha256::digest(bytes).into() }
pub fn hex(d: &Digest) -> String { d.iter().map(|v| format!("{v:02x}")).collect() }
/// Length-framed, domain-separated canonical hashing, not JSON serialization.
pub struct Canonical(Vec<u8>);
impl Canonical {
    pub fn new(domain: &str) -> Self { let mut x=Self(Vec::new()); x.bytes(domain.as_bytes()); x }
    pub fn bytes(&mut self, x: &[u8]) -> &mut Self { self.0.extend_from_slice(&(x.len() as u64).to_be_bytes()); self.0.extend_from_slice(x); self }
    pub fn text(&mut self, x: &str) -> &mut Self { self.bytes(x.as_bytes()) }
    pub fn u64(&mut self, x:u64) -> &mut Self { self.0.extend_from_slice(&x.to_be_bytes()); self }
    pub fn u128(&mut self, x:u128) -> &mut Self { self.0.extend_from_slice(&x.to_be_bytes()); self }
    pub fn finish(self) -> Digest { hash(&self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub enum Primitive { Choice, Noul, Score }
impl Primitive {
    pub fn tag(self)->u64 { match self { Self::Choice=>0, Self::Noul=>1, Self::Score=>2 } }
    pub fn text(self)->&'static str { match self { Self::Choice=>"choice", Self::Noul=>"noul", Self::Score=>"score" } }
    pub fn count_valid(self,n:usize)->bool { match self { Self::Choice=>(2..=5).contains(&n),Self::Noul=>n==2,Self::Score=>(3..=7).contains(&n) } }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct OptionDef { pub id:String, pub text:String }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct Schema {
    pub id:String, pub version:u64, pub primitive:Primitive,
    /// Full instruction including the direction/meaning of an ordinal scale.
    pub instructions:String, pub options:Vec<OptionDef>,
}
impl Schema {
    pub fn digest(&self, tokenizer:&Digest)->Digest {
        let mut h=Canonical::new("ic-laya/schema/v1");
        h.text(&self.id).u64(self.version).u64(self.primitive.tag()).text(&self.instructions)
            .u64(self.options.len() as u64);
        for o in &self.options { h.text(&o.id).text(&o.text); }
        h.bytes(tokenizer).text(PROFILE); h.finish()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct SpecialTokens { pub cls:u32, pub sep:u32, pub mask:u32, pub pad:u32, pub literals:Vec<String> }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct CompiledSchema {
    pub schema:Schema, pub schema_hash:Digest, pub tokenizer_hash:Digest,
    pub prefix:Vec<u32>, pub markers:Vec<u32>, pub special:SpecialTokens,
    /// Explicit export mapping. Never infer the original qtype order from enum order.
    pub qtype_id:u32,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct TokenInput { pub input_ids:Vec<u32>, pub markers:Vec<u32>, pub qtype_id:u32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub enum BackendKind { Checkpoint, SyntheticFixture }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, CandidType)]
pub struct Calibration {
    pub id:Digest, pub model:Digest, pub schema:Digest, pub tokenizer:Digest,
    pub temperature:f64, pub expires_at_ns:u64,
    pub holdout_hash:Digest, pub sample_count:u64,
    /// Fixture calibrations can never enable real funds.
    pub test_only:bool,
}
impl Calibration {
    pub fn validate(&self,now:u64)->Result<()> {
        if self.id==[0;32] || self.holdout_hash==[0;32] || self.sample_count==0 { return Err(Error::Uncalibrated); }
        if self.expires_at_ns<=now { return Err(Error::Uncalibrated); }
        if !self.temperature.is_finite() || !(1e-6..=1e6).contains(&self.temperature) { return Err(Error::Numeric); }
        Ok(())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct Binding { pub workflow:Digest, pub slot:u8, pub attempt:u32, pub snapshot:Digest }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct DecisionRequest {
    pub evaluation_id:Digest, pub schema_hash:Digest, pub model:Digest,
    pub calibration:Option<Digest>, pub binding:Option<Binding>,
    pub state:String, pub expires_at_ns:u64,
}
impl DecisionRequest {
    pub fn digest(&self)->Digest {
        let mut h=Canonical::new("ic-laya/evaluation/v1");
        h.bytes(&self.evaluation_id).bytes(&self.schema_hash).bytes(&self.model);
        match self.calibration { Some(x)=>{h.u64(1).bytes(&x);},None=>{h.u64(0);} }
        match &self.binding { Some(b)=>{h.u64(1).bytes(&b.workflow).u64(b.slot as u64).u64(b.attempt as u64).bytes(&b.snapshot);},None=>{h.u64(0);} }
        h.text(&self.state).u64(self.expires_at_ns); h.finish()
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct Stamp {
    pub evaluation_id:Digest, pub schema_hash:Digest, pub tokenizer_hash:Digest,
    pub model:Digest, pub calibration:Option<Digest>, pub binding:Option<Binding>,
    pub state_hash:Digest, pub profile:String, pub backend:BackendKind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub enum DecisionValue {
    Choice { option_ids:Vec<String>, mass_ppm:Vec<u32>, selected_id:String },
    Noul { false_ppm:u32, true_ppm:u32 },
    Score { bin_ids:Vec<String>, mass_ppm:Vec<u32>, expected_level_microunits:u64, mean_ppm:u32 },
}
impl DecisionValue {
    pub fn primitive(&self)->Primitive { match self {Self::Choice{..}=>Primitive::Choice,Self::Noul{..}=>Primitive::Noul,Self::Score{..}=>Primitive::Score} }
    pub fn masses(&self)->Vec<u32> { match self {Self::Choice{mass_ppm,..}|Self::Score{mass_ppm,..}=>mass_ppm.clone(),Self::Noul{false_ppm,true_ppm}=>vec![*false_ppm,*true_ppm]} }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct Diagnostics { pub top1_ppm:u32, pub margin_ppm:u32, pub tied:bool }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub enum EvaluationOutcome { Assessed(DecisionValue), Abstained(String), Error(Error) }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct Receipt {
    pub stamp:Stamp, pub outcome:EvaluationOutcome, pub diagnostics:Option<Diagnostics>,
    pub input_tokens:u32, pub measured_instructions:u64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, CandidType)]
pub struct Account { pub owner:Principal, pub subaccount:[u8;32] }
impl Account {
    pub fn write(&self,h:&mut Canonical){h.bytes(self.owner.as_slice()).bytes(&self.subaccount);}
    pub fn validate(&self)->Result<()> {if self.owner==Principal::anonymous() || self.owner==Principal::management_canister(){Err(Error::Invalid("recipient principal".into()))}else{Ok(())}}
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, CandidType)]
pub struct TransferProposal {
    pub ledger:Principal, pub from_subaccount:[u8;32], pub to:Account,
    pub amount:u128, pub fee:u128,
}
impl TransferProposal {
    pub fn digest(&self)->Digest {
        let mut h=Canonical::new("ic-laya/proposal/v1");
        h.bytes(self.ledger.as_slice()).bytes(&self.from_subaccount); self.to.write(&mut h);
        h.u128(self.amount).u128(self.fee); h.finish()
    }
    pub fn charge(&self)->Result<u128>{ self.amount.checked_add(self.fee).ok_or(Error::Budget) }
    pub fn validate(&self)->Result<()>{
        self.to.validate()?;
        if self.amount==0 || self.ledger==Principal::anonymous() || self.ledger==Principal::management_canister(){return Err(Error::Invalid("transfer proposal".into()));}
        self.charge()?;Ok(())
    }
}
impl fmt::Display for Primitive {fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{f.write_str(self.text())}}
