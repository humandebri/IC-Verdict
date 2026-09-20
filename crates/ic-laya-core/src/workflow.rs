//! ML-independent workflow, budget reservations and outcome-unknown handling.
//! Mutating methods validate first and then commit their in-memory transition.
//! Canister adapters MUST persist before issuing an inter-canister call.
use crate::{math, schema::validate_compiled, *};
use candid::{CandidType, Principal};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_REQUESTS:usize=512;
const MAX_REGISTRY:usize=64;
pub const MOCK_DEDUP_WINDOW_NS:u64=60_000_000_000;

#[derive(Debug,Clone,Copy,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum Mode { ReportOnly, Shadow, Mock, LimitedLive }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum Rule {
    NoulTrue { minimum_ppm:u32 },
    ChoiceIs { option_id:String, minimum_ppm:u32 },
    ScoreTailAtMost { first_bad_bin:u32, maximum_ppm:u32 },
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct RequiredSignal { pub schema:CompiledSchema,pub calibration:Digest,pub rule:Rule }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Plan { pub id:Digest,pub version:u64,pub model:Digest,pub signals:Vec<RequiredSignal> }
impl Plan {
    pub fn validate(&self)->Result<()> {
        if self.version==0 || self.id==[0;32] || self.model==[0;32] || self.signals.is_empty() || self.signals.len()>MAX_SLOTS{return Err(Error::Invalid("plan".into()));}
        for s in &self.signals {
            validate_compiled(&s.schema)?;
            if s.calibration==[0;32]{return Err(Error::Uncalibrated);}
            match &s.rule {
                Rule::NoulTrue{minimum_ppm} if s.schema.schema.primitive==Primitive::Noul && *minimum_ppm<=PPM=>{},
                Rule::ChoiceIs{option_id,minimum_ppm} if s.schema.schema.primitive==Primitive::Choice && *minimum_ppm<=PPM && s.schema.schema.options.iter().any(|o|&o.id==option_id)=>{},
                Rule::ScoreTailAtMost{first_bad_bin,maximum_ppm} if s.schema.schema.primitive==Primitive::Score && (*first_bad_bin as usize)<s.schema.schema.options.len() && *maximum_ppm<=PPM=>{},
                _=>return Err(Error::Invalid("rule/schema mismatch".into())),
            }
        } Ok(())
    }
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum OperationStatus { Available, Reserved(Digest), Consumed(Digest) }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Operation { pub id:Digest,pub revision:u64,pub evidence:String,pub proposal:TransferProposal,pub status:OperationStatus }
#[derive(Debug,Clone,Default,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Usage { pub spent:u128,pub reserved:u128 }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Grant {
    pub id:Digest,pub revision:u64,pub delegate:Principal,pub plan:Digest,
    pub ledger:Principal,pub from_subaccount:[u8;32],pub recipients:Vec<Account>,
    pub max_amount:u128,pub max_fee:u128,pub total_cap:u128,pub window_cap:u128,
    pub window_ns:u64,pub expires_at_ns:u64,pub revoked:bool,
    pub total:Usage,pub windows:BTreeMap<u64,Usage>,
}
impl Grant {
    fn eligible(&self,caller:Principal,p:&TransferProposal,now:u64)->Result<()> {
        if caller!=self.delegate || self.revoked || self.expires_at_ns<=now{return Err(Error::Unauthorized);}
        if self.ledger!=p.ledger || self.from_subaccount!=p.from_subaccount || !self.recipients.contains(&p.to)
            || p.amount>self.max_amount || p.fee>self.max_fee{return Err(Error::Denied("ledger/account/amount/fee".into()));}
        p.validate()?; Ok(())
    }
    fn reserve(&mut self,charge:u128,epoch:u64)->Result<()> {
        if !self.windows.contains_key(&epoch)&&self.windows.len()>=64{return Err(Error::Capacity);}
        let window=self.windows.get(&epoch).cloned().unwrap_or_default();
        let total=self.total.spent.checked_add(self.total.reserved).and_then(|x|x.checked_add(charge)).ok_or(Error::Budget)?;
        let win=window.spent.checked_add(window.reserved).and_then(|x|x.checked_add(charge)).ok_or(Error::Budget)?;
        if total>self.total_cap || win>self.window_cap{return Err(Error::Budget);}
        let total_reserved=self.total.reserved.checked_add(charge).ok_or(Error::Budget)?;
        let window_reserved=window.reserved.checked_add(charge).ok_or(Error::Budget)?;
        self.total.reserved=total_reserved;self.windows.insert(epoch,Usage{spent:window.spent,reserved:window_reserved});Ok(())
    }
    fn settle(&mut self,charge:u128,epoch:u64,success:bool)->Result<()> {
        let old=self.windows.get(&epoch).cloned().ok_or(Error::Storage)?;
        let tr=self.total.reserved.checked_sub(charge).ok_or(Error::Storage)?;
        let wr=old.reserved.checked_sub(charge).ok_or(Error::Storage)?;
        let ts=if success{self.total.spent.checked_add(charge).ok_or(Error::Budget)?}else{self.total.spent};
        let ws=if success{old.spent.checked_add(charge).ok_or(Error::Budget)?}else{old.spent};
        self.total=Usage{spent:ts,reserved:tr};self.windows.insert(epoch,Usage{spent:ws,reserved:wr});Ok(())
    }
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum Status { Received, Evaluating, ReadyToDispatch, Reported, NeedsReview(String), Stale, Cancelled, Submitted, OutcomeUnknown(String), Succeeded(String), FailedDefinitive(String) }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct PendingEvaluation { pub request:DecisionRequest,pub in_flight:bool,pub sends:u8 }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct Reservation { pub charge:u128,pub epoch:u64 }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct FrozenTransfer { pub proposal:TransferProposal,pub memo:Digest,pub created_at_time_ns:u64 }
impl FrozenTransfer {
    pub fn digest(&self)->Digest {let mut h=Canonical::new("ic-laya/frozen-transfer/v1");h.bytes(&self.proposal.digest()).bytes(&self.memo).u64(self.created_at_time_ns);h.finish()}
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct RequestRecord {
    pub id:Digest,pub caller:Principal,pub nonce:u64,pub operation:Digest,pub grant:Digest,pub plan:Digest,
    pub operation_revision:u64,pub grant_revision:u64,pub snapshot:Digest,pub expires_at_ns:u64,
    pub status:Status,pub receipts:Vec<Receipt>,pub pending:Option<PendingEvaluation>,
    pub reservation:Option<Reservation>,pub frozen:Option<FrozenTransfer>,pub ledger_attempt:u8,pub ever_unknown:bool,
}
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub struct DispatchCommand { pub request:Digest,pub attempt:u8,pub transfer:FrozenTransfer }
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize,CandidType)]
pub enum LedgerOutcome { Success(String), Duplicate(String), DefinitiveError(String), Unknown(String) }

/// A local, non-serializable one-use value. Network DTOs cannot construct this.
///
/// ```compile_fail
/// use ic_laya_core::workflow::AuthorizedTransfer;
/// let forged = AuthorizedTransfer { request: [0;32], snapshot: [0;32] };
/// ```
#[derive(Debug)]
pub struct AuthorizedTransfer { request:Digest,snapshot:Digest }

#[derive(Debug,Clone,Serialize,Deserialize,CandidType)]
pub struct ExecutorState {
    pub instance:Principal,pub owner:Principal,pub engine:Principal,pub mode:Mode,pub paused:bool,
    pub plans:BTreeMap<Digest,Plan>,pub operations:BTreeMap<Digest,Operation>,pub grants:BTreeMap<Digest,Grant>,
    pub requests:BTreeMap<Digest,RequestRecord>,pub next_nonce:BTreeMap<Principal,u64>,
    /// Only ledgers that passed the mock-only handshake may receive this release's outcalls.
    pub mock_ledgers:Vec<Principal>,
}
impl ExecutorState {
    pub fn new(instance:Principal,owner:Principal,engine:Principal)->Self {Self{instance,owner,engine,mode:Mode::ReportOnly,paused:false,plans:BTreeMap::new(),operations:BTreeMap::new(),grants:BTreeMap::new(),requests:BTreeMap::new(),next_nonce:BTreeMap::new(),mock_ledgers:Vec::new()}}
    pub fn assert_owner(&self,caller:Principal)->Result<()> {if caller!=self.owner || caller==Principal::anonymous(){Err(Error::Unauthorized)}else{Ok(())}}
    pub fn set_mode(&mut self,caller:Principal,mode:Mode)->Result<()> {self.assert_owner(caller)?;if mode==Mode::LimitedLive{return Err(Error::LiveDisabled);}self.mode=mode;Ok(())}
    pub fn set_paused(&mut self,caller:Principal,pause:bool)->Result<()> {self.assert_owner(caller)?;self.paused=pause;Ok(())}
    pub fn install_plan(&mut self,caller:Principal,p:Plan)->Result<()> {
        self.assert_owner(caller)?;p.validate()?;
        if let Some(old)=self.plans.get(&p.id){return if old==&p{Ok(())}else{Err(Error::IdConflict)};}
        if self.plans.len()>=MAX_REGISTRY{return Err(Error::Capacity);}self.plans.insert(p.id,p);Ok(())
    }
    pub fn install_operation(&mut self,caller:Principal,o:Operation)->Result<()> {
        self.assert_owner(caller)?;o.proposal.validate()?;
        if o.id==[0;32] || o.revision==0 || o.evidence.is_empty() || o.evidence.len()>MAX_STATE_BYTES || o.status!=OperationStatus::Available{return Err(Error::Invalid("operation".into()));}
        if self.operations.contains_key(&o.id){return Err(Error::IdConflict);}
        if self.operations.len()>=MAX_REQUESTS{return Err(Error::Capacity);}self.operations.insert(o.id,o);Ok(())
    }
    pub fn revise_operation(&mut self,caller:Principal,id:Digest,evidence:String)->Result<()> {
        self.assert_owner(caller)?;
        if evidence.is_empty() || evidence.len()>MAX_STATE_BYTES{return Err(Error::TooLong);}
        let o=self.operations.get_mut(&id).ok_or(Error::NotFound)?;
        if o.status!=OperationStatus::Available{return Err(Error::OperationUsed);}
        let rev=o.revision.checked_add(1).ok_or(Error::Capacity)?;o.evidence=evidence;o.revision=rev;Ok(())
    }
    pub fn install_grant(&mut self,caller:Principal,g:Grant)->Result<()> {
        self.assert_owner(caller)?;
        if g.id==[0;32] || g.revision==0 || g.delegate==Principal::anonymous() || g.window_ns==0 || g.total!=Usage::default() || !g.windows.is_empty()
            || g.revoked || g.recipients.is_empty() || g.recipients.len()>16 || g.max_amount==0 || g.total_cap==0 || g.window_cap==0 {return Err(Error::Invalid("grant".into()));}
        for r in &g.recipients{r.validate()?;}
        if !self.plans.contains_key(&g.plan){return Err(Error::NotFound);}
        if self.grants.contains_key(&g.id){return Err(Error::IdConflict);}
        if self.grants.len()>=MAX_REGISTRY{return Err(Error::Capacity);}self.grants.insert(g.id,g);Ok(())
    }
    pub fn revoke(&mut self,caller:Principal,id:Digest)->Result<()> {self.assert_owner(caller)?;let g=self.grants.get_mut(&id).ok_or(Error::NotFound)?;let rev=g.revision.checked_add(1).ok_or(Error::Capacity)?;g.revoked=true;g.revision=rev;Ok(())}
    fn request_id(&self,caller:Principal,nonce:u64)->Digest {let mut h=Canonical::new("ic-laya/workflow/v1");h.bytes(self.instance.as_slice()).bytes(caller.as_slice()).u64(nonce);h.finish()}
    fn snapshot(o:&Operation,g:&Grant,p:&Plan)->Digest {
        let mut h=Canonical::new("ic-laya/snapshot/v1");h.bytes(&o.id).u64(o.revision).text(&o.evidence).bytes(&o.proposal.digest())
            .bytes(&g.id).u64(g.revision).bytes(&p.id).u64(p.version).bytes(&p.model);
        for s in &p.signals{h.bytes(&s.schema.schema_hash).bytes(&s.calibration);}h.finish()
    }
    pub fn submit(&mut self,caller:Principal,nonce:u64,operation:Digest,grant:Digest,now:u64)->Result<Digest> {
        if self.paused{return Err(Error::Denied("paused".into()));}
        if caller==Principal::anonymous(){return Err(Error::Unauthorized);}
        let next=self.next_nonce.get(&caller).copied().unwrap_or(0);let id=self.request_id(caller,nonce);
        if nonce<next {return match self.requests.get(&id){Some(r) if r.operation==operation&&r.grant==grant=>Ok(id),Some(_)=>Err(Error::IdConflict),None=>Err(Error::NonceConsumed)}}
        if nonce>next{return Err(Error::NonceGap);}
        if self.requests.len()>=MAX_REQUESTS || (!self.next_nonce.contains_key(&caller)&&self.next_nonce.len()>=MAX_REGISTRY){return Err(Error::Capacity);}
        let g=self.grants.get(&grant).ok_or(Error::NotFound)?;let o=self.operations.get(&operation).ok_or(Error::NotFound)?;
        g.eligible(caller,&o.proposal,now)?;if o.status!=OperationStatus::Available{return Err(Error::OperationUsed);}
        let p=self.plans.get(&g.plan).ok_or(Error::NotFound)?;
        let expires=now.checked_add(300_000_000_000).ok_or(Error::Expired)?.min(g.expires_at_ns);
        let new_nonce=next.checked_add(1).ok_or(Error::Capacity)?;
        let r=RequestRecord{id,caller,nonce,operation,grant,plan:g.plan,operation_revision:o.revision,grant_revision:g.revision,snapshot:Self::snapshot(o,g,p),expires_at_ns:expires,
            status:Status::Received,receipts:Vec::new(),pending:None,reservation:None,frozen:None,ledger_attempt:0,ever_unknown:false};
        self.requests.insert(id,r);self.next_nonce.insert(caller,new_nonce);Ok(id)
    }
    fn owned(&self,caller:Principal,id:&Digest)->Result<RequestRecord> {
        let r=self.requests.get(id).ok_or(Error::NotFound)?;
        if caller!=r.caller&&caller!=self.owner{return Err(Error::Unauthorized);}Ok(r.clone())
    }
    fn current(&self,r:&RequestRecord,now:u64)->Result<()> {
        if self.paused{return Err(Error::Denied("paused".into()));}
        if now>=r.expires_at_ns{return Err(Error::Expired);}
        let o=self.operations.get(&r.operation).ok_or(Error::NotFound)?;let g=self.grants.get(&r.grant).ok_or(Error::NotFound)?;let p=self.plans.get(&r.plan).ok_or(Error::NotFound)?;
        g.eligible(r.caller,&o.proposal,now)?;
        if o.revision!=r.operation_revision || g.revision!=r.grant_revision || Self::snapshot(o,g,p)!=r.snapshot{return Err(Error::Stale);}
        Ok(())
    }
    pub fn begin_evaluation(&mut self,caller:Principal,id:Digest,now:u64)->Result<DecisionRequest> {
        let mut r=self.owned(caller,&id)?;
        if !matches!(r.status,Status::Received|Status::Evaluating){return Err(Error::Transition);}
        if let Err(e)=self.current(&r,now){r.status=Status::Stale;r.pending=None;self.requests.insert(id,r);return Err(e);}
        if let Some(p)=r.pending.as_mut(){
            if p.in_flight{return Err(Error::Busy);}if p.sends>=2{return Err(Error::Capacity);}
            p.in_flight=true;p.sends+=1;let req=p.request.clone();self.requests.insert(id,r);return Ok(req);
        }
        let p=self.plans.get(&r.plan).ok_or(Error::NotFound)?;let slot=r.receipts.len();
        let required=p.signals.get(slot).ok_or(Error::Transition)?;let o=self.operations.get(&r.operation).ok_or(Error::NotFound)?;
        let mut h=Canonical::new("ic-laya/slot/v1");h.bytes(&id).u64(slot as u64).bytes(&r.snapshot);
        let req=DecisionRequest{evaluation_id:h.finish(),schema_hash:required.schema.schema_hash,model:p.model,calibration:Some(required.calibration),
            binding:Some(Binding{workflow:id,slot:slot as u8,attempt:0,snapshot:r.snapshot}),state:o.evidence.clone(),expires_at_ns:r.expires_at_ns};
        r.pending=Some(PendingEvaluation{request:req.clone(),in_flight:true,sends:1});r.status=Status::Evaluating;self.requests.insert(id,r);Ok(req)
    }
    pub fn engine_transport_failed(&mut self,id:Digest,eval_id:Digest)->Result<()> {
        let r=self.requests.get_mut(&id).ok_or(Error::NotFound)?;
        if r.status!=Status::Evaluating{return Err(Error::Transition);}
        let p=r.pending.as_mut().ok_or(Error::Transition)?;if p.request.evaluation_id!=eval_id{return Err(Error::BindingMismatch);}
        p.in_flight=false;if p.sends>=2{r.status=Status::NeedsReview("engine retry limit".into());r.pending=None;}Ok(())
    }
    fn expected_stamp(&self,r:&RequestRecord,slot:usize,req:&DecisionRequest,backend:BackendKind)->Result<Stamp> {
        let s=self.plans.get(&r.plan).and_then(|p|p.signals.get(slot)).ok_or(Error::NotFound)?;
        Ok(Stamp{evaluation_id:req.evaluation_id,schema_hash:s.schema.schema_hash,tokenizer_hash:s.schema.tokenizer_hash,model:req.model,calibration:req.calibration,
            binding:req.binding.clone(),state_hash:hash(req.state.as_bytes()),profile:PROFILE.into(),backend})
    }
    fn check_rule(required:&RequiredSignal,receipt:&Receipt)->Result<bool> {
        let v=match &receipt.outcome{EvaluationOutcome::Assessed(v)=>v,EvaluationOutcome::Abstained(_)=>return Ok(false),EvaluationOutcome::Error(e)=>return Err(e.clone())};
        let d=math::validate_value(&required.schema.schema,v)?;
        if receipt.diagnostics.as_ref()!=Some(&d.diagnostics()){return Err(Error::BindingMismatch);}
        Ok(match &required.rule {
            Rule::NoulTrue{minimum_ppm}=>d.as_slice()[1]>=*minimum_ppm && !d.diagnostics().tied,
            Rule::ChoiceIs{option_id,minimum_ppm}=>required.schema.schema.options[d.argmax()].id==*option_id && d.as_slice()[d.argmax()]>=*minimum_ppm && !d.diagnostics().tied,
            Rule::ScoreTailAtMost{first_bad_bin,maximum_ppm}=>d.tail(*first_bad_bin as usize)?<=*maximum_ppm,
        })
    }
    /// Called only by the adapter holding the outstanding engine call; not a public endpoint.
    pub fn finish_evaluation(&mut self,id:Digest,eval_id:Digest,result:Result<Receipt>,now:u64)->Result<Status> {
        let mut r=self.requests.get(&id).cloned().ok_or(Error::NotFound)?;
        if let Ok(incoming)=&result {
            if let Some(old)=r.receipts.iter().find(|v|v.stamp.evaluation_id==eval_id){return if old==incoming{Ok(r.status)}else{Err(Error::IdConflict)}}
        }
        if r.status!=Status::Evaluating{return Err(Error::Transition);}
        let pending=r.pending.clone().ok_or(Error::Transition)?;
        if pending.request.evaluation_id!=eval_id{return Err(Error::BindingMismatch);}
        if let Err(e)=self.current(&r,now){r.status=Status::Stale;r.pending=None;self.requests.insert(id,r);return Err(e);}
        let slot=r.receipts.len();
        let required=self.plans.get(&r.plan).and_then(|p|p.signals.get(slot)).ok_or(Error::NotFound)?;
        let outcome=(||{
            let receipt=result?;
            let expected=self.expected_stamp(&r,slot,&pending.request,receipt.stamp.backend)?;
            if receipt.stamp!=expected || receipt.input_tokens==0 || receipt.input_tokens>MAX_TOKENS as u32{return Err(Error::BindingMismatch);}
            let pass=Self::check_rule(required,&receipt)?;Ok((receipt,pass))
        })();
        r.pending=None;
        match outcome {
            Ok((receipt,pass))=>{
                r.receipts.push(receipt);
                r.status=if !pass{Status::NeedsReview("registered rule did not pass".into())}else if r.receipts.len()==self.plans[&r.plan].signals.len(){Status::ReadyToDispatch}else{Status::Received};
            },
            Err(e)=>r.status=Status::NeedsReview(e.to_string()),
        }
        let status=r.status.clone();self.requests.insert(id,r);Ok(status)
    }
    pub fn authorize(&self,caller:Principal,id:Digest,now:u64)->Result<AuthorizedTransfer> {
        let r=self.owned(caller,&id)?;self.current(&r,now)?;
        if r.status!=Status::ReadyToDispatch || r.pending.is_some(){return Err(Error::Transition);}
        let p=self.plans.get(&r.plan).ok_or(Error::NotFound)?;
        if r.receipts.len()!=p.signals.len(){return Err(Error::BindingMismatch);}
        for (i,(s,v)) in p.signals.iter().zip(&r.receipts).enumerate(){
            if v.stamp.binding.as_ref()!=Some(&Binding{workflow:id,slot:i as u8,attempt:0,snapshot:r.snapshot})
                || v.stamp.model!=p.model || v.stamp.schema_hash!=s.schema.schema_hash || v.stamp.calibration!=Some(s.calibration)
                || !Self::check_rule(s,v)? {return Err(Error::BindingMismatch);}
        }
        let o=self.operations.get(&r.operation).ok_or(Error::NotFound)?;
        if o.status!=OperationStatus::Available{return Err(Error::OperationUsed);}
        Ok(AuthorizedTransfer{request:id,snapshot:r.snapshot})
    }
    pub fn prepare_dispatch(&mut self,authorization:AuthorizedTransfer,now:u64)->Result<DispatchCommand> {
        let mut r=self.requests.get(&authorization.request).cloned().ok_or(Error::NotFound)?;
        self.authorize(r.caller,r.id,now)?;
        if authorization.snapshot!=r.snapshot{return Err(Error::BindingMismatch);}
        match self.mode {
            Mode::ReportOnly|Mode::Shadow=>{r.status=Status::Reported;self.requests.insert(r.id,r);return Err(Error::ReportOnly);},
            Mode::LimitedLive=>return Err(Error::LiveDisabled),Mode::Mock=>{},
        }
        let mut op=self.operations.get(&r.operation).cloned().ok_or(Error::NotFound)?;
        if !self.mock_ledgers.contains(&op.proposal.ledger){return Err(Error::Denied("unregistered mock ledger".into()));}
        let mut grant=self.grants.get(&r.grant).cloned().ok_or(Error::NotFound)?;
        let charge=op.proposal.charge()?;let epoch=now/grant.window_ns;grant.reserve(charge,epoch)?;
        let frozen=FrozenTransfer{proposal:op.proposal.clone(),memo:r.id,created_at_time_ns:now};
        r.reservation=Some(Reservation{charge,epoch});r.frozen=Some(frozen.clone());r.status=Status::Submitted;r.ledger_attempt=1;
        op.status=OperationStatus::Reserved(r.id);
        let cmd=DispatchCommand{request:r.id,attempt:1,transfer:frozen};
        self.grants.insert(grant.id,grant);self.operations.insert(op.id,op);self.requests.insert(r.id,r);Ok(cmd)
    }
    fn settle_record(&mut self,mut r:RequestRecord,success:bool,text:String)->Result<Status> {
        let reservation=r.reservation.clone().ok_or(Error::Storage)?;
        let mut g=self.grants.get(&r.grant).cloned().ok_or(Error::Storage)?;
        let mut o=self.operations.get(&r.operation).cloned().ok_or(Error::Storage)?;
        if o.status!=OperationStatus::Reserved(r.id){return Err(Error::Storage);}
        g.settle(reservation.charge,reservation.epoch,success)?;
        o.status=if success{OperationStatus::Consumed(r.id)}else{OperationStatus::Available};
        r.reservation=None;r.status=if success{Status::Succeeded(text)}else{Status::FailedDefinitive(text)};
        let status=r.status.clone();self.grants.insert(g.id,g);self.operations.insert(o.id,o);self.requests.insert(r.id,r);Ok(status)
    }
    pub fn finish_ledger(&mut self,cmd:&DispatchCommand,outcome:LedgerOutcome)->Result<Status> {
        let mut r=self.requests.get(&cmd.request).cloned().ok_or(Error::NotFound)?;
        if r.frozen.as_ref()!=Some(&cmd.transfer) || cmd.attempt==0 || cmd.attempt>r.ledger_attempt{return Err(Error::BindingMismatch);}
        // Terminal successes cannot be undone by a late error callback.
        if matches!(r.status,Status::Succeeded(_)){return Ok(r.status);}
        if !matches!(r.status,Status::Submitted|Status::OutcomeUnknown(_)){return Err(Error::Transition);}
        match outcome {
            LedgerOutcome::Success(index)|LedgerOutcome::Duplicate(index)=>{
                if index.is_empty() || index.len()>128 || !index.bytes().all(|c|c.is_ascii_digit()){return Err(Error::Invalid("block index".into()));}
                self.settle_record(r,true,index)
            },
            LedgerOutcome::DefinitiveError(e) if !r.ever_unknown && cmd.attempt==r.ledger_attempt=>self.settle_record(r,false,e),
            LedgerOutcome::DefinitiveError(e)|LedgerOutcome::Unknown(e)=>{
                r.ever_unknown=true;r.status=Status::OutcomeUnknown(e);let status=r.status.clone();self.requests.insert(r.id,r);Ok(status)
            },
        }
    }
    /// Mock-only identical-payload retry; never releases or doubles the reservation.
    pub fn retry_unknown(&mut self,caller:Principal,id:Digest,now:u64)->Result<DispatchCommand> {
        let mut r=self.owned(caller,&id)?;self.current(&r,now)?;
        if self.mode!=Mode::Mock || !matches!(r.status,Status::OutcomeUnknown(_)) || r.ledger_attempt>=2{return Err(Error::Transition);}
        let frozen=r.frozen.clone().ok_or(Error::Storage)?;
        if now<frozen.created_at_time_ns || now-frozen.created_at_time_ns>MOCK_DEDUP_WINDOW_NS{return Err(Error::Expired);}
        r.ledger_attempt+=1;r.status=Status::Submitted;
        let cmd=DispatchCommand{request:id,attempt:r.ledger_attempt,transfer:frozen};self.requests.insert(id,r);Ok(cmd)
    }
    pub fn cancel(&mut self,caller:Principal,id:Digest)->Result<()> {
        let mut r=self.owned(caller,&id)?;
        if r.frozen.is_some() || r.reservation.is_some(){return Err(Error::OutcomeUnknown);}
        r.pending=None;r.status=Status::Cancelled;self.requests.insert(id,r);Ok(())
    }
    pub fn get(&self,caller:Principal,id:Digest)->Result<RequestRecord>{self.owned(caller,&id)}
    /// Conservative restart recovery. Never rewind a nonce or release an unknown transfer.
    pub fn recover_after_upgrade(&mut self){
        for r in self.requests.values_mut(){
            if r.status==Status::Evaluating {r.status=Status::NeedsReview("upgrade interrupted inference".into());r.pending=None;}
            if r.status==Status::Submitted {r.status=Status::OutcomeUnknown("upgrade interrupted callback".into());r.ever_unknown=true;}
        }
        self.paused=true;
    }
    pub fn check_invariants(&self)->Result<()> {
        for g in self.grants.values(){
            let expected=self.requests.values().filter(|r|r.grant==g.id).filter_map(|r|r.reservation.as_ref()).try_fold(0u128,|a,r|a.checked_add(r.charge).ok_or(Error::Budget))?;
            if expected!=g.total.reserved{return Err(Error::Storage);}
            if g.total.spent.checked_add(g.total.reserved).ok_or(Error::Budget)?>g.total_cap{return Err(Error::Budget);}
            for (&epoch,u) in &g.windows {
                let actual=self.requests.values().filter(|r|r.grant==g.id).filter_map(|r|r.reservation.as_ref()).filter(|r|r.epoch==epoch).map(|r|r.charge).sum::<u128>();
                if u.reserved!=actual || u.spent.checked_add(u.reserved).ok_or(Error::Budget)?>g.window_cap{return Err(Error::Storage);}
            }
        }
        for r in self.requests.values(){
            let op=self.operations.get(&r.operation).ok_or(Error::Storage)?;
            if r.reservation.is_some() && (op.status!=OperationStatus::Reserved(r.id) || !matches!(r.status,Status::Submitted|Status::OutcomeUnknown(_))){return Err(Error::Storage);}
            if matches!(r.status,Status::Succeeded(_)) && (r.reservation.is_some() || op.status!=OperationStatus::Consumed(r.id)){return Err(Error::Storage);}
            if self.next_nonce.get(&r.caller).copied().unwrap_or(0)<=r.nonce{return Err(Error::Storage);}
        }Ok(())
    }
}
