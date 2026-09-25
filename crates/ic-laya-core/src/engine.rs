use crate::{math,schema::{self,TextTokenizer},*};
use candid::{CandidType,Principal};
use serde::{Deserialize,Serialize};
use std::collections::BTreeMap;

pub trait InferenceBackend {
    fn bundle_id(&self)->Digest;
    fn kind(&self)->BackendKind;
    /// Returns unnormalized option logits in the compiled marker order.
    fn infer(&mut self,input:&TokenInput)->Result<Vec<f32>>;
}
#[derive(Debug,Clone,Serialize,Deserialize,CandidType)]
pub struct Cached { pub fingerprint:Digest, pub result:Result<Receipt>, pub stored_at:u64 }
#[derive(Debug,Clone,Serialize,Deserialize,CandidType)]
pub struct CallerQuota { pub epoch:u64,pub used:u32,pub max_per_minute:u32 }
#[derive(Debug,Clone,Serialize,Deserialize,CandidType)]
pub struct EngineState {
    pub active_model:Digest,
    pub schemas:BTreeMap<Digest,CompiledSchema>,
    pub calibrations:BTreeMap<Digest,Calibration>,
    pub callers:BTreeMap<Principal,CallerQuota>,
    pub cache:BTreeMap<(Principal,Digest),Cached>,
    pub max_cache_entries:u32,
}
impl EngineState {
    pub fn new(model:Digest)->Self {Self{active_model:model,schemas:BTreeMap::new(),calibrations:BTreeMap::new(),callers:BTreeMap::new(),cache:BTreeMap::new(),max_cache_entries:1024}}
    pub fn allow_caller(&mut self,p:Principal,per_minute:u32)->Result<()> {
        if p==Principal::anonymous() || p==Principal::management_canister() || per_minute==0 || per_minute>1000{return Err(Error::Invalid("caller/quota".into()));}
        if !self.callers.contains_key(&p)&&self.callers.len()>=64{return Err(Error::Capacity);}
        self.callers.insert(p,CallerQuota{epoch:0,used:0,max_per_minute:per_minute});Ok(())
    }
    pub fn register(&mut self,s:CompiledSchema)->Result<()> {
        schema::validate_compiled(&s)?;
        if let Some(old)=self.schemas.get(&s.schema_hash){if old!=&s{return Err(Error::IdConflict);}return Ok(());}
        if self.schemas.len()>=64{return Err(Error::Capacity);}
        self.schemas.insert(s.schema_hash,s);Ok(())
    }
    pub fn register_calibration(&mut self,c:Calibration,now:u64)->Result<()> {
        c.validate(now)?;
        let s=self.schemas.get(&c.schema).ok_or(Error::NotFound)?;
        if c.model!=self.active_model || c.tokenizer!=s.tokenizer_hash{return Err(Error::BindingMismatch);}
        if let Some(old)=self.calibrations.get(&c.id){return if old==&c{Ok(())}else{Err(Error::IdConflict)};}
        if self.calibrations.len()>=128{return Err(Error::Capacity);}
        self.calibrations.insert(c.id,c);Ok(())
    }
    pub fn evaluate<T:TextTokenizer,B:InferenceBackend>(&mut self,caller:Principal,req:DecisionRequest,now:u64,tokenizer:&T,backend:&mut B)->Result<Receipt> {
        if backend.bundle_id()!=req.model{return Err(Error::BindingMismatch);}
        self.evaluate_with(caller,req,now,|s,_,state|{
            let input=schema::render(s,tokenizer,state)?;
            let logits=backend.infer(&input)?;
            Ok((logits,input.input_ids.len() as u32,backend.kind(),0))
        })
    }
    /// Shared durable contract for local and canister-specific inference backends.
    /// The callback owns prompt rendering and returns logits in registered option order.
    pub fn evaluate_with<F>(&mut self,caller:Principal,req:DecisionRequest,now:u64,infer:F)->Result<Receipt>
    where F:FnOnce(&CompiledSchema,f64,&str)->Result<(Vec<f32>,u32,BackendKind,u64)> {
        if !self.callers.contains_key(&caller){return Err(Error::Unauthorized);}
        if req.state.len()>MAX_STATE_BYTES || req.state.is_empty(){return Err(Error::TooLong);}
        // Bound expiration to avoid permanent attacker-selected cache lifetimes.
        if req.expires_at_ns<=now || req.expires_at_ns-now>MAX_EVALUATION_WINDOW_NS{return Err(Error::Expired);}
        let key=(caller,req.evaluation_id);let fingerprint=req.digest();
        if req.model!=self.active_model{return Err(Error::BindingMismatch);}
        let s=self.schemas.get(&req.schema_hash).cloned().ok_or(Error::NotFound)?;
        let temp=if let Some(id)=req.calibration {
            let c=self.calibrations.get(&id).ok_or(Error::Uncalibrated)?;c.validate(now)?;
            if c.model!=req.model || c.schema!=req.schema_hash || c.tokenizer!=s.tokenizer_hash{return Err(Error::BindingMismatch);}
            if req.expires_at_ns>c.expires_at_ns{return Err(Error::Uncalibrated);} c.temperature
        }else{1.0};
        if let Some(c)=self.cache.get(&key){return if c.fingerprint==fingerprint{c.result.clone()}else{Err(Error::IdConflict)};}
        if self.cache.len()>=self.max_cache_entries as usize{
            // Without eviction the cap was a lifetime limit: after `max_cache_entries`
            // distinct evaluations the engine refused every new one forever. Results
            // older than the longest acceptable acceptance window are dropped first;
            // inside that window the cache keeps its idempotency guarantee.
            let cutoff=now.saturating_sub(MAX_EVALUATION_WINDOW_NS);
            self.cache.retain(|_,c|c.stored_at>=cutoff);
            if self.cache.len()>=self.max_cache_entries as usize{return Err(Error::Capacity);}
        }
        let quota=self.callers.get_mut(&caller).ok_or(Error::Unauthorized)?;
        let epoch=now/60_000_000_000;
        if quota.epoch!=epoch {quota.epoch=epoch;quota.used=0;}
        if quota.used>=quota.max_per_minute{return Err(Error::Capacity);}quota.used+=1;
        let result=(||{
            let (logits,input_tokens,backend,measured_instructions)=infer(&s,temp,&req.state)?;
            if logits.len()!=s.schema.options.len(){return Err(Error::BindingMismatch);}
            let d=math::from_logits(&logits,temp)?;
            let stamp=Stamp{evaluation_id:req.evaluation_id,schema_hash:req.schema_hash,tokenizer_hash:s.tokenizer_hash,model:req.model,calibration:req.calibration,binding:req.binding.clone(),state_hash:hash(req.state.as_bytes()),profile:PROFILE.into(),backend};
            Ok(Receipt{stamp,outcome:EvaluationOutcome::Assessed(math::value(&s.schema,&d)?),diagnostics:Some(d.diagnostics()),input_tokens,measured_instructions})
        })();
        // Cache failures too: a bad backend cannot force unlimited retries under one ID.
        self.cache.insert(key,Cached{fingerprint,stored_at:now,result:result.clone()});result
    }
}
