//! Deterministic TEST FIXTURES. No language understanding, no real model weights.
//! Do not use these predictions or thresholds as business decisions.
use crate::{engine::{InferenceBackend,EngineState},schema::{self,TextTokenizer},workflow::*,*};
use candid::Principal;
use std::collections::BTreeMap;
pub const NOW:u64=1_000_000_000_000;
pub fn actor(n:u8)->Principal{Principal::from_slice(&[1,n])}
pub fn fixture_bundle()->Digest{hash(b"TEST-ONLY-STATE-INDEPENDENT-FIXTURE-v1")}
pub struct FixtureTokenizer;
impl TextTokenizer for FixtureTokenizer {
    fn encode_piece(&self,text:&str)->Result<Vec<u32>>{
        Ok(text.split_whitespace().map(|s|{let d=hash(s.as_bytes());10+u16::from_be_bytes([d[0],d[1]]) as u32}).collect())
    }
    fn fingerprint(&self)->Digest{hash(b"TEST-ONLY-WHITESPACE-TOKENIZER-v1")}
    fn special_tokens(&self)->SpecialTokens{SpecialTokens{cls:1,sep:2,mask:3,pad:0,literals:vec!["[CLS]".into(),"[SEP]".into(),"[MASK]".into(),"[PAD]".into()]}}
}
#[derive(Default)]
pub struct FixtureBackend {pub calls:u64}
impl InferenceBackend for FixtureBackend {
    fn bundle_id(&self)->Digest{fixture_bundle()}
    fn kind(&self)->BackendKind{BackendKind::SyntheticFixture}
    fn infer(&mut self,input:&TokenInput)->Result<Vec<f32>>{
        self.calls+=1;
        let v=match (input.qtype_id,input.markers.len()){
            (0,3)=>vec![4.0,-2.0,-3.0],(1,2)=>vec![-4.0,4.0],(2,5)=>vec![5.0,1.0,0.0,-2.0,-4.0],
            _=>return Err(Error::Invalid("fixture supports only demonstration schemas".into())),
        };Ok(v)
    }
}
fn option(id:&str,text:&str)->OptionDef{OptionDef{id:id.into(),text:text.into()}}
pub fn schemas()->Vec<Schema>{vec![
    Schema{id:"RefundRequested".into(),version:1,primitive:Primitive::Noul,instructions:"Does the customer ask for a refund?".into(),options:vec![option("false","false"),option("true","true")]},
    Schema{id:"PaymentAction".into(),version:1,primitive:Primitive::Choice,instructions:"Choose the handling action for this request.".into(),options:vec![option("execute","process refund"),option("reject","reject request"),option("review","human review")]},
    Schema{id:"PaymentRisk".into(),version:1,primitive:Primitive::Score,instructions:"Rate request risk from minimal to severe.".into(),options:vec![option("minimal","minimal risk"),option("low","low risk"),option("medium","moderate risk"),option("high","high risk"),option("severe","severe risk")]},
]}
pub fn setup()->Result<(ExecutorState,EngineState,Digest,Digest)> {
    let owner=actor(1);let delegate=actor(2);let ledger=actor(3);let engine_id=actor(4);let executor_id=actor(5);
    let mut engine=EngineState::new(fixture_bundle());engine.allow_caller(executor_id,30)?;
    let mut required=Vec::new();
    for s in schemas(){
        let tag=s.primitive;let compiled=schema::compile(s,&FixtureTokenizer,tag.tag() as u32)?;
        let calibration=hash(format!("TEST-ONLY-calibration-{}",compiled.schema.id).as_bytes());
        engine.register(compiled.clone())?;
        engine.register_calibration(Calibration{id:calibration,model:fixture_bundle(),schema:compiled.schema_hash,tokenizer:compiled.tokenizer_hash,temperature:1.0,expires_at_ns:NOW+600_000_000_000,holdout_hash:hash(b"SYNTHETIC-NOT-REAL-HOLDOUT"),sample_count:1,test_only:true},NOW)?;
        let rule=match tag{Primitive::Noul=>Rule::NoulTrue{minimum_ppm:900_000},Primitive::Choice=>Rule::ChoiceIs{option_id:"execute".into(),minimum_ppm:900_000},Primitive::Score=>Rule::ScoreTailAtMost{first_bad_bin:3,maximum_ppm:50_000}};
        required.push(RequiredSignal{schema:compiled,calibration,rule});
    }
    let mut executor=ExecutorState::new(executor_id,owner,engine_id);
    let plan_id=hash(b"TEST-ONLY-refund-plan-v1");
    executor.install_plan(owner,Plan{id:plan_id,version:1,model:fixture_bundle(),signals:required})?;
    let op_id=hash(b"trusted-demo-invoice-1");let grant_id=hash(b"TEST-ONLY-grant-1");
    let account=Account{owner:actor(6),subaccount:[0;32]};
    executor.install_operation(owner,Operation{id:op_id,revision:1,evidence:"The customer reports a duplicate payment and requests a refund.".into(),proposal:TransferProposal{ledger,from_subaccount:[0;32],to:account.clone(),amount:100,fee:10},status:OperationStatus::Available})?;
    executor.install_grant(owner,Grant{id:grant_id,revision:1,delegate,plan:plan_id,ledger,from_subaccount:[0;32],recipients:vec![account],max_amount:1000,max_fee:10,total_cap:1000,window_cap:500,window_ns:60_000_000_000,expires_at_ns:NOW+600_000_000_000,revoked:false,total:Usage::default(),windows:BTreeMap::new()})?;
    executor.mock_ledgers.push(ledger);
    Ok((executor,engine,op_id,grant_id))
}
pub fn complete(ex:&mut ExecutorState,en:&mut EngineState,id:Digest)->Result<()> {
    let caller=actor(2);let mut backend=FixtureBackend::default();
    for _ in 0..3 {
        let req=ex.begin_evaluation(caller,id,NOW+1)?;
        let result=en.evaluate(ex.instance,req.clone(),NOW+2,&FixtureTokenizer,&mut backend);
        ex.finish_evaluation(id,req.evaluation_id,result,NOW+3)?;
    }Ok(())
}
