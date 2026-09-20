//! Schema-checked typed views. No typed decision is an authorization capability.
use crate::{math::{self,Distribution},*};
use std::marker::PhantomData;

pub trait DecisionSchema {
    const ID:&'static str;
    const VERSION:u64;
    const PRIMITIVE:Primitive;
    const OPTIONS:&'static [&'static str];
}
pub trait ChoiceSchema:DecisionSchema {
    type Value;
    fn decode_option(id:&str)->Option<Self::Value>;
}
#[derive(Debug,Clone)]
pub struct Choice<T:DecisionSchema> { distribution:Distribution,stamp:Stamp,_type:PhantomData<T> }
#[derive(Debug,Clone)]
pub struct Noul<T:DecisionSchema> { distribution:Distribution,stamp:Stamp,_type:PhantomData<T> }
#[derive(Debug,Clone)]
pub struct Score<T:DecisionSchema> { distribution:Distribution,stamp:Stamp,_type:PhantomData<T> }

fn checked<T:DecisionSchema>(schema:&CompiledSchema,r:&Receipt,expected:&Stamp,kind:Primitive)->Result<Distribution> {
    if schema.schema.id!=T::ID || schema.schema.version!=T::VERSION || T::PRIMITIVE!=kind
        || schema.schema.primitive!=kind || schema.schema.options.iter().map(|o|o.id.as_str()).collect::<Vec<_>>()!=T::OPTIONS
        || r.stamp!=*expected || r.stamp.schema_hash!=schema.schema_hash || r.stamp.tokenizer_hash!=schema.tokenizer_hash
        || r.stamp.profile!=PROFILE {return Err(Error::BindingMismatch);}
    let v=match &r.outcome {EvaluationOutcome::Assessed(v)=>v,EvaluationOutcome::Abstained(_)=>return Err(Error::Uncalibrated),EvaluationOutcome::Error(e)=>return Err(e.clone())};
    let d=math::validate_value(&schema.schema,v)?;
    if r.diagnostics.as_ref()!=Some(&d.diagnostics()){return Err(Error::BindingMismatch);}
    Ok(d)
}
impl<T:ChoiceSchema> Choice<T> {
    pub fn try_from_receipt(schema:&CompiledSchema,r:&Receipt,expected:&Stamp)->Result<Self>{Ok(Self{distribution:checked::<T>(schema,r,expected,Primitive::Choice)?,stamp:r.stamp.clone(),_type:PhantomData})}
    pub fn selected(&self)->Result<T::Value>{T::decode_option(T::OPTIONS[self.distribution.argmax()]).ok_or(Error::BindingMismatch)}
    pub fn distribution(&self)->&Distribution{&self.distribution}
    pub fn stamp(&self)->&Stamp{&self.stamp}
}
impl<T:DecisionSchema> Noul<T> {
    pub fn try_from_receipt(schema:&CompiledSchema,r:&Receipt,expected:&Stamp)->Result<Self>{Ok(Self{distribution:checked::<T>(schema,r,expected,Primitive::Noul)?,stamp:r.stamp.clone(),_type:PhantomData})}
    pub fn p_true_ppm(&self)->u32{self.distribution.as_slice()[1]}
    pub fn stamp(&self)->&Stamp{&self.stamp}
}
impl<T:DecisionSchema> Score<T> {
    pub fn try_from_receipt(schema:&CompiledSchema,r:&Receipt,expected:&Stamp)->Result<Self>{Ok(Self{distribution:checked::<T>(schema,r,expected,Primitive::Score)?,stamp:r.stamp.clone(),_type:PhantomData})}
    pub fn mean_ppm(&self)->u32{self.distribution.mean_ppm()}
    pub fn expected_level_microunits(&self)->u64{self.distribution.expected_level_microunits()}
    pub fn tail_ppm(&self,from_bin:usize)->Result<u32>{self.distribution.tail(from_bin)}
    pub fn cdf_ppm(&self,through_bin:usize)->Result<u32>{self.distribution.cdf(through_bin)}
    pub fn distribution(&self)->&Distribution{&self.distribution}
    pub fn stamp(&self)->&Stamp{&self.stamp}
}
