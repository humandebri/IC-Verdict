use crate::*;
use std::collections::BTreeSet;

/// Must encode individual pieces without automatic special tokens or truncation.
pub trait TextTokenizer {
    fn encode_piece(&self,text:&str)->Result<Vec<u32>>;
    fn fingerprint(&self)->Digest;
    fn special_tokens(&self)->SpecialTokens;
}
fn identifier(s:&str)->bool { !s.is_empty() && s.len()<=64 && s.bytes().all(|c|c.is_ascii_alphanumeric() || b"_-.:".contains(&c)) }
fn contains_reserved(s:&str,t:&SpecialTokens)->bool { t.literals.iter().any(|x|!x.is_empty()&&s.contains(x)) }
fn control(ids:&[u32],s:&SpecialTokens)->bool {ids.iter().any(|x|[s.cls,s.sep,s.mask,s.pad].contains(x))}

pub fn validate_schema(s:&Schema)->Result<()> {
    if !identifier(&s.id) || s.version==0 || s.instructions.trim().is_empty() || s.instructions.len()>2048 || !s.primitive.count_valid(s.options.len()) {return Err(Error::Invalid("schema header".into()));}
    let mut ids=BTreeSet::new();
    for o in &s.options {
        if !identifier(&o.id) || o.text.trim().is_empty() || o.text.len()>1024 || !ids.insert(o.id.as_str()){return Err(Error::Invalid("option id/text".into()));}
    }
    if s.primitive==Primitive::Noul && (s.options[0].id!="false" || s.options[1].id!="true"){return Err(Error::Invalid("Noul order must be false,true".into()));}
    Ok(())
}

pub fn compile<T:TextTokenizer>(s:Schema,t:&T,qtype_id:u32)->Result<CompiledSchema> {
    validate_schema(&s)?;
    if qtype_id>=3 {return Err(Error::Invalid("qtype mapping".into()));}
    let special=t.special_tokens();
    let unique:BTreeSet<u32>=[special.cls,special.sep,special.mask,special.pad].into_iter().collect();
    if unique.len()!=4 || special.literals.is_empty(){return Err(Error::Invalid("special token config".into()));}
    if contains_reserved(&s.instructions,&special) || s.options.iter().any(|o|contains_reserved(&o.text,&special)){return Err(Error::Invalid("reserved token literal".into()));}
    let question=format!("{} question: {}",s.primitive.text(),s.instructions);
    let instruction_ids=t.encode_piece(&question)?;
    if instruction_ids.is_empty() || control(&instruction_ids,&special){return Err(Error::Invalid("instruction tokens".into()));}
    let mut prefix=vec![special.cls]; prefix.extend(instruction_ids);prefix.push(special.sep);
    let mut markers=Vec::new();
    for o in &s.options {
        let ids=t.encode_piece(&o.text)?;
        if ids.is_empty() || ids.len()>48 {return Err(Error::TooLong);}
        if control(&ids,&special){return Err(Error::Invalid("option control token".into()));}
        markers.push(prefix.len() as u32);prefix.push(special.mask);prefix.extend(ids);
    }
    prefix.push(special.sep);
    if prefix.len()>MAX_PREFIX {return Err(Error::TooLong);}
    let tokenizer_hash=t.fingerprint();
    let schema_hash=s.digest(&tokenizer_hash);
    Ok(CompiledSchema{schema:s,schema_hash,tokenizer_hash,prefix,markers,special,qtype_id})
}

pub fn validate_compiled(s:&CompiledSchema)->Result<()> {
    validate_schema(&s.schema)?;
    if s.schema_hash!=s.schema.digest(&s.tokenizer_hash) || s.prefix.len()>MAX_PREFIX || s.prefix.len()<5
        || s.prefix.first()!=Some(&s.special.cls) || s.prefix.last()!=Some(&s.special.sep)
        || s.markers.len()!=s.schema.options.len() || s.qtype_id>=3 {return Err(Error::BindingMismatch);}
    let mut prev=None;
    for &p in &s.markers {
        if s.prefix.get(p as usize)!=Some(&s.special.mask) || prev.is_some_and(|x|p<=x) {return Err(Error::BindingMismatch);}
        prev=Some(p);
    }
    if s.prefix.iter().filter(|&&x|x==s.special.mask).count()!=s.markers.len(){return Err(Error::BindingMismatch);}
    Ok(())
}

pub fn render<T:TextTokenizer>(s:&CompiledSchema,t:&T,state:&str)->Result<TokenInput> {
    validate_compiled(s)?;
    if t.fingerprint()!=s.tokenizer_hash || t.special_tokens()!=s.special {return Err(Error::BindingMismatch);}
    if state.is_empty() || state.len()>MAX_STATE_BYTES {return Err(Error::TooLong);}
    if contains_reserved(state,&s.special){return Err(Error::Invalid("reserved token literal".into()));}
    let ids=t.encode_piece(state)?;
    if ids.is_empty() || control(&ids,&s.special){return Err(Error::Invalid("state tokens".into()));}
    if s.prefix.len()+ids.len()+1>MAX_TOKENS {return Err(Error::TooLong);}
    let mut input=s.prefix.clone();input.extend(ids);input.push(s.special.sep);
    Ok(TokenInput{input_ids:input,markers:s.markers.clone(),qtype_id:s.qtype_id})
}
