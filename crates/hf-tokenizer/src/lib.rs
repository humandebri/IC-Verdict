//! Tokenizer adapter. No network, filesystem, JS imports or automatic truncation.
use ic_laya_core::{schema::TextTokenizer,*};
pub struct HfTokenizer {inner:tokenizers::Tokenizer,fingerprint:Digest,special:SpecialTokens}
impl HfTokenizer {
    pub fn from_bytes(bytes:&[u8],special:SpecialTokens)->Result<Self>{
        if bytes.is_empty() || bytes.len()>32*1024*1024{return Err(Error::TooLong);}
        tokenizers::utils::parallelism::set_parallelism(false);
        let mut inner=tokenizers::Tokenizer::from_bytes(bytes).map_err(|e|Error::Invalid(e.to_string()))?;
        inner.with_truncation(None).map_err(|e|Error::Invalid(e.to_string()))?;inner.with_padding(None);
        for id in [special.cls,special.sep,special.mask,special.pad] {
            let token=inner.id_to_token(id).ok_or_else(||Error::Invalid("special token id".into()))?;
            if !special.literals.contains(&token){return Err(Error::Invalid("special literal list".into()));}
        }
        // Reject ALL added special-token literals, not only MASK/CLS.
        let json:serde_json::Value=serde_json::from_slice(bytes).map_err(|e|Error::Invalid(e.to_string()))?;
        let mut special=special;
        if let Some(tokens)=json.get("added_tokens").and_then(|x|x.as_array()) {
            for t in tokens {
                if t.get("special").and_then(|x|x.as_bool())==Some(true){
                    let text=t.get("content").and_then(|x|x.as_str()).ok_or_else(||Error::Invalid("added token".into()))?;
                    if !special.literals.iter().any(|x|x==text){special.literals.push(text.into());}
                }
            }
        }
        special.literals.sort();special.literals.dedup();
        Ok(Self{inner,fingerprint:hash(bytes),special})
    }
}
impl TextTokenizer for HfTokenizer {
    fn encode_piece(&self,text:&str)->Result<Vec<u32>>{self.inner.encode(text,false).map(|x|x.get_ids().to_vec()).map_err(|e|Error::Invalid(e.to_string()))}
    fn fingerprint(&self)->Digest{self.fingerprint}
    fn special_tokens(&self)->SpecialTokens{self.special.clone()}
}
