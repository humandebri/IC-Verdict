use ic_laya_core::{schema::*,*};
#[test]fn tiny_wordlevel_adapter(){
    let bytes=include_bytes!("../../../fixtures/tiny-prenorm/tokenizer.json");
    let special=SpecialTokens{cls:1,sep:2,mask:3,pad:0,literals:vec!["[PAD]".into(),"[CLS]".into(),"[SEP]".into(),"[MASK]".into()]};
    let t=hf_tokenizer::HfTokenizer::from_bytes(bytes,special).unwrap();
    assert_eq!(t.encode_piece("t5 t7").unwrap(),vec![5,7]);
    assert_eq!(t.encode_piece("unseen_word").unwrap(),vec![4]);
    assert!(t.special_tokens().literals.contains(&"[UNK]".into()));
    let s=Schema{id:"Tiny".into(),version:1,primitive:Primitive::Noul,instructions:"t9".into(),options:vec![OptionDef{id:"false".into(),text:"t10".into()},OptionDef{id:"true".into(),text:"t11".into()}]};
    let c=compile(s,&t,1).unwrap();let inp=render(&c,&t,"t20 t21").unwrap();assert_eq!(&inp.input_ids[inp.input_ids.len()-3..],&[20,21,2]);
    assert!(render(&c,&t,"[UNK]").is_err());
}
