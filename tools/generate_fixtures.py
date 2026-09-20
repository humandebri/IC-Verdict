#!/usr/bin/env python3
"""Generate small, redistributable random-weight fixtures and numerical vectors."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from reference_math import from_logits,score_stats
from model_reference import shapes,torch_forward,numpy_forward
ROOT=Path(__file__).resolve().parents[1]

def sha(b:bytes)->list[int]:return list(hashlib.sha256(b).digest())

def generate():
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg={"vocab_size":128,"hidden_size":8,"layers":2,"attention_heads":2,"intermediate_size":12,"norm_eps":1e-5,
         "global_every":2,"local_attention":4,"global_rope_theta":10000.0,"local_rope_theta":1000.0,"first_layer_attention_norm":False,
         "decision_layers":2,"decision_heads":2,"decision_ff":16,"decision_norm_eps":1e-5,"decision_norm_first":True,
         "decision_activation":"Relu","scorer_norm_eps":1e-5,"qtypes":3,"mask_token_id":3}
    torch.manual_seed(19)
    w={}
    for name,shape in shapes(cfg).items():
        x=torch.randn(shape)*0.07
        if name.endswith(".weight") and "norm" in name:x=1+x*.1
        w[name]=x.contiguous()
    tokenizer={"version":"1.0","truncation":None,"padding":None,
      "added_tokens":[{"id":i,"content":t,"single_word":False,"lstrip":False,"rstrip":False,"normalized":False,"special":True} for i,t in enumerate(["[PAD]","[CLS]","[SEP]","[MASK]","[UNK]"])],
      "normalizer":None,"pre_tokenizer":{"type":"WhitespaceSplit"},"post_processor":None,"decoder":None,
      "model":{"type":"WordLevel","vocab":{**{t:i for i,t in enumerate(["[PAD]","[CLS]","[SEP]","[MASK]","[UNK]"])},**{f"t{i}":i for i in range(5,128)}},"unk_token":"[UNK]"}}
    tokenizer_raw=(json.dumps(tokenizer,ensure_ascii=False,separators=(",",":"))+"\n").encode()
    metrics=[]
    for pre in [True,False]:
        c={**cfg,"decision_norm_first":pre}
        directory=ROOT/"fixtures"/("tiny-prenorm" if pre else "tiny-postnorm")
        directory.mkdir(parents=True,exist_ok=True)
        entries=[];blob=bytearray()
        for name,x in sorted(w.items()):
            raw=x.numpy().astype("<f4").tobytes()
            entries.append({"name":name,"shape":list(x.shape),"offset":len(blob),"length":len(raw),"sha256":sha(raw)})
            blob.extend(raw)
        manifest={"format":"ic-laya-f32-pack-v1","source_repo":"synthetic/random-weights-not-Laya","source_revision":"seed-19","test_only":True,
          "tokenizer_sha256":sha(tokenizer_raw),"primitive_to_qtype":[0,1,2],"config":c,"total_bytes":len(blob),"tensors":entries}
        (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (directory/"model.bin").write_bytes(blob)
        (directory/"tokenizer.json").write_bytes(tokenizer_raw)
        cases=[]
        for qtype,n,t in [(0,3,16),(1,2,13),(2,3,19),(2,5,21),(2,7,29)]:
            ids=[1]+[5+i%100 for i in range(t-2)]+[2]
            markers=[2+i*2 for i in range(n)]
            for m in markers:ids[m]=3
            inp={"input_ids":ids,"markers":markers,"qtype_id":qtype}
            with torch.no_grad():logits=torch_forward(c,w,inp)
            independent=numpy_forward(c,{k:v.numpy() for k,v in w.items()},inp)
            delta=float(np.max(np.abs(logits-independent)))
            if delta>2e-5:raise AssertionError((pre,qtype,n,delta))
            cases.append({"input":inp,"expected_logits":logits.tolist(),"atol":1e-3,"rtol":1e-3,"provenance":"PyTorch synthetic reference; NOT pretrained Laya"})
            metrics.append({"pre_norm":pre,"qtype":qtype,"options":n,"tokens":t,"pytorch_numpy_max_abs":delta})
        (directory/"cases.json").write_text(json.dumps(cases,indent=2)+"\n")
        (directory/"input.json").write_text(json.dumps(cases[0]["input"],indent=2)+"\n")
    vectors=[]
    for logits,temp in [([0.,0.],1.),([0.,0.,0.],1.),([0.]*5,1.),([0.]*7,1.),([4.,-2.,-3.],1.),([-4.,4.],1.),([5.,1.,0.,-2.,-4.],1.),([2.,1.,0.],2.),([100.,-100.],.5),([-1.,-2.,-3.,-4.,-5.,-6.,-7.],1.)]:
        mass=from_logits(logits,temp)
        v={"logits":logits,"temperature":temp,"mass_ppm":mass}
        if len(mass)>=3:v["score"]=score_stats(mass)
        vectors.append(v)
    (ROOT/"fixtures/numerics.json").write_text(json.dumps(vectors,indent=2)+"\n")
    (ROOT/"artifacts/synthetic_neural_checks.json").write_text(json.dumps({"kind":"PyTorch versus NumPy on synthetic weights","torch":torch.__version__,"cases":metrics},indent=2)+"\n")
    print(json.dumps({"synthetic_model_cases":len(metrics),"maximum_abs_error":max(x["pytorch_numpy_max_abs"] for x in metrics),"numeric_vectors":len(vectors)},indent=2))
if __name__=="__main__":generate()
