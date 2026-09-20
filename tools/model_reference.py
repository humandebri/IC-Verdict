"""Synthetic ModernBERT/decision-head reference, independently evaluated in PyTorch/NumPy.
No pretrained weights, training, real-language benchmark or original Laya parity is implied.
"""
from __future__ import annotations
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn.functional as F

def shapes(c: dict) -> dict[str, list[int]]:
    h = c["hidden_size"]
    m = {"embeddings.weight": [c["vocab_size"], h], "embeddings.norm.weight": [h], "final_norm.weight": [h]}
    for i in range(c["layers"]):
        p = f"encoder.{i}"
        if i or c["first_layer_attention_norm"]:
            m[p+".attn_norm.weight"] = [h]
        for k, shape in {"qkv.weight": [3*h,h], "out.weight": [h,h], "mlp_norm.weight": [h],
                         "wi.weight": [2*c["intermediate_size"],h], "wo.weight": [h,c["intermediate_size"]]}.items():
            m[p+"."+k] = shape
    m["qtype.weight"] = [3,h]
    for i in range(c["decision_layers"]):
        p = f"decision.{i}"
        for k, shape in {"qkv.weight": [3*h,h], "qkv.bias": [3*h], "out.weight": [h,h], "out.bias": [h],
                         "norm1.weight": [h], "norm1.bias": [h], "norm2.weight": [h], "norm2.bias": [h],
                         "linear1.weight": [c["decision_ff"],h], "linear1.bias": [c["decision_ff"]],
                         "linear2.weight": [h,c["decision_ff"]], "linear2.bias": [h]}.items():
            m[p+"."+k] = shape
    for k, shape in {"norm.weight": [h], "norm.bias": [h], "dense.weight": [h,h], "dense.bias": [h], "out.weight": [1,h], "out.bias": [1]}.items():
        m["scorer."+k] = shape
    return dict(sorted(m.items()))

def torch_forward(c: dict, w: dict[str, torch.Tensor], inp: dict) -> np.ndarray:
    h = c["hidden_size"]
    def norm(x,p,eps,bias=False): return F.layer_norm(x,(h,),w[p+".weight"],w.get(p+".bias") if bias else None,eps)
    def linear(x,p): return F.linear(x,w[p+".weight"],w.get(p+".bias"))
    def attention(x,p,heads,theta=None,distance=None):
        t,d = len(x),h//heads
        q,k,v = linear(x,p+".qkv").reshape(t,3,heads,d).permute(1,2,0,3)
        if theta is not None:
            a = torch.tensor([[pos/theta**(2*j/d) for j in range(d//2)] for pos in range(t)], dtype=torch.float64)
            co,si = a.cos().float()[None], a.sin().float()[None]
            def rope(x):
                left,right = x.chunk(2,dim=-1)
                return torch.cat((left*co-right*si, right*co+left*si),dim=-1)
            q,k = rope(q),rope(k)
        scores = q @ k.transpose(-1,-2) / math.sqrt(d)
        if distance is not None:
            mask = torch.arange(t)[:,None] - torch.arange(t)[None,:]
            scores = scores.masked_fill(mask.abs()[None]>distance, -float("inf"))
        y=(scores.softmax(-1) @ v).transpose(0,1).reshape(t,h)
        return linear(y,p+".out")
    x=norm(w["embeddings.weight"][inp["input_ids"]],"embeddings.norm",c["norm_eps"])
    for i in range(c["layers"]):
        p=f"encoder.{i}"
        y=norm(x,p+".attn_norm",c["norm_eps"]) if i or c["first_layer_attention_norm"] else x
        local=i%c["global_every"]!=0
        x=x+attention(y,p,c["attention_heads"],c["local_rope_theta"] if local else c["global_rope_theta"], c["local_attention"]//2 if local else None)
        a,b=linear(norm(x,p+".mlp_norm",c["norm_eps"]),p+".wi").chunk(2,dim=-1)
        x=x+linear(F.gelu(a,approximate="none")*b,p+".wo")
    x=norm(x,"final_norm",c["norm_eps"])+w["qtype.weight"][inp["qtype_id"]]
    # Independent PyTorch library implementation of TransformerEncoderLayer.
    for i in range(c["decision_layers"]):
        layer=torch.nn.TransformerEncoderLayer(h,c["decision_heads"],dim_feedforward=c["decision_ff"],dropout=0,
            activation=c["decision_activation"].lower(),layer_norm_eps=c["decision_norm_eps"],batch_first=True,norm_first=c["decision_norm_first"])
        p=f"decision.{i}"
        mapping={"self_attn.in_proj_weight":"qkv.weight","self_attn.in_proj_bias":"qkv.bias","self_attn.out_proj.weight":"out.weight","self_attn.out_proj.bias":"out.bias"}
        state={k:w[p+"."+mapping.get(k,k)] for k in layer.state_dict()}
        layer.load_state_dict(state);layer.eval()
        x=layer(x.unsqueeze(0)).squeeze(0)
    x=norm(x[inp["markers"]],"scorer.norm",c["scorer_norm_eps"],True)
    return linear(F.gelu(linear(x,"scorer.dense"),approximate="none"),"scorer.out").squeeze(-1).detach().cpu().numpy()

def numpy_forward(c: dict, w: dict[str,np.ndarray], inp: dict) -> np.ndarray:
    h=c["hidden_size"]
    def norm(x,p,eps,bias=False):
        y=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+eps)*w[p+".weight"]
        return y+w[p+".bias"] if bias else y
    def linear(x,p): return x@w[p+".weight"].T + w.get(p+".bias",0.)
    def gelu(x): return .5*x*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
    def attention(x,p,heads,theta=None,distance=None):
        t,d=len(x),h//heads
        q,k,v=linear(x,p+".qkv").reshape(t,3,heads,d).transpose(1,2,0,3)
        if theta is not None:
            a=np.array([[pos/theta**(2*j/d) for j in range(d//2)] for pos in range(t)])
            co,si=np.cos(a)[None],np.sin(a)[None]
            def rope(x):
                l,r=np.split(x,2,-1)
                return np.concatenate((l*co-r*si,r*co+l*si),-1)
            q,k=rope(q),rope(k)
        scores=q@k.transpose(0,2,1)/math.sqrt(d)
        if distance is not None:
            mask=np.abs(np.arange(t)[:,None]-np.arange(t)[None,:])>distance
            scores=np.where(mask[None],-np.inf,scores)
        p_mass=np.exp(scores-np.max(scores,axis=-1,keepdims=True));p_mass/=p_mass.sum(-1,keepdims=True)
        return linear((p_mass@v).transpose(1,0,2).reshape(t,h),p+".out")
    x=norm(w["embeddings.weight"][inp["input_ids"]],"embeddings.norm",c["norm_eps"])
    for i in range(c["layers"]):
        p=f"encoder.{i}"
        y=norm(x,p+".attn_norm",c["norm_eps"]) if i or c["first_layer_attention_norm"] else x
        local=i%c["global_every"]!=0
        x=x+attention(y,p,c["attention_heads"],c["local_rope_theta"] if local else c["global_rope_theta"],c["local_attention"]//2 if local else None)
        a,b=np.split(linear(norm(x,p+".mlp_norm",c["norm_eps"]),p+".wi"),2,-1)
        x=x+linear(gelu(a)*b,p+".wo")
    x=norm(x,"final_norm",c["norm_eps"])+w["qtype.weight"][inp["qtype_id"]]
    for i in range(c["decision_layers"]):
        p=f"decision.{i}"
        def ff(y):
            y=linear(y,p+".linear1")
            return linear(np.maximum(y,0) if c["decision_activation"]=="Relu" else gelu(y),p+".linear2")
        if c["decision_norm_first"]:
            x=x+attention(norm(x,p+".norm1",c["decision_norm_eps"],True),p,c["decision_heads"])
            x=x+ff(norm(x,p+".norm2",c["decision_norm_eps"],True))
        else:
            x=norm(x+attention(x,p,c["decision_heads"]),p+".norm1",c["decision_norm_eps"],True)
            x=norm(x+ff(x),p+".norm2",c["decision_norm_eps"],True)
    x=norm(x[inp["markers"]],"scorer.norm",c["scorer_norm_eps"],True)
    return linear(gelu(linear(x,"scorer.dense")),"scorer.out").reshape(-1)
