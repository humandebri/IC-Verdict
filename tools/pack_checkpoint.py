#!/usr/bin/env python3
"""Offline canonical F32 pack exporter. Does not download or execute model code.

An explicit mapping is REQUIRED: we have not verified upstream Laya tensor names,
QKV order, normalization, qtype mapping, or checkpoint parity in this environment.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, tempfile
from pathlib import Path
import numpy as np
from model_reference import shapes

FORMAT = "ic-laya-f32-pack-v1"
MAX_TENSOR = 256 * 1024 * 1024
MAX_TOTAL = 2 * 1024**3

def sha(raw: bytes) -> list[int]:
    return list(hashlib.sha256(raw).digest())

def validate_config(c: dict) -> None:
    h=c["hidden_size"]
    if not (0<h<=2048 and 5<=c["vocab_size"]<=300000 and 1<=c["layers"]<=64):
        raise ValueError("unsupported backbone dimensions")
    if c["attention_heads"]<1 or h%c["attention_heads"] or (h//c["attention_heads"])%2:
        raise ValueError("invalid rotary head dimensions")
    if c["decision_heads"]<1 or h%c["decision_heads"] or not 1<=c["decision_layers"]<=4:
        raise ValueError("invalid decision head dimensions")
    if c["global_every"]<1 or c["local_attention"]<1 or c["qtypes"]!=3:
        raise ValueError("invalid attention/qtype configuration")
    if not 0<=c["mask_token_id"]<c["vocab_size"]:
        raise ValueError("invalid mask id")
    for k in ["intermediate_size", "decision_ff"]:
        if not 1<=c[k]<=16384: raise ValueError(k)
    for k in ["norm_eps", "decision_norm_eps", "scorer_norm_eps"]:
        if not np.isfinite(c[k]) or not 0<c[k]<=1: raise ValueError(k)
    for k in ["global_rope_theta", "local_rope_theta"]:
        if not np.isfinite(c[k]) or not 1<c[k]<=1e12: raise ValueError(k)
    if c["decision_activation"] not in ["Relu", "Gelu"]:
        raise ValueError("unsupported activation")

def source_reader(source: Path):
    """Index only; tensors are loaded one at a time. No pickle loading."""
    from safetensors import safe_open
    paths = sorted(source.glob("*.safetensors")) if source.is_dir() else [source]
    if not paths or any(p.suffix!=".safetensors" for p in paths):
        raise ValueError("source must contain safetensors, never a pickle checkpoint")
    index={}
    for path in paths:
        with safe_open(str(path),framework="pt",device="cpu") as f:
            for key in f.keys():
                if key in index: raise ValueError(f"duplicate source tensor {key}")
                index[key]=path
    def read(name):
        import torch
        if name not in index: raise ValueError(f"missing source tensor {name}")
        with safe_open(str(index[name]),framework="pt",device="cpu") as f:
            return f.get_tensor(name).to(dtype=torch.float32).contiguous().numpy()
    return index,read

def resolve(spec, read):
    if isinstance(spec,str): return read(spec)
    if set(spec)=={"source","transpose"} and spec["transpose"] is True:
        a=read(spec["source"])
        if a.ndim!=2: raise ValueError("transpose requires a matrix")
        return a.T
    if set(spec)=={"concat","axis"} and isinstance(spec["concat"],list) and spec["axis"]==0:
        # Caller supplies the VERIFIED Q,K,V order. No automatic architecture guess.
        return np.concatenate([read(name) for name in spec["concat"]],axis=0)
    raise ValueError("mapping permits source name, explicit transpose, or axis-0 concat only")

def export(source:Path,config:Path,mapping:Path,tokenizer:Path,out:Path,repo:str,revision:str,qtypes:list[int],test_only:bool):
    if out.exists(): raise ValueError("output already exists; refusing to overwrite")
    if not repo or not revision: raise ValueError("source identity required")
    if not test_only and not re.fullmatch(r"[0-9a-fA-F]{40}",revision):
        raise ValueError("real model export requires a 40-character immutable revision")
    if sorted(qtypes)!=[0,1,2]: raise ValueError("qtypes must map Choice,Noul,Score to a permutation of 0,1,2")
    c=json.loads(config.read_text());validate_config(c)
    mapped=json.loads(mapping.read_text());expected=shapes(c)
    if set(mapped)!=set(expected):
        raise ValueError(f"mapping mismatch: missing={set(expected)-set(mapped)}, extra={set(mapped)-set(expected)}")
    tok=tokenizer.read_bytes()
    if not 0<len(tok)<=32*1024**2: raise ValueError("tokenizer size")
    json.loads(tok)
    _,read=source_reader(source)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=".laya-pack-",dir=out.parent))
    try:
        entries=[];position=0
        with (temporary/"model.bin").open("wb") as f:
            for name,shape in sorted(expected.items()):
                values=np.asarray(resolve(mapped[name],read),dtype="<f4",order="C")
                if list(values.shape)!=list(shape): raise ValueError(f"shape mismatch for {name}")
                if not np.isfinite(values).all(): raise ValueError(f"nonfinite tensor {name}")
                raw=values.tobytes()
                if len(raw)>MAX_TENSOR or position+len(raw)>MAX_TOTAL: raise ValueError("pack exceeds v0.1 limits")
                f.write(raw)
                entries.append(dict(name=name,shape=list(shape),offset=position,length=len(raw),sha256=sha(raw)))
                position+=len(raw)
        manifest=dict(format=FORMAT,source_repo=repo,source_revision=revision,test_only=test_only,
          tokenizer_sha256=sha(tok),primitive_to_qtype=qtypes,config=c,total_bytes=position,tensors=entries)
        (temporary/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (temporary/"tokenizer.json").write_bytes(tok)
        shutil.copyfile(mapping,temporary/"export_mapping.json")
        shutil.copyfile(config,temporary/"export_config.json")
        (temporary/"EXPORT_NOTES.txt").write_text(
          "Canonical export only. Upstream checkpoint parity and ICP performance NOT established.\n"
          "Source mapping/config were supplied by the caller, not inferred or verified by this tool.\n")
        temporary.rename(out)
    except Exception:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    i=sub.add_parser("inspect");i.add_argument("source",type=Path)
    e=sub.add_parser("export")
    for flag in ["source","config","mapping","tokenizer","out"]:e.add_argument("--"+flag,type=Path,required=True)
    e.add_argument("--repo",required=True);e.add_argument("--revision",required=True)
    e.add_argument("--qtypes",type=int,nargs=3,required=True,metavar=("CHOICE","NOUL","SCORE"))
    e.add_argument("--test-only",action="store_true")
    args=p.parse_args()
    if args.command=="inspect":
        index,read=source_reader(args.source)
        print(json.dumps({name:list(read(name).shape) for name in sorted(index)},indent=2));return
    manifest=export(args.source,args.config,args.mapping,args.tokenizer,args.out,args.repo,args.revision,args.qtypes,args.test_only)
    print(json.dumps({"output":str(args.out),"tensor_count":len(manifest["tensors"]),"bytes":manifest["total_bytes"],
                      "bundle_sha256":hashlib.sha256((args.out/"manifest.json").read_bytes()).hexdigest(),"parity_verified":False},indent=2))
if __name__=="__main__":main()
