#!/usr/bin/env python3
"""Compile W8A16 tile variants and measure exact IC instruction counts in PocketIC.
Source edits are temporary within CI checkout and restored in finally.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time
from pocket_ic import PocketIC

SHAPE=(120,2304,768)
EXPECTED_CHECKSUM=641785196358105
VARIANTS=[('16x16',16,16),('8x16',8,16),('4x8',4,8),('4x2',4,2)]
SOURCE=Path('crates/verdict-simd/src/lib.rs')
WASM=Path('target/wasm32-unknown-unknown/release/examples/metering_canister.wasm')
OLD='matmul_i8_simd_tile::<768,16,16>'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--artifact-dir',type=Path,required=True);args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True);args.artifact_dir.mkdir(parents=True,exist_ok=True)
    original=SOURCE.read_bytes();source=original.decode();assert source.count(OLD)==1
    report={'shape_m_n_k':SHAPE,'expected_checksum':EXPECTED_CHECKSUM,'method':'same production Rust implementation, only 768 dispatch R/C changed; PocketIC before/after performance_counter; one weight bank','variants':{}}
    try:
        for name,r,c in VARIANTS:
            SOURCE.write_text(source.replace(OLD,f'matmul_i8_simd_tile::<768,{r},{c}>'))
            subprocess.run(['cargo','build','--locked','--release','--target','wasm32-unknown-unknown','-p','verdict-simd','--example','metering-canister'],check=True)
            data=WASM.read_bytes();path=args.artifact_dir/f'metering-{name}.wasm';path.write_bytes(data)
            report['variants'][name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'tile_rows':r,'tile_cols':c,'samples':[]}
    finally:
        SOURCE.write_bytes(original)
    assert SOURCE.read_bytes()==original
    pic=PocketIC()
    for name,_,_ in VARIANTS:
        v=report['variants'][name];cid=pic.create_canister();pic.add_cycles(cid,100_000_000_000_000)
        pic.install_code(cid,(args.artifact_dir/f'metering-{name}.wasm').read_bytes(),[])
        prepare=struct.pack('<4I',*SHAPE,1)
        pic.update_call(cid,'prepare',prepare)
        for iterations in (0,1,4,8,8,8):
            start=time.perf_counter();payload=pic.update_call(cid,'run',struct.pack('<I',iterations));wall=time.perf_counter()-start
            total,kernel,checksum=struct.unpack('<3Q',bytes(payload))
            assert checksum==EXPECTED_CHECKSUM,(name,iterations,checksum)
            sample={'iterations':iterations,'total_metered':total,'kernel_metered':kernel,'wall_s':wall,'checksum':checksum}
            v['samples'].append(sample)
        one=v['samples'][1]['kernel_metered'];four=v['samples'][2]['kernel_metered'];eight=v['samples'][3]['kernel_metered']
        assert (four-one)%3==0,(name,one,four)
        per=(four-one)//3
        assert eight-four==4*per,(name,one,four,eight,per)
        assert all(s['kernel_metered']==eight for s in v['samples'][3:]),name
        v['kernel_metered_per_iteration']=per
        v['first_nonzero_fixed_cost']=one-per
        args.out.write_text(json.dumps(report,indent=2)+'\n')
        print(name,v['kernel_metered_per_iteration'],v['sha256'],flush=True)
    base=report['variants']['16x16']['kernel_metered_per_iteration']
    for v in report['variants'].values():v['reduction_vs_16x16_percent']=100*(1-v['kernel_metered_per_iteration']/base)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
