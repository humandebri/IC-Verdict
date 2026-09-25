"""Separate Wasm local bookkeeping from x86 machine code and register spills.
Standalone Wasmtime, NOT the fully instrumented IC executable. No timing claims.
"""
import argparse, hashlib, io, json, os, platform, re
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import wasmtime
from elftools.elf.elffile import ELFFile
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone.x86 import X86_OP_MEM, X86_OP_IMM, X86_REG_RSP, X86_REG_RBP, X86_REG_ESP, X86_REG_EBP

TARGETS = {'matmul_i8_simd_tileKj300_Kj10_KBW_': 'tile16', 'matmul_i8_simd_tileKj300_Kj8_Kj10_': 'tile8'}
SHA='3bbb0359cd243985ccff6879ba003563cc7f7646f52b9d035deaac971c03799f'
def sha(b): return hashlib.sha256(b).hexdigest()
def compile_symbols(engine, wasm):
    elf=ELFFile(io.BytesIO(wasmtime.Module(engine,wasm).serialize()))
    text=elf.get_section_by_name('.text').data()
    return {s.name:text[s['st_value']:s['st_value']+s['st_size']] for s in elf.get_section_by_name('.symtab').iter_symbols() if s['st_size'] and '::function[' in s.name}
def analyse(code, path):
    md=Cs(CS_ARCH_X86,CS_MODE_64);md.detail=True;ops=list(md.disasm(code,0))
    path.write_text('\n'.join(f'{x.address:08x}: {x.mnemonic} {x.op_str}' for x in ops)+'\n')
    def counts(xs):
        c=Counter(x.mnemonic for x in xs)
        stack=[x for x in xs if any(o.type==X86_OP_MEM and o.mem.base in (X86_REG_RSP,X86_REG_RBP,X86_REG_ESP,X86_REG_EBP) for o in x.operands)]
        return {'instructions':len(xs),'mnemonics':dict(c),'stack_address_instructions':len(stack),'stack_mnemonics':dict(Counter(x.mnemonic for x in stack)), 'stack_reads':sum(any(o.type==X86_OP_MEM and o.mem.base in (X86_REG_RSP,X86_REG_RBP,X86_REG_ESP,X86_REG_EBP) and o.access & 1 for o in x.operands) for x in stack), 'stack_writes':sum(any(o.type==X86_OP_MEM and o.mem.base in (X86_REG_RSP,X86_REG_RBP,X86_REG_ESP,X86_REG_EBP) and o.access & 2 for o in x.operands) for x in stack)}
    loops=[]
    for x in ops:
        if x.mnemonic.startswith('j') and x.operands and x.operands[0].type==X86_OP_IMM and x.operands[0].imm<x.address:
            body=[y for y in ops if x.operands[0].imm<=y.address<=x.address]
            if sum(y.mnemonic in ('pmaddwd','vpmaddwd') for y in body)>=100:
                loops.append({'start':x.operands[0].imm,'end':x.address,**counts(body)})
    return {'code_bytes':len(code),'sha256':sha(code),'whole_function':counts(ops),'dot_loops':loops}
def main():
    p=argparse.ArgumentParser();p.add_argument('--wasm',type=Path,required=True);p.add_argument('--wat',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert platform.machine() in ('x86_64','AMD64'), 'Run on x86 Linux to inspect its native code'
    assert sha(a.wasm.read_bytes())==SHA
    c=wasmtime.Config();c.cranelift_opt_level='none';c.cranelift_nan_canonicalization=True;c.wasm_relaxed_simd=False
    e=wasmtime.Engine(c)
    # The same arithmetic with 40 extra local operations (20 set/get pairs).
    probe={}
    for copies in (0,20):
        wat='(module (func $probe (export "probe") (param i32 i32) (result i32) (local i32) local.get 0 '+('local.set 2 local.get 2 '*copies)+'local.get 1 i32.add))'
        syms=compile_symbols(e,wat);code=next(iter(syms.values()));probe[str(copies)]={'code':code.hex(),'sha256':sha(code),'body_cost_excluding_entry':3+2*copies}
        store=wasmtime.Store(e);inst=wasmtime.Instance(store,wasmtime.Module(e,wat),[])
        assert inst.exports(store)['probe'](store,123,456)==579
        analyse(code,a.out/f'probe-{copies}.asm')
    assert probe['0']['code']==probe['20']['code']
    # Preserve the value on the operand stack and assign it back to itself.
    # Apply only to the two production tile functions, not other Wasm code.
    active=None;added=Counter();types={};lines=[]
    for line in a.wat.read_text().splitlines(keepends=True):
        if line.startswith('  (func '):
            active=next((v for k,v in TARGETS.items() if k in line),None);types={}
        if active:types.update(re.findall(r'\((?:param|local) (\$\w+) (\w+)\)',line))
        lines.append(line)
        m=re.fullmatch(r'\s*local.get (\$\w+)\s*',line)
        if active and m and types.get(m[1])=='v128':
            lines.append(f'    local.tee {m[1]}\n');added[active]+=1
    assert set(added)=={'tile16','tile8'}
    report={'wasm_sha256':SHA,'wasmtime_python_version':version('wasmtime'),'platform':platform.platform(),'config':'opt_level=none; nan_canonicalization=true; relaxed_simd=false; host ISA defaults','ic_reference_wasmtime_version':'48.0.1','native_library_version':os.environ.get('LOCALS_NATIVE_LIBRARY_VERSION','48.0.0'),'native_library_sha256':sha(Path(wasmtime._ffi.filename).read_bytes()),'instrumentation':'none; original compute functions; IC metering insertion is not reproduced','probe':probe,'inserted_static_local_tee':dict(added),'functions':{}}
    for variant,wasm in [('original',a.wasm.read_bytes()),('extra_locals',''.join(lines))]:
        print('compiling',variant,flush=True)
        syms=compile_symbols(e,wasm)
        for key,label in TARGETS.items():
            match=[(name,b) for name,b in syms.items() if key in name];assert len(match)==1
            name,code=match[0];report['functions'].setdefault(label,{})[variant]={'symbol':name,**analyse(code,a.out/f'{label}-{variant}.asm')}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for label,variants in report['functions'].items():
        variants['identical_machine_code']=variants['original']['sha256']==variants['extra_locals']['sha256']
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['identical_machine_code'] for k,v in report['functions'].items()}))
if __name__=='__main__':main()
