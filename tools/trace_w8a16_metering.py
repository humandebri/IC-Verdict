#!/usr/bin/env python3
"""Count executed operations in the exact production Wasm, then apply IC prices.

This is an execution-path audit, NOT a timing benchmark. Added audit counters
are excluded from the prices. Three original functions are exported/instrumented:
matmul_i8 and its 16-row and 8-row specializations. All ic0 imports trap if called.
The original Wasm hash, numerical outputs, and eight PocketIC totals are checked.
"""
import argparse
import array
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import struct
import wasmtime

SHA = '3bbb0359cd243985ccff6879ba003563cc7f7646f52b9d035deaac971c03799f'
TARGETS = {'matmul_i8_simd_tileKj300_Kj10_KBW_': 'tile16',
           'matmul_i8_simd_tileKj300_Kj8_Kj10_': 'tile8',
           'verdict_simd9matmul_i8 ': 'dispatch'}
# Original flat WAT line ranges (wabt 1.0.39); hash checked before use.
HOT = {'tile16': (1092041, 1101366), 'tile8': (1106727, 1111858)}
OUTPUT = {'tile16': (1101367, 1105046), 'tile8': (1111859, 1113716)}
# dfinity/ic d26cd031176beec51b39fbb9e39e80a3a46a748e,
# rs/embedders/src/wasm_utils/instrumentation.rs, Wasm32 instruction_to_cost.
COST = {op: 1 for op in '''function.entry local.get local.set local.tee
 i32.add i32.and i32.const i32.eq i32.eqz i32.lt_u i32.mul i32.ne i32.or
 i32.shl i32.shr_u i32.sub i32.wrap_i64 i64.const i64.extend_i32_u i64.mul i64.shr_u
 i32x4.add i32x4.dot_i16x8_s i32x4.extract_lane i32x4.replace_lane i32x4.splat
 f32x4.convert_i32x4_s return v128.const v128.load v128.load32_splat
 v128.load8x8_s v128.store'''.split()}
COST.update({'br': 2, 'br_if': 2, 'if': 2, 'f32x4.mul': 2, 'call': 5})


def instrument(wat, output):
    active, entry, types, previous = None, False, {}, None
    keys, lines, exports = {}, {}, {}
    with wat.open() as source, output.open('w') as dest:
        for num, line in enumerate(source, 1):
            if line.startswith('  (func '):
                active = next((label for key, label in TARGETS.items() if key in line), None)
                entry, types, previous = bool(active), {}, None
                if active:
                    exports[active] = line.strip().split()[1]
            s = line.strip()
            if active:
                types.update(re.findall(r'\((?:param|local) (\$\w+) (\w+)\)', line))
            if active and s and not s.startswith('(') and s != ')':
                op = s.split()[0].rstrip(')')

                def count(key):
                    index = keys.setdefault(key, len(keys))
                    lines.setdefault(key, num)
                    # Net-zero operand-stack effect; original operands are preserved.
                    dest.write(f'    global.get $audit{index}\n    i64.const 1\n'
                               f'    i64.add\n    global.set $audit{index}\n')

                if entry:
                    count((active, 'entry', 'function.entry', ''))
                    entry = False
                if op not in ('block', 'loop', 'else', 'end'):
                    phase = ('hot' if active in HOT and HOT[active][0] < num < HOT[active][1]
                             else 'output' if active in OUTPUT and OUTPUT[active][0] <= num <= OUTPUT[active][1]
                             else 'setup_control')
                    detail = ''
                    if op.startswith('local.'):
                        detail = types[s.split()[1].rstrip(')')]
                    elif op == 'i32.add' and previous == 'i32x4.extract_lane':
                        detail = 'lane_reduction'
                    count((active, phase, op, detail))
                previous = op
            dest.write(line)
    assert set(exports) == set(TARGETS.values())
    tail = ''.join(f'  (global $audit{i} (export "audit{i}") (mut i64) (i64.const 0))\n' for i in keys.values())
    tail += ''.join(f'  (export "audit_{label}" (func {name}))\n' for label, name in exports.items())
    text = output.read_text().rstrip()
    assert text.endswith(')')
    output.write_text(text[:-1] + '\n' + tail + ')\n')
    return keys, lines


def category(row):
    op, phase, detail = row['op'], row['phase'], row['detail']
    if op.startswith('local.'):
        return op
    if op == 'i32x4.dot_i16x8_s': return 'simd_dot'
    if op == 'i32x4.add': return 'simd_accumulate'
    if op == 'v128.load8x8_s': return 'weight_load_and_sign_extension'
    if op == 'v128.load' and phase == 'hot': return 'activation_load'
    if op in ('v128.load', 'v128.load32_splat'): return 'scale_load'
    if op == 'i32x4.extract_lane' or detail == 'lane_reduction': return 'horizontal_reduction'
    if op in ('i32x4.splat', 'i32x4.replace_lane'): return 'repack_output_vectors'
    if op == 'f32x4.convert_i32x4_s': return 'integer_to_float'
    if op == 'f32x4.mul': return 'apply_quantization_scales'
    if op == 'v128.store': return 'output_store'
    if op.endswith('.const'): return 'constants'
    if op in ('br', 'br_if', 'if'): return 'branches'
    if op in ('call', 'return', 'function.entry'): return 'function_control'
    if op in ('i32.eq', 'i32.eqz', 'i32.lt_u', 'i32.ne'): return 'integer_comparisons'
    if op in ('i32.wrap_i64', 'i64.extend_i32_u'): return 'integer_width_conversion'
    if op.startswith(('i32.', 'i64.')): return 'address_size_loop_arithmetic'
    raise ValueError(op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wasm', type=Path, required=True)
    parser.add_argument('--wat', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.wasm.read_bytes()).hexdigest() == SHA
    args.work_dir.mkdir(parents=True, exist_ok=True)
    traced = args.work_dir / 'traced.wat'
    keys, keylines = instrument(args.wat, traced)
    config = wasmtime.Config()
    config.cranelift_opt_level = 'none'
    engine = wasmtime.Engine(config)
    module = wasmtime.Module(engine, wasmtime.wat2wasm(traced.read_text()))
    report = {'wasm_sha256': SHA, 'wat_sha256': hashlib.sha256(args.wat.read_bytes()).hexdigest(), 'wasmtime_python_version': version('wasmtime'), 'ic_source_revision': 'd26cd031176beec51b39fbb9e39e80a3a46a748e',
              'method': 'dynamic opcode counters; original functions, not IC wall/CPU timing', 'shapes': []}
    for m in (40, 120):
        n, k = 2304, 768
        store, linker = wasmtime.Store(engine), wasmtime.Linker(engine)
        for imp in module.imports:
            assert isinstance(imp.type, wasmtime.FuncType)
            def trap(*values, _name=imp.name):
                raise RuntimeError('unexpected imported function ' + _name)
            linker.define_func(imp.module, imp.name, imp.type, trap)
        exports = linker.instantiate(store, module).exports(store)
        memory = exports['memory']
        buffers = [array.array('h', ((i % 257)-128 for i in range(m*k))).tobytes(),
                   array.array('b', ((i % 31)-15 for i in range(n*k))).tobytes(),
                   struct.pack('<f', 1.0)*m, struct.pack('<f', 1.0)*n, b'\x00'*(m*n*4)]
        addresses, end = [], 8*1024*1024
        for buf in buffers:
            addresses.append(end)
            end += (len(buf)+63)//64*64
        memory.grow(store, (end+65535)//65536-memory.size(store))
        for address, buf in zip(addresses, buffers):
            memory.write(store, buf, address)
        ap, wp, sxp, swp, outp = addresses
        assert exports['audit_dispatch'](store, ap, m*k, wp, n*k, sxp, m, swp, n, m, k, n, outp, m*n) == 1
        for i in (0, 15, 16, m-9, m-8, m-1):
            for j in (0, 15, 16, 2303):
                expected = sum(((i*k+t) % 257-128)*((j*k+t) % 31-15) for t in range(k))
                offset = outp+(i*n+j)*4
                assert struct.unpack('<f', memory.read(store, offset, offset+4))[0] == expected
        rows, groups, opcode_costs, phases = [], Counter(), Counter(), Counter()
        for key, index in keys.items():
            count = exports[f'audit{index}'].value(store)
            if not count: continue
            function, phase, op, detail = key
            row = {'function': function, 'phase': phase, 'op': op, 'detail': detail, 'count': count,
                   'unit_cost': COST[op], 'cost': count*COST[op], 'example_wat_line': keylines[key]}
            row['category'] = category(row)
            rows.append(row)
            groups[row['category']] += row['cost']
            opcode_costs[op] += row['cost']
            phases[phase] += row['cost']
        kernel = sum(groups.values())
        assert opcode_costs['i32x4.dot_i16x8_s']*8 == m*n*k
        assert sum(r['count'] for r in rows if r['detail'] == 'lane_reduction') == m*n*3
        assert opcode_costs['v128.store']*4 == m*n
        # IC charges a basic block before executing it. The post-counter block
        # costs 37, including operations after the counter call; see the report.
        driver = {'performance_counter_api': 200, 'loop_setup': 2, 'per_iteration': 25, 'post_loop_block': 37}
        observed = json.loads((args.measurements / f'm{m}.json').read_text())['rows']
        checks = []
        for level in (1, 4, 8, 16):
            actuals = {r['charged_instructions'] for r in observed if r['kind']=='gemm' and r['level']==level}
            predicted = level*kernel + level*25 + 239
            assert actuals == {predicted}, (m, level, actuals, predicted)
            checks.append({'iterations': level, 'measured': predicted, 'reconstructed': predicted, 'error': 0})
        report['shapes'].append({'m_n_k': [m,n,k], 'checked_output_elements': 24,
                                'kernel_cost': kernel, 'groups': groups, 'opcode_costs': opcode_costs,
                                'phases': phases, 'benchmark_overhead': driver, 'reconciliation': checks, 'rows': rows})
        print(m, kernel, checks, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
