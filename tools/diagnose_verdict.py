#!/usr/bin/env python3
"""Offline error attribution; independent NumPy F32 reference, never a deployment gate.

Compare checkpoint F32, decoded INT8 weights with F32 activations, and decoded
INT8 weights with the production i16 activation quantizer. The last mode uses
BLAS accumulation rather than the production block kernel. No downloads.
"""
import argparse
import json
import math
import struct
import subprocess
from pathlib import Path

try:
    import numpy as np
except ImportError as error:
    raise SystemExit("diagnose_verdict.py requires NumPy; the standard test suite does not") from error


def source_name(name):
    if name == 'embeddings.weight':
        return 'model.encoder_model.embeddings.tok_embeddings.weight'
    if name.startswith(('embeddings.', 'final_norm.')):
        return 'model.encoder_model.' + name
    if name.startswith('encoder.'):
        _, layer, tail = name.split('.', 2)
        mapping = {'qkv.weight': 'attn.Wqkv.weight', 'out.weight': 'attn.Wo.weight',
                   'wi.weight': 'mlp.Wi.weight', 'wo.weight': 'mlp.Wo.weight'}
        return f'model.encoder_model.layers.{layer}.{mapping.get(tail, tail)}'
    return 'model.' + name


def load_weights(checkpoint, pack):
    with checkpoint.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(length))
    raw = np.memmap(checkpoint, mode='r', dtype=np.uint8)
    manifest = json.loads((pack / 'manifest.json').read_text())
    quant_raw = np.memmap(pack / 'model.bin', mode='r', dtype=np.uint8)
    original, decoded = {}, {}
    for entry in manifest['tensors']:
        name, shape = entry['name'], entry['shape']
        src = header[source_name(name)]
        if src['dtype'] != 'F32' or src['shape'] != shape:
            raise ValueError(f'unsupported checkpoint tensor: {name}')
        begin, end = src['data_offsets']
        original[name] = raw[8+length+begin:8+length+end].view('<f4').reshape(shape)
        begin = entry['offset']
        n = math.prod(shape)
        if entry['encoding'] == 'f32_le':
            decoded[name] = quant_raw[begin:begin+4*n].view('<f4').reshape(shape)
        elif entry['encoding'] == 'i8_row_symmetric':
            rows, cols = shape
            q = quant_raw[begin:begin+n].view(np.int8).reshape(shape)
            scales = quant_raw[begin+n:begin+entry['length']].view('<f4').reshape(rows, 1)
            decoded[name] = q.astype(np.float32) * scales
        else:
            rows, cols = shape
            blocks = (cols+31)//32
            q = quant_raw[begin:begin+n].view(np.int8).reshape(shape)
            scales = quant_raw[begin+n:begin+entry['length']].view('<f4').reshape(rows, blocks)
            decoded[name] = q.astype(np.float32) * np.repeat(scales, 32, axis=1)[:, :cols]
    return manifest['config'], original, decoded


def forward(ids, c, w, quantize_activations=False):
    def linear(x, prefix):
        if quantize_activations:
            width = x.shape[-1]
            bound = min(8192., math.floor(np.float32(2**31-1)/(np.float32(127)*width)))
            maximum = np.max(np.abs(x), axis=-1, keepdims=True)
            inv = np.divide(bound, maximum, out=np.ones_like(maximum), where=maximum > 0)
            scale = np.where(maximum > 0, maximum/bound, 1.).astype(np.float32)
            x = np.clip(np.rint(x*inv), -bound, bound)*scale
        y = x @ w[prefix+'.weight'].T
        return y+w[prefix+'.bias'] if prefix+'.bias' in w else y

    def norm(x, prefix):
        centered = x-np.mean(x, axis=-1, keepdims=True)
        return centered/np.sqrt(np.mean(centered*centered, axis=-1, keepdims=True)+c['norm_eps'])*w[prefix+'.weight']

    def gelu(x):
        erf = np.frompyfunc(math.erf, 1, 1)(x/np.float32(math.sqrt(2))).astype(np.float32)
        return np.float32(0.5)*x*(np.float32(1)+erf)

    def project(x, prefix):
        x = linear(x, prefix+'.linear_1')
        return linear(gelu(x) if c['projector_activation'] == 'Gelu' else np.maximum(x, 0), prefix+'.linear_2')

    t, h, heads = len(ids), c['hidden_size'], c['attention_heads']
    d = h//heads
    x = norm(w['embeddings.weight'][ids], 'embeddings.norm')
    for i in range(c['layers']):
        p = f'encoder.{i}'
        a = norm(x, p+'.attn_norm') if i or c['first_layer_attention_norm'] else x
        q, k, v = [z.reshape(t, heads, d).transpose(1, 0, 2) for z in np.split(linear(a, p+'.qkv'), 3, axis=-1)]
        local = i % c['global_every'] != 0
        theta = c['local_rope_theta'] if local else c['global_rope_theta']
        angles = np.arange(t, dtype=np.float64)[:, None]/(theta**(2*np.arange(d//2, dtype=np.float64)/d))
        cos, sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        def rope(z):
            a, b = np.split(z, 2, axis=-1)
            return np.concatenate([a*cos-b*sin, b*cos+a*sin], axis=-1)
        scores = (rope(q) @ rope(k).transpose(0, 2, 1))*np.float32(1/math.sqrt(d))
        if local:
            scores[:, np.abs(np.arange(t)[:, None]-np.arange(t)) > c['local_attention']//2] = -np.inf
        probs = np.exp(scores-np.max(scores, axis=-1, keepdims=True))
        probs /= np.sum(probs, axis=-1, keepdims=True)
        x = x+linear((probs @ v).transpose(1, 0, 2).reshape(t, h), p+'.out')
        gate, value = np.split(linear(norm(x, p+'.mlp_norm'), p+'.wi'), 2, axis=-1)
        x = x+linear(gelu(gate)*value, p+'.wo')
    x = norm(x, 'final_norm')
    text = project(x[:1], 'text_projector')
    classes = project(x[np.asarray(ids) == c['class_token_id']], 'classes_projector')
    return np.sum(classes*text, axis=-1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/verdict-151m/model.safetensors'))
    parser.add_argument('--pack', type=Path, default=Path('models/verdict-pack'))
    parser.add_argument('--tokenizer', type=Path, default=Path('models/verdict-151m/tokenizer.json'))
    parser.add_argument('--cases', type=Path, default=Path('models/verdict-parity/cases.jsonl'))
    parser.add_argument('--case-ids', default='test_oos_00033,test_oos_00139,test_in_00673')
    parser.add_argument('--infer', type=Path, default=Path('target/release/verdict-infer'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    wanted = set(args.case_ids.split(','))
    cases = [x for line in args.cases.read_text().splitlines() if (x := json.loads(line))['id'] in wanted]
    if {c['id'] for c in cases} != wanted:
        raise ValueError('missing diagnostic case')
    config, original, decoded = load_weights(args.checkpoint, args.pack)
    results = []
    for case in cases:
        output = subprocess.check_output([str(args.infer), 'tokens', '--pack', str(args.pack), '--tokenizer', str(args.tokenizer), '--case-json', json.dumps(case)], text=True)
        ids = [int(x) for x in next(line[4:] for line in output.splitlines() if line.startswith('ids=')).split(',')]
        row = {'case': case['id'], 'tokens': len(ids), 'modes': {}}
        labels = [c['id'] for c in case['candidates']]
        for mode, weights, qa in [('checkpoint_f32', original, False), ('weight_only', decoded, False), ('weight_and_activation', decoded, True)]:
            logits = forward(ids, config, weights, qa)
            row['modes'][mode] = {'logits': logits, 'selected': labels[int(np.argmax(logits))]}
        output = subprocess.check_output([str(args.infer), 'ids', '--pack', str(args.pack), '--ids', ','.join(map(str, ids))], text=True)
        logits = [float(line.split(' logit ')[1].split()[0]) for line in output.splitlines() if line.startswith('class ')]
        row['modes']['native_block32'] = {'logits': logits, 'selected': labels[int(np.argmax(logits))]}
        results.append(row)
        args.out.write_text(json.dumps({'reference': 'independent NumPy; F32 BLAS accumulation', 'results': results}, indent=2)+'\n')
        print(case['id'], {k: v['selected'] for k, v in row['modes'].items()}, flush=True)


if __name__ == '__main__':
    main()
