#!/usr/bin/env python3
"""Offline checkpoint / quantization / prompt diagnosis. Never calls a canister."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from diagnose_verdict import load_weights,forward
from game_feasibility import Probe,ROOT
from semantic_game_study import PREFIX

OUT=ROOT/'artifacts/model-quality-diagnosis'

def softmax(values,temp=1.):
    x=np.asarray(values,dtype=np.float64)/temp
    e=np.exp(x-x.max());return (e/e.sum()).tolist()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c,w,q=load_weights(ROOT/'models/verdict-151m/model.safetensors',ROOT/'models/verdict-pack')
    probe=Probe()
    files=['models/verdict-151m/model.safetensors','models/verdict-pack/model.bin','models/verdict-pack/manifest.json',
           'models/verdict-151m/tokenizer.json','models/verdict-parity/cases.jsonl','models/verdict-parity/predictions.jsonl',
           'target/release/tetris_probe','target/release/verdict-infer','tools/diagnose_verdict.py','tools/verdict_quality_diagnosis.py']
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    (OUT/'protocol.json').write_text(json.dumps(dict(files=provenance,bank_sample='first 100 records, existing fixed order',games='all 288 descriptive-label requests',format_sample='72 original cases, wording 0, forward label order',reference='independent NumPy F32; validate against author predictions first',canister_calls=0),indent=2)+'\n')
    def tokenize(text,labels):return probe.rpc(dict(text=text,labels=labels))
    def native(ids):
        text=subprocess.check_output([str(ROOT/'target/release/verdict-infer'),'ids','--pack',str(ROOT/'models/verdict-pack'),'--ids',','.join(map(str,ids))],text=True)
        return [float(line.split(' logit ')[1].split()[0]) for line in text.splitlines() if line.startswith('class ')]
    def save(name,rows): (OUT/name).write_text(json.dumps(rows,indent=2)+'\n')
    try:
        predictions={x['id']:x for line in (ROOT/'models/verdict-parity/predictions.jsonl').read_text().splitlines() if (x:=json.loads(line))}
        cases=[json.loads(line) for line in (ROOT/'models/verdict-parity/cases.jsonl').read_text().splitlines()][:100]
        bank=[];temp=json.loads((ROOT/'models/verdict-151m/calibrator.json').read_text())['temperature']
        for case in cases:
            text=f"Question: {case['question']}\n\nContext:\n{case['text']}"
            tok=tokenize(text,[x['description'] for x in case['candidates']]);ids=tok['ids']
            ref=forward(ids,c,w);rust=native(ids);pred=predictions[case['id']]
            labels=[x['id'] for x in case['candidates']]
            bank.append(dict(id=case['id'],target=case['target_id'],author=pred['predicted_id'],author_confidence=pred['confidence'],
                             f32=labels[int(np.argmax(ref))],native=labels[int(np.argmax(rust))],tokens=len(ids),
                             f32_confidence=max(softmax(ref,temp)),native_confidence=max(softmax(rust,temp)),f32_logits=ref,native_logits=rust))
            if len(bank)%20==0:save('bank.json',bank);print('bank',len(bank),flush=True)
        raw=json.loads((ROOT/'artifacts/semantic-game-study/descriptive/inferences.json').read_text())
        by_request={json.dumps([r['request']['text'],r['request']['labels']]):r['result'] for r in raw}
        cases=json.loads((ROOT/'artifacts/semantic-game-study/descriptive/responses.json').read_text())
        games=[]
        for case in cases:
            archived=by_request[json.dumps([case['text'],case['input_labels']])];ids=archived['ids']
            ref=forward(ids,c,w);decoded=forward(ids,c,q)
            result=dict(id=case['id'],game=case['game'],expected=case['expected'],native=case['answer'],
                        f32=case['labels'][int(np.argmax(ref))],decoded=case['labels'][int(np.argmax(decoded))],
                        f32_logits=ref,decoded_logits=decoded,native_scores=archived['scores'],tokens=len(ids))
            if result['decoded']!=result['native']:
                acts=forward(ids,c,q,True);result['activation_quantized']=case['labels'][int(np.argmax(acts))]
                result['activation_logits']=acts
            games.append(result)
            if len(games)%24==0:save('games.json',games);print('games',len(games),flush=True)
        formatted=[]
        for case in cases:
            if case['wording'] or case['order']:continue
            context=case['text'][len(PREFIX[case['game']]):]
            question=PREFIX[case['game']].strip()
            for mode in ('question_context','label_prefix','sdk_choice'):
                labels=case['input_labels'][:];keys=case['labels'][:]
                if mode in ('label_prefix','sdk_choice'):
                    labels=['It is '+label for label in labels]
                if mode=='sdk_choice':
                    labels+=['insufficient evidence'];keys+=['__insufficient_evidence__']
                text=f'Question: {question}\n\nContext:\n{context}'
                tok=tokenize(text,labels);ref=forward(tok['ids'],c,w)
                formatted.append(dict(id=case['id'],game=case['game'],mode=mode,expected=case['expected'],answer=keys[int(np.argmax(ref))],tokens=tok['tokens'],text=text,labels=labels,logits=ref))
            if len(formatted)%24==0:save('formatted.json',formatted);print('formatted',len(formatted),flush=True)
        summary=dict(bank=dict(total=len(bank),f32_author_match=sum(r['f32']==r['author'] for r in bank),native_author_match=sum(r['native']==r['author'] for r in bank),
                              f32_correct=sum(r['f32']==r['target'] for r in bank),native_correct=sum(r['native']==r['target'] for r in bank),
                              max_f32_author_confidence_delta=max(abs(r['f32_confidence']-r['author_confidence']) for r in bank)),games={},formatted={})
        for game in PREFIX:
            rows=[r for r in games if r['game']==game]
            summary['games'][game]=dict(total=len(rows),**{mode+'_correct':sum(r[mode]==r['expected'] for r in rows) for mode in ('f32','decoded','native')},
                                       f32_native_agreement=sum(r['f32']==r['native'] for r in rows),decoded_native_agreement=sum(r['decoded']==r['native'] for r in rows))
            for mode in ('question_context','label_prefix','sdk_choice'):
                rows=[r for r in formatted if r['game']==game and r['mode']==mode]
                summary['formatted'][game+'/'+mode]=dict(total=len(rows),correct=sum(r['answer']==r['expected'] for r in rows),abstain=sum(r['answer']=='__insufficient_evidence__' for r in rows),max_tokens=max(r['tokens'] for r in rows))
        save('result.json',summary);print(json.dumps(summary,indent=2),flush=True)
    finally:probe.close()

if __name__=='__main__':main()
