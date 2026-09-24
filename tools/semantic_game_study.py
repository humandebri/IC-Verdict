#!/usr/bin/env python3
"""Offline screening of semantic game decisions; frozen before model evaluation."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from game_feasibility import Probe, ROOT

OUT = ROOT/'artifacts/semantic-game-study'
LABELS = {
 'terrarium': ['eat','flee','investigate','ignore'],
 'last_exit': ['consistent','contradictory','insufficient information'],
 'hundred': ['eat','rest','work','help'],
}
PREFIX = {
 'terrarium': 'Choose the animal action. ',
 'last_exit': 'Compare the claim with the evidence. ',
 'hundred': 'Choose the person\'s next action. ',
}
# Each pair changes a consequential fact; each row has two authored phrasings.
# Labels are idealized intended responses, not universal real-world judgments.
DATA = {
 'terrarium': [
 ('eat','Hungry rabbit finds a fresh carrot.','A rabbit needs food. A fresh carrot is nearby.'),
 ('ignore','Hungry rabbit finds a plastic carrot.','A rabbit needs food. The nearby carrot is a plastic replica.'),
 ('eat','Hungry cat finds cooked fish.','A cat needs food and finds a piece of cooked fish.'),
 ('ignore','Hungry cat finds a picture of fish.','A cat needs food. The fish nearby is only a photograph.'),
 ('eat','Hungry snake finds a live mouse.','A snake needs food. A living mouse is beside it.'),
 ('ignore','Hungry snake finds a rubber mouse.','A snake needs food. The mouse nearby is made of rubber.'),
 ('eat','Hungry rabbit finds fresh grass.','A rabbit needs food. Fresh grass grows beside it.'),
 ('flee','Rabbit sees a cat chasing it.','A cat is pursuing a rabbit. The rabbit must act.'),
 ('eat','Hungry cat finds fresh meat.','A cat needs food. Fresh meat is within reach.'),
 ('flee','Cat sees a dog charging at it.','A dog is running toward a cat to attack it.'),
 ('eat','Hungry rabbit finds a fresh apple.','A rabbit needs food. A fresh apple lies nearby.'),
 ('flee','Rabbit sees a snake lunging at it.','A snake is about to bite a rabbit.'),
 ('flee','Mouse sees a real cat hunting it.','A living cat is hunting a mouse.'),
 ('investigate','Curious mouse sees a harmless toy cat.','A curious mouse encounters a safe stuffed cat.'),
 ('flee','Rabbit sees a real wolf approaching to attack.','A wolf is approaching a rabbit with intent to attack.'),
 ('investigate','Curious rabbit sees a harmless stuffed wolf.','A curious rabbit finds a safe plush wolf.'),
 ('flee','Cat sees flames spreading toward it.','Fire is spreading toward a cat.'),
 ('investigate','Curious cat sees a safe flickering light toy.','A curious cat finds a harmless toy with blinking lights.'),
 ('investigate','Curious cat finds an unfamiliar safe box.','A curious cat encounters a new harmless cardboard box.'),
 ('ignore','Sleeping cat is content. An old box sits far away.','A satisfied cat sleeps while a familiar box remains far away.'),
 ('investigate','Curious rabbit finds a new safe tunnel.','A curious rabbit discovers a harmless unfamiliar tunnel.'),
 ('ignore','Resting rabbit is content. A familiar tunnel is far away.','A satisfied rabbit rests far from its familiar tunnel.'),
 ('investigate','Curious mouse finds a new harmless wooden object.','A curious mouse notices a safe unfamiliar wooden object.'),
 ('ignore','Resting mouse is content. An old wooden object is far away.','A satisfied mouse rests far from a familiar wooden object.'),
 ],
 'last_exit': [
 ('consistent','Claim: only fruit. Scan: apples only.','The driver reports fruit only. The scanner finds only apples.'),
 ('contradictory','Claim: only fruit. Scan: rifles only.','The driver reports fruit only. The scanner finds only rifles.'),
 ('consistent','Claim: no passengers. Search: empty seats.','The driver says nobody else is aboard. Every passenger seat is empty.'),
 ('contradictory','Claim: no passengers. Search: two passengers.','The driver says nobody else is aboard. Two people occupy the back seat.'),
 ('consistent','Claim: heading north. GPS: northbound.','The driver says they are going north. GPS confirms travel north.'),
 ('contradictory','Claim: heading north. GPS: southbound.','The driver says they are going north. GPS confirms travel south.'),
 ('consistent','Claim: no animals. Search: books only.','The driver denies carrying animals. The search finds only books.'),
 ('contradictory','Claim: no animals. Search: live dog.','The driver denies carrying animals. A living dog is found.'),
 ('consistent','Claim: carrying medicine. Search: labeled medicine.','The driver reports medical supplies. A search confirms medical supplies.'),
 ('insufficient information','Claim: carrying medicine. Cargo not inspected.','The driver reports medical supplies. Nobody has examined the cargo.'),
 ('consistent','Claim: arrived yesterday. Entry record: yesterday.','The driver says they arrived yesterday. The entry log confirms yesterday.'),
 ('insufficient information','Claim: arrived yesterday. Entry record unavailable.','The driver says they arrived yesterday. The entry log cannot be accessed.'),
 ('consistent','Claim: permit is valid. Database confirms valid permit.','The driver claims a valid permit. The registry confirms it is valid.'),
 ('insufficient information','Claim: permit is valid. Database offline.','The driver claims a valid permit. The registry is unavailable.'),
 ('consistent','Claim: transporting glass. Search confirms glass.','The driver says the load is glass. Inspection confirms glass.'),
 ('insufficient information','Claim: transporting glass. Crate remains sealed.','The driver says the load is glass. The crate has not been opened.'),
 ('contradictory','Claim: carrying no weapons. Search finds a pistol.','The driver denies having weapons. Inspection finds a handgun.'),
 ('insufficient information','Claim: carrying no weapons. No search performed.','The driver denies having weapons. No inspection has been made.'),
 ('contradictory','Claim: vehicle is empty. Search finds furniture.','The driver says the vehicle contains nothing. Inspectors find furniture.'),
 ('insufficient information','Claim: vehicle is empty. Windows are opaque.','The driver says the vehicle contains nothing. Its contents are not visible.'),
 ('contradictory','Claim: only frozen goods. Scan: live birds.','The driver reports exclusively frozen cargo. The scan reveals living birds.'),
 ('insufficient information','Claim: only frozen goods. Scanner is broken.','The driver reports exclusively frozen cargo. The scanner is not working.'),
 ('contradictory','Claim: no liquids. Search: bottled water.','The driver denies carrying liquids. Inspection finds bottles of water.'),
 ('insufficient information','Claim: no liquids. No cargo evidence available.','The driver denies carrying liquids. There is no evidence about the cargo.'),
 ],
 'hundred': [
 ('eat','Mira is starving, rested, and has food ready. Nobody needs help.','Mira urgently needs food, has energy, and a meal is available. Others are safe.'),
 ('rest','Mira is exhausted, fed, and has a bed ready. Nobody needs help.','Mira urgently needs sleep, has eaten, and a bed is available. Others are safe.'),
 ('eat','Noah has not eaten all day. He is rested and lunch is ready.','Noah has plenty of energy but urgently needs the meal beside him.'),
 ('rest','Noah has not slept all night. He is fed and a bed is ready.','Noah has eaten enough but urgently needs the bed beside him.'),
 ('eat','Ava is faint from hunger. She slept well and has soup.','Ava is well rested but urgently needs food. Soup is available.'),
 ('rest','Ava is faint from exhaustion. She ate well and has a bed.','Ava has eaten well but urgently needs sleep. A bed is available.'),
 ('eat','Leo is starving. Work can wait; food is ready.','Leo urgently needs food. A meal is ready and his job is not urgent.'),
 ('work','Leo is fed and rested. His shift starts now; nobody needs help.','Leo has no urgent needs and must start his scheduled job. Others are safe.'),
 ('eat','Iris is starving. She has food; work starts tomorrow.','Iris needs to eat immediately. A meal is ready; her shift is tomorrow.'),
 ('work','Iris is fed and rested. Her shift starts now; nobody needs help.','Iris has eaten and slept enough. It is time to start work. Others are safe.'),
 ('eat','Owen is hungry and rested. Lunch is ready; work is finished.','Owen has energy but needs food. His job is done and a meal awaits.'),
 ('work','Owen is fed and rested. His shift starts now; nobody needs help.','Owen has no unmet needs. His scheduled work begins now. Others are safe.'),
 ('rest','Zoe is exhausted. Everyone is safe and a bed is available.','Zoe urgently needs sleep. A bed is ready and others need nothing.'),
 ('help','Zoe is fed and rested. A child is trapped and calling for help.','Zoe has energy and has eaten. A trapped child urgently needs rescue.'),
 ('rest','Eli is exhausted. Everyone is safe and a bed is available.','Eli urgently needs sleep. Others are safe and a bed is ready.'),
 ('help','Eli is fed and rested. A neighbor fell and cannot get up.','Eli has no urgent personal needs. A fallen neighbor needs assistance.'),
 ('rest','Ada is exhausted. Everyone is safe and a bed is available.','Ada urgently needs sleep. Nobody needs assistance and a bed is ready.'),
 ('help','Ada is fed and rested. A friend is drowning nearby.','Ada has eaten and slept enough. A nearby friend urgently needs rescue from water.'),
 ('work','Finn is fed and rested. Work starts now. Everyone is safe.','Finn has no unmet needs and should begin his shift. Others need no assistance.'),
 ('help','Finn is fed and rested. Work can wait; someone is trapped.','Finn has no unmet needs. His job is not urgent but a person needs rescue.'),
 ('work','Nia is fed and rested. Work starts now. Everyone is safe.','Nia has eaten and slept enough. Her shift begins and others need nothing.'),
 ('help','Nia is fed and rested. Work can wait; a lost child needs help.','Nia has no unmet needs. A child urgently needs assistance; her job can wait.'),
 ('work','Ben is fed and rested. Work starts now. Everyone is safe.','Ben has no urgent personal needs. It is time for his job and others are safe.'),
 ('help','Ben is fed and rested. Work can wait; an injured friend needs help.','Ben has energy and has eaten. His injured friend needs assistance before work.'),
 ],
}

def fixtures():
    rows=[]
    for game,cases in DATA.items():
        for i,(expected,*texts) in enumerate(cases):
            for wording,text in enumerate(texts):
                for order in range(2):
                    labels=LABELS[game][::1 if order==0 else -1]
                    rows.append(dict(id=f'{game}/{i}/{wording}/{order}',game=game,case=i,pair=i//2,wording=wording,order=order,
                                     expected=expected,text=PREFIX[game]+text,labels=labels))
    return rows

def summarize(rows):
    output={}
    for game in DATA:
        group=[r for r in rows if r['game']==game]
        completed=[r for r in group if 'answer' in r]
        pairs=sum(all(r.get('answer')==r['expected'] for r in group if r['pair']==p) for p in range(12))
        output[game]=dict(correct=sum(r.get('answer')==r['expected'] for r in group),total=len(group),
                         completed=len(completed),all_eight_pair_variants_correct=pairs,pairs=12,
                         per_label={label:dict(correct=sum(r.get('answer')==label and r['expected']==label for r in group),total=sum(r['expected']==label for r in group)) for label in LABELS[game]},
                         selected=dict(Counter(r['answer'] for r in completed)),
                         order_consistent=sum(next(r.get('answer') for r in group if r['case']==i and r['wording']==w and r['order']==0) is not None and next(r.get('answer') for r in group if r['case']==i and r['wording']==w and r['order']==0)==next(r.get('answer') for r in group if r['case']==i and r['wording']==w and r['order']==1) for i in range(24) for w in range(2)),order_comparisons=48)
    return output

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptive',action='store_true',help='Exploratory follow-up with descriptive candidate labels, same cases.')
    args=parser.parse_args()
    if args.descriptive:
        OUT=OUT/'descriptive'
    OUT.mkdir(parents=True,exist_ok=True)
    cases=fixtures()
    descriptions={
      'terrarium':dict(zip(LABELS['terrarium'],['eat the food','escape danger','inspect a novel object','leave it alone'])),
      'last_exit':dict(zip(LABELS['last_exit'],['evidence supports the claim','evidence contradicts the claim','evidence cannot verify the claim'])),
      'hundred':dict(zip(LABELS['hundred'],['eat a meal','sleep to recover','start the job','assist someone'])),
    }
    for case in cases:
        case['input_labels']=[descriptions[case['game']][label] for label in case['labels']] if args.descriptive else case['labels']
    fixture_bytes=(json.dumps(cases,indent=2)+'\n').encode()
    (OUT/'fixtures.json').write_bytes(fixture_bytes)
    protocol=dict(mode='native CPU only; no canister calls',tokens=52,descriptive_followup=args.descriptive,
                  threshold='At least 90% overall and 80% each label for a small prototype; not a production or full-game guarantee.',
                  scope='English, 24 authored cases per project, two paraphrases and two label orders. 96 responses per project are correlated, not 96 independent cases.',
                  fixture_sha256=hashlib.sha256(fixture_bytes).hexdigest(),
                  hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['tools/semantic_game_study.py','target/release/tetris_probe','models/verdict-pack/model.bin','models/verdict-151m/tokenizer.json']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    probe=Probe(); rows=[]
    try:
        for c in cases:
            row=dict(c)
            try:
                selected=probe.ask(c['text'],c['input_labels'])
                row['answer']=c['labels'][selected]
                result=probe.cache[json.dumps([c['text'],c['input_labels']])]
                assert len(result['scores'])==len(c['labels'])
                assert all(math.isfinite(x) for x in result['scores'])
                assert abs(sum(result['scores'])-1)<1e-5
                row['tokens']=result['tokens']
            except ValueError as e:
                row['error']=str(e)
            rows.append(row)
            if len(rows)%24==0:
                print(f"{len(rows)}/{len(cases)} complete",flush=True)
                (OUT/'responses.json').write_text(json.dumps(rows,indent=2)+'\n')
        summary=summarize(rows)
        for s in summary.values():
            s['passes_screen']=s['correct']/s['total']>=.9 and all(x['correct']/x['total']>=.8 for x in s['per_label'].values())
        (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        (OUT/'inferences.json').write_text(json.dumps(probe.rows,indent=2)+'\n')
        probe.close()

if __name__=='__main__':main()
