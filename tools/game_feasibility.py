#!/usr/bin/env python3
"""Offline frozen-model Snake/2048 screening; no network or canister calls."""
import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/game-feasibility'
NAMES = ('up', 'right', 'down', 'left')
DELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))
SEEDS = (11, 23, 37, 53, 71)
LIMIT = 150

class Probe:
    def __init__(self):
        self.p = subprocess.Popen([str(ROOT/'target/release/tetris_probe'), str(ROOT)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.cache = {}
        self.rows = []
    def ask(self, text, labels):
        key = json.dumps([text, labels])
        if key not in self.cache:
            request = dict(text=text, labels=labels)
            tokenized = self.rpc(request)
            if tokenized['tokens'] > 52:
                raise ValueError(f"token budget: {tokenized['tokens']}")
            start = time.monotonic()
            result = self.rpc(dict(request, infer=True))
            self.cache[key] = result
            self.rows.append(dict(request=request, result=result, seconds=time.monotonic()-start))
        return self.cache[key]['selected']
    def rpc(self, request):
        self.p.stdin.write(json.dumps(request)+'\n'); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('native probe exited')
        return json.loads(line)
    def close(self):
        self.p.stdin.close(); self.p.wait(timeout=30); self.p.stdout.close()

def slide(board, action):
    """Standard 2048: each original tile merges at most once per move."""
    out = list(board); score = 0
    for i in range(4):
        indices = ([i+4*j for j in range(4)] if action == 0 else
                   [4*i+j for j in reversed(range(4))] if action == 1 else
                   [i+4*j for j in reversed(range(4))] if action == 2 else
                   [4*i+j for j in range(4)])
        values = [board[j] for j in indices if board[j]]
        merged = []; k = 0
        while k < len(values):
            if k+1 < len(values) and values[k] == values[k+1]:
                merged.append(values[k]*2); score += values[k]*2; k += 2
            else:
                merged.append(values[k]); k += 1
        for j, value in zip(indices, merged+[0]*(4-len(merged))):
            out[j] = value
    return out, score

def spawn(board, rng):
    empty = [i for i, v in enumerate(board) if not v]
    if empty:
        board[rng.choice(empty)] = 2 if rng.random() < .9 else 4

def request2048(board, mode):
    if mode == 'state':
        grid = ';'.join(','.join(map(str, board[i:i+4])) for i in range(0,16,4))
        return '2048 rows: '+grid, list(NAMES)
    labels = []
    for action, name in enumerate(NAMES):
        after, gain = slide(board, action)
        labels.append(name+' invalid' if after == board else f'{name} empty{after.count(0)} score{gain}')
    return '2048. Prefer more empty cells, then greater merge score.', labels

def play2048(seed, policy, probe):
    rng = random.Random(seed); choice_rng = random.Random(seed+1000)
    board = [0]*16; spawn(board, rng); spawn(board, rng)
    score = invalid = streak = steps = 0; trace = []; stop = 'step_limit'
    for step in range(LIMIT):
        options = [slide(board,a) for a in range(4)]
        legal = [a for a in range(4) if options[a][0] != board]
        if not legal:
            stop = 'game_over'; break
        if policy == 'random':
            action = choice_rng.randrange(4)
        elif policy == 'random_legal':
            action = choice_rng.choice(legal)
        elif policy == 'greedy':
            action = max(legal, key=lambda a:(options[a][0].count(0), options[a][1]))
        else:
            try:
                text, labels = request2048(board, policy)
                offered = legal if policy == 'legal_outcome' else list(range(4))
                action = offered[probe.ask(text, [labels[a] for a in offered])]
            except ValueError as e:
                stop = str(e); break
        after, gain = options[action]; steps += 1
        trace.append(dict(board=board[:], action=action, gain=gain, valid=action in legal))
        if action not in legal:
            invalid += 1; streak += 1
            if streak >= 16:
                stop = '16_consecutive_noops'; break
        else:
            streak = 0; score += gain; board = after; spawn(board, rng)
    return dict(game='2048', seed=seed, policy=policy, steps=steps, score=score,
                max_tile=max(board), invalid=invalid, stop=stop, trace=trace)

def snake_options(body, food, direction):
    result = {}
    for action, (dx,dy) in enumerate(DELTA):
        if action == (direction+2)%4:
            continue  # Standard Snake forbids reversing into the neck.
        head = (body[0][0]+dx, body[0][1]+dy)
        eating = head == food
        death = not (0 <= head[0] < 6 and 0 <= head[1] < 6) or head in (body if eating else body[:-1])
        distance = abs(head[0]-food[0])+abs(head[1]-food[1])
        result[action] = dict(head=head, eating=eating, death=death, distance=distance)
    return result

def request_snake(body, food, direction, mode):
    options = snake_options(body,food,direction)
    names = [NAMES[a] for a in options]
    if mode == 'state':
        horizontal = 'left' if food[0] < body[0][0] else 'right' if food[0] > body[0][0] else ''
        vertical = 'up' if food[1] < body[0][1] else 'down' if food[1] > body[0][1] else ''
        danger = ','.join(NAMES[a] for a,o in options.items() if o['death']) or 'none'
        return f"Snake. Food {','.join(filter(None,(horizontal,vertical)))}. Danger {danger}. Choose safe move toward food.", names
    old = abs(body[0][0]-food[0])+abs(body[0][1]-food[1])
    labels = [NAMES[a]+' '+('death' if o['death'] else 'food' if o['eating'] else 'closer' if o['distance']<old else 'farther') for a,o in options.items()]
    return 'Snake. Avoid death. Prefer food, then closer to food.', labels

def food_at(body, rng):
    free = [(x,y) for y in range(6) for x in range(6) if (x,y) not in body]
    return rng.choice(free) if free else None

def play_snake(seed, policy, probe):
    rng = random.Random(seed); choice_rng = random.Random(seed+1000)
    body = [(2,3),(1,3),(0,3)]; direction = 1; food = food_at(body,rng)
    eaten = steps = 0; trace = []; stop = 'step_limit'
    for step in range(LIMIT):
        options = snake_options(body,food,direction); actions = list(options)
        if policy == 'random':
            action = choice_rng.choice(actions)
        elif policy == 'random_legal':
            safe = [a for a in actions if not options[a]['death']]
            action = choice_rng.choice(safe or actions)
        elif policy == 'greedy':
            action = min(actions, key=lambda a:(options[a]['death'], options[a]['distance']))
        else:
            try:
                text, labels = request_snake(body,food,direction,policy)
                offered = [i for i,a in enumerate(actions) if not options[a]['death']] if policy == 'legal_outcome' else list(range(len(actions)))
                offered = offered or list(range(len(actions)))
                action = actions[offered[probe.ask(text, [labels[i] for i in offered])]]
            except ValueError as e:
                stop = str(e); break
        o = options[action]; steps += 1
        trace.append(dict(body=body[:], food=food, action=action, death=o['death'], eating=o['eating']))
        if o['death']:
            stop = 'collision'; break
        direction = action; body.insert(0,o['head'])
        if o['eating']:
            eaten += 1; food = food_at(body,rng)
            if food is None:
                stop = 'win'; break
        else:
            body.pop()
    return dict(game='snake', seed=seed, policy=policy, steps=steps, food=eaten, stop=stop, trace=trace)

def gates(probe):
    rows = []
    # Fully balanced immediate-food decisions. All other offered actions collide.
    for direction in range(4):
        actions = [a for a in range(4) if a != (direction+2)%4]
        for winner in actions:
            for order in (actions, list(reversed(actions))):
                for mode in ('state','outcome'):
                    if mode == 'state':
                        text = f"Snake. Food {NAMES[winner]}. Danger {','.join(NAMES[a] for a in actions if a != winner)}. Choose safe move toward food."
                        labels = [NAMES[a] for a in order]
                    else:
                        text = 'Snake. Avoid death. Prefer food, then closer to food.'
                        labels = [NAMES[a]+(' food' if a==winner else ' death') for a in order]
                    selected = order[probe.ask(text,labels)]
                    rows.append(dict(game='snake',mode=mode,expected=[winner],selected=selected,correct=selected==winner))
    # Horizontal/vertical adjacent pairs at all rows/columns; only merges correct.
    for axis in range(2):
        for line in range(4):
            for value in (2,8,32):
                board = [0]*16
                indices = [line*4,line*4+1] if axis==0 else [line,line+4]
                for i in indices: board[i]=value
                expected = [a for a in range(4) if slide(board,a)[1]>0]
                for mode in ('state','outcome'):
                    selected = probe.ask(*request2048(board,mode))
                    rows.append(dict(game='2048',mode=mode,board=board,expected=expected,selected=selected,correct=selected in expected))
    return rows

def main():
    global OUT, LIMIT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assisted', action='store_true', help='Separate follow-up: mask immediate death/no-op actions, with equally masked random baseline.')
    parser.add_argument('--long-2048', action='store_true', help='Separate 1000-step follow-up of outcome input and baselines.')
    args = parser.parse_args()
    if args.assisted and args.long_2048:
        parser.error('choose one follow-up')
    if args.long_2048:
        OUT = OUT/'long-2048'
        LIMIT = 1000
    if args.assisted:
        OUT = OUT/'assisted'
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = dict(seeds=SEEDS,max_decisions=LIMIT,snake_board=[6,6],policies=['outcome','random_legal','greedy'] if args.long_2048 else ['legal_outcome','random_legal'] if args.assisted else ['state','outcome','random','greedy'],
                    gate_threshold=.9,queries='native only',token_limit=52,
                    caveat='Snake is turn-based and sees only immediate danger and food direction; outcome mode supplies one-step mechanics. No long-range planner.',
                    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['target/release/tetris_probe','models/verdict-pack/model.bin','models/verdict-151m/tokenizer.json']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    probe = Probe(); games = []
    if args.long_2048:
        for row in json.loads((OUT.parent/'inferences.json').read_text()):
            probe.cache[json.dumps([row['request']['text'],row['request']['labels']])] = row['result']
    try:
        checks = [] if args.assisted or args.long_2048 else gates(probe)
        (OUT/'gates.json').write_text(json.dumps(checks,indent=2)+'\n')
        for game, run in ([('2048',play2048)] if args.long_2048 else [('snake',play_snake),('2048',play2048)]):
            for policy in protocol['policies']:
                for seed in SEEDS:
                    result = run(seed,policy,probe); games.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
                (OUT/'games.json').write_text(json.dumps(games,indent=2)+'\n')
        summary = dict(gates={},games={},native_inferences=len(probe.rows),max_tokens=max(r['result']['tokens'] for r in probe.rows))
        for game in (('2048',) if args.long_2048 else ('snake','2048')):
            for mode in ('state','outcome'):
                subset = [r for r in checks if r['game']==game and r['mode']==mode]
                if subset:
                    summary['gates'][game+'/'+mode] = dict(correct=sum(r['correct'] for r in subset),total=len(subset))
            for policy in protocol['policies']:
                subset = [r for r in games if r['game']==game and r['policy']==policy]
                keys = ('steps','food') if game=='snake' else ('steps','score','max_tile','invalid')
                summary['games'][game+'/'+policy] = {k:sum(r[k] for r in subset)/len(subset) for k in keys}
                summary['games'][game+'/'+policy]['stops'] = [r['stop'] for r in subset]
        (OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    finally:
        (OUT/'inferences.json').write_text(json.dumps(probe.rows,indent=2)+'\n')
        probe.close()

if __name__=='__main__': main()
