import {describe,it,expect} from 'vitest';
import {RealtimeGame,FRAME_MS,gravityFrames,clearPoints,levelForLines,pieceGenerator,fits,route,landingsFor,decisionInput,type Action} from './realtime';
import {emptyBoard} from './game';
const frames=(g:RealtimeGame,n:number)=>{for(let i=0;i<n;i++)g.tick();};
const send=(g:RealtimeGame,action:Action,now=0)=>{const t=g.beginQuery(now)!;expect(t).toBeDefined();g.response(t.request,action,10);g.tick();return t;};
describe('query v3 controls',()=>{
  it('sends terrain and pose, excludes blocked controls but always allows down/wait',()=>{
    const board=emptyBoard();board[18][0]=1;
    const input=decisionInput(board,1,2,{x:1,y:0,r:0});
    expect(input.heights).toEqual([2,0,0,0,0,0,0,0,0,0]);expect(input.holes).toBe(1);expect(input.legal_mask).toBe(26);
    expect(input).toMatchObject({piece:1,next:2,x:1,y:0,rotation:0});
  });
  it('executes once and queries the same piece repeatedly without an automatic route',()=>{
    const g=new RealtimeGame(1,0),a=g.active!,x=a.pose.x;
    const t=send(g,0);expect(t.outcome).toBe('executed');expect(a.pose.x).toBe(x-1);
    frames(g,20);expect(a.pose.x).toBe(x-1);expect(g.beginQuery(999)).toBeUndefined();
    send(g,1,1000);expect(a.pose.x).toBe(x);expect(g.metrics.executed).toBe(2);
  });
  it('carries previous pose/action/outcome and two older executed actions, resetting on spawn',()=>{
    const g=new RealtimeGame(1,0),a=g.active!;
    send(g,0);frames(g,6);send(g,1,1000);frames(g,6);send(g,4,2000);
    const t=g.beginQuery(3000)!;
    expect(t.input.previous).toEqual([{x:5,y:0,rotation:0,action:4,outcome:0}]);expect(t.input.history).toEqual([0,1]);
    g.response(t.request,4,0);g.tick();g.active=undefined;g.spawn();
    const next=g.beginQuery(4000)!;expect(next.piece).not.toBe(a.id);expect(next.input.previous).toEqual([]);expect(next.input.history).toEqual([]);
  });
  it('reports an unexecuted previous choice as blocked, not as an executed history action',()=>{
    const g=new RealtimeGame(1,0);g.active!.piece=1;const t=g.beginQuery(0)!;
    g.active!.pose.y=10;g.board[10][3]=1;g.response(t.request,0,500);g.tick();
    const next=g.beginQuery(1000)!;expect(next.input.previous[0]?.outcome).toBe(1);expect(next.input.history).toEqual([]);
  });
  it('blocks concurrent requests and duplicate replies',()=>{
    const g=new RealtimeGame(1,0),t=g.beginQuery(0)!;
    expect(g.beginQuery(2000)).toBeUndefined();g.response(t.request,0,50);
    expect(g.beginQuery(2000)).toBeUndefined();g.response(t.request,1,50);g.tick();
    expect(g.metrics.responses).toBe(1);expect(g.active!.pose.x).toBe(4);
  });
  it('rotates clockwise once and wait only permits natural gravity',()=>{
    const g=new RealtimeGame(1,0);g.active!.piece=2;
    send(g,2);expect(g.active!.pose.r).toBe(1);
    const p={...g.active!.pose};send(g,4,1000);expect(g.active!.pose).toEqual(p);
    frames(g,46);expect(g.active!.pose.y).toBe(p.y+1);
  });
  it('down moves one row, awards one point on lock, and does not double-drop with gravity',()=>{
    const g=new RealtimeGame(1,0);g.active!.fall=47;send(g,3);
    expect(g.active!.pose.y).toBe(1);expect(g.active!.softPoints).toBe(1);expect(g.active!.fall).toBe(0);
    frames(g,47);expect(g.active!.pose.y).toBe(1);g.tick();expect(g.active!.pose.y).toBe(2);
    g.active!.pose={x:5,y:18,r:0};g.active!.piece=1;
    send(g,3,1000);expect(g.active).toBeUndefined();expect(g.count).toBe(1);expect(g.score).toBe(1);
  });
  it('respects six-frame movement and two-frame down cadence',()=>{
    const g=new RealtimeGame(1,0);g.active!.input=6;
    const t=g.beginQuery(0)!;g.response(t.request,0,0);frames(g,5);expect(t.outcome).toBe('queued');g.tick();expect(t.outcome).toBe('executed');
    g.active!.dropInput=2;const d=g.beginQuery(1000)!;g.response(d.request,3,0);g.tick();expect(d.outcome).toBe('queued');g.tick();expect(d.outcome).toBe('executed');
  });
  it('revalidates collision after the piece falls during query latency',()=>{
    const g=new RealtimeGame(1,0);g.active!.piece=1;const t=g.beginQuery(0)!;
    g.active!.pose.y=10;g.board[10][3]=1;
    g.response(t.request,0,500);g.tick();expect(t.outcome).toBe('blocked');expect(g.metrics.blocked).toBe(1);expect(g.active!.pose.x).toBe(5);
    expect(g.beginQuery(1000)).toBeDefined();
  });
  it('continues gravity while waiting and discards answers after lock',()=>{
    const g=new RealtimeGame(1,0),t=g.beginQuery(0)!;frames(g,48);expect(g.active!.pose.y).toBe(1);
    g.lines=290;while(g.active?.id===t.piece)g.tick();expect(g.metrics.late).toBe(1);
    frames(g,20);const p={...g.active!.pose};g.response(t.request,0,20000);expect(t.outcome).toBe('late');expect(g.metrics.late).toBe(1);expect(g.active!.pose).toEqual(p);
  });
  it('discards a queued input when gravity locks before its input slot',()=>{
    const g=new RealtimeGame(1,0);Object.assign(g.active!,{piece:1,pose:{x:5,y:18,r:0},fall:47,input:6});
    const t=g.beginQuery(0)!;g.response(t.request,0,0);g.tick();
    expect(t.outcome).toBe('late');expect(g.queued).toBeUndefined();expect(g.metrics.executed).toBe(0);
    frames(g,g.delay);expect(g.beginQuery(1000)).toBeDefined();
  });
  it('drops replies across a hidden/resumed epoch and queued inputs on pause',()=>{
    const g=new RealtimeGame(1,0),t=g.beginQuery(0)!;g.pause(true,0);g.advance(100000);expect(g.active!.pose.y).toBe(0);
    g.pause(false,100000);g.response(t.request,0,100000);expect(t.outcome).toBe('stale');expect(g.metrics.valid).toBe(false);
    const next=g.beginQuery(100001)!;g.response(next.request,1,0);g.pause(true,100001);expect(next.outcome).toBe('stale');expect(g.queued).toBeUndefined();
  });
  it('retries after errors with gravity running and a one-second dispatch gap',()=>{
    const g=new RealtimeGame(1,0),t=g.beginQuery(0)!;g.fail(t.request);frames(g,48);
    expect(g.metrics.errors).toBe(1);expect(g.active!.pose.y).toBe(1);expect(g.beginQuery(999)).toBeUndefined();expect(g.beginQuery(1000)).toBeDefined();
  });
  it('accounts for all elapsed frames and flags long stalls',()=>{
    const a=new RealtimeGame(1,0),b=new RealtimeGame(1,0);
    for(let t=0;t<=2000;t+=8)a.advance(t);b.advance(2000);expect(a.active!.pose).toEqual(b.active!.pose);expect(b.metrics.valid).toBe(false);
  });
  it('keeps history bounded without imposing an action limit',()=>{
    const g=new RealtimeGame(1,0);for(let i=0;i<105;i++)send(g,4,i*1000);
    expect(g.history.length).toBe(100);expect(g.metrics.executed).toBe(105);
  });
});
describe('retained NES physics and clockwise-only baseline',()=>{
  it('preserves NTSC timing and scoring',()=>{
    expect([0,1,9,10,12,13,15,16,18,19,28,29].map(gravityFrames)).toEqual([48,43,6,5,5,4,4,3,3,2,2,1]);
    expect(gravityFrames(0)*FRAME_MS).toBeCloseTo(798.68,1);expect(levelForLines(10)).toBe(1);expect(clearPoints(4,9)).toBe(12000);
  });
  it('clears lines, scores with the new level and preserves the entry delay',()=>{
    const g=new RealtimeGame(1,0);g.board[19].fill(1);for(let x=3;x<7;x++)g.board[19][x]=0;g.lines=9;
    Object.assign(g.active!,{piece:0,pose:{x:5,y:19,r:0},fall:47});g.tick();
    expect(g.lines).toBe(10);expect(g.score).toBe(80);expect(g.clearing).toEqual([19]);frames(g,g.delay);expect(g.board.every(r=>r.every(c=>c===0))).toBe(true);
  });
  it('has only legal clockwise route steps',()=>{
    const board=emptyBoard(),start={x:5,y:0,r:0};
    for(const o of landingsFor(board,2,start)){const path=route(board,2,start,o.pose)!;expect(o.path).toEqual(path);
      for(let i=1;i<path.length;i++){expect(fits(board,2,path[i])).toBe(true);if(path[i].r!==path[i-1].r)expect(path[i].r).toBe((path[i-1].r+1)%4);}}
  });
  it('tops out without a 100-piece cap and keeps seeded streams repeatable',()=>{
    const g=new RealtimeGame(1,0);g.count=100;g.spawn();expect(g.over).toBe(false);g.board[0].fill(1);g.spawn();expect(g.over).toBe(true);
    const a=pieceGenerator(42),b=pieceGenerator(42);expect(Array.from({length:100},a)).toEqual(Array.from({length:100},b));
  });
});
