import {emptyBoard,clearLines,features,type Board,type Cell,type Features} from './game';

// NES NTSC speed/score rules; see docs/TETRIS_REALTIME.md for sources and deviations.
export const FPS=60.0988, FRAME_MS=1000/FPS;
export const gravityFrames=(level:number)=>[48,43,38,33,28,23,18,13,8,6,5,5,5,4,4,4,3,3,3,2,2,2,2,2,2,2,2,2,2][level]??1;
export const levelForLines=(lines:number)=>Math.floor(lines/10);
export const clearPoints=(lines:number,level:number)=>[0,40,100,300,1200][lines]*(level+1);
export const PIECE_NAMES=['I','O','T','S','Z','J','L'];
export const RULES_VERSION='nes-ntsc-query-v3-controls-history';
// Cell offsets use fixed NES pivots, not normalized rotations. Piece IDs match UI colors.
export const SHAPES:Cell[][][]=[
  [[[-2,0],[-1,0],[0,0],[1,0]],[[0,-2],[0,-1],[0,0],[0,1]]],
  [[[-1,0],[0,0],[-1,1],[0,1]]],
  [[[-1,0],[0,0],[1,0],[0,1]],[[-1,0],[0,-1],[0,0],[0,1]],[[-1,0],[0,0],[1,0],[0,-1]],[[0,-1],[0,0],[1,0],[0,1]]],
  [[[0,0],[1,0],[-1,1],[0,1]],[[0,-1],[0,0],[1,0],[1,1]]],
  [[[-1,0],[0,0],[0,1],[1,1]],[[1,-1],[0,0],[1,0],[0,1]]],
  [[[-1,0],[0,0],[1,0],[1,1]],[[0,-1],[0,0],[-1,1],[0,1]],[[-1,-1],[-1,0],[0,0],[1,0]],[[0,-1],[1,-1],[0,0],[0,1]]],
  [[[-1,0],[0,0],[1,0],[-1,1]],[[-1,-1],[0,-1],[0,0],[0,1]],[[1,-1],[-1,0],[0,0],[1,0]],[[0,-1],[0,0],[0,1],[1,1]]],
];
export type Pose={x:number;y:number;r:number};
export type Option={pose:Pose;cells:Cell[];features:Features;score:number;path:Pose[]};
export type Active={id:number;piece:number;pose:Pose;fall:number;input:number;dropInput:number;softPoints:number};
export type Metrics={queries:number;late:number;blocked:number;stale:number;executed:number;errors:number;responses:number;latencyTotal:number;latencyMax:number;valid:boolean};
export const cellsFor=(piece:number,pose:Pose):Cell[]=>SHAPES[piece][pose.r].map(([x,y])=>[x+pose.x,y+pose.y]);
const key=(p:Pose)=>`${p.x},${p.y},${p.r}`;
export const fits=(board:Board,piece:number,p:Pose)=>cellsFor(piece,p).every(([x,y])=>x>=0&&x<10&&y>=-2&&y<20&&(y<0||board[y][x]===0));
const neighbors=(p:Pose,piece:number):Pose[]=>[
  {...p,r:(p.r+1)%SHAPES[piece].length},
  {...p,x:p.x-1},{...p,x:p.x+1},{...p,y:p.y+1},
];
function reachable(board:Board,piece:number,start:Pose){
  const poses=[start],parents=[-1],seen=new Map([[key(start),0]]);
  for(let i=0;i<poses.length;i++)for(const n of neighbors(poses[i],piece)){
    if(!seen.has(key(n))&&fits(board,piece,n)){seen.set(key(n),poses.length);poses.push(n);parents.push(i);}
  }
  return {poses,parents,seen};
}
export function route(board:Board,piece:number,start:Pose,target:Pose):Pose[]|undefined{
  if(!fits(board,piece,start))return undefined;
  const {poses,parents,seen}=reachable(board,piece,start);let i=seen.get(key(target));
  if(i===undefined)return undefined;
  const path:Pose[]=[];while(i>=0){path.push(poses[i]);i=parents[i];}return path.reverse();
}
// Enumerate legal destinations without a quality ranking. The score is used only by the benchmark baseline.
export function landingsFor(board:Board,piece:number,start:Pose):Option[]{
  const all:Option[]=[];
  const {poses,parents}=reachable(board,piece,start);
  for(let i=0;i<poses.length;i++){
    const pose=poses[i];
    const cells=cellsFor(piece,pose);
    if(cells.some(c=>c[1]<0)||fits(board,piece,{...pose,y:pose.y+1}))continue;
    const landed=board.map(r=>[...r]);cells.forEach(([x,y])=>landed[y][x]=piece+1);
    const cleared=clearLines(landed),f=features(cleared.board,cleared.lines);
    const path:Pose[]=[];for(let j=i;j>=0;j=parents[j])path.push(poses[j]);path.reverse();
    all.push({pose,cells,features:f,score:10*f.lines-8*f.holes-f.height,path});
  }
  return all.sort((a,b)=>a.pose.r-b.pose.r||a.pose.x-b.pose.x||a.pose.y-b.pose.y);
}
export type Action=0|1|2|3|4;
export const ACTION_SYMBOLS=['←','→','↑','↓','·'];
export const ACTION_NAMES=['Left','Right','Rotate','Down','Wait'];
export const CONTROL_TOKEN_LIMIT=52;
export type PreviousInput={x:number;y:number;rotation:number;action:Action;outcome:0|1|2};
export type DecisionInput={piece:number;next:number;x:number;y:number;rotation:number;heights:number[];holes:number;legal_mask:number;previous:[]|[PreviousInput];history:Action[]};
export const actionPose=(piece:number,p:Pose,action:Action):Pose=>action===0?{...p,x:p.x-1}:action===1?{...p,x:p.x+1}:action===2?{...p,r:(p.r+1)%SHAPES[piece].length}:action===3?{...p,y:p.y+1}:{...p};
export const actionsFor=(mask:number):Action[]=>([0,1,2,3,4] as Action[]).filter(a=>mask&(1<<a));
export function decisionInput(board:Board,piece:number,next:number,pose:Pose):DecisionInput {
  let legal_mask=24;
  for(const action of [0,1,2] as Action[])if((action!==2||SHAPES[piece].length>1)&&fits(board,piece,actionPose(piece,pose,action)))legal_mask|=1<<action;
  const heights=Array.from({length:10},(_,x)=>{const y=board.findIndex(row=>row[x]!==0);return y<0?0:20-y;});
  return {piece,next,x:pose.x,y:pose.y,rotation:pose.r,heights,holes:features(board,0).holes,legal_mask,previous:[],history:[]};
}
export type Outcome='pending'|'queued'|'executed'|'blocked'|'late'|'stale'|'error';
export type ControlTrace={request:number;piece:number;epoch:number;input:DecisionInput;action?:Action;outcome:Outcome;frame:number};
// Seeded one-reroll randomizer: repeated pieces/droughts are possible (not seven-bag).
// Deterministic benchmark stream, not a frame-clock-exact emulation of the NES RNG.
export function pieceGenerator(seed:number){
  let state=(seed>>>0)||1,previous=-1;
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return state>>>0;};
  return ()=>{let p=random()%8;if(p===7||p===previous)p=random()%7;previous=p;return p;};
}

export class RealtimeGame {
  board=emptyBoard();active?:Active;next:number;lines=0;score=0;count=0;over=false;
  clearing:number[]=[];delay=0;frame=0;revision=0;paused=false;
  metrics:Metrics={queries:0,late:0,blocked:0,stale:0,executed:0,errors:0,responses:0,latencyTotal:0,latencyMax:0,valid:true};
  history:ControlTrace[]=[];pending?:ControlTrace;queued?:ControlTrace;
  private pick:()=>number;private serial=0;private requestSerial=0;private epoch=0;private lastStart=-Infinity;
  private lastTime:number;private remainder=0;
  constructor(seed:number,now:number){this.pick=pieceGenerator(seed);this.next=this.pick();this.lastTime=now;this.spawn();}
  get level(){return levelForLines(this.lines);}
  spawn(){
    const piece=this.next;this.next=this.pick();const pose={x:5,y:0,r:0};
    if(!fits(this.board,piece,pose)){this.over=true;this.active=undefined;this.revision++;return;}
    this.active={id:++this.serial,piece,pose,fall:0,input:0,dropInput:0,softPoints:0};this.revision++;
  }
  pause(hidden:boolean,now:number){
    if(hidden!==this.paused){
      this.epoch++;this.metrics.valid=false;this.paused=hidden;this.lastTime=now;this.remainder=0;
      if(this.queued)this.finish(this.queued,'stale');this.queued=undefined;this.revision++;
    }
  }
  advance(now:number){
    const elapsed=Math.max(0,now-this.lastTime);this.lastTime=now;
    if(this.paused||this.over)return;
    if(elapsed>1000)this.metrics.valid=false;
    this.remainder+=elapsed;const frames=Math.floor(this.remainder/FRAME_MS);this.remainder-=frames*FRAME_MS;
    for(let i=0;i<frames&&!this.over;i++)this.tick();
  }
  beginQuery(now:number):ControlTrace|undefined {
    const a=this.active;
    if(!a||this.over||this.paused||this.pending||this.queued||now-this.lastStart<1000)return;
    const input=decisionInput(this.board,a.piece,this.next,a.pose);
    const prior=this.history.filter(t=>t.piece===a.id&&t.action!==undefined);
    const previous=prior[prior.length-1];
    if(previous){
      input.previous=[{x:previous.input.x,y:previous.input.y,rotation:previous.input.rotation,action:previous.action!,
        outcome:previous.outcome==='executed'?0:previous.outcome==='blocked'?1:2}];
      input.history=prior.slice(0,-1).filter(t=>t.outcome==='executed').slice(-2).map(t=>t.action!);
    }
    const trace:ControlTrace={request:++this.requestSerial,piece:a.id,epoch:this.epoch,
      input,outcome:'pending',frame:this.frame};
    this.pending=trace;this.lastStart=now;this.metrics.queries++;
    this.history.push(trace);if(this.history.length>100)this.history.shift();this.revision++;return trace;
  }
  response(request:number,action:Action,latency:number){
    const t=this.pending;if(!t||t.request!==request)return;
    this.pending=undefined;t.action=action;
    this.metrics.responses++;this.metrics.latencyTotal+=latency;this.metrics.latencyMax=Math.max(this.metrics.latencyMax,latency);
    if(t.outcome==='late'||!this.active||this.active.id!==t.piece||this.over){if(t.outcome!=='late')this.finish(t,'late');}
    else if(t.epoch!==this.epoch||this.paused)this.finish(t,'stale');
    else if(!actionsFor(t.input.legal_mask).includes(action))this.finish(t,'blocked');
    else {t.outcome='queued';this.queued=t;}
    this.revision++;
  }
  fail(request:number){
    const t=this.pending;if(!t||t.request!==request)return;
    this.pending=undefined;this.metrics.errors++;if(t.outcome!=='late')t.outcome='error';this.revision++;
  }
  private finish(t:ControlTrace,outcome:Outcome){
    t.outcome=outcome;
    if(outcome==='executed')this.metrics.executed++;
    if(outcome==='blocked')this.metrics.blocked++;
    if(outcome==='late')this.metrics.late++;
    if(outcome==='stale')this.metrics.stale++;
    this.revision++;
  }
  tick(){
    if(this.over||this.paused)return;
    this.frame++;
    if(this.delay>0){if(--this.delay===0){this.board=clearLines(this.board).board;this.clearing=[];this.spawn();}return;}
    const a=this.active;if(!a)return;
    if(a.input>0)a.input--;if(a.dropInput>0)a.dropInput--;
    const t=this.queued;let dropped=false;
    if(t){
      const action=t.action!;
      if(t.piece!==a.id||t.epoch!==this.epoch){this.finish(t,'stale');this.queued=undefined;}
      else if(action===4||(action===3?a.dropInput===0:a.input===0)){
        this.queued=undefined;
        const pose=actionPose(a.piece,a.pose,action);
        if(action===4)this.finish(t,'executed');
        else if(action===3){
          this.finish(t,'executed');a.fall=0;a.dropInput=2;dropped=true;
          if(fits(this.board,a.piece,pose)){a.pose=pose;a.softPoints++;}else {this.lock();return;}
        }else if((action!==2||SHAPES[a.piece].length>1)&&fits(this.board,a.piece,pose)){
          a.pose=pose;a.input=6;this.finish(t,'executed');
        }else this.finish(t,'blocked');
      }
    }
    if(!dropped&&++a.fall>=gravityFrames(this.level)){
      a.fall=0;const below={...a.pose,y:a.pose.y+1};
      if(fits(this.board,a.piece,below)){a.pose=below;this.revision++;}else this.lock();
    }
  }
  private lock(){
    const a=this.active!,cells=cellsFor(a.piece,a.pose);
    if(this.pending?.piece===a.id&&this.pending.outcome==='pending')this.finish(this.pending,'late');
    if(this.queued?.piece===a.id){this.finish(this.queued,'late');this.queued=undefined;}
    if(cells.some(c=>c[1]<0)){this.over=true;this.active=undefined;this.revision++;return;}
    cells.forEach(([x,y])=>this.board[y][x]=a.piece+1);
    this.clearing=this.board.flatMap((row,y)=>row.every(Boolean)?[y]:[]);
    this.lines+=this.clearing.length;
    this.score+=clearPoints(this.clearing.length,this.level)+a.softPoints;this.count++;
    const bottom=Math.max(...cells.map(c=>c[1]));
    this.delay=10+2*Math.min(4,Math.floor((19-bottom+2)/4))+(this.clearing.length?17+(4-this.frame%4)%4:0);
    this.active=undefined;this.revision++;
  }
}
