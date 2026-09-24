// Offline study only. No imports into the public application.
import {SHAPES,PIECE_NAMES,actionsFor,actionPose,cellsFor,fits,decisionInput,type Action,type Pose,type DecisionInput} from '../src/realtime';
import {emptyBoard,clearLines,features,type Board} from '../src/game';
export const FORMATS=['history','no-history','columns','rows'] as const;
export type Format=typeof FORMATS[number];
export const LABELS={short:['left','right','rotate','down','wait'],explicit:['move left','move right','rotate clockwise','move down','wait']} as const;
export type Labels=keyof typeof LABELS;
export type Situation={board:Board;input:DecisionInput};
export function columnHoles(board:Board){return Array.from({length:10},(_,x)=>{let covered=false,n=0;for(const row of board){if(row[x])covered=true;else if(covered)n++;}return n;});}
export function rowMasks(board:Board){return board.map(row=>row.reduce((mask,cell,x)=>mask|(cell?1<<x:0),0));}
export function prompt(s:Situation,format:Format,labelStyle:Labels){
  const i=s.input,actions=actionsFor(i.legal_mask);
  let text=`Clear lines. ${format==='history'||format==='no-history'?'Tetris ':''}${PIECE_NAMES[i.piece]} x${i.x} y${i.y} r${i.rotation} next ${PIECE_NAMES[i.next]}`;
  if(format==='rows')text+=` rows ${rowMasks(s.board).join(' ')}`;
  else {
    text+=` heights ${i.heights.join(' ')}`;
    text+=format==='columns'?` holes ${columnHoles(s.board).join(' ')}`:` holes${i.holes}`;
  }
  if(format==='history'&&i.previous.length){const p=i.previous[0];text+=` prev x${p.x} y${p.y} r${p.rotation} ${LABELS.short[p.action]} ${['executed','blocked','discarded'][p.outcome]}`;if(i.history.length)text+=` history ${i.history.map(a=>LABELS.short[a]).join(' ')}`;}
  return {text,labels:actions.map(a=>LABELS[labelStyle][a]),actions};
}
// Test-only oracle: one legal control, then straight gravity to lock; no oracle data enters prompts.
export function landingValue(board:Board,piece:number,pose:Pose,action:Action){
  let p={...actionPose(piece,pose,action)};
  if(!fits(board,piece,p)){if(action===3&&fits(board,piece,pose))p={...pose};else return undefined;}
  while(fits(board,piece,{...p,y:p.y+1}))p.y++;
  const cells=cellsFor(piece,p);if(cells.some(([,y])=>y<0))return undefined;
  const next=board.map(row=>[...row]);for(const [x,y] of cells)next[y][x]=piece+1;
  const cleared=clearLines(next),f=features(cleared.board,cleared.lines);
  return {lines:f.lines,holes:f.holes,height:f.height,score:100000*f.lines-100*f.holes-f.height};
}
export type Fixture=Situation&{id:string;split:'dev'|'test';category:string;acceptable:Action[];rationale:string};
export function fixtures():Fixture[]{
  const groups=new Map<string,Fixture[]>(['rotate','left','right','aligned','collision'].map(k=>[k,[]]));
  const seen=new Set<string>();
  for(let piece=0;piece<7;piece++)for(let rotation=0;rotation<SHAPES[piece].length;rotation++)for(let x=1;x<10;x++)for(let rise=0;rise<3;rise++){
    const target={x,y:19-Math.max(...SHAPES[piece][rotation].map(c=>c[1]))-rise,r:rotation};
    const cells=cellsFor(piece,target);if(cells.some(([cx,y])=>cx<0||cx>=10||y<0||y>=20))continue;
    const top=Math.min(...cells.map(c=>c[1])),board=emptyBoard();
    // Solid support below the cavity, with a different gap to keep each support row uncleared.
    for(let y=top;y<20;y++)board[y].fill(1);
    for(let y=20-rise;y<20;y++)board[y][(x+5)%10]=0;
    for(const [cx,y] of cells)board[y][cx]=0;
    for(let r=0;r<SHAPES[piece].length;r++)for(const dx of [-1,0,1])for(let lift=0;lift<5;lift++){
      const pose={x:x+dx,y:target.y-lift,r};if(pose.x<0||pose.x>9||!fits(board,piece,pose))continue;
      const input=decisionInput(board,piece,(piece+3)%7,pose),actions=actionsFor(input.legal_mask);
      const values=actions.map(a=>({a,v:landingValue(board,piece,pose,a)})).filter(e=>e.v!==undefined);
      const best=Math.max(...values.map(e=>e.v!.score)),acceptable=values.filter(e=>e.v!.score===best).map(e=>e.a);
      if(!values.some(e=>e.v!.lines>0)||values.every(e=>e.v!.score===best))continue;
      let category=acceptable.length===1&&acceptable[0]===2?'rotate':acceptable.length===1&&acceptable[0]===0?'left':acceptable.length===1&&acceptable[0]===1?'right':acceptable.includes(3)&&acceptable.includes(4)&&!acceptable.some(a=>a<3)?'aligned':'';
      if(!category||(category==='aligned'&&!fits(board,piece,actionPose(piece,pose,3))))continue;
      const split=x<=4?'dev':'test';
      const categories=[category];
      if(input.legal_mask!==31&&category!=='rotate')categories.push('collision');
      const key=JSON.stringify({board,piece,pose});if(seen.has(key))continue;seen.add(key);
      // A valid prior wait one row above this state, shared across all variants.
      if(fits(board,piece,{...pose,y:pose.y-1})&&pose.y>0){input.previous=[{x:pose.x,y:pose.y-1,rotation:r,action:4,outcome:0}];input.history=[4,4];}
      for(const cat of categories)groups.get(cat)!.push({id:'',split,category:cat,board:board.map(row=>[...row]),input,acceptable,rationale:'Maximize cleared lines, then minimize holes and height after one control followed by gravity. Ties accepted; oracle never sent.'});
    }
  }
  const out:Fixture[]=[],selected=new Set<string>(),boardSplit=new Map<string,string>();
  for(const [category,rows] of groups)for(const split of ['dev','test'] as const){
    const buckets=Array.from({length:7},(_,piece)=>rows.filter(f=>f.split===split&&f.input.piece===piece));
    let count=0;
    for(let round=0;count<10&&round<100;round++)for(let piece=0;piece<7&&count<10;piece++){
      const row=buckets[piece].find(f=>{const key=JSON.stringify({board:f.board,input:f.input});const assigned=boardSplit.get(JSON.stringify(f.board));return !selected.has(key)&&(!assigned||assigned===split);});
      if(!row)continue;
      selected.add(JSON.stringify({board:row.board,input:row.input}));boardSplit.set(JSON.stringify(row.board),split);
      out.push({...row,id:`${category}-${split}-${String(count++).padStart(2,'0')}`});
    }
    if(count!==10)throw new Error(`Insufficient ${category}/${split}: ${count}`);
  }
  return out;
}
