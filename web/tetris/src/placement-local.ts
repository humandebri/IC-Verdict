import type {Candidate,Game,Pose} from './placement-api';

export const RULES='placement-v1-normalized-cw-no-kicks-7bag-query-feature-cost-v1';
export const MAX_TURNS=128;
type Cell=[number,number];
const SHAPES:Cell[][]=[
 [[0,0],[1,0],[2,0],[3,0]],[[0,0],[1,0],[0,1],[1,1]],
 [[0,0],[1,0],[2,0],[1,1]],[[1,0],[2,0],[0,1],[1,1]],
 [[0,0],[1,0],[1,1],[2,1]],[[0,0],[0,1],[1,1],[2,1]],
 [[2,0],[0,1],[1,1],[2,1]],
];
const cellOrder=(a:Cell,b:Cell)=>a[0]-b[0]||a[1]-b[1];
const normalize=(s:Cell[]):Cell[]=>{
 const x=Math.min(...s.map(c=>c[0])),y=Math.min(...s.map(c=>c[1]));
 return s.map(([cx,cy]):Cell=>[cx-x,cy-y]).sort(cellOrder);
};
export function shapes(piece:number):Cell[][]{
 let current=SHAPES[piece];if(!current)throw new Error('Invalid piece');
 const result:Cell[][]=[];
 for(let i=0;i<4;i++){
  current=normalize(current);
  if(!result.some(v=>JSON.stringify(v)===JSON.stringify(current)))result.push(current);
  current=current.map(([x,y]):Cell=>[-y,x]);
 }
 return result;
}
const cells=(s:Cell[][],p:Pose):Cell[]=>s[p.r].map(([x,y]):Cell=>[x+p.x,y+p.y]);
const fits=(board:number[],s:Cell[][],p:Pose)=>cells(s,p).every(([x,y])=>x>=0&&x<10&&y>=0&&y<20&&board[y*10+x]===0);
export function landed(board:number[],piece:number,cs:Cell[]):{board:number[];lines:number}{
 const next=board.slice();for(const [x,y] of cs)next[y*10+x]=piece+1;
 const kept:number[]=[];
 for(let y=0;y<20;y++){const row=next.slice(y*10,y*10+10);if(row.includes(0))kept.push(...row);}
 const lines=(200-kept.length)/10;
 return {board:[...Array(200-kept.length).fill(0),...kept],lines};
}
export function features(board:number[]):{holes:number;height:number}{
 let holes=0,height=0;
 for(let x=0;x<10;x++){
  let covered=false;
  for(let y=0;y<20;y++){
   if(board[y*10+x]!==0){covered=true;height=Math.max(height,20-y);}
   else if(covered)holes++;
  }
 }
 return {holes,height};
}
export function placements(board:number[],piece:number):Candidate[]{
 if(board.length!==200)throw new Error('Invalid board');
 const s=shapes(piece),width=Math.max(...s[0].map(c=>c[0]))+1;
 const start:Pose={r:0,x:Math.floor((10-width)/2),y:0};
 if(!fits(board,s,start))return [];
 const nodes:{pose:Pose;parent:number}[]=[{pose:start,parent:-1}];
 const key=(p:Pose)=>`${p.r},${p.x},${p.y}`;
 const seen=new Set([key(start)]),result:Candidate[]=[];
 for(let i=0;i<nodes.length;i++){
  const p=nodes[i].pose,down={...p,y:p.y+1};
  if(!fits(board,s,down)){
   const cs=cells(s,p),after=landed(board,piece,cs),f=features(after.board),path:Pose[]=[];
   for(let at=i;at!==-1;at=nodes[at].parent)path.push(nodes[at].pose);
   path.reverse();
   result.push({pose:p,cells:cs,holes:f.holes,height:f.height,lines:after.lines,
    evaluation:10*after.lines-8*f.holes-f.height,path});
  }
  for(const n of [{...p,r:(p.r+1)%s.length},{...p,x:p.x-1},{...p,x:p.x+1},down]){
   const k=key(n);if(!seen.has(k)&&fits(board,s,n)){seen.add(k);nodes.push({pose:n,parent:i});}
  }
 }
 return result.sort((a,b)=>a.pose.r-b.pose.r||a.pose.x-b.pose.x||a.pose.y-b.pose.y);
}
export function pieceAt(seed:number,index:number):number{
 let state=(seed>>>0)||0x9e3779b9,bag=[0,1,2,3,4,5,6];
 for(let b=0;b<=Math.floor(index/7);b++){
  bag=[0,1,2,3,4,5,6];
  for(let i=6;i>0;i--){state^=state<<13;state^=state>>>17;state^=state<<5;
   const j=Math.floor((state>>>0)*(i+1)/4294967296);[bag[i],bag[j]]=[bag[j],bag[i]];}
 }
 return bag[index%7];
}
export function diverseCandidates(all:Candidate[]):Candidate[]{
 const seen=new Set<string>(),unique=all.filter(c=>{const key=`${c.lines},${c.holes},${c.height}`;
  if(seen.has(key))return false;seen.add(key);return true;});
 if(!unique.length)return [];
 const selected=[0];
 while(selected.length<Math.min(4,unique.length)){
  let best=-1,bestDistance=-1;
  for(let i=0;i<unique.length;i++){
   if(selected.includes(i))continue;
   const distance=Math.min(...selected.map(j=>Math.abs(unique[i].lines-unique[j].lines)
    +Math.abs(unique[i].holes-unique[j].holes)+Math.abs(unique[i].height-unique[j].height)));
   if(distance>=bestDistance){best=i;bestDistance=distance;}
  }
  selected.push(best);
 }
 return selected.map(i=>unique[i]);
}
export const recommend=(options:Candidate[])=>options.reduce((best,c,i)=>c.evaluation>options[best].evaluation?i:best,0);
const rank=(value:number,values:number[])=>{
 const min=Math.min(...values),max=Math.max(...values);
 return min===max?'equal':value===min?'min':value===max?'max':'mid';
};
export function decisionRequest(options:Candidate[]){
 const maxLines=Math.max(...options.map(c=>c.lines));
 return {question:'',state:'Min holes',abstention:false,temperature:1,
  options:options.map((c,i)=>({id:String(i),text:`${rank(c.holes,options.map(v=>v.holes))} holes,${rank(c.height,options.map(v=>v.height))} height,${rank(maxLines-c.lines,options.map(v=>maxLines-v.lines))} missed`}))};
}
export const renderPrompt=(request:ReturnType<typeof decisionRequest>)=>request.options.map(o=>`<<LABEL>>${o.text}`).join('')+`<<SEP>>${request.state}`;
export function prepareTurn(game:Game){
 if(game.over||game.turn>=MAX_TURNS)throw new Error('Run complete');
 const piece=pieceAt(game.seed,game.turn),next=pieceAt(game.seed,game.turn+1);
 const all=placements(game.board,piece),candidates=diverseCandidates(all);
 if(!candidates.length)throw new Error('No legal placements');
 return {piece,next,legal_count:all.length,candidates,recommended:recommend(candidates),request:decisionRequest(candidates)};
}
