export const WIDTH = 10, HEIGHT = 20;
export type Board = number[][];
export type Cell = [number, number];
export type Features = { holes: number; height: number; lines: number };
export type Placement = { board: Board; cells: Cell[]; rotation: number; x: number; features: Features; score: number; path: Cell[][] };
export type Game = { board: Board; lines: number; count: number; ended: boolean };
export const emptyBoard = (): Board => Array.from({ length: HEIGHT }, () => Array(WIDTH).fill(0));
export const newGame = (): Game => ({ board: emptyBoard(), lines: 0, count: 0, ended: false });
const SHAPES: Cell[][] = [
  [[0,0],[1,0],[2,0],[3,0]], [[0,0],[1,0],[0,1],[1,1]],
  [[0,0],[1,0],[2,0],[1,1]], [[1,0],[2,0],[0,1],[1,1]],
  [[0,0],[1,0],[1,1],[2,1]], [[0,0],[0,1],[1,1],[2,1]], [[2,0],[0,1],[1,1],[2,1]],
];
const normalize = (s: Cell[]): Cell[] => {
  const minX = Math.min(...s.map(c => c[0])), minY = Math.min(...s.map(c => c[1]));
  return s.map(([x,y]): Cell => [x-minX,y-minY]).sort((a,b) => a[1]-b[1] || a[0]-b[0]);
};
export function rotations(piece: number): Cell[][] {
  let shape = normalize(SHAPES[piece]); const out: Cell[][] = [];
  for (let i=0;i<4;i++) {
    if (!out.some(s => JSON.stringify(s)===JSON.stringify(shape))) out.push(shape);
    shape = normalize(shape.map(([x,y]) => [-y,x]));
  }
  return out;
}
export function clearLines(board: Board): { board: Board; lines: number } {
  const remaining = board.filter(row => row.some(c => c===0)).map(row => [...row]);
  const lines = HEIGHT - remaining.length;
  return { board: [...Array.from({length:lines},()=>Array(WIDTH).fill(0)), ...remaining], lines };
}
export function features(board: Board, lines: number): Features {
  let height=0, holes=0;
  for(let x=0;x<WIDTH;x++) { let covered=false;
    for(let y=0;y<HEIGHT;y++) {
      if(board[y][x]) { covered=true; height=Math.max(height,HEIGHT-y); }
      else if(covered) holes++;
    }
  }
  return { holes,height,lines };
}
export function placements(board: Board, piece: number): Placement[] {
  const out: Placement[]=[];
  const shapes=rotations(piece);
  // Reachable moves from a centered spawn; no kicks or upward moves.
  type Pose={x:number;y:number;r:number;parent:number};
  const spawn={x:Math.floor((WIDTH-Math.max(...shapes[0].map(c=>c[0]))-1)/2),y:0,r:0,parent:-1};
  const cellsOf=(p:Pose)=>shapes[p.r].map(([x,y]):Cell=>[x+p.x,y+p.y]);
  const valid=(p:Pose)=>cellsOf(p).every(([x,y])=>x>=0&&x<WIDTH&&y>=0&&y<HEIGHT&&board[y][x]===0);
  if(!valid(spawn))return [];
  const poses:Pose[]=[spawn],key=(p:Pose)=>`${p.x},${p.y},${p.r}`;
  const seen=new Map([[key(spawn),0]]);
  for(let i=0;i<poses.length;i++){
    const p=poses[i];
    for(const n of [{...p,r:(p.r+1)%shapes.length},{...p,x:p.x-1},{...p,x:p.x+1},{...p,y:p.y+1}]){
      if(!seen.has(key(n))&&valid(n)){n.parent=i;seen.set(key(n),poses.length);poses.push(n);}
    }
  }
  shapes.forEach((shape,rotation) => {
    const width=Math.max(...shape.map(c=>c[0]))+1;
    for(let x=0;x<=WIDTH-width;x++) {
      const fits=(y:number)=>shape.every(([dx,dy]) => y+dy<HEIGHT && board[y+dy][x+dx]===0);
      if(!fits(0)) continue;
      let y=0; while(fits(y+1)) y++;
      const cells=shape.map(([dx,dy]):Cell=>[x+dx,y+dy]); const landed=board.map(row=>[...row]);
      let index=seen.get(key({x,y,r:rotation,parent:-1}));if(index===undefined)continue;
      const path:Cell[][]=[];
      while(index>=0){path.push(cellsOf(poses[index]));index=poses[index].parent;}
      path.reverse();
      cells.forEach(([cx,cy])=>landed[cy][cx]=piece+1);
      const cleared=clearLines(landed), f=features(cleared.board,cleared.lines);
      out.push({board:cleared.board,cells,rotation,x,features:f,score:10*f.lines-8*f.holes-f.height,path});
    }
  });
  return out.sort((a,b)=>b.score-a.score || a.rotation-b.rotation || a.x-b.x);
}
export function candidates(board: Board, piece: number): Placement[] {
  const all=placements(board,piece); if(!all.length) return [];
  const other=all.find(p=>JSON.stringify(p.features)!==JSON.stringify(all[0].features));
  return other ? [all[0],other] : [all[0]];
}
export const apply = (g: Game, p: Placement): Game => ({board:p.board,lines:g.lines+p.features.lines,count:g.count+1,ended:false});
export function sequence(seed: number, count=100): number[] {
  let state=(seed>>>0)||0x9e3779b9;
  const random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return (state>>>0)/4294967296;};
  const out:number[]=[];
  while(out.length<count) {const bag=[0,1,2,3,4,5,6];for(let i=6;i>0;i--){const j=Math.floor(random()*(i+1));[bag[i],bag[j]]=[bag[j],bag[i]];}out.push(...bag);}
  return out.slice(0,count);
}
