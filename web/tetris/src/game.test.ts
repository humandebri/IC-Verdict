import {describe,it,expect} from 'vitest';
import {apply,candidates,clearLines,emptyBoard,features,newGame,placements,rotations,sequence} from './game';
describe('bounded deterministic game',()=>{
  it('repeats seed and uses seven-bags',()=>{
    const a=sequence(42,98);expect(a).toEqual(sequence(42,98));expect(a).not.toEqual(sequence(43,98));
    for(let i=0;i<98;i+=7)expect(new Set(a.slice(i,i+7)).size).toBe(7);
  });
  it('deduplicates normalized rotations',()=>{expect(rotations(0)).toHaveLength(2);expect(rotations(1)).toHaveLength(1);expect(rotations(2)).toHaveLength(4);});
  it('enumerates only legal hard drops without mutating input',()=>{
    const board=emptyBoard();board[19][4]=2;
    const before=JSON.stringify(board);
    for(const p of placements(board,0)) {
      expect(p.cells).toHaveLength(4);
      for(const [x,y] of p.cells){expect(x).toBeGreaterThanOrEqual(0);expect(x).toBeLessThan(10);expect(y).toBeLessThan(20);expect(board[y][x]).toBe(0);}
      expect(p.cells.some(([x,y])=>y===19||board[y+1][x]!==0)).toBe(true);
    }
    expect(JSON.stringify(board)).toBe(before);
  });
  it('clears multiple rows and measures holes after clearing',()=>{
    const b=emptyBoard();b[19].fill(1);b[18].fill(2);b[17][0]=1;
    const c=clearLines(b);expect(c.lines).toBe(2);expect(features(c.board,c.lines)).toEqual({holes:0,height:1,lines:2});
    c.board[17][1]=1;expect(features(c.board,0).holes).toBe(2);
  });
  it('ends on a full board',()=>{expect(candidates(Array.from({length:20},()=>Array(10).fill(1)),0)).toEqual([]);});
  it('never teleports through the stack during animation',()=>{
    const b=emptyBoard();for(let y=14;y<20;y++)b[y][4]=3;
    for(let piece=0;piece<7;piece++)for(const p of placements(b,piece)){
      expect(p.path.at(-1)).toEqual(p.cells);
      for(const frame of p.path){expect(frame).toHaveLength(4);for(const [x,y]of frame){expect(x>=0&&x<10&&y>=0&&y<20).toBe(true);expect(b[y][x]).toBe(0);}}
      for(let i=1;i<p.path.length;i++){
        const min=(f:number[][],axis:number)=>Math.min(...f.map(c=>c[axis]));
        const dx=min(p.path[i],0)-min(p.path[i-1],0),dy=min(p.path[i],1)-min(p.path[i-1],1);
        expect(Math.abs(dx)+Math.abs(dy)).toBeLessThanOrEqual(1);expect(dy).toBeGreaterThanOrEqual(0);
      }
    }
  });
  it('tops out if the center spawn is blocked even with open side columns',()=>{
    const b=emptyBoard();b[0][4]=1;expect(placements(b,0)).toEqual([]);
  });
  it('uses distinct feature candidates in a stable ranking',()=>{
    const a=candidates(emptyBoard(),0);expect(a).toHaveLength(2);expect(a[0].features).not.toEqual(a[1].features);
    expect(a[0].score).toBeGreaterThanOrEqual(a[1].score);expect(a).toEqual(candidates(emptyBoard(),0));
    expect(candidates(emptyBoard(),1)).toHaveLength(1);
  });
  it('plays bounded legal games for ten seeds',()=>{
    for(let seed=1;seed<=10;seed++){let g=newGame();for(const p of sequence(seed)){const c=candidates(g.board,p);if(!c.length)break;g=apply(g,c[0]);expect(g.board).toHaveLength(20);expect(g.board.every(r=>r.length===10)).toBe(true);expect(g.count).toBeLessThanOrEqual(100);}}
  });
});
