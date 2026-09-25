import {describe,it,expect} from 'vitest';
import {fixtures,columnHoles,rowMasks,prompt,landingValue} from '../scripts/input-study-core';
import {emptyBoard} from './game';
import {actionsFor,fits,decisionInput} from './realtime';
describe('offline input study',()=>{
 it('freezes 50 development and 50 held-out tactical states with disjoint boards',()=>{
  const fs=fixtures();expect(fs).toHaveLength(100);expect(fixtures()).toEqual(fs);
  const boardKey=(f:typeof fs[number])=>JSON.stringify(f.board);
  const dev=new Set(fs.filter(f=>f.split==='dev').map(boardKey));
  expect(fs.filter(f=>f.split==='test').every(f=>!dev.has(boardKey(f)))).toBe(true);
  for(const split of ['dev','test'])for(const category of ['rotate','left','right','aligned','collision'])expect(fs.filter(f=>f.split===split&&f.category===category)).toHaveLength(10);
  for(const split of ['dev','test'])for(const category of ['rotate','left','right','aligned','collision'])expect(new Set(fs.filter(f=>f.split===split&&f.category===category).map(f=>f.input.piece)).size).toBe(category==='rotate'?6:7);
  for(const f of fs){
   const i=f.input,p={x:i.x,y:i.y,r:i.rotation};expect(fits(f.board,i.piece,p)).toBe(true);
   expect(f.acceptable.every(a=>actionsFor(i.legal_mask).includes(a))).toBe(true);
   const before=JSON.stringify({board:f.board,p});
   const scores=actionsFor(i.legal_mask).map(a=>({a,v:landingValue(f.board,i.piece,p,a)}));
   const max=Math.max(...scores.map(s=>s.v?.score??-Infinity));
   expect(scores.filter(s=>s.v?.score===max).map(s=>s.a)).toEqual(f.acceptable);
   expect(JSON.stringify({board:f.board,p})).toBe(before);
   if(f.category==='rotate')expect(f.acceptable).toEqual([2]);
   if(f.category==='collision')expect(i.legal_mask).not.toBe(31);
  }
 });
 it('treats down against the floor as locking the current pose',()=>{
  const board=emptyBoard(),pose={x:5,y:18,r:0};
  expect(landingValue(board,1,pose,3)).toEqual(landingValue(board,1,pose,4));
  expect(pose.y).toBe(18);
 });
 it('distinguishes same heights/total holes at different columns and preserves row orientation',()=>{
  const a=emptyBoard(),b=emptyBoard();a[17][0]=a[17][1]=b[17][0]=b[17][1]=1;
  a[18][0]=a[19][0]=1;b[18][1]=b[19][1]=1;
  expect(decisionInput(a,2,0,{x:5,y:0,r:0}).heights).toEqual(decisionInput(b,2,0,{x:5,y:0,r:0}).heights);
  expect(columnHoles(a)).not.toEqual(columnHoles(b));expect(rowMasks(a)[17]).toBe(3);expect(rowMasks(a)[19]).toBe(1);expect(rowMasks(b)[19]).toBe(2);
 });
 it('never leaks oracle labels or future outcomes and keeps stable action IDs',()=>{
  const f=fixtures()[0],short=prompt(f,'no-history','short');
  expect(short.text).not.toContain('prev');expect(short.text).not.toContain('history');
  expect(prompt(f,'history','short').text).toContain('prev');
  for(const format of ['history','no-history','columns','rows'] as const){
   const before=prompt(f,format,'explicit');
   const changed={...f,acceptable:[4 as const],rationale:'SECRET_ORACLE'};
   expect(prompt(changed,format,'explicit')).toEqual(before);
   expect(before.actions).toEqual(actionsFor(f.input.legal_mask));
  }
 });
});
