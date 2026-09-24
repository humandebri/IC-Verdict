async page => {
  const checks=[];const check=(ok,label)=>{if(!ok)throw new Error(label);checks.push(label);};
  const mock=`export const hex=b=>Array.from(b).map(x=>x.toString(16).padStart(2,'0')).join('');
    export async function connect(){return {status:async()=>({enabled:true,warmed:true,max_tokens:52,model:Array(32).fill(7)}),
    decide:async input=>{window.lastInput=input;(window.inputs??=[]).push(input);window.calls=(window.calls||0)+1;window.flight=(window.flight||0)+1;window.peak=Math.max(window.peak||0,window.flight);
      await new Promise(r=>{window.release=r;});window.flight--;
      if(window.fail)throw new Error('test network failure');
      const actions=[0,1,2,3,4].filter(a=>input.legal_mask&(1<<a));
      return {action:4,actions,scores:actions.map(a=>a===4?.6:.1),input_tokens:52,model:Array(32).fill(7),measured_instructions:2734000000n,prompt:'MOCK real-time UI check'};
    }}};`;
  await page.route('**/src/api.ts*',r=>r.fulfill({contentType:'application/javascript',body:mock}));
  await page.addInitScript(()=>{crypto.getRandomValues=a=>{a[0]=1;return a;};});
  try{
    await page.goto('http://127.0.0.1:5174');
    check(await page.locator('html').getAttribute('lang')==='en','English document language');
    check(!/[ぁ-んァ-ヶ一-龠]/.test(await page.locator('body').textContent()),'English UI copy including collapsed rules');
    check(await page.getByRole('button').count()===1,'one start button');
    await page.getByRole('button',{name:'Start',exact:true}).click();
    await page.waitForFunction(()=>window.calls===1,null,{timeout:10000});
    const minY=()=>Array.from(document.querySelectorAll('.active-piece .mino>g')).map(n=>Number(n.getAttribute('transform').split(',')[1].replace(')',''))).reduce((a,b)=>Math.min(a,b),Infinity);
    const y=await page.evaluate(minY);
    await page.waitForFunction(y=>Array.from(document.querySelectorAll('.active-piece .mino>g')).some(n=>Number(n.getAttribute('transform').split(',')[1].replace(')',''))>y+48),y,{timeout:5000});
    check(await page.evaluate(()=>window.flight)===1,'gravity moves while query is still unresolved');
    check(await page.locator('.choice').count()===0,'no invented query reply during wait');
    await page.evaluate(()=>window.release());
    await page.getByText('View full response',{exact:true}).click();
    const raw=JSON.parse(await page.locator('.decision .raw pre').first().innerText());
    check(raw.action===4&&raw.measured_instructions==='2734000000','real returned fields preserved');
    check(await page.locator('.scores>div').count()>=4,'legal control scores rendered');
    check((await page.locator('.choice').innerText()).includes('Wait'),'wait action displayed');
    await page.getByText('View decision input',{exact:true}).click();
    const sent=await page.evaluate(()=>window.inputs[0]);
    const shown=JSON.parse(await page.locator('.decision-input pre').first().innerText());
    check(JSON.stringify(sent)===JSON.stringify(shown),'displayed input matches actual request including current piece, NEXT and terrain');
    await page.waitForFunction(()=>window.calls>=2,null,{timeout:10000});
    check((await page.locator('.decision-heading').innerText()).includes('PIECE 01'),'same piece queried again');
    check((await page.getByTestId('action-history').innerText()).includes('·'),'operation history displayed');
    check(await page.evaluate(()=>window.lastInput.previous.length===1&&window.lastInput.previous[0].action===4&&window.lastInput.previous[0].outcome===0),'previous input/action/outcome reaches the next query');
    await page.waitForFunction(()=>Number(document.querySelector('[data-testid="late"]').textContent)>0,null,{timeout:23000});
    check(await page.evaluate(()=>window.peak)===1,'expired request never overlaps a new query');
    const before=Number(await page.getByTestId('pieces').innerText());
    check(before>0,'piece locks without a response');
    await page.evaluate(()=>window.release());
    await page.locator('.outcome').filter({hasText:'late'}).waitFor();
    check(true,'late result is shown but not applied');
    await page.waitForFunction(()=>window.flight===1,null,{timeout:10000});
    await page.evaluate(()=>{window.fail=true;window.release();});
    await page.locator('.fault').filter({hasText:'test network failure'}).waitFor();
    check(await page.getByRole('button').isDisabled(),'network error does not stop the game');
    await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});
    await page.locator('.benchmark').filter({hasText:'Non-comparable run'}).waitFor();
    check(true,'hidden tab is explicitly marked unsuitable for benchmark comparison');
    await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));});
    await page.setViewportSize({width:390,height:844});
    check(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'mobile no overflow');
    await page.screenshot({path:'/tmp/openjev-query-v3-history-mobile.png',fullPage:true});
    return checks;
  }finally{await page.goto('about:blank');await page.unroute('**/src/api.ts*');}
}
