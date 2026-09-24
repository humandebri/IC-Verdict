async page => {
  const errors=[],requests=[];
  const onError=e=>errors.push(e.message),onRequest=r=>{if(r.url().includes('/api/v'))requests.push({url:r.url(),method:r.method()});};
  page.on('pageerror',onError);page.on('request',onRequest);
  try {
    await page.setViewportSize({width:1200,height:1000});
    await page.goto('http://127.0.0.1:5174/?seed=42');
    await page.getByRole('button',{name:'Start',exact:true}).click();
    await page.waitForFunction(()=>Number(document.querySelector('[data-testid="pieces"]')?.textContent)>=2,null,{timeout:60000});
    const read=async selector=>JSON.parse(await page.locator(selector).first().textContent());
    const benchmark=await read('.benchmark .raw pre'),reply=await read('.decision .raw pre'),input=await read('.decision-input pre');
    const history=await page.locator('.decision ol').innerText();
    if(benchmark.queries<=benchmark.pieces||benchmark.executed<3)throw new Error('Repeated controls not demonstrated');
    if(benchmark.errors||errors.length)throw new Error(JSON.stringify({benchmark,errors}));
    if(reply.input_tokens>52||reply.actions.length!==reply.scores.length||!reply.actions.includes(reply.action))throw new Error('Invalid live reply');
    if(input.previous.length!==1||!reply.prompt.includes(' prev '))throw new Error('Missing previous input/action context');
    if(input.heights.length!==10)throw new Error('Missing terrain');
    if(requests.some(r=>/\/call$/.test(r.url)))throw new Error('Unexpected update request');
    await page.screenshot({path:'/tmp/openjev-query-v3-history-live-desktop.png',fullPage:true});
    await page.setViewportSize({width:390,height:844});
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
    if(overflow)throw new Error('Mobile overflow');
    await page.screenshot({path:'/tmp/openjev-query-v3-history-live-mobile.png',fullPage:true});
    return {benchmark,reply,input,history,errors,query_requests:requests.filter(r=>/\/query$/.test(r.url)).length,overflow};
  } finally {await page.goto('about:blank');page.off('pageerror',onError);page.off('request',onRequest);}
}
