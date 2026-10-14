const $ = id => document.getElementById(id);
let grid=[], previous=[], generation=0, running=false, stepping=false, timer=null, revision=0, playToken=0;
let report=null, benchmarking=false;
async function api(path, body){
  const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||'Request failed');
  return data;
}
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').hidden=true,6500);}
function configuration(){return {size:Number($('size').value),pattern:$('pattern').value,seed:Number($('seed').value),density:Number($('density').value)/100,boundary:$('boundary').value};}
function setRunning(value){
  running=value;clearTimeout(timer);++playToken;
  $('play').textContent=value?'Ⅱ  Pause simulation':'▶  Start simulation';
  $('play-state').textContent=value?'RUNNING':'PAUSED';
  $('live-indicator').classList.toggle('running',value);
  $('step').disabled=value||benchmarking;
  if(value) tick(playToken);
}
function draw(){
  if(!grid.length)return;
  const canvas=$('grid'),ctx=canvas.getContext('2d'),n=grid.length;
  const cell=canvas.width/n;
  ctx.fillStyle='#0a1521';ctx.fillRect(0,0,canvas.width,canvas.height);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    ctx.fillStyle=grid[y][x]?(previous[y]?.[x]?'#83edc0':'#ff967f'):'#132231';
    const gap=n<=128?Math.max(1,cell*.18):0;
    ctx.fillRect(x*cell+gap/2,y*cell+gap/2,cell-gap,cell-gap);
  }
  const alive=grid.reduce((s,row)=>s+row.reduce((a,b)=>a+b,0),0);
  $('alive').textContent=alive.toLocaleString();$('generation').textContent=String(generation).padStart(4,'0');
  $('population').textContent=(100*alive/(n*n)).toFixed(1)+'% of the universe';$('grid-label').textContent=`${n} × ${n}`;
  $('boundary-label').textContent=$('boundary').value==='wrap'?'TOROIDAL':'DEAD OUTSIDE';
}
async function reset(empty=false){
  setRunning(false);const version=++revision;
  try{const config=configuration();if(empty)config.pattern='empty';
    const data=await api('/api/grid',config);if(version!==revision)return;
    grid=data.grid;previous=grid.map(row=>row.slice());generation=0;draw();
  }catch(error){toast(error.message);}
}
async function step(){
  if(stepping||!grid.length||benchmarking)return;
  stepping=true;const version=revision;const old=grid;
  try{const data=await api('/api/step',{grid,boundary:$('boundary').value});
    if(version===revision){previous=old;grid=data.grid;generation++;draw();}
  }catch(error){setRunning(false);toast(error.message);}finally{stepping=false;}
}
async function tick(token){if(!running||token!==playToken)return;const start=performance.now();await step();if(running&&token===playToken)timer=setTimeout(()=>tick(token),Math.max(0,1000/Number($('speed').value)-(performance.now()-start)));}
function changeView(name){if(name!=='simulation')setRunning(false);document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==name);document.querySelectorAll('.nav').forEach(b=>b.classList.toggle('active',b.dataset.view===name));}
document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>changeView(button.dataset.view)));
$('play').onclick=()=>setRunning(!running);$('step').onclick=step;$('reset').onclick=()=>reset();$('clear').onclick=()=>reset(true);
['size','pattern','seed','boundary'].forEach(id=>$(id).addEventListener('change',()=>reset()));
$('density').oninput=()=>{$('density-value').textContent=$('density').value+'%';};$('density').onchange=()=>reset();
$('speed').oninput=()=>{$('speed-value').textContent=$('speed').value+' gen/s';};
$('grid').addEventListener('pointerdown',event=>{
  if(running||stepping||benchmarking||!grid.length)return;
  const rect=$('grid').getBoundingClientRect(),side=Math.min(rect.width,rect.height);
  const left=rect.left+(rect.width-side)/2,top=rect.top+(rect.height-side)/2;
  const x=Math.floor((event.clientX-left)/side*grid.length),y=Math.floor((event.clientY-top)/side*grid.length);
  if(x<0||y<0||x>=grid.length||y>=grid.length)return;
  ++revision;previous=grid.map(row=>row.slice());grid[y][x]=1-grid[y][x];draw();
});
function lockBenchmark(value){benchmarking=value;$('run-benchmark').disabled=value;$('play').disabled=value;$('step').disabled=value;$('bench-progress').hidden=!value;}
const fmt=value=>value===null||value===undefined?'—':value.toFixed(3);
function addCell(row,text){const td=document.createElement('td');td.textContent=text;row.append(td);}
function renderResults(data){
  report=data;$('no-results').hidden=true;$('results-content').hidden=false;$('bars').replaceChildren();$('results-body').replaceChildren();
  const max=Math.max(...data.results.flatMap(r=>[r.cpu?.compute_ms||0,r.cuda?.compute_ms||0]),.001);
  data.results.forEach(result=>{
    const row=document.createElement('div');row.className='chart-row';const label=document.createElement('span');label.textContent=result.config.size+'²';row.append(label);
    const tracks=document.createElement('div');
    for(const backend of ['cpu','cuda']){if(!result[backend])continue;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.className='bar'+(backend==='cuda'?' gpu':'');bar.style.width=(result[backend].compute_ms/max*100)+'%';const text=document.createElement('span');text.textContent=fmt(result[backend].compute_ms)+' ms';track.append(bar,text);tracks.append(track);}
    row.append(tracks);$('bars').append(row);
    const tr=document.createElement('tr');[result.config.size+'²',fmt(result.cpu?.compute_ms),fmt(result.cuda?.compute_ms),result.speedup_total===null?'—':result.speedup_total.toFixed(2)+'×',result.verified===null?'Not tested':result.verified?'Match ✓':'MISMATCH'].forEach(t=>addCell(tr,t));$('results-body').append(tr);
  });
  const config=data.results[0].config;
  $('results-note').textContent=`${config.iterations} generations · ${config.repeats} repeats · seed ${config.seed} · ${config.boundary} boundary. `+ [...new Set(data.results.map(r=>r.note).filter(Boolean))].join(' ');
  $('export-csv').disabled=false;$('export-json').disabled=false;
}
async function pollJob(){
  try{const job=await api('/api/job');$('bench-state').textContent=job.message||'Ready';$('bench-progress').value=job.progress||0;
    if(job.state==='running'){setTimeout(pollJob,600);return;}
    lockBenchmark(false);
    if(job.state==='done')renderResults(job.report);
    else if(job.state==='error')toast(job.message);
  }catch(error){lockBenchmark(false);toast(error.message);}
}
$('run-benchmark').onclick=async()=>{
  setRunning(false);++revision;
  // Wait for an already requested preview step before starting timed work.
  while(stepping)await new Promise(resolve=>setTimeout(resolve,20));
  lockBenchmark(true);$('bench-state').textContent='Preparing experiment…';
  try{const [block_x,block_y]=$('block').value.split('x').map(Number);
    const config={...configuration(),pattern:'random',iterations:Number($('iterations').value),repeats:Number($('repeats').value),block_x,block_y};
    const sizes=$('bench-size').value==='suite'?[128,256,512,1024]:[Number($('bench-size').value)];
    await api('/api/benchmark',{config,sizes,mode:$('mode').value});pollJob();
  }catch(error){lockBenchmark(false);$('bench-state').textContent='Experiment could not start';toast(error.message);}
};
async function download(ext){try{const response=await fetch('/api/export.'+ext);if(!response.ok)throw new Error('Export unavailable');const url=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=url;a.download='life-lab-benchmark.'+ext;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){toast(error.message);}}
$('export-csv').onclick=()=>download('csv');$('export-json').onclick=()=>download('json');
async function init(){
  await reset();
  try{const {cuda}=await api('/api/status');$('cuda-label').textContent=cuda.available?'GPU ready':cuda.simulator?'Simulator only':'CPU mode';$('cuda-detail').textContent=cuda.available?cuda.name:'No CUDA device available';$('cuda-detail').title=cuda.reason;
    if(cuda.available)$('cuda-label').style.color='var(--mint)';
    const job=await api('/api/job');if(job.state==='done')renderResults(job.report);else if(job.state==='running'){lockBenchmark(true);pollJob();}
  }catch(error){$('cuda-label').textContent='Disconnected';toast(error.message);}
}
init();
