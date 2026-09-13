const $ = (id) => document.getElementById(id);
const fileInput=$('fileInput'), preview=$('preview'), emptyState=$('emptyState'), analyzeBtn=$('analyzeBtn'), resetBtn=$('resetBtn'), dropzone=$('dropzone'), fileMeta=$('fileMeta');
const placeholder=$('placeholder'), loading=$('loading'), result=$('result'), loadingText=$('loadingText');
let selectedFile=null, previewUrl=null;

function humanSize(bytes){ return bytes < 1024*1024 ? `${(bytes/1024).toFixed(0)} KB` : `${(bytes/1024/1024).toFixed(2)} MB`; }
function setFile(file){
  if(!file || !file.type.startsWith('image/')) return;
  if(file.size > 10*1024*1024){ alert('Ukuran file maksimal 10 MB.'); return; }
  selectedFile=file;
  if(previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file); preview.src=previewUrl; preview.hidden=false; emptyState.hidden=true;
  fileMeta.textContent=`${file.name} · ${humanSize(file.size)}`; analyzeBtn.disabled=false; placeholder.hidden=false; loading.hidden=true; result.hidden=true;
}
fileInput.addEventListener('change',e=>setFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.add('drag')}));
['dragleave','drop'].forEach(ev=>dropzone.addEventListener(ev,e=>{e.preventDefault();dropzone.classList.remove('drag')}));
dropzone.addEventListener('drop',e=>setFile(e.dataTransfer.files[0]));
resetBtn.addEventListener('click',()=>{
  selectedFile=null;fileInput.value='';if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;preview.src='';preview.hidden=true;emptyState.hidden=false;fileMeta.textContent='Belum ada file dipilih.';analyzeBtn.disabled=true;placeholder.hidden=false;loading.hidden=true;result.hidden=true;
});
const messages=['Memvalidasi file…','Resize citra ke 96 × 96…','Mengekstraksi fitur CNN…','Menghitung 7 sigmoid scores…','Menerapkan threshold per kelas…'];
analyzeBtn.addEventListener('click',async()=>{
  if(!selectedFile)return; placeholder.hidden=true;result.hidden=true;loading.hidden=false;analyzeBtn.disabled=true;
  let i=0;loadingText.textContent=messages[0];const timer=setInterval(()=>{i=(i+1)%messages.length;loadingText.textContent=messages[i]},550);
  const form=new FormData();form.append('image',selectedFile);
  try{
    const r=await fetch('/api/predict',{method:'POST',body:form});const data=await r.json();if(!r.ok||!data.ok)throw new Error(data.error||'Prediksi gagal');render(data);
  }catch(err){result.innerHTML=`<div class="error-box"><b>Analisis gagal</b><p>${escapeHtml(err.message)}</p></div>`;result.hidden=false;}
  finally{clearInterval(timer);loading.hidden=true;analyzeBtn.disabled=false;}
});
function escapeHtml(v){const d=document.createElement('div');d.textContent=v;return d.innerHTML;}
function render(data){
  const stateMap={"normal":["NORMAL","Tidak ada label penyakit melewati threshold"],"perlu-evaluasi":["PERLU EVALUASI","Model menemukan satu atau lebih pola penyakit"],"tidak-pasti":["TIDAK PASTI","Tidak ada label yang cukup kuat"]};
  const [badge,title]=stateMap[data.state]||['HASIL','Hasil model'];$('stateBadge').textContent=badge;$('stateBadge').className=`badge ${data.state}`;$('resultTitle').textContent=title;$('summary').textContent=data.summary;
  $('topProbability').textContent=`${data.top_probability.toFixed(2)}%`;$('topLabel').textContent=data.top_label;
  const qw=$('qualityWarnings');qw.innerHTML='';(data.quality.warnings||[]).forEach(w=>qw.insertAdjacentHTML('beforeend',`<div class="quality-warning">⚠ ${escapeHtml(w)}</div>`));
  const dl=$('detectedList');dl.innerHTML='';
  if(!data.detected.length){dl.innerHTML='<div class="no-detection">Tidak ada label yang melewati threshold model.</div>'}else{
    data.detected.forEach(x=>dl.insertAdjacentHTML('beforeend',`<article class="detected-item"><div><span>${escapeHtml(x.name)}</span><strong>${x.probability.toFixed(2)}%</strong></div><p>${escapeHtml(x.description)}</p><small><b>Tindak lanjut:</b> ${escapeHtml(x.next_step)}</small></article>`));
  }
  const probs=$('probabilities');probs.innerHTML='';
  data.probabilities.slice().sort((a,b)=>b.value-a.value).forEach(x=>{
    const width=Math.max(1,Math.min(100,x.value)); const threshold=Math.min(100,x.threshold);
    probs.insertAdjacentHTML('beforeend',`<div class="prob-row ${x.positive?'positive':''}"><div class="prob-name"><span>${escapeHtml(x.name)}</span><b>${x.value.toFixed(2)}%</b></div><div class="track"><div class="fill" style="width:${width}%"></div><i style="left:${threshold}%" title="Threshold ${x.threshold}%"></i></div><small>threshold ${x.threshold.toFixed(1)}%</small></div>`);
  });
  result.hidden=false;
}
