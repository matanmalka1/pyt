// ─── Navigation ───────────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
    item.classList.add('active');
    const page = document.getElementById('page-' + item.dataset.page);
    page.classList.remove('hidden');
    page.classList.add('active');
    if (item.dataset.page === 'results') refreshResults();
    if (item.dataset.page === 'history') renderHistory();
  });
});

// ─── Training ─────────────────────────────────────────────────────────────
let totalEpochs = 5;
let currentBackbone = 'resnet18';
let modelRuns = JSON.parse(sessionStorage.getItem('modelRuns') || '[]');

async function startTraining() {
  const btn = document.getElementById('btn-train');
  totalEpochs     = parseInt(document.getElementById('epochs').value);
  currentBackbone = document.getElementById('backbone').value;

  const fd = new FormData();
  fd.append('epochs',   totalEpochs);
  fd.append('batch',    document.getElementById('batch').value);
  fd.append('lr',       document.getElementById('lr').value);
  fd.append('backbone', currentBackbone);
  fd.append('workers',  4);

  const res = await fetch('/train/start', { method: 'POST', body: fd });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }

  btn.disabled = true;
  document.getElementById('train-spinner').classList.remove('hidden');
  document.getElementById('progress-card').style.display = '';
  document.getElementById('log-box').innerHTML = '';
  document.getElementById('model-status').textContent = 'מאמן...';

  listenToStream();
}

function listenToStream() {
  const es = new EventSource('/train/stream');
  let lastValAcc = 0, lastTestAcc = 0;

  es.onmessage = e => {
    const msg = e.data;
    const box = document.getElementById('log-box');

    if (msg.startsWith('EPOCH:')) {
      const [cur, tot] = msg.replace('EPOCH:','').split('/');
      const pct = (parseInt(cur) / parseInt(tot)) * 100;
      document.getElementById('bar-epoch').style.width = pct + '%';
      document.getElementById('val-epoch').textContent = cur + ' / ' + tot;
      addLog(`── Epoch ${cur}/${tot}`, 'epoch');

    } else if (msg.startsWith('RESULT:')) {
      const [ep, trL, trA, vlL, vlA, sec, best] = msg.replace('RESULT:','').split('|');
      document.getElementById('bar-tacc').style.width = (parseFloat(trA)*100) + '%';
      document.getElementById('bar-vacc').style.width = (parseFloat(vlA)*100) + '%';
      document.getElementById('val-tacc').textContent = (parseFloat(trA)*100).toFixed(1) + '%';
      document.getElementById('val-vacc').textContent = (parseFloat(vlA)*100).toFixed(1) + '%';
      lastValAcc = parseFloat(vlA);
      const star = best === '1' ? ' ★' : '';
      addLog(`  train acc=${(parseFloat(trA)*100).toFixed(1)}%  val acc=${(parseFloat(vlA)*100).toFixed(1)}%  ${sec}s${star}`, 'result');
      const sideEl = document.getElementById('best-acc-side');
      const cur = parseFloat(sideEl.textContent) || 0;
      if (parseFloat(vlA)*100 > cur) sideEl.textContent = (parseFloat(vlA)*100).toFixed(2) + '%';

    } else if (msg.startsWith('TEST:')) {
      const [tl, ta] = msg.replace('TEST:','').split('|');
      lastTestAcc = parseFloat(ta);
      addLog(`  Test accuracy: ${(parseFloat(ta)*100).toFixed(2)}%`, 'result');
      document.getElementById('s-test-acc').textContent = (parseFloat(ta)*100).toFixed(1) + '%';

    } else if (msg === 'DONE') {
      addLog('✓ אימון הושלם בהצלחה', 'done');
      es.close();
      saveModelRun(currentBackbone, lastValAcc, lastTestAcc, totalEpochs);
      onTrainingDone();

    } else if (msg.startsWith('ERROR:')) {
      addLog(msg.replace('ERROR:',''), 'err');
      es.close();
      onTrainingDone();
    } else {
      addLog(msg, '');
    }
  };
  es.onerror = () => es.close();
}

function onTrainingDone() {
  document.getElementById('btn-train').disabled = false;
  document.getElementById('train-spinner').classList.add('hidden');
  document.getElementById('model-status').textContent = 'מאומן';
  fetchStatus();
}

async function fetchStatus() {
  const res = await fetch('/train/status');
  const d   = await res.json();
  if (d.best_acc > 0) {
    document.getElementById('best-acc-side').textContent = (d.best_acc*100).toFixed(2) + '%';
    document.getElementById('s-best-acc').textContent    = (d.best_acc*100).toFixed(1) + '%';
    document.getElementById('s-epochs').textContent      = d.history.train_loss.length;
    document.getElementById('s-backbone').textContent    = currentBackbone.replace('resnet','R');
  }
}

function addLog(text, cls) {
  const box  = document.getElementById('log-box');
  const line = document.createElement('div');
  line.className = 'log-line' + (cls ? ' ' + cls : '');
  line.textContent = text;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function resetLog() {
  document.getElementById('log-box').innerHTML = '<div class="log-line" style="color:var(--muted)">ממתין להרצה...</div>';
}

// ─── Results / Charts ─────────────────────────────────────────────────────
let chartLoss = null, chartAcc = null;

async function refreshResults() {
  const res = await fetch('/train/status');
  const d   = await res.json();
  const h   = d.history;
  if (!h.train_loss.length) return;

  const epochs = h.train_loss.map((_,i) => i+1);
  const cfg = (label1, data1, label2, data2) => ({
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: label1, data: data1, borderColor: '#5a9e3a', backgroundColor: 'rgba(90,158,58,.1)', tension: .3, pointRadius: 4, pointBackgroundColor: '#5a9e3a' },
        { label: label2, data: data2, borderColor: '#b8f07a', backgroundColor: 'rgba(184,240,122,.1)', tension: .3, pointRadius: 4, pointBackgroundColor: '#b8f07a' },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#d4c4a8', font: { family: "'DM Mono'" } } } },
      scales: {
        x: { ticks: { color: '#6b7c6b' }, grid: { color: 'rgba(255,255,255,.04)' } },
        y: { ticks: { color: '#6b7c6b' }, grid: { color: 'rgba(255,255,255,.04)' } }
      }
    }
  });

  if (chartLoss) chartLoss.destroy();
  if (chartAcc)  chartAcc.destroy();
  chartLoss = new Chart(document.getElementById('chart-loss'), cfg('Train Loss', h.train_loss, 'Val Loss', h.val_loss));
  chartAcc  = new Chart(document.getElementById('chart-acc'),  cfg('Train Acc', h.train_acc,  'Val Acc',  h.val_acc));

  document.getElementById('s-best-acc').textContent = (d.best_acc*100).toFixed(1) + '%';
  document.getElementById('s-epochs').textContent   = h.train_loss.length;
  document.getElementById('s-backbone').textContent = currentBackbone.replace('resnet','R');

  renderCompareTable();
}

// ─── [NEW] Model Comparison ────────────────────────────────────────────────
function saveModelRun(backbone, valAcc, testAcc, epochs) {
  const idx = modelRuns.findIndex(r => r.backbone === backbone);
  const run = { backbone, valAcc, testAcc, epochs, ts: Date.now() };
  if (idx >= 0) modelRuns[idx] = run; else modelRuns.push(run);
  sessionStorage.setItem('modelRuns', JSON.stringify(modelRuns));
}

function renderCompareTable() {
  const tbody = document.getElementById('compare-tbody');
  if (!modelRuns.length) return;

  const bestAcc = Math.max(...modelRuns.map(r => r.valAcc));
  tbody.innerHTML = modelRuns.map(r => {
    const isBest = r.valAcc === bestAcc;
    const barW   = Math.round(r.valAcc * 120);
    return `<tr class="${isBest ? 'best-row' : ''}">
      <td>${r.backbone}${isBest ? ' ★' : ''}</td>
      <td>${(r.valAcc*100).toFixed(2)}%</td>
      <td>${r.testAcc ? (r.testAcc*100).toFixed(2)+'%' : '—'}</td>
      <td>${r.epochs}</td>
      <td><div class="mini-bar-wrap"><div class="mini-bar" style="width:${barW}px"></div></div></td>
    </tr>`;
  }).join('');
}

// ─── [NEW] Export CSV ──────────────────────────────────────────────────────
async function exportCSV() {
  const res = await fetch('/train/status');
  const d   = await res.json();
  const h   = d.history;

  if (!h.train_loss.length) { alert('אין נתוני אימון לייצוא'); return; }

  const rows = [['epoch','train_loss','val_loss','train_acc','val_acc']];
  h.train_loss.forEach((_, i) => {
    rows.push([i+1, h.train_loss[i], h.val_loss[i], h.train_acc[i], h.val_acc[i]]);
  });

  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `training_${currentBackbone}_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── [NEW] Image Preview ───────────────────────────────────────────────────
let pendingFile = null;

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) showPreview(file);
}

function handleFile(file) {
  if (!file) return;
  showPreview(file);
}

function showPreview(file) {
  pendingFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('preview-filename').textContent = file.name;
    document.getElementById('upload-zone').classList.add('hidden');
    document.getElementById('preview-section').classList.remove('hidden');
    document.getElementById('predict-result-card').classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  pendingFile = null;
  document.getElementById('preview-section').classList.add('hidden');
  document.getElementById('upload-zone').classList.remove('hidden');
  document.getElementById('predict-result-card').classList.add('hidden');
  document.getElementById('file-input').value = '';
}

async function runPredict() {
  if (!pendingFile) return;

  const card    = document.getElementById('predict-result-card');
  const loading = document.getElementById('predict-loading');
  const result  = document.getElementById('predict-result');

  card.classList.remove('hidden');
  loading.classList.remove('hidden');
  result.classList.add('hidden');

  const fd = new FormData();
  fd.append('file',  pendingFile);
  fd.append('top_k', 5);

  const res  = await fetch('/predict', { method: 'POST', body: fd });
  const data = await res.json();

  loading.classList.add('hidden');

  if (data.error) {
    result.classList.remove('hidden');
    document.getElementById('predictions').innerHTML =
      `<div style="color:var(--rust);font-family:'DM Mono',monospace;font-size:.8rem">${data.error}</div>`;
    document.getElementById('health-badge').innerHTML = '';
    return;
  }

  const imgSrc = 'data:image/jpeg;base64,' + data.image;
  document.getElementById('predict-img').src = imgSrc;

  const top = data.predictions[0];
  const isHealthy = top.label.toLowerCase().includes('healthy');
  const conf      = parseFloat(top.confidence);
  let badgeClass, badgeText, badgeIcon;
  if (isHealthy) {
    badgeClass = 'healthy'; badgeText = 'צמח בריא'; badgeIcon = '✓';
  } else if (conf >= 70) {
    badgeClass = 'sick'; badgeText = 'מחלה זוהתה'; badgeIcon = '✗';
  } else {
    badgeClass = 'warn'; badgeText = 'אי-ודאות גבוהה'; badgeIcon = '⚠';
  }
  document.getElementById('health-badge').innerHTML =
    `<div class="health-badge ${badgeClass}">${badgeIcon} ${badgeText} — ${top.label.replace(/_/g,' ').replace(/___/g,' › ')}</div>`;

  document.getElementById('predictions').innerHTML = data.predictions.map(p => {
    const c = parseFloat(p.confidence);
    const confClass = c >= 70 ? 'conf-high' : c >= 40 ? 'conf-mid' : 'conf-low';
    return `<div class="pred-row">
      <div class="pred-rank">${p.rank}</div>
      <div class="pred-bar-wrap">
        <div class="pred-bar ${confClass}" style="width:${p.confidence}%"></div>
        <div class="pred-label">${p.label.replace(/_/g,' ').replace(/___/g,' › ')}</div>
      </div>
      <div class="pred-pct ${confClass}">${p.confidence}%</div>
    </div>`;
  }).join('');

  result.classList.remove('hidden');

  const gradcamWrap  = document.getElementById('gradcam-wrap');
  const gradcamError = document.getElementById('gradcam-error');
  const gradcamOrig  = document.getElementById('gradcam-orig');
  const gradcamHeat  = document.getElementById('gradcam-heat');

  if (data.gradcam) {
    gradcamOrig.src = imgSrc;
    gradcamHeat.src = 'data:image/jpeg;base64,' + data.gradcam;
    gradcamWrap.style.display  = '';
    gradcamError.classList.add('hidden');
  } else {
    gradcamWrap.style.display = 'none';
    gradcamError.classList.remove('hidden');
  }

  addToHistory(imgSrc, top.label, top.confidence, isHealthy, badgeClass);
}

// ─── [NEW] Prediction History ─────────────────────────────────────────────
let predHistory = [];

function addToHistory(imgSrc, label, confidence, isHealthy, badgeClass) {
  predHistory.unshift({
    imgSrc, label, confidence, isHealthy, badgeClass,
    ts: new Date().toLocaleTimeString('he-IL', { hour:'2-digit', minute:'2-digit' })
  });
  if (predHistory.length > 20) predHistory.pop();
  document.getElementById('history-count') &&
    (document.getElementById('history-count').textContent = predHistory.length + ' תחזיות');
}

function renderHistory() {
  const list = document.getElementById('history-list');
  document.getElementById('history-count').textContent = predHistory.length + ' תחזיות';

  if (!predHistory.length) {
    list.innerHTML = '<div class="history-empty">עדיין לא נותחו תמונות</div>';
    return;
  }

  list.innerHTML = predHistory.map((h, i) => `
    <div class="history-item" onclick="jumpToPredict(${i})">
      <img class="history-thumb" src="${h.imgSrc}" alt="thumb"/>
      <div class="history-info">
        <div class="history-label">${h.label.replace(/_/g,' ').replace(/___/g,' › ')}</div>
        <div class="history-meta">${h.confidence}% ביטחון · ${h.ts}</div>
      </div>
      <div class="history-badge ${h.badgeClass}">${h.isHealthy ? 'בריא' : 'חולה'}</div>
    </div>
  `).join('');
}

function jumpToPredict(idx) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
  document.querySelector('[data-page="predict"]').classList.add('active');
  const page = document.getElementById('page-predict');
  page.classList.remove('hidden');
  page.classList.add('active');
}

function clearHistory() {
  predHistory = [];
  renderHistory();
}

// ─── Init ─────────────────────────────────────────────────────────────────
fetchStatus();
renderCompareTable();
