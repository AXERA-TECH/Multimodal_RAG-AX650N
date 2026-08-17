/* ============================================================
   Multimodal RAG — Frontend Application
   ============================================================ */

// ============================================================
// Constants
// ============================================================
const modIcons = { text: 'T', image: 'I', audio: 'A', video: 'V' };
const modLabels = { text: '文本', image: '图片', audio: '音频', video: '视频' };

// ============================================================
// API Client
// ============================================================
const api = {
  async request(method, path, body) {
    const opts = { method };
    if (!(body instanceof FormData)) opts.headers = { 'Content-Type': 'application/json' };
    if (body) opts.body = body instanceof FormData ? body : JSON.stringify(body);
    const resp = await fetch('/api' + path, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || data.error || `HTTP ${resp.status}`);
    return data;
  },
  ingestFiles(files, incremental) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    fd.append('incremental', incremental ? 'true' : 'false');
    return this.request('POST', '/ingest/files', fd);
  },
  ask(p)      { return this.request('POST', '/query', p); },
  crossModalSearch(query, topK, threshold = 0) {
    const fd = new FormData(); fd.append('query', query); fd.append('top_k_per_modality', topK);
    if (threshold > 0) fd.append('min_similarity', threshold);
    return this.request('POST', '/query/cross-modal', fd);
  },
  crossModalSearchFile(file, topK, threshold = 0) {
    const fd = new FormData(); fd.append('file', file); fd.append('top_k_per_modality', topK);
    if (threshold > 0) fd.append('min_similarity', threshold);
    return this.request('POST', '/query/cross-modal', fd);
  },
  getStats()   { return this.request('GET', '/stats'); },
  getSources() { return this.request('GET', '/sources'); },
  getChunks(source) { return this.request('GET', '/sources/chunks?source_file=' + encodeURIComponent(source)); },
  deleteSource(s) {
    const fd = new FormData(); fd.append('source_file', s);
    return this.request('DELETE', '/sources', fd);
  },
  clearCollection() { return this.request('DELETE', '/collection'); },
  getConfig()  { return this.request('GET', '/config'); },
  saveConfig(p) { return this.request('POST', '/config', p); },
};

// ============================================================
// Navigation
// ============================================================
function initNavigation() {
  document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.getElementById('page-' + btn.dataset.page).classList.add('active');
      if (btn.dataset.page === 'management') { loadManagement(); loadConfig(); }
      if (btn.dataset.page === 'ingestion') { loadIngestSources(); }
    });
  });
}

// ============================================================
// Toast
// ============================================================
const _toastTimers = new WeakMap();

function showToast(msg, type) {
  const c = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = 'toast ' + (type || 'success');
  t.textContent = msg;
  c.appendChild(t);

  // 清理旧定时器
  const old = _toastTimers.get(t);
  if (old) { clearTimeout(old[0]); clearTimeout(old[1]); }

  const t1 = setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; }, 3000);
  const t2 = setTimeout(() => t.remove(), 3400);
  _toastTimers.set(t, [t1, t2]);
}

function showSpinner(el) {
  el.innerHTML = '<div class="loading"><div class="spinner"></div>处理中…</div>';
}

function escapeHtml(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#x27;');
}

// ============================================================
// QA Page
// ============================================================
function initQA() {
  const submit = () => {
    const q = document.getElementById('qa-question').value.trim();
    if (!q) return;
    const topK = parseInt(document.getElementById('qa-topk').value) || 5;
    const mod = document.getElementById('qa-modality').value;
    const xm = document.getElementById('qa-crossmodal').checked;
    const ans = document.getElementById('qa-answer-container');
    const src = document.getElementById('qa-sources-container');
    showSpinner(ans); src.innerHTML = '';

    const threshold = parseFloat(document.getElementById('qa-threshold').value) || 0;
    const p = { question: q, top_k: topK, cross_modal: xm, min_similarity: threshold };
    if (mod !== 'all') p.modality_filter = mod;

    api.ask(p).then(data => {
      const lb = data.latency_breakdown || {};
      ans.innerHTML = `<div class="answer-card">
        <div class="answer-label">回答</div>
        <div class="answer-content">${escapeHtml(data.answer || '')}</div>
        <div class="meta-bar">
          ${lb.total_ms != null ? `<div class="meta-item">总延迟 <span>${lb.total_ms} ms</span></div>` : ''}
          ${lb.embed_ms != null ? `<div class="meta-item">嵌入 <span>${lb.embed_ms} ms</span></div>` : ''}
          ${lb.retrieve_ms != null ? `<div class="meta-item">检索 <span>${lb.retrieve_ms} ms</span></div>` : ''}
          ${lb.generate_ms != null ? `<div class="meta-item">生成 <span>${lb.generate_ms} ms</span></div>` : ''}
        </div></div>`;
      renderSources(src, data.sources || [], data.modality_breakdown);
    }).catch(err => { ans.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`; });
  };
  document.getElementById('qa-submit').addEventListener('click', submit);
  document.getElementById('qa-question').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  });
}

function renderSources(el, sources, breakdown) {
  if (!sources.length) { el.innerHTML = '<div class="empty-state"><div class="text">无结果</div></div>'; return; }
  const bd = breakdown ? Object.entries(breakdown).map(([m,c]) =>
    `<span style="font-size:12px;background:var(--bg-hover);padding:2px 8px;border-radius:10px;">${modIcons[m]||'?'} ${c}</span>`).join(' ') : '';
  let h = `<div class="card"><div class="card-header">检索结果 (${sources.length}) ${bd}</div><div class="source-list">`;
  sources.forEach((s, i) => {
    const mod = s.modality || 'text';
    const name = s.source_file_name || s.source_file || '';
    const score = s.score != null ? Number(s.score).toFixed(4) : '';
    const text = s.text_content || s.content_preview || '';
    h += `<div class="source-item" onclick="this.classList.toggle('expanded')">
      <div class="source-item-header">
        <div class="source-name"><span class="source-badge">${modIcons[mod]||'?'}</span> ${escapeHtml(name || '#'+(i+1))}</div>
        <div class="source-score">${score}</div></div>
      <div class="source-content">
        ${text ? '<p class="source-text">'+escapeHtml(text)+'</p>' : ''}
        ${mod==='image' && s.media_url ? `<img class="source-image" src="${s.media_url}" loading="lazy" />` : ''}
        ${mod==='video' && s.media_url ? `<video class="source-image" controls preload="none" style="width:100%;max-height:300px;margin-top:6px;"><source src="${s.media_url}" type="video/mp4"></video>` : ''}
        ${mod==='audio' && s.media_url ? `<audio controls preload="none" style="width:100%;margin-top:6px;height:30px;"><source src="${s.media_url}" type="audio/wav"></audio>` : ''}
      </div></div>`;
  });
  h += '</div></div>'; el.innerHTML = h;
}

// ============================================================
// Ingestion Page
// ============================================================
let ingestFiles = [];

function initIngestion() {
  const dz = document.getElementById('ingest-dropzone');
  const inp = document.getElementById('ingest-file-input');
  dz.addEventListener('click', () => inp.click());
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('dragover'); addFiles(e.dataTransfer.files); });
  inp.addEventListener('change', () => { addFiles(inp.files); inp.value = ''; });
  document.getElementById('ingest-submit').addEventListener('click', doIngest);
  document.getElementById('ingest-refresh').addEventListener('click', loadIngestSources);
  document.getElementById('ingest-clear-all').addEventListener('click', () => {
    showModal('清空向量库','确定删除所有已索引数据？不可恢复。', () => {
      api.clearCollection().then(()=>{showToast('已清空');loadIngestSources();}).catch(e=>showToast(e.message,'error'));
    });
  });
  loadIngestSources();
}

function addFiles(fl) { for (const f of fl) { if (!ingestFiles.find(x=>x.name===f.name&&x.size===f.size)) ingestFiles.push(f); } renderFiles(); }
function removeFile(name) { ingestFiles = ingestFiles.filter(f => f.name !== name); renderFiles(); }
function fmtSize(b) { return b<1024?b+' B':b<1024*1024?(b/1024).toFixed(1)+' KB':(b/1024/1024).toFixed(1)+' MB'; }

function renderFiles() {
  document.getElementById('file-count-text').textContent = ingestFiles.length + ' 个文件';
  const c = document.getElementById('ingest-file-list');
  if (!ingestFiles.length) { c.innerHTML = '<div class="file-item" style="color:var(--text-muted)">暂无文件</div>'; return; }
  const icons = {txt:'T',md:'M',pdf:'P',jpg:'I',jpeg:'I',png:'I',gif:'I',webp:'I',mp3:'A',wav:'A',ogg:'A',flac:'A',mp4:'V',avi:'V',mov:'V',webm:'V',mkv:'V'};
  c.innerHTML = ingestFiles.map((f, idx) => {
    const ext = f.name.split('.').pop().toLowerCase();
    return `<div class="file-item">
      <div class="file-item-name"><span style="font-size:16px">${icons[ext]||'?'}</span> ${escapeHtml(f.name)}</div>
      <div style="display:flex;align-items:center;gap:12px;">
        <span class="file-item-size">${fmtSize(f.size)}</span>
        <span class="file-item-status status-pending">待上传</span>
        <button class="btn btn-sm btn-secondary file-remove-btn" data-file-index="${idx}" style="font-size:12px">移除</button>
      </div></div>`;
  }).join('');

  // 事件委托：用 data-file-index 安全传递
  c.querySelectorAll('.file-remove-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.fileIndex);
      if (!isNaN(i) && i < ingestFiles.length) {
        ingestFiles.splice(i, 1);
        renderFiles();
      }
    });
  });
}

function doIngest() {
  if (!ingestFiles.length) return;
  const inc = document.getElementById('ingest-incremental').checked;
  const c = document.getElementById('ingest-results-container');
  showSpinner(c); document.getElementById('ingest-submit').disabled = true;

  api.ingestFiles(ingestFiles, inc).then(data => {
    document.getElementById('ingest-submit').disabled = false;
    ingestFiles = []; renderFiles();
    const s = data.summary || data;
    let h = '<div class="result-summary">';
    h += `<div class="result-stat"><div class="value">${s.files_processed||0}</div><div class="label">成功</div></div>`;
    h += `<div class="result-stat"><div class="value">${s.files_failed||0}</div><div class="label">失败</div></div>`;
    h += `<div class="result-stat"><div class="value">${s.chunks_created||0}</div><div class="label">分块</div></div>`;
    h += '</div>';
    if (s.chunks_by_modality && Object.keys(s.chunks_by_modality).length) {
      h += '<div class="card" style="margin-top:12px"><div class="card-header">按模态分布</div>';
      h += Object.entries(s.chunks_by_modality).map(([m,c]) => `<span style="font-size:14px;margin-right:14px">${modLabels[m]||m}: ${c}</span>`).join('');
      h += '</div>';
    }
    if (s.errors && s.errors.length) {
      h += '<div class="card" style="margin-top:12px"><div class="card-header">错误</div>';
      s.errors.forEach(e => { h += `<div class="error-msg">${escapeHtml(String(e))}</div>`; });
      h += '</div>';
    }
    c.innerHTML = h;
    showToast('摄取完成', s.files_failed ? 'error' : 'success');
    // 刷新数据源列表
    loadIngestSources();
  }).catch(err => {
    document.getElementById('ingest-submit').disabled = false;
    c.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`;
    // 部分成功也可能有数据入库，刷新列表
    loadIngestSources();
  });
}

// Chunk Preview (on Ingestion page)
function previewChunks(name) {
  const content = document.getElementById('ingest-chunks-content');
  showSpinner(content);

  api.getChunks(name).then(data => {
    const chunks = data.chunks || [];
    if (!chunks.length) { content.innerHTML = '<div class="empty-state"><div class="text">无分块</div></div>'; return; }

    let h = `<div class="card-header">${escapeHtml(name)} (${chunks.length} 个分块)</div><div class="chunk-list">`;
    chunks.forEach((c, i) => {
      const mod = c.modality || 'text';
      const text = c.text_content || c.content_preview || '';
      const ts = c.timestamp_sec != null ? c.timestamp_sec.toFixed(1)+'s' : '';
      const off = c.start_offset != null ? `${c.start_offset}s-${c.end_offset}s` : '';

      h += `<div class="chunk-item">
        <div class="chunk-item-header">
          <span class="chunk-index">#${i+1}</span>
          <span class="source-badge">${modIcons[mod]||'?'} ${modLabels[mod]||mod}</span>
          <span style="font-size:12px;color:var(--text-muted);margin-left:auto">${c.chunk_id||''}</span>
        </div>
        <div class="chunk-item-meta">
          ${ts?`<span>${ts}</span>`:''} ${off?`<span>${off}</span>`:''}
          ${c.frame_index!=null?`<span>帧#${c.frame_index}</span>`:''}
          <span>维度 ${c.embedding_dim||'-'}</span>
        </div>`;

      // Media preview (use media_url for images/audio, thumbnail for quick preview)
      const mediaUrl = c.media_url || '';
      const thumb = c.thumbnail_base64 || '';
      if (mod === 'image' && mediaUrl) {
        h += `<img class="chunk-media" src="${mediaUrl}" alt="图片预览" loading="lazy" />`;
      } else if (mod === 'video' && mediaUrl) {
        h += `<video class="chunk-media" controls preload="none" style="width:100%;max-height:300px;margin-top:6px;"><source src="${mediaUrl}" type="video/mp4"></video>`;
      } else if (mod === 'image' && thumb) {
        h += `<img class="chunk-media" src="data:image/jpeg;base64,${thumb}" alt="图片预览" />`;
      } else if (mod === 'audio') {
        h += `<div style="margin-top:8px;padding:12px;background:var(--bg);border-radius:4px;font-size:13px;color:var(--text-muted);">
          🔊 音频片段 ${c.start_offset!=null?`(${c.start_offset}s - ${c.end_offset}s)` :''}
        </div>`;
        if (mediaUrl) {
          h += `<audio controls preload="none" style="width:100%;margin-top:4px;height:32px;"><source src="${mediaUrl}" type="audio/wav"></audio>`;
        }
      } else if (mod === 'video') {
        h += `<div style="margin-top:8px;padding:12px;background:var(--bg);border-radius:4px;font-size:13px;color:var(--text-muted);">
          🎬 视频 ${c.timestamp_sec!=null?`(${c.timestamp_sec.toFixed(1)}s)` :''} ${c.frame_index!=null?`帧#${c.frame_index}` :''}
          ${thumb ? `<img class="chunk-media" src="data:image/jpeg;base64,${thumb}" alt="帧预览" style="margin-top:6px;" />` : ''}
        </div>`;
      }

      if (text) {
        h += `<div class="chunk-item-content">${escapeHtml(text).substring(0, 400)}</div>`;
      }
      h += '</div>';
    });
    h += '</div>';
    content.innerHTML = h;
  }).catch(err => { content.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`; });
}

function loadIngestSources() {
  const countEl = document.getElementById('ingest-source-count');
  const list = document.getElementById('ingest-sources-list');
  const empty = document.getElementById('ingest-sources-empty');
  // DOM 元素不存在时静默跳过 (如在其他页面被调用)
  if (!list || !empty) return;

  api.getSources().then(d => {
    const srcs = d.sources || [];
    if (countEl) countEl.textContent = '('+srcs.length+')';
    if (!srcs.length) {
      list.innerHTML = '';
      empty.style.display = 'block';
      return;
    }
    empty.style.display = 'none';
    list.innerHTML = srcs.map(s => {
      const n = s.source_file || '';
      const dn = s.source_file_name || n;
      const cnt = s.count != null ? s.count : '-';
      const mods = s.modalities ? (Array.isArray(s.modalities)?s.modalities.join(', '):Object.keys(s.modalities).join(', ')) : '-';
      return `<div class="source-row" data-source="${escapeHtml(n)}" onclick="selectSource('${escapeHtml(n).replace(/'/g,"\\'")}')">
        <span class="src-name" title="${escapeHtml(dn)}">${escapeHtml(dn)}</span>
        <span class="src-meta">${cnt} 块 | ${escapeHtml(String(mods))}</span>
        <div class="src-actions">
          <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation();selectSource('${escapeHtml(n).replace(/'/g,"\\'")}')">查看</button>
          <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deleteIngestSource('${escapeHtml(n).replace(/'/g,"\\'")}')">删除</button>
        </div>
      </div>`;
    }).join('');
  }).catch(err => {
    console.error('加载数据源列表失败:', err);
  });
}

function selectSource(name) {
  // Highlight active row
  document.querySelectorAll('.source-row').forEach(r => r.classList.remove('active'));
  const row = document.querySelector(`.source-row[data-source="${name}"]`);
  if (row) row.classList.add('active');
  previewChunks(name);
}

function deleteIngestSource(name) {
  showModal('删除数据源',`确定删除 "${name}" 的所有数据？`,()=>{
    api.deleteSource(name).then(()=>{showToast('已删除');loadIngestSources();}).catch(e=>showToast(e.message,'error'));
  });
}

// ============================================================
// Cross-Modal Page
// ============================================================
let xmMode = 'text';    // 'text' | 'image' | 'audio' | 'video'
let xmFile = null;      // uploaded File object

function initCrossModal() {
  const topK = () => parseInt(document.getElementById('xm-topk').value) || 5;
  const container = document.getElementById('xm-results-container');

  // Mode-specific dropzone hints
  const modeHints = {
    text:  { icon: '📝', text: '输入搜索关键词', hint: '文本查询将检索所有模态的内容' },
    image: { icon: '🖼️', text: '拖拽图片或点击选择', hint: '支持 JPG / PNG / GIF / WebP / BMP' },
    audio: { icon: '🎵', text: '拖拽音频或点击选择', hint: '支持 MP3 / WAV / FLAC / OGG / M4A' },
    video: { icon: '🎬', text: '拖拽视频或点击选择', hint: '支持 MP4 / MOV / AVI / MKV / WebM' },
  };

  // ---- Mode Tabs ----
  document.querySelectorAll('.xm-mode-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      xmMode = tab.dataset.mode;
      document.querySelectorAll('.xm-mode-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('xm-text-panel').style.display = xmMode === 'text' ? '' : 'none';
      document.getElementById('xm-file-panel').style.display = xmMode !== 'text' ? '' : 'none';

      // Update dropzone hints for file modes
      if (xmMode !== 'text') {
        const hint = modeHints[xmMode] || modeHints.image;
        document.getElementById('xm-dropzone-icon').textContent = hint.icon;
        document.getElementById('xm-dropzone-text').textContent = hint.text;
        document.getElementById('xm-dropzone-hint-text').textContent = hint.hint;
      }
    });
  });

  // ---- Text Submit ----
  document.getElementById('xm-submit').addEventListener('click', () => {
    const q = document.getElementById('xm-query').value.trim();
    if (!q) return;
    const threshold = parseFloat(document.getElementById('xm-threshold').value) || 0;
    showSpinner(container);
    api.crossModalSearch(q, topK(), threshold).then(data => renderXMResults(container, data))
      .catch(err => { container.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`; });
  });
  document.getElementById('xm-query').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('xm-submit').click(); }
  });

  // ---- File Upload ----
  const dropzone = document.getElementById('xm-dropzone');
  const fileInput = document.getElementById('xm-file-input');

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) setXMFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) setXMFile(fileInput.files[0]);
    fileInput.value = '';
  });
  document.getElementById('xm-file-remove').addEventListener('click', () => setXMFile(null));

  // ---- File Submit ----
  document.getElementById('xm-file-submit').addEventListener('click', () => {
    if (!xmFile) { showToast('请先选择文件', 'error'); return; }
    const threshold = parseFloat(document.getElementById('xm-threshold').value) || 0;
    showSpinner(container);
    api.crossModalSearchFile(xmFile, topK(), threshold).then(data => renderXMResults(container, data))
      .catch(err => { container.innerHTML = `<div class="error-msg">${escapeHtml(err.message)}</div>`; });
  });
}

function setXMFile(file) {
  xmFile = file;
  const preview = document.getElementById('xm-file-preview');
  const hint = document.getElementById('xm-dropzone-hint');
  const img = document.getElementById('xm-preview-img');
  const audioBadge = document.getElementById('xm-preview-audio');
  const videoBadge = document.getElementById('xm-preview-video');
  const nameEl = document.getElementById('xm-file-name');

  img.style.display = 'none';
  audioBadge.style.display = 'none';
  videoBadge.style.display = 'none';

  if (file) {
    nameEl.textContent = file.name;
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    const imgExts = ['jpg','jpeg','png','gif','webp','bmp'];
    const vidExts = ['mp4','mov','avi','mkv','webm','flv','wmv'];

    if (imgExts.includes(ext)) {
      const url = URL.createObjectURL(file);
      img.src = url; img.style.display = '';
    } else if (vidExts.includes(ext)) {
      videoBadge.style.display = '';
    } else {
      audioBadge.style.display = '';
    }

    preview.style.display = ''; hint.style.display = 'none';
  } else {
    preview.style.display = 'none'; hint.style.display = '';
  }
}

function renderXMResults(container, data) {
  const mods = data.modalities || {};
  const order = [
    {k:'text',  icon:'📝', label:'文本'},
    {k:'image', icon:'🖼️', label:'图片'},
    {k:'audio', icon:'🎵', label:'音频'},
    {k:'video', icon:'🎬', label:'视频'}
  ];

  let h = `<div style="font-size:var(--fs-base);color:var(--text-secondary);margin-bottom:12px;">
    查询模式: <b>${data.query_mode === 'text' ? '📝 文本' : data.query_mode === 'image' ? '🖼️ 图片' : data.query_mode === 'audio' ? '🎵 音频' : '🎬 视频'}</b>
    ${data.query_file ? ' — ' + escapeHtml(data.query_file) : ''}
    &nbsp;|&nbsp; 共 <b>${data.total_results || 0}</b> 条结果
  </div><div class="modality-grid">`;

  order.forEach(({k, icon, label}) => {
    const md = mods[k] || {}; const items = md.results || [];
    h += `<div class="modality-card"><div class="modality-card-header">${icon} ${label} (${md.count || items.length})</div><div class="modality-card-body">`;
    if (!items.length) h += '<div style="color:var(--text-muted);font-size:var(--fs-sm)">无结果</div>';

    items.forEach(item => {
      const score = item.score != null ? Number(item.score).toFixed(4) : '';
      const text = item.text_content || item.content_preview || '';
      const mod = item.modality || 'text';
      const name = item.source_file_name || '';
      const mediaUrl = item.media_url || '';

      h += `<div class="modality-result">`;
      h += `<div class="src">${escapeHtml(name)}<span class="score">${score}</span></div>`;

      // Media preview based on result modality
      if (mod === 'image' && mediaUrl) {
        h += `<img class="result-media" src="${mediaUrl}" alt="${escapeHtml(name)}" loading="lazy" onclick="this.style.maxHeight=this.style.maxHeight==='100%'?'180px':'100%'">`;
      }
      if (mod === 'video' && mediaUrl) {
        h += `<video class="result-media" controls preload="none" style="width:100%;max-height:240px;margin-top:4px;" onclick="this.style.maxHeight=this.style.maxHeight==='100%'?'240px':'100%'"><source src="${mediaUrl}" type="video/mp4"></video>`;
      }
      if (mod === 'video' && item.timestamp_sec != null) {
        h += `<div class="result-video-info">🎬 时间戳: ${item.timestamp_sec.toFixed(1)}s ${item.frame_index != null ? '帧#' + item.frame_index : ''}</div>`;
      }
      if (mod === 'audio' && mediaUrl) {
        const start = item.start_offset != null ? item.start_offset.toFixed(1) + 's' : '';
        const end = item.end_offset != null ? item.end_offset.toFixed(1) + 's' : '';
        h += `<div class="result-audio-info">🔊 ${start} — ${end}</div>`;
        h += `<audio controls preload="none" style="width:100%;margin-top:4px;height:32px;"><source src="${mediaUrl}" type="audio/wav"></audio>`;
      } else if (mod === 'audio') {
        const start = item.start_offset != null ? item.start_offset.toFixed(1) + 's' : '';
        const end = item.end_offset != null ? item.end_offset.toFixed(1) + 's' : '';
        h += `<div class="result-audio-info">🔊 ${start} — ${end} (无播放源)</div>`;
      }

      // Text preview
      if (text) {
        h += `<div class="preview">${escapeHtml(text).substring(0, 250)}</div>`;
      }
      h += '</div>';
    });
    h += '</div></div>';
  });
  h += '</div>';
  container.innerHTML = h;
}

// ============================================================
// Management Page
// ============================================================
function loadManagement() { loadMgStats(); }

function loadMgStats() {
  api.getStats().then(d => {
    document.getElementById('mg-stats-grid').innerHTML = `
      <div class="stat-card"><div class="stat-value">${d.total_chunks||0}</div><div class="stat-label">总分块</div></div>
      <div class="stat-card"><div class="stat-value">${d.unique_sources||0}</div><div class="stat-label">数据源</div></div>
      <div class="stat-card"><div class="stat-value">${d.embedding_dim||'-'}</div><div class="stat-label">嵌入维度</div></div>
      <div class="stat-card"><div class="stat-value">${d.active_modalities||0}</div><div class="stat-label">活跃模态</div></div>`;
    const bd = d.modality_breakdown || {};
    const total = Object.values(bd).reduce((s,v)=>s+Number(v),0)||1;
    document.getElementById('mg-modality-breakdown').innerHTML = Object.entries(bd).map(([m,c]) =>
      `<div style="margin-top:8px;font-size:13px;display:flex;justify-content:space-between"><span>${modLabels[m]||m}</span><span>${c}</span></div>
      <div class="progress-bar"><div class="progress-fill ${m}" style="width:${(Number(c)/total*100).toFixed(0)}%"></div></div>`).join('');
  }).catch(()=>{});
}

// ============================================================
// Model Config
// ============================================================
function loadConfig() {
  api.getConfig().then(d => {
    document.getElementById('cfg-llm-base').value = d.llm_api_base || '';
    document.getElementById('cfg-llm-model').value = d.llm_model_name || '';
    document.getElementById('cfg-emb-base').value = d.embedding_api_base || '';
    document.getElementById('cfg-emb-model').value = d.embedding_model_name || '';
    // 预处理参数
    const pp = d.preprocessing || {};
    document.getElementById('cfg-text-chunk-size').value = pp.text_chunk_size ?? 512;
    document.getElementById('cfg-text-chunk-overlap').value = pp.text_chunk_overlap ?? 64;
    document.getElementById('cfg-image-target-size').value = pp.image_target_size ?? 256;
    document.getElementById('cfg-image-quality').value = pp.image_quality ?? 85;
    document.getElementById('cfg-audio-max-dur').value = pp.audio_max_duration_sec ?? 25;
    document.getElementById('cfg-audio-overlap').value = pp.audio_overlap_sec ?? 1;
    document.getElementById('cfg-video-target-size').value = pp.video_target_size ?? 256;
    document.getElementById('cfg-video-max-frames').value = pp.video_max_frames ?? 32;
    document.getElementById('cfg-video-qa-frames').value = pp.video_qa_frames ?? 4;
  }).catch(()=>{});
}

document.getElementById('cfg-save').addEventListener('click', () => {
  const p = {
    llm_api_base: document.getElementById('cfg-llm-base').value.trim() || null,
    llm_model_name: document.getElementById('cfg-llm-model').value.trim() || null,
    embedding_api_base: document.getElementById('cfg-emb-base').value.trim() || null,
    embedding_model_name: document.getElementById('cfg-emb-model').value.trim() || null,
  };
  api.saveConfig(p).then(d => showToast(d.message || '配置已保存')).catch(e => showToast(e.message, 'error'));
});

// Preprocessing config save
document.getElementById('cfg-preprocess-save').addEventListener('click', () => {
  const p = {
    text_chunk_size: parseInt(document.getElementById('cfg-text-chunk-size').value) || null,
    text_chunk_overlap: parseInt(document.getElementById('cfg-text-chunk-overlap').value) || null,
    image_target_size: parseInt(document.getElementById('cfg-image-target-size').value) || null,
    image_quality: parseInt(document.getElementById('cfg-image-quality').value) || null,
    audio_max_duration_sec: parseFloat(document.getElementById('cfg-audio-max-dur').value) || null,
    audio_overlap_sec: parseFloat(document.getElementById('cfg-audio-overlap').value) || null,
    video_target_size: parseInt(document.getElementById('cfg-video-target-size').value) || null,
    video_max_frames: parseInt(document.getElementById('cfg-video-max-frames').value) || null,
    video_qa_frames: parseInt(document.getElementById('cfg-video-qa-frames').value) || null,
  };
  api.saveConfig(p).then(d => showToast(d.message || '预处理参数已保存')).catch(e => showToast(e.message, 'error'));
});

// ============================================================
// Modal
// ============================================================
let modalCb = null;
function showModal(title, msg, cb) {
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-message').textContent = msg;
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.add('active');
  overlay.setAttribute('aria-hidden', 'false');
  modalCb = cb;
}
function hideModal() {
  const overlay = document.getElementById('modal-overlay');
  overlay.classList.remove('active');
  overlay.setAttribute('aria-hidden', 'true');
  modalCb = null;
}
document.getElementById('modal-cancel').addEventListener('click', hideModal);
document.getElementById('modal-confirm').addEventListener('click', () => { if(modalCb){modalCb();modalCb=null;} hideModal(); });
document.getElementById('modal-overlay').addEventListener('click', e => { if(e.target===document.getElementById('modal-overlay')) hideModal(); });

// ============================================================
// Init
// ============================================================
function init() {
  const loader = document.getElementById('app-loader');
  const hideLoader = () => { if (loader) loader.style.display = 'none'; };

  try {
    initNavigation();
    initQA();
    initIngestion();
    initCrossModal();
  } catch (e) {
    console.error('Init error:', e);
    if (loader) loader.innerHTML = `<div class="error-msg">初始化失败: ${escapeHtml(e.message)}</div>`;
    return;
  }
  hideLoader();
}
document.addEventListener('DOMContentLoaded', init);
