const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let activeTab = 'journal';
let currentFile = null;
let currentSession = null;

const logError = $('#log-error');
const journalUnit = $('#journal-unit');
const journalLines = $('#journal-lines');
const journalOutput = $('#journal-output');
const fileList = $('#log-file-list');
const fileOutput = $('#file-output');
const historyList = $('#history-list');
const historyDetail = $('#history-detail');

function setError(message) {
  logError.textContent = message || '';
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

async function requestJSON(url, options = {}) {
  const resp = await fetch(url, options);
  let data = {};
  try {
    data = await resp.json();
  } catch (_) {
    data = {};
  }
  if (!resp.ok) {
    throw new Error(data.error || `HTTP ${resp.status}`);
  }
  return data;
}

function formatSize(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(iso) {
  if (!iso) return '-';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const mm = d.getMonth() + 1;
  const dd = d.getDate();
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  return `${mm}/${dd} ${hh}:${mi}`;
}

// --- サービスログ ---

async function loadJournal() {
  const unit = journalUnit.value || 'subpc-web';
  journalOutput.textContent = '読み込み中...';
  try {
    const data = await requestJSON(
      `/api/logs/journal?unit=${encodeURIComponent(unit)}&lines=${journalLines.value}`
    );
    if (!journalUnit.childElementCount) {
      for (const u of data.units) {
        const opt = document.createElement('option');
        opt.value = u;
        opt.textContent = u;
        if (u === data.unit) opt.selected = true;
        journalUnit.appendChild(opt);
      }
    }
    journalOutput.textContent = data.lines.join('\n') || '(ログなし)';
    journalOutput.scrollTop = journalOutput.scrollHeight;
  } catch (err) {
    journalOutput.textContent = '';
    setError(`サービスログ取得失敗: ${err.message}`);
  }
}

// --- アプリログファイル ---

async function loadFiles() {
  try {
    const data = await requestJSON('/api/logs/files');
    if (!data.files.length) {
      fileList.innerHTML = '<div class="task-empty-row"><span class="empty-hint">ログファイルはまだありません。</span></div>';
      fileOutput.hidden = true;
      return;
    }
    fileList.innerHTML = data.files.map((f) => `
      <button class="log-file-row ${f.name === currentFile ? 'active' : ''}" data-file="${escapeHtml(f.name)}" type="button">
        <span class="log-file-name">${escapeHtml(f.name)}</span>
        <span class="log-file-meta">${formatSize(f.size_bytes)} ・ ${formatDate(f.mtime)}</span>
      </button>
    `).join('');
    if (currentFile) await loadFileTail(currentFile);
  } catch (err) {
    setError(`ログ一覧取得失敗: ${err.message}`);
  }
}

async function loadFileTail(name) {
  currentFile = name;
  $$('.log-file-row').forEach((row) => {
    row.classList.toggle('active', row.dataset.file === name);
  });
  fileOutput.hidden = false;
  fileOutput.textContent = '読み込み中...';
  try {
    const data = await requestJSON(`/api/logs/files/${encodeURIComponent(name)}?lines=500`);
    fileOutput.textContent = data.lines.join('\n') || '(空)';
    fileOutput.scrollTop = fileOutput.scrollHeight;
  } catch (err) {
    fileOutput.textContent = '';
    setError(`ログ取得失敗: ${err.message}`);
  }
}

// --- 会話ログ ---

async function loadHistory() {
  try {
    const data = await requestJSON('/api/history/sessions');
    if (!data.sessions.length) {
      historyList.innerHTML = '<div class="task-empty-row"><span class="empty-title">会話の記録はまだありません</span><span class="empty-hint">話した内容はここから振り返れます。</span></div>';
      historyDetail.hidden = true;
      return;
    }
    historyList.innerHTML = data.sessions.map((s) => `
      <div class="history-row ${s.file === currentSession ? 'active' : ''}" data-file="${escapeHtml(s.file)}">
        <button class="history-open" data-file="${escapeHtml(s.file)}" type="button">
          <span class="history-preview">${escapeHtml(s.preview || '(発言なし)')}</span>
          <span class="history-meta">${formatDate(s.saved_at)}・${s.turn_count ?? '?'}往復・${formatSize(s.size_bytes)}</span>
        </button>
        <button class="action-btn danger" data-delete="${escapeHtml(s.file)}" type="button">削除</button>
      </div>
    `).join('');
  } catch (err) {
    setError(`会話ログ一覧取得失敗: ${err.message}`);
  }
}

async function openSession(file) {
  currentSession = file;
  $$('.history-row').forEach((row) => {
    row.classList.toggle('active', row.dataset.file === file);
  });
  historyDetail.hidden = false;
  historyDetail.innerHTML = '<div class="task-empty-row"><span class="empty-hint">読み込み中...</span></div>';
  try {
    const data = await requestJSON(`/api/history/sessions/${encodeURIComponent(file)}`);
    const messages = (data.messages || []).map((m) => `
      <div class="message ${m.role === 'user' ? 'user' : 'assistant'}">
        <div class="message-bubble">${escapeHtml(m.content)}</div>
      </div>
    `).join('');
    historyDetail.innerHTML = `
      <div class="cal-day-head">
        ${escapeHtml(data.session_id || file)}
        <span class="task-muted">${formatDate(data.created_at)}から・${data.turn_count ?? '?'}往復</span>
      </div>
      <div class="history-messages">${messages || '<div class="task-empty-row"><span class="empty-hint">メッセージなし</span></div>'}</div>
    `;
  } catch (err) {
    historyDetail.innerHTML = '';
    setError(`会話ログ取得失敗: ${err.message}`);
  }
}

async function deleteSession(file) {
  if (!confirm('この会話の記録を削除しますか？')) return;
  try {
    await requestJSON(`/api/history/sessions/${encodeURIComponent(file)}`, { method: 'DELETE' });
    if (currentSession === file) {
      currentSession = null;
      historyDetail.hidden = true;
    }
    await loadHistory();
  } catch (err) {
    setError(`削除失敗: ${err.message}`);
  }
}

// --- タブ切替 ---

function setTab(tab) {
  activeTab = tab;
  $$('#log-tab-seg .segmented-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.tab === tab);
  });
  $('#log-journal-pane').hidden = tab !== 'journal';
  $('#log-files-pane').hidden = tab !== 'files';
  $('#log-history-pane').hidden = tab !== 'history';
  setError('');
  refresh();
}

function refresh() {
  if (activeTab === 'journal') loadJournal();
  else if (activeTab === 'files') loadFiles();
  else loadHistory();
}

function init() {
  $$('#log-tab-seg .segmented-btn').forEach((btn) => {
    btn.addEventListener('click', () => setTab(btn.dataset.tab));
  });
  $('#log-refresh-btn').addEventListener('click', refresh);
  journalUnit.addEventListener('change', loadJournal);
  journalLines.addEventListener('change', loadJournal);
  fileList.addEventListener('click', (event) => {
    const row = event.target.closest('.log-file-row[data-file]');
    if (row) loadFileTail(row.dataset.file);
  });
  historyList.addEventListener('click', (event) => {
    const del = event.target.closest('button[data-delete]');
    if (del) {
      deleteSession(del.dataset.delete);
      return;
    }
    const open = event.target.closest('.history-open[data-file]');
    if (open) openSession(open.dataset.file);
  });
  loadJournal();
}

document.addEventListener('DOMContentLoaded', init);
