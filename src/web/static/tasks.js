const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let tasks = [];
let editingTaskId = null;
let activeFilter = 'all';
let loading = false;
let initialized = false;
let viewMode = localStorage.getItem('subpc_tasks_view') === 'calendar' ? 'calendar' : 'list';
let calCursor = null;
let selectedDayKey = null;

const addForm = $('#task-add-form');
const titleInput = $('#task-title-input');
const dueInput = $('#task-due-input');
const priorityInput = $('#task-priority-input');
const noteInput = $('#task-note-input');
const refreshBtn = $('#task-refresh-btn');
const taskList = $('#task-list');
const taskError = $('#task-error');
const taskCount = $('#task-count');
const metricOpen = $('#metric-open');
const metricOverdue = $('#metric-overdue');
const metricToday = $('#metric-today');
const metricHigh = $('#metric-high');
const filterSeg = $('#task-filter-seg');
const listShell = $('#task-list-shell');
const calShell = $('#task-calendar-shell');
const calTitle = $('#cal-title');
const calMonthCount = $('#cal-month-count');
const calWeekdays = $('#cal-weekdays');
const calGrid = $('#cal-grid');
const calDayPanel = $('#cal-day-panel');

const WEEKDAY_LABELS = ['日', '月', '火', '水', '木', '金', '土'];

const FILTER_LABELS = {
  all: 'すべて',
  overdue: '超過',
  today: '今日',
  nodue: '期限なし',
};

function setError(message) {
  taskError.textContent = message || '';
}

async function requestJSON(url, options = {}) {
  const resp = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  let data = {};
  try {
    data = await resp.json();
  } catch (_) {
    data = {};
  }
  if (!resp.ok) {
    const hint = data.hint ? ` (${data.hint})` : '';
    throw new Error(data.error || `HTTP ${resp.status}${hint}`);
  }
  return data;
}

function escapeHtml(text) {
  return String(text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function dueDate(task) {
  return task.due_at ? new Date(task.due_at) : null;
}

function isOverdue(task) {
  const due = dueDate(task);
  return !!due && due.getTime() < Date.now();
}

function isToday(task) {
  const due = dueDate(task);
  if (!due) return false;
  const now = new Date();
  return due.getFullYear() === now.getFullYear()
    && due.getMonth() === now.getMonth()
    && due.getDate() === now.getDate();
}

function formatDue(task) {
  const due = dueDate(task);
  if (!due) return '期限なし';
  const month = due.getMonth() + 1;
  const day = due.getDate();
  if (task.due_granularity === 'date') return `${month}/${day}`;
  const hh = String(due.getHours()).padStart(2, '0');
  const mm = String(due.getMinutes()).padStart(2, '0');
  return `${month}/${day} ${hh}:${mm}`;
}

function formatDueInput(task) {
  return task.due_at ? formatDue(task) : '';
}

function remainingText(task) {
  const due = dueDate(task);
  if (!due) return '';
  const diff = due.getTime() - Date.now();
  const overdue = diff < 0;
  const secs = Math.abs(diff) / 1000;
  const days = Math.floor(secs / 86400);
  const hours = Math.floor((secs % 86400) / 3600);
  const mins = Math.max(1, Math.floor((secs % 3600) / 60));
  const span = days >= 1 ? `${days}日` : hours >= 1 ? `${hours}時間` : `${mins}分`;
  return `${overdue ? '超過' : 'あと'}${span}`;
}

function priorityLabel(priority) {
  if (priority === 'high') return '高';
  if (priority === 'low') return '低';
  return '普通';
}

function visibleTasks() {
  if (activeFilter === 'overdue') return tasks.filter(isOverdue);
  if (activeFilter === 'today') return tasks.filter(isToday);
  if (activeFilter === 'nodue') return tasks.filter((task) => !task.due_at);
  return tasks;
}

function setMetric(el, value) {
  el.textContent = value;
  const card = el.closest('.metric');
  if (card) card.classList.toggle('is-zero', value === 0);
}

function updateMetrics() {
  setMetric(metricOpen, tasks.length);
  setMetric(metricOverdue, tasks.filter(isOverdue).length);
  setMetric(metricToday, tasks.filter(isToday).length);
  setMetric(metricHigh, tasks.filter((task) => task.priority === 'high').length);
}

function renderSkeleton() {
  const rows = Array.from({ length: 4 }, () => `
    <div class="task-skeleton" aria-hidden="true">
      <div class="skel-line long"></div>
      <div class="skel-line mid"></div>
      <div class="skel-line tiny"></div>
      <div class="skel-line short"></div>
    </div>
  `).join('');
  taskList.innerHTML = rows;
}

function renderEmpty() {
  const hasTasks = tasks.length > 0;
  let title;
  let hint;
  if (!hasTasks) {
    title = 'タスクはありません';
    hint = '上のフォームからタスクを追加できます。';
  } else {
    const label = FILTER_LABELS[activeFilter] || 'この絞り込み';
    title = '該当するタスクはありません';
    hint = `${label}に該当するタスクはありません。別の絞り込みを試してください。`;
  }
  taskList.innerHTML = `
    <div class="task-empty-row">
      <span class="empty-title">${escapeHtml(title)}</span>
      <span class="empty-hint">${escapeHtml(hint)}</span>
    </div>
  `;
}

async function loadTasks() {
  if (loading) return;
  loading = true;
  refreshBtn.disabled = true;
  setError('');
  renderSkeleton();
  try {
    const data = await requestJSON('/api/tasks?status=open&limit=200');
    tasks = data.tasks || [];
    updateMetrics();
    initialized = true;
    render();
  } catch (err) {
    setError(`読み込み失敗: ${err.message}`);
    if (!initialized) {
      taskList.innerHTML = '';
    } else {
      render();
    }
  } finally {
    loading = false;
    refreshBtn.disabled = false;
  }
}

function render() {
  syncViewUI();
  if (viewMode === 'calendar') {
    renderCalendar();
    return;
  }
  const shown = visibleTasks();
  taskCount.textContent = `${shown.length} / ${tasks.length} 件`;
  if (!shown.length) {
    renderEmpty();
    return;
  }
  taskList.innerHTML = shown.map((task) => (
    editingTaskId === task.id ? renderEditRow(task) : renderRow(task)
  )).join('');
}

// ============================================
//  カレンダービュー
// ============================================

function syncViewUI() {
  $$('#task-view-seg .segmented-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewMode);
  });
  const isList = viewMode === 'list';
  filterSeg.hidden = !isList;
  taskCount.hidden = !isList;
  listShell.hidden = !isList;
  calShell.hidden = isList;
}

function setView(view) {
  viewMode = view;
  localStorage.setItem('subpc_tasks_view', view);
  if (view === 'calendar') {
    const now = new Date();
    if (!calCursor) calCursor = { y: now.getFullYear(), m: now.getMonth() };
    if (!selectedDayKey) selectedDayKey = dayKeyOf(now);
  }
  render();
}

function dayKeyOf(date) {
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  return `${date.getFullYear()}-${mm}-${dd}`;
}

function tasksByDay() {
  const map = new Map();
  tasks.forEach((task) => {
    const due = dueDate(task);
    if (!due) return;
    const key = dayKeyOf(due);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(task);
  });
  const byTime = (a, b) => (dueDate(a)?.getTime() || 0) - (dueDate(b)?.getTime() || 0);
  map.forEach((list) => list.sort(byTime));
  return map;
}

function moveMonth(delta) {
  const d = new Date(calCursor.y, calCursor.m + delta, 1);
  calCursor = { y: d.getFullYear(), m: d.getMonth() };
  renderCalendar();
}

function calChip(task) {
  const cls = ['cal-chip'];
  if (isOverdue(task)) cls.push('overdue');
  else if (task.priority === 'high') cls.push('high');
  else if (task.priority === 'low') cls.push('low');
  return `<span class="${cls.join(' ')}" title="${escapeHtml(task.title)}">${escapeHtml(task.title)}</span>`;
}

function calDot(task) {
  const cls = ['cal-dot'];
  if (isOverdue(task)) cls.push('overdue');
  else if (task.priority === 'high') cls.push('high');
  else if (task.priority === 'low') cls.push('low');
  return `<i class="${cls.join(' ')}"></i>`;
}

function renderCalendar() {
  if (!calCursor) {
    const now = new Date();
    calCursor = { y: now.getFullYear(), m: now.getMonth() };
    if (!selectedDayKey) selectedDayKey = dayKeyOf(now);
  }
  const { y, m } = calCursor;
  calTitle.textContent = `${y}年${m + 1}月`;

  if (!calWeekdays.childElementCount) {
    calWeekdays.innerHTML = WEEKDAY_LABELS.map((label, i) => {
      const cls = i === 0 ? 'cal-weekday sun' : i === 6 ? 'cal-weekday sat' : 'cal-weekday';
      return `<span class="${cls}">${label}</span>`;
    }).join('');
  }

  const byDay = tasksByDay();
  const monthCount = tasks.filter((task) => {
    const due = dueDate(task);
    return due && due.getFullYear() === y && due.getMonth() === m;
  }).length;
  calMonthCount.textContent = monthCount ? `${monthCount} 件` : '';

  const first = new Date(y, m, 1);
  const daysInMonth = new Date(y, m + 1, 0).getDate();
  const weeks = Math.ceil((first.getDay() + daysInMonth) / 7);
  const start = new Date(y, m, 1 - first.getDay());
  const todayKey = dayKeyOf(new Date());

  const cells = [];
  for (let i = 0; i < weeks * 7; i++) {
    const d = new Date(start.getFullYear(), start.getMonth(), start.getDate() + i);
    const key = dayKeyOf(d);
    const dayTasks = byDay.get(key) || [];
    const cls = ['cal-cell'];
    if (d.getMonth() !== m) cls.push('out');
    if (key === todayKey) cls.push('today');
    if (key === selectedDayKey) cls.push('selected');
    if (d.getDay() === 0) cls.push('sun');
    if (d.getDay() === 6) cls.push('sat');
    const chips = dayTasks.slice(0, 3).map(calChip).join('');
    const more = dayTasks.length > 3 ? `<span class="cal-more">+${dayTasks.length - 3}件</span>` : '';
    const dots = dayTasks.length
      ? `<span class="cal-dots">${dayTasks.slice(0, 4).map(calDot).join('')}</span>`
      : '';
    cells.push(`
      <button class="${cls.join(' ')}" type="button" data-day="${key}" aria-label="${key} のタスク ${dayTasks.length}件">
        <span class="cal-date">${d.getDate()}</span>
        ${chips}${more}${dots}
      </button>
    `);
  }
  calGrid.innerHTML = cells.join('');
  renderDayPanel(byDay);
}

function renderDayPanel(byDay) {
  if (!selectedDayKey) {
    calDayPanel.hidden = true;
    calDayPanel.innerHTML = '';
    return;
  }
  const [yy, mm, dd] = selectedDayKey.split('-').map(Number);
  const d = new Date(yy, mm - 1, dd);
  const heading = `${mm}/${dd} (${WEEKDAY_LABELS[d.getDay()]})`;
  const dayTasks = byDay.get(selectedDayKey) || [];
  const body = dayTasks.length
    ? dayTasks.map((task) => (
      editingTaskId === task.id ? renderEditRow(task) : renderRow(task)
    )).join('')
    : '<div class="task-empty-row"><span class="empty-hint">この日のタスクはありません</span></div>';
  calDayPanel.hidden = false;
  calDayPanel.innerHTML = `
    <div class="cal-day-head">${escapeHtml(heading)} のタスク <span class="task-muted">${dayTasks.length} 件</span></div>
    ${body}
  `;
}

function handleCalendarClick(event) {
  if (event.target.closest('button[data-action]')) {
    handleActionClick(event);
    return;
  }
  const cell = event.target.closest('.cal-cell[data-day]');
  if (cell) {
    selectedDayKey = cell.dataset.day;
    renderCalendar();
  }
}

function renderRow(task) {
  const overdue = isOverdue(task);
  const today = isToday(task);
  const remaining = remainingText(task);
  const dueClass = overdue ? 'overdue' : today ? 'today' : '';
  return `
    <article class="task-row ${overdue ? 'overdue' : ''}" data-id="${task.id}">
      <div class="task-cell title-cell">
        <span class="priority-dot ${escapeHtml(task.priority)}" aria-hidden="true"></span>
        <div class="task-title-block">
          <div class="task-title-line">
            <strong>${escapeHtml(task.title)}</strong>
            <span class="task-id">#${task.id}</span>
          </div>
          ${task.note ? `<p class="task-note">${escapeHtml(task.note)}</p>` : ''}
        </div>
      </div>
      <div class="task-cell due-cell">
        <span class="due-text ${dueClass}">${escapeHtml(formatDue(task))}</span>
        ${remaining ? `<span class="remaining-text ${overdue ? 'overdue' : ''}">${escapeHtml(remaining)}</span>` : ''}
      </div>
      <div class="task-cell priority-cell">
        <span class="priority-badge ${escapeHtml(task.priority)}">${escapeHtml(priorityLabel(task.priority))}</span>
      </div>
      <div class="task-cell actions-cell">
        <button class="action-btn done" data-action="done" data-id="${task.id}" type="button">完了</button>
        <button class="action-btn" data-action="edit" data-id="${task.id}" type="button">編集</button>
        <button class="action-btn" data-action="snooze" data-when="30m" data-id="${task.id}" type="button">+30分</button>
        <button class="action-btn danger" data-action="drop" data-id="${task.id}" type="button">削除</button>
      </div>
    </article>
  `;
}

function renderEditRow(task) {
  return `
    <article class="task-row editing" data-id="${task.id}">
      <form class="task-edit-form" data-id="${task.id}">
        <div class="edit-grid">
          <input name="title" type="text" value="${escapeHtml(task.title)}" placeholder="タイトル" required>
          <input name="due" type="text" value="${escapeHtml(formatDueInput(task))}" placeholder="期限 例: 明日 / 7/10">
          <select name="priority" aria-label="優先度">
            <option value="normal" ${task.priority === 'normal' ? 'selected' : ''}>普通</option>
            <option value="high" ${task.priority === 'high' ? 'selected' : ''}>高</option>
            <option value="low" ${task.priority === 'low' ? 'selected' : ''}>低</option>
          </select>
          <input name="note" type="text" value="${escapeHtml(task.note || '')}" placeholder="メモ (任意)">
          <div class="edit-actions">
            <button class="primary-btn compact" type="submit">保存</button>
            <button class="secondary-btn compact" data-action="cancel-edit" type="button">キャンセル</button>
          </div>
        </div>
      </form>
    </article>
  `;
}

async function addTask(event) {
  event.preventDefault();
  const title = titleInput.value.trim();
  if (!title) return;
  setError('');
  const submitBtn = addForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  try {
    await requestJSON('/api/tasks', {
      method: 'POST',
      body: JSON.stringify({
        title,
        due: dueInput.value.trim(),
        priority: priorityInput.value,
        note: noteInput.value.trim(),
      }),
    });
    addForm.reset();
    priorityInput.value = 'normal';
    await loadTasks();
    titleInput.focus();
  } catch (err) {
    setError(`追加失敗: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
  }
}

async function handleActionClick(event) {
  const btn = event.target.closest('button[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  const id = Number(btn.dataset.id);
  setError('');

  if (action === 'edit') {
    editingTaskId = id;
    render();
    const input = taskList.querySelector(`.task-edit-form[data-id="${id}"] input[name="title"]`);
    if (input) {
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
    }
    return;
  }
  if (action === 'cancel-edit') {
    editingTaskId = null;
    render();
    return;
  }
  if (action === 'drop' && !confirm(`タスク #${id} を削除しますか？`)) {
    return;
  }

  btn.disabled = true;
  try {
    if (action === 'done') {
      await requestJSON(`/api/tasks/${id}/done`, { method: 'POST', body: '{}' });
    } else if (action === 'drop') {
      await requestJSON(`/api/tasks/${id}/drop`, { method: 'POST', body: '{}' });
    } else if (action === 'snooze') {
      await requestJSON(`/api/tasks/${id}/snooze`, {
        method: 'POST',
        body: JSON.stringify({ when: btn.dataset.when }),
      });
    }
    await loadTasks();
  } catch (err) {
    setError(`操作失敗: ${err.message}`);
  } finally {
    btn.disabled = false;
  }
}

async function saveEdit(event) {
  const form = event.target.closest('.task-edit-form');
  if (!form) return;
  event.preventDefault();

  const id = Number(form.dataset.id);
  const current = tasks.find((task) => task.id === id);
  const data = new FormData(form);
  const due = String(data.get('due') || '').trim();
  const payload = {
    title: String(data.get('title') || '').trim(),
    priority: String(data.get('priority') || 'normal'),
    note: String(data.get('note') || '').trim(),
  };
  if (!payload.title) {
    setError('タイトルは必須です');
    return;
  }
  if (!current || due !== formatDueInput(current)) {
    payload.due = due;
  }

  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;
  try {
    await requestJSON(`/api/tasks/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
    editingTaskId = null;
    await loadTasks();
  } catch (err) {
    setError(`保存失敗: ${err.message}`);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

function handleEditKeydown(event) {
  const form = event.target.closest('.task-edit-form');
  if (!form) return;
  if (event.key === 'Escape') {
    event.preventDefault();
    editingTaskId = null;
    render();
  }
}

function setFilter(filter) {
  activeFilter = filter;
  $$('#task-filter-seg .segmented-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.filter === filter);
  });
  render();
}

function handleGlobalKeydown(event) {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
    return;
  }
  if (event.key === 'n' || event.key === 'N') {
    event.preventDefault();
    titleInput.focus();
  } else if (event.key === 'r' || event.key === 'R') {
    event.preventDefault();
    loadTasks();
  } else if (viewMode === 'calendar' && event.key === 'ArrowLeft') {
    event.preventDefault();
    moveMonth(-1);
  } else if (viewMode === 'calendar' && event.key === 'ArrowRight') {
    event.preventDefault();
    moveMonth(1);
  }
}

function init() {
  addForm.addEventListener('submit', addTask);
  refreshBtn.addEventListener('click', loadTasks);
  taskList.addEventListener('click', handleActionClick);
  taskList.addEventListener('submit', saveEdit);
  taskList.addEventListener('keydown', handleEditKeydown);
  calShell.addEventListener('click', handleCalendarClick);
  calShell.addEventListener('submit', saveEdit);
  calShell.addEventListener('keydown', handleEditKeydown);
  $('#cal-prev').addEventListener('click', () => moveMonth(-1));
  $('#cal-next').addEventListener('click', () => moveMonth(1));
  $('#cal-today-btn').addEventListener('click', () => {
    const now = new Date();
    calCursor = { y: now.getFullYear(), m: now.getMonth() };
    selectedDayKey = dayKeyOf(now);
    renderCalendar();
  });
  $$('#task-filter-seg .segmented-btn').forEach((btn) => {
    btn.addEventListener('click', () => setFilter(btn.dataset.filter));
  });
  $$('#task-view-seg .segmented-btn').forEach((btn) => {
    btn.addEventListener('click', () => setView(btn.dataset.view));
  });
  document.addEventListener('keydown', handleGlobalKeydown);
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/service-worker.js').catch((err) => {
      console.warn('[PWA] Service worker registration failed:', err);
    });
  }
  loadTasks();
}

document.addEventListener('DOMContentLoaded', init);
