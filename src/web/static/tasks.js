const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let tasks = [];
let editingTaskId = null;
let loading = false;
let initialized = false;
let viewMode = localStorage.getItem('subpc_tasks_view') === 'calendar' ? 'calendar' : 'list';
let calCursor = null;
let selectedDayKey = null;
let previewDebounceTimer = null;
let detailsOpen = false;
// Google Calendar の予定 (upcoming.json キャッシュ由来)
let calEvents = [];
let calEventsRange = null; // {start, end} 取得済み範囲
let calWritable = false;
let calEventsLoaded = false;
let editingEventId = null;

const addForm = $('#task-add-form');
const textInput = $('#task-text-input');
const priorityInput = $('#task-priority-input');
const noteInput = $('#task-note-input');
const detailsToggle = $('#task-details-toggle');
const detailsSection = $('#task-details-section');
const previewLine = $('#task-preview-line');
const refreshBtn = $('#task-refresh-btn');
const taskList = $('#task-list');
const taskError = $('#task-error');
const metricOpen = $('#metric-open');
const metricOverdue = $('#metric-overdue');
const metricToday = $('#metric-today');
const metricHigh = $('#metric-high');
const listShell = $('#task-list-shell');
const calShell = $('#task-calendar-shell');
const calTitle = $('#cal-title');
const calMonthCount = $('#cal-month-count');
const calWeekdays = $('#cal-weekdays');
const calGrid = $('#cal-grid');
const calDayPanel = $('#cal-day-panel');

const WEEKDAY_LABELS = ['日', '月', '火', '水', '木', '金', '土'];
const DUE_DATE_CHIPS = ['今日', '明日', '明後日', '金曜', '来週月曜'];
const DUE_TIME_CHIPS = ['9:00', '12:00', '15:00', '18:00', '21:00'];
const DUE_TIME_TAIL_RE = /(\d{1,2}(?::\d{1,2}|時(?:半|\d{1,2}分?)?))\s*$/;

function dueChipsHtml() {
  const chip = (kind, value) => `<button class="due-chip" type="button" data-due-kind="${kind}" data-due-value="${value}">${value}</button>`;
  return [
    ...DUE_DATE_CHIPS.map((v) => chip('date', v)),
    '<span class="due-quick-sep" aria-hidden="true"></span>',
    ...DUE_TIME_CHIPS.map((v) => chip('time', v)),
    '<span class="due-quick-sep" aria-hidden="true"></span>',
    '<button class="due-chip clear" type="button" data-due-kind="clear">クリア</button>',
  ].join('');
}

function handleDueChipClick(event) {
  const btn = event.target.closest('.due-chip');
  if (!btn) return;
  const input = textInput;
  if (!input) return;
  const kind = btn.dataset.dueKind;
  const value = btn.dataset.dueValue || '';
  const current = input.value.trim();
  if (kind === 'clear') {
    input.value = '';
  } else if (kind === 'date') {
    const time = current.match(DUE_TIME_TAIL_RE);
    input.value = time ? `${value} ${time[1]}` : value;
  } else if (kind === 'time') {
    const datePart = current.replace(DUE_TIME_TAIL_RE, '').trim();
    input.value = datePart ? `${datePart} ${value}` : value;
  }
  input.focus();
  updatePreview();
}

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

function isTomorrow(task) {
  const due = dueDate(task);
  if (!due) return false;
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  return due.getFullYear() === tomorrow.getFullYear()
    && due.getMonth() === tomorrow.getMonth()
    && due.getDate() === tomorrow.getDate();
}

function isThisWeek(task) {
  const due = dueDate(task);
  if (!due || isToday(task) || isTomorrow(task)) return false;
  const now = new Date();
  const dayOfWeek = now.getDay();
  const daysUntilSunday = (7 - dayOfWeek) % 7 || 7;
  const endOfWeek = new Date();
  endOfWeek.setDate(endOfWeek.getDate() + daysUntilSunday);
  endOfWeek.setHours(23, 59, 59, 999);
  return due.getTime() <= endOfWeek.getTime() && !isOverdue({ due_at: task.due_at });
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
  if (priority === 'high') return 'だいじ';
  if (priority === 'low') return 'あとで';
  return 'ふつう';
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
    </div>
  `).join('');
  taskList.innerHTML = rows;
}

function groupTasksBySection() {
  const overdue = [];
  const today = [];
  const tomorrow = [];
  const thisWeek = [];
  const nextWeek = [];
  const noDue = [];

  tasks.forEach((task) => {
    if (isOverdue(task)) overdue.push(task);
    else if (isToday(task)) today.push(task);
    else if (isTomorrow(task)) tomorrow.push(task);
    else if (isThisWeek(task)) thisWeek.push(task);
    else if (task.due_at) nextWeek.push(task);
    else noDue.push(task);
  });

  const byTime = (a, b) => (dueDate(a)?.getTime() || 0) - (dueDate(b)?.getTime() || 0);
  [overdue, today, tomorrow, thisWeek, nextWeek, noDue].forEach((arr) => arr.sort(byTime));

  return { overdue, today, tomorrow, thisWeek, nextWeek, noDue };
}

function renderTaskRow(task) {
  const overdue = isOverdue(task);
  const today = isToday(task);
  const remaining = remainingText(task);
  const dueClass = overdue ? 'overdue' : today ? 'today' : '';
  const steps = Array.isArray(task.steps) ? task.steps.filter(Boolean).slice(0, 3) : [];
  const stepDone = Array.isArray(task.step_done) ? task.step_done : [];
  const chatPrompt = encodeURIComponent(
    `タスク#${task.id}「${task.title}」の最初の一歩を一緒に始めたい。まずは「${task.action_hint || '完了条件を1行で書く'}」`
  );
  return `
    <article class="task-row ${overdue ? 'overdue' : ''}" data-id="${task.id}" data-touch-x="0">
      <button class="task-check-btn" data-action="done" data-id="${task.id}" type="button" aria-label="タスク完了"></button>
      <div class="task-row-content">
        <div class="task-title-block">
          <strong>${escapeHtml(task.title)}</strong>
          ${task.note ? `<p class="task-note">${escapeHtml(task.note)}</p>` : ''}
        </div>
        ${task.action_hint ? `
          <div class="task-first-step">
            <span class="task-first-step-label">まずこれ</span>
            <span>${escapeHtml(task.action_hint)}</span>
            <a href="/?prompt=${chatPrompt}" class="task-start-link">一緒に始める</a>
          </div>
        ` : ''}
        ${steps.length ? `
          <details class="task-breakdown">
            <summary>${steps.length}ステップを見る</summary>
            <ol>${steps.map((step, index) => `
              <li>
                <label class="task-step-check">
                  <input type="checkbox" data-action="step-done" data-id="${task.id}" data-step-index="${index}" ${stepDone[index] ? 'checked' : ''}>
                  <span>${escapeHtml(step)}</span>
                </label>
              </li>
            `).join('')}</ol>
          </details>
        ` : ''}
        <div class="task-meta">
          ${task.due_at ? `<span class="due-text ${dueClass}">${escapeHtml(formatDue(task))}</span>` : '<span class="due-text">期限なし</span>'}
          ${remaining ? `<span class="remaining-text ${overdue ? 'overdue' : ''}">${escapeHtml(remaining)}</span>` : ''}
          <span class="priority-badge ${escapeHtml(task.priority)}">${escapeHtml(priorityLabel(task.priority))}</span>
        </div>
      </div>
      <button class="task-action-menu" data-action="menu" data-id="${task.id}" type="button" aria-label="操作メニュー">⋯</button>
      <div class="task-actions-expanded" hidden>
        <button class="action-item edit" data-action="edit" data-id="${task.id}" type="button">編集</button>
        <button class="action-item" data-action="regenerate" data-id="${task.id}" type="button">手順を作り直す</button>
        <button class="action-item" data-action="snooze" data-when="30m" data-id="${task.id}" type="button">+30分</button>
        <button class="action-item" data-action="snooze-tomorrow" data-id="${task.id}" type="button">明日へ</button>
        <button class="action-item danger" data-action="drop" data-id="${task.id}" type="button">削除</button>
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
          <input name="due" type="text" value="${escapeHtml(formatDue(task))}" placeholder="期限 例: 明日 18時 / 金曜">
          <select name="priority" aria-label="優先度">
            <option value="normal" ${task.priority === 'normal' ? 'selected' : ''}>ふつう</option>
            <option value="high" ${task.priority === 'high' ? 'selected' : ''}>だいじ</option>
            <option value="low" ${task.priority === 'low' ? 'selected' : ''}>あとで</option>
          </select>
          <input name="note" type="text" value="${escapeHtml(task.note || '')}" placeholder="メモ (任意)">
          <input name="action_hint" type="text" value="${escapeHtml(task.action_hint || '')}" placeholder="最初の一歩 (5分以内)">
          <div class="due-quick compact">${dueChipsHtml()}</div>
          <div class="edit-actions">
            <button class="primary-btn compact" type="submit">保存</button>
            <button class="secondary-btn compact" data-action="cancel-edit" type="button">キャンセル</button>
          </div>
        </div>
      </form>
    </article>
  `;
}

function render() {
  syncViewUI();
  if (viewMode === 'calendar') {
    renderCalendar();
    return;
  }

  if (!tasks.length) {
    taskList.innerHTML = `
      <div class="task-empty-row">
        <span class="empty-title">まだ何もありません</span>
        <span class="empty-hint">思いついたら上から追加できます。</span>
      </div>
    `;
    return;
  }

  const sections = groupTasksBySection();
  const html = [];

  const sectionDef = [
    { key: 'overdue', label: '超過', list: sections.overdue },
    { key: 'today', label: '今日', list: sections.today },
    { key: 'tomorrow', label: '明日', list: sections.tomorrow },
    { key: 'thisWeek', label: '今週', list: sections.thisWeek },
    { key: 'nextWeek', label: '来週以降', list: sections.nextWeek },
    { key: 'noDue', label: '期限なし', list: sections.noDue },
  ];

  sectionDef.forEach(({ label, list }) => {
    if (!list.length) return;
    html.push(`
      <div class="task-section">
        <div class="task-section-header">
          <span class="section-title">${label}</span>
          <span class="section-count">${list.length}件</span>
        </div>
        ${list.map((task) => (
          editingTaskId === task.id ? renderEditRow(task) : renderTaskRow(task)
        )).join('')}
      </div>
    `);
  });

  taskList.innerHTML = html.length ? html.join('') : `
    <div class="task-empty-row">
      <span class="empty-title">ここにはまだありません</span>
      <span class="empty-hint">別の日を見るか、新しく追加してみましょう。</span>
    </div>
  `;
}

async function updatePreview() {
  const text = textInput.value.trim();
  if (!text) {
    previewLine.hidden = true;
    previewLine.innerHTML = '';
    return;
  }

  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(async () => {
    try {
      const result = await requestJSON('/api/tasks/preview', {
        method: 'POST',
        body: JSON.stringify({ text }),
      });
      const parts = [];
      if (result.title) parts.push(escapeHtml(result.title));
      if (result.due_display) parts.push(escapeHtml(result.due_display));
      if (result.priority && result.priority !== 'normal') {
        parts.push(priorityLabel(result.priority));
      }
      if (parts.length) {
        previewLine.innerHTML = `<span class="preview-text">→ ${parts.join(' ・ ')}</span>`;
        previewLine.hidden = false;
      } else {
        previewLine.hidden = true;
      }
    } catch (_) {
      // fetch失敗は静かに無視
    }
  }, 300);
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

async function addTask(event) {
  event.preventDefault();
  const text = textInput.value.trim();
  if (!text) return;
  setError('');
  const submitBtn = addForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  try {
    const body = { text };
    if (priorityInput.value !== 'normal') {
      body.priority = priorityInput.value;
    }
    if (noteInput.value.trim()) {
      body.note = noteInput.value.trim();
    }
    await requestJSON('/api/tasks', {
      method: 'POST',
      body: JSON.stringify(body),
    });
    addForm.reset();
    priorityInput.value = 'normal';
    noteInput.value = '';
    detailsOpen = false;
    detailsSection.hidden = true;
    detailsToggle.classList.remove('active');
    detailsToggle.textContent = 'こまかく';
    previewLine.hidden = true;
    await loadTasks();
    textInput.focus();
  } catch (err) {
    setError(`追加失敗: ${err.message}`);
  } finally {
    submitBtn.disabled = false;
  }
}

function handleMenuClick(event) {
  const btn = event.target.closest('button[data-action="menu"]');
  if (!btn) return;
  const row = btn.closest('.task-row');
  const expanded = row.querySelector('.task-actions-expanded');
  if (!expanded) return;
  expanded.hidden = !expanded.hidden;
}

async function handleActionClick(event) {
  const btn = event.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  const id = Number(btn.dataset.id);
  const previousChecked = action === 'step-done' ? !btn.checked : null;

  if (action === 'menu') {
    handleMenuClick(event);
    return;
  }

  if (action === 'edit') {
    editingTaskId = id;
    render();
    setTimeout(() => {
      const input = taskList.querySelector(`.task-edit-form[data-id="${id}"] input[name="title"]`);
      if (input) {
        input.focus();
        input.setSelectionRange(input.value.length, input.value.length);
      }
    }, 0);
    return;
  }

  if (action === 'cancel-edit') {
    editingTaskId = null;
    render();
    return;
  }

  if (action === 'drop' && !confirm('この「やること」を削除しますか？')) {
    return;
  }

  btn.disabled = true;
  setError('');
  try {
    if (action === 'done') {
      const row = btn.closest('.task-row');
      row.classList.add('fading-out');
      await new Promise(r => setTimeout(r, 200));
      await requestJSON(`/api/tasks/${id}/done`, { method: 'POST', body: '{}' });
    } else if (action === 'drop') {
      await requestJSON(`/api/tasks/${id}/drop`, { method: 'POST', body: '{}' });
    } else if (action === 'snooze') {
      await requestJSON(`/api/tasks/${id}/snooze`, {
        method: 'POST',
        body: JSON.stringify({ when: btn.dataset.when }),
      });
    } else if (action === 'snooze-tomorrow') {
      await requestJSON(`/api/tasks/${id}/snooze`, {
        method: 'POST',
        body: JSON.stringify({ when: '明日' }),
      });
    } else if (action === 'regenerate') {
      await requestJSON(`/api/tasks/${id}/breakdown`, {
        method: 'POST',
        body: '{}',
      });
    } else if (action === 'step-done') {
      const stepIndex = Number(btn.dataset.stepIndex);
      await requestJSON(`/api/tasks/${id}/steps/${stepIndex}`, {
        method: 'POST',
        body: JSON.stringify({ done: btn.checked }),
      });
      const task = tasks.find((item) => item.id === id);
      if (task) {
        if (!Array.isArray(task.step_done)) task.step_done = [];
        task.step_done[stepIndex] = btn.checked;
      }
      return;
    }
    await loadTasks();
  } catch (err) {
    if (action === 'step-done') btn.checked = previousChecked;
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
    action_hint: String(data.get('action_hint') || '').trim(),
  };
  if (!payload.title) {
    setError('タイトルは必須です');
    return;
  }
  if (due) {
    payload.due = due;
  }

  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;
  setError('');
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

function syncViewUI() {
  $$('#task-view-seg .segmented-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === viewMode);
  });
  const isList = viewMode === 'list';
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
    if (!calEventsLoaded) {
      loadCalendarEvents().then(() => {
        if (viewMode === 'calendar') renderCalendar();
      });
    }
  }
  render();
}

function dayKeyOf(date) {
  const mm = String(date.getMonth() + 1).padStart(2, '0');
  const dd = String(date.getDate()).padStart(2, '0');
  return `${date.getFullYear()}-${mm}-${dd}`;
}

// --- Google Calendar 予定 ---

async function loadCalendarEvents() {
  try {
    const data = await requestJSON('/api/calendar/events');
    calEvents = (data.events || []).filter((e) => e && e.start);
    calEventsRange = data.range && data.range.start ? data.range : null;
    calWritable = !!data.writable;
    calEventsLoaded = true;
  } catch (err) {
    // 予定が取れなくてもタスク表示は続行する
    calEvents = [];
    calEventsLoaded = true;
  }
}

function eventDayKey(ev) {
  return String(ev.start || '').slice(0, 10);
}

function eventTimeLabel(ev) {
  const start = String(ev.start || '');
  if (!start.includes('T')) return '終日';
  const d = new Date(start);
  if (Number.isNaN(d.getTime())) return '';
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function eventsByDay() {
  const map = new Map();
  calEvents.forEach((ev) => {
    const key = eventDayKey(ev);
    if (!key) return;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(ev);
  });
  map.forEach((list) => list.sort((a, b) => String(a.start).localeCompare(String(b.start))));
  return map;
}

function calEventChip(ev) {
  const label = `${eventTimeLabel(ev) === '終日' ? '' : eventTimeLabel(ev) + ' '}${ev.title}`;
  return `<span class="cal-chip gcal" title="${escapeHtml(ev.title)}">${escapeHtml(label)}</span>`;
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
  const evByDay = eventsByDay();
  const monthPrefix = `${y}-${String(m + 1).padStart(2, '0')}`;
  const monthCount = tasks.filter((task) => {
    const due = dueDate(task);
    return due && due.getFullYear() === y && due.getMonth() === m;
  }).length;
  const monthEvCount = calEvents.filter((ev) => eventDayKey(ev).startsWith(monthPrefix)).length;
  const countParts = [];
  if (monthCount) countParts.push(`やること ${monthCount}件`);
  if (monthEvCount) countParts.push(`予定${monthEvCount}件`);
  // 取得済み範囲外の月は「予定未取得」と明示する (キャッシュは過去2週〜先45日)
  if (calEventsRange
      && (`${monthPrefix}-31` < calEventsRange.start || `${monthPrefix}-01` > calEventsRange.end)) {
    countParts.push('予定は未取得の範囲');
  }
  calMonthCount.textContent = countParts.join(' / ');

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
    const dayEvents = evByDay.get(key) || [];
    const cls = ['cal-cell'];
    if (d.getMonth() !== m) cls.push('out');
    if (key === todayKey) cls.push('today');
    if (key === selectedDayKey) cls.push('selected');
    if (d.getDay() === 0) cls.push('sun');
    if (d.getDay() === 6) cls.push('sat');
    // 予定 (Google Calendar) を先に、タスクを後に。チップは合計3件まで。
    const chipItems = [
      ...dayEvents.map((ev) => calEventChip(ev)),
      ...dayTasks.map((task) => calChip(task)),
    ];
    const chips = chipItems.slice(0, 3).join('');
    const moreCount = chipItems.length - 3;
    const more = moreCount > 0 ? `<span class="cal-more">+${moreCount}件</span>` : '';
    const dotItems = [
      ...dayEvents.map(() => '<i class="cal-dot gcal"></i>'),
      ...dayTasks.map((task) => calDot(task)),
    ];
    const dots = dotItems.length
      ? `<span class="cal-dots">${dotItems.slice(0, 4).join('')}</span>`
      : '';
    cells.push(`
      <button class="${cls.join(' ')}" type="button" data-day="${key}" aria-label="${key} やること${dayTasks.length}件 予定${dayEvents.length}件">
        <span class="cal-date">${d.getDate()}</span>
        ${chips}${more}${dots}
      </button>
    `);
  }
  calGrid.innerHTML = cells.join('');
  renderDayPanel(byDay, evByDay);
}

function renderEventRow(ev) {
  const time = eventTimeLabel(ev);
  const location = ev.location ? `<span class="task-muted"> @${escapeHtml(ev.location)}</span>` : '';
  const actions = calWritable ? `
    <span class="cal-event-actions">
      <button class="secondary-btn compact" type="button" data-event-action="edit" data-event-id="${escapeHtml(ev.event_id)}">編集</button>
      <button class="secondary-btn compact" type="button" data-event-action="delete" data-event-id="${escapeHtml(ev.event_id)}">削除</button>
    </span>` : '';
  return `
    <div class="cal-event-row">
      <span class="cal-event-time">${escapeHtml(time)}</span>
      <span class="cal-event-title">📅 ${escapeHtml(ev.title)}${location}</span>
      ${actions}
    </div>
  `;
}

function renderEventEditRow(ev) {
  const time = eventTimeLabel(ev);
  return `
    <form class="cal-event-row editing cal-event-edit-form" data-event-id="${escapeHtml(ev.event_id)}">
      <input name="time" type="text" value="${time === '終日' ? '' : escapeHtml(time)}" placeholder="HH:MM (空=終日)" class="cal-event-time-input">
      <input name="title" type="text" value="${escapeHtml(ev.title)}" placeholder="予定名" required>
      <span class="cal-event-actions">
        <button class="primary-btn compact" type="submit">保存</button>
        <button class="secondary-btn compact" type="button" data-event-action="cancel-edit">キャンセル</button>
      </span>
    </form>
  `;
}

function renderDayPanel(byDay, evByDay) {
  if (!selectedDayKey) {
    calDayPanel.hidden = true;
    calDayPanel.innerHTML = '';
    return;
  }
  const [yy, mm, dd] = selectedDayKey.split('-').map(Number);
  const d = new Date(yy, mm - 1, dd);
  const heading = `${mm}/${dd} (${WEEKDAY_LABELS[d.getDay()]})`;
  const dayTasks = byDay.get(selectedDayKey) || [];
  const dayEvents = (evByDay && evByDay.get(selectedDayKey)) || [];

  const eventRows = dayEvents.map((ev) => (
    editingEventId && ev.event_id === editingEventId ? renderEventEditRow(ev) : renderEventRow(ev)
  )).join('');
  const eventBlock = dayEvents.length
    ? `<div class="cal-event-list">${eventRows}</div>`
    : '';

  const body = dayTasks.length
    ? dayTasks.map((task) => (
      editingTaskId === task.id ? renderEditRow(task) : renderTaskRow(task)
    )).join('')
    : (dayEvents.length ? '' : '<div class="task-empty-row"><span class="empty-hint">この日はまだ何もありません</span></div>');
  const addEventBtn = calWritable
    ? `<button class="secondary-btn compact" type="button" data-add-event-day="${selectedDayKey}">＋予定</button>`
    : '';
  calDayPanel.hidden = false;
  calDayPanel.innerHTML = `
    <div class="cal-day-head">
      <span>${escapeHtml(heading)} <span class="task-muted">やること ${dayTasks.length}件・予定 ${dayEvents.length}件</span></span>
      <span class="cal-day-head-actions">
        ${addEventBtn}
        <button class="secondary-btn compact" type="button" data-add-day="${selectedDayKey}">＋やること</button>
      </span>
    </div>
    ${eventBlock}
    ${body}
  `;
}

function handleCalendarClick(event) {
  if (event.target.closest('button[data-event-action]')) {
    handleEventActionClick(event);
    return;
  }
  if (event.target.closest('button[data-action]')) {
    handleActionClick(event);
    return;
  }
  const addEventBtn = event.target.closest('button[data-add-event-day]');
  if (addEventBtn) {
    handleCalendarDayAddEvent(addEventBtn);
    return;
  }
  const addBtn = event.target.closest('button[data-add-day]');
  if (addBtn) {
    handleCalendarDayAdd(addBtn);
    return;
  }
  const cell = event.target.closest('.cal-cell[data-day]');
  if (cell) {
    selectedDayKey = cell.dataset.day;
    editingEventId = null;
    renderCalendar();
  }
}

async function handleEventActionClick(event) {
  const btn = event.target.closest('button[data-event-action]');
  if (!btn) return;
  const action = btn.dataset.eventAction;
  const id = btn.dataset.eventId || '';

  if (action === 'edit') {
    editingEventId = id;
    renderCalendar();
    return;
  }
  if (action === 'cancel-edit') {
    editingEventId = null;
    renderCalendar();
    return;
  }
  if (action === 'delete') {
    const ev = calEvents.find((e) => e.event_id === id);
    if (!confirm(`予定「${ev ? ev.title : ''}」を Google カレンダーから削除しますか？`)) return;
    btn.disabled = true;
    setError('');
    try {
      await requestJSON(`/api/calendar/events/${encodeURIComponent(id)}`, { method: 'DELETE' });
      await loadCalendarEvents();
      renderCalendar();
    } catch (err) {
      setError(`予定の削除失敗: ${err.message}`);
      btn.disabled = false;
    }
  }
}

async function saveEventEdit(event) {
  const form = event.target.closest('.cal-event-edit-form');
  if (!form) return;
  event.preventDefault();
  const id = form.dataset.eventId;
  const data = new FormData(form);
  const title = String(data.get('title') || '').trim();
  const time = String(data.get('time') || '').trim();
  if (!title) {
    setError('予定名は必須です');
    return;
  }
  const payload = { title, date: selectedDayKey, time };
  // 時刻付き予定は元の長さを維持する (未指定だとサーバー既定の60分になる)
  const current = calEvents.find((e) => e.event_id === id);
  if (time && current && String(current.start).includes('T') && String(current.end).includes('T')) {
    const dur = (new Date(current.end) - new Date(current.start)) / 60000;
    if (Number.isFinite(dur) && dur > 0) payload.duration_min = Math.round(dur);
  }
  const submitBtn = form.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.disabled = true;
  setError('');
  try {
    await requestJSON(`/api/calendar/events/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    });
    editingEventId = null;
    await loadCalendarEvents();
    renderCalendar();
  } catch (err) {
    setError(`予定の保存失敗: ${err.message}`);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

function handleCalendarDayAddEvent(btn) {
  const dayKey = btn.dataset.addEventDay;
  const panel = btn.closest('.cal-day-head').parentElement;
  const existing = panel.querySelector('.cal-day-add-input');
  if (existing) {
    existing.remove();
    return;
  }
  const wrapper = document.createElement('div');
  wrapper.innerHTML = `
    <div class="cal-day-add-input">
      <input type="text" placeholder="予定を追加 (例: 15:00 歯医者 / 飲み会)" class="cal-day-input">
    </div>
  `;
  const input = wrapper.querySelector('.cal-day-input');
  btn.closest('.cal-day-head').insertAdjacentElement('afterend', wrapper.firstElementChild);

  input.focus();
  input.addEventListener('keydown', async (e) => {
    if (e.key === 'Enter') {
      const text = input.value.trim();
      if (!text) return;
      const m = text.match(/^(\d{1,2}):(\d{2})\s+(.+)$/);
      const payload = m
        ? { title: m[3].trim(), date: dayKey, time: `${m[1].padStart(2, '0')}:${m[2]}` }
        : { title: text, date: dayKey, time: '' };
      setError('');
      try {
        await requestJSON('/api/calendar/events', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
        await loadCalendarEvents();
        renderCalendar();
      } catch (err) {
        setError(`予定の追加失敗: ${err.message}`);
      }
    } else if (e.key === 'Escape') {
      input.parentElement.remove();
    }
  });
  input.addEventListener('blur', () => {
    if (input.parentElement) {
      input.parentElement.remove();
    }
  });
}

function handleCalendarDayAdd(btn) {
  const dayKey = btn.dataset.addDay;
  const [yy, mm, dd] = dayKey.split('-').map(Number);
  const datePrefix = `${mm}/${dd}`;

  const inputHtml = `
    <div class="cal-day-add-input">
      <input type="text" placeholder="この日のやることを追加" class="cal-day-input" data-date-prefix="${datePrefix}">
    </div>
  `;
  const panel = btn.closest('.cal-day-head').parentElement;
  const existing = panel.querySelector('.cal-day-add-input');
  if (existing) {
    existing.remove();
    return;
  }

  const wrapper = document.createElement('div');
  wrapper.innerHTML = inputHtml;
  const input = wrapper.querySelector('.cal-day-input');
  btn.parentElement.appendChild(wrapper.firstElementChild);
  input.focus();

  input.addEventListener('keydown', async (e) => {
    if (e.key === 'Enter') {
      const text = input.value.trim();
      if (!text) return;
      const datePrefix = input.dataset.datePrefix;
      setError('');
      try {
        await requestJSON('/api/tasks', {
          method: 'POST',
          body: JSON.stringify({ text: `${datePrefix} ${text}` }),
        });
        await loadTasks();
        setView('calendar');
      } catch (err) {
        setError(`追加失敗: ${err.message}`);
      }
    } else if (e.key === 'Escape') {
      input.parentElement.remove();
    }
  });

  input.addEventListener('blur', () => {
    if (input.parentElement) {
      input.parentElement.remove();
    }
  });
}

function handleTaskSwipe(event) {
  const row = event.target.closest('.task-row:not(.editing)');
  if (!row || row.classList.contains('editing')) return;

  const startX = event.touches[0].clientX;
  const startY = event.touches[0].clientY;
  let currentX = startX;

  function handleTouchMove(moveEvent) {
    currentX = moveEvent.touches[0].clientX;
    const deltaX = currentX - startX;
    const deltaY = Math.abs(moveEvent.touches[0].clientY - startY);

    if (deltaY > Math.abs(deltaX)) {
      row.removeEventListener('touchmove', handleTouchMove);
      return;
    }

    row.style.transform = `translateX(${deltaX}px)`;
    row.dataset.touchX = deltaX;
  }

  function handleTouchEnd() {
    row.removeEventListener('touchmove', handleTouchMove);
    row.removeEventListener('touchend', handleTouchEnd);

    const deltaX = parseInt(row.dataset.touchX || 0);
    const threshold = 60;

    if (deltaX > threshold) {
      const checkBtn = row.querySelector('button[data-action="done"]');
      if (checkBtn) checkBtn.click();
      row.style.transform = '';
    } else if (deltaX < -threshold) {
      const menuBtn = row.querySelector('button[data-action="menu"]');
      if (menuBtn) menuBtn.click();
      row.style.transform = '';
    } else {
      row.style.transform = '';
    }
    row.dataset.touchX = 0;
  }

  row.addEventListener('touchmove', handleTouchMove);
  row.addEventListener('touchend', handleTouchEnd);
}

function handleGlobalKeydown(event) {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA' || event.target.tagName === 'SELECT') {
    return;
  }
  if (event.key === 'n' || event.key === 'N') {
    event.preventDefault();
    textInput.focus();
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
  $('#due-quick').innerHTML = dueChipsHtml();

  document.addEventListener('click', handleDueChipClick);
  addForm.addEventListener('submit', addTask);
  textInput.addEventListener('input', updatePreview);

  detailsToggle.addEventListener('click', (e) => {
    e.preventDefault();
    detailsOpen = !detailsOpen;
    detailsSection.hidden = !detailsOpen;
    detailsToggle.classList.toggle('active', detailsOpen);
    detailsToggle.textContent = detailsOpen ? 'とじる' : 'こまかく';
  });

  refreshBtn.addEventListener('click', async () => {
    await Promise.all([loadTasks(), loadCalendarEvents()]);
    render();
  });
  taskList.addEventListener('click', handleActionClick);
  taskList.addEventListener('submit', saveEdit);
  taskList.addEventListener('keydown', handleEditKeydown);
  taskList.addEventListener('touchstart', handleTaskSwipe);

  calShell.addEventListener('click', handleCalendarClick);
  calShell.addEventListener('submit', (e) => {
    if (e.target.closest('.cal-event-edit-form')) {
      saveEventEdit(e);
    } else {
      saveEdit(e);
    }
  });
  calShell.addEventListener('keydown', handleEditKeydown);

  $('#cal-prev').addEventListener('click', () => moveMonth(-1));
  $('#cal-next').addEventListener('click', () => moveMonth(1));
  $('#cal-today-btn').addEventListener('click', () => {
    const now = new Date();
    calCursor = { y: now.getFullYear(), m: now.getMonth() };
    selectedDayKey = dayKeyOf(now);
    renderCalendar();
  });

  $$('#task-view-seg .segmented-btn').forEach((btn) => {
    btn.addEventListener('click', () => setView(btn.dataset.view));
  });

  document.addEventListener('keydown', handleGlobalKeydown);


  loadTasks();

  const routeParams = new URLSearchParams(window.location.search);
  if (routeParams.get('new') === '1') {
    textInput.focus();
    textInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    window.history.replaceState({}, '', window.location.pathname);
  }
}

document.addEventListener('DOMContentLoaded', init);
