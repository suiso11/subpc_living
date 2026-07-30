(() => {
  'use strict';

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/service-worker.js', { scope: '/' }).catch((error) => {
      console.warn('[PWA] Service worker registration failed:', error);
    });
  }

  const ROUTES = {
    '1': '/',
    '2': '/tasks',
    '3': '/logs',
    '4': '/achievements',
  };

  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const isEditable = (target) => {
    if (!(target instanceof Element)) return false;
    return target.matches('input, textarea, select, [contenteditable="true"]');
  };

  // 設定ダイアログなど、シェル外のモーダルが開いているかを判定する。
  // シェルのコマンドパレット自体は除外し、パレット操作中は邪魔しない。
  function settingsModalOpen() {
    if (document.querySelector('dialog[open]')) return true;
    const panel = document.querySelector('#settings-panel');
    if (!panel) return false;
    if (panel.getAttribute('aria-hidden') === 'false') return true;
    if (panel.hasAttribute('open')) return true;
    if (panel.classList && panel.classList.contains('open')) return true;
    return false;
  }

  const navigate = (path) => window.location.assign(path);

  function baseCommands() {
    return [
      { id: 'chat', icon: '●', label: '話す', detail: 'いつもの会話を続ける', shortcut: 'Alt 1', run: () => navigate('/') },
      { id: 'tasks', icon: '✓', label: 'やること', detail: '今日のタスクと最初の一歩', shortcut: 'Alt 2', run: () => navigate('/tasks') },
      { id: 'logs', icon: '≡', label: '記録', detail: '会話とシステムの履歴', shortcut: 'Alt 3', run: () => navigate('/logs') },
      { id: 'achievements', icon: '◆', label: '実績', detail: '積み上げと解除条件を見る', shortcut: 'Alt 4', run: () => navigate('/achievements') },
      { id: 'new-task', icon: '＋', label: 'タスク追加', detail: '入力欄を開いてすぐ記録する', shortcut: '', run: () => navigate('/tasks?new=1') },
    ];
  }

  function contextualCommands() {
    const commands = [];
    const messageInput = document.querySelector('#message-input');
    const taskRefresh = document.querySelector('#task-refresh-btn');
    const listView = document.querySelector('.segmented-btn[data-view="list"]');
    const calendarView = document.querySelector('.segmented-btn[data-view="calendar"]');
    const logRefresh = document.querySelector('#log-refresh-btn');

    if (messageInput) {
      commands.push({
        id: 'focus-chat', icon: '⌁', label: '入力欄に集中', detail: '会話の続きを入力する', shortcut: '/',
        run: () => messageInput.focus(),
      });
    }
    if (taskRefresh) {
      commands.push({
        id: 'refresh-tasks', icon: '↻', label: 'タスクを読み直す', detail: '最新の状態に更新する', shortcut: '',
        run: () => taskRefresh.click(),
      });
    }
    if (listView) {
      commands.push({
        id: 'task-list', icon: '☷', label: '表示をリストにする', detail: '期限ごとにタスクを並べる', shortcut: '',
        run: () => listView.click(),
      });
    }
    if (calendarView) {
      commands.push({
        id: 'task-calendar', icon: '□', label: '表示をカレンダーにする', detail: '月の予定とタスクを見る', shortcut: '',
        run: () => calendarView.click(),
      });
    }
    if (logRefresh) {
      commands.push({
        id: 'refresh-logs', icon: '↻', label: '記録を読み直す', detail: '表示中の記録を更新する', shortcut: '',
        run: () => logRefresh.click(),
      });
    }
    return commands;
  }

  function buildPalette(commands) {
    const backdrop = create('div', 'shell-command-backdrop');
    backdrop.hidden = true;
    backdrop.setAttribute('aria-hidden', 'true');

    const dialog = create('section', 'shell-command-dialog');
    dialog.setAttribute('role', 'dialog');
    dialog.setAttribute('aria-modal', 'true');
    dialog.setAttribute('aria-label', 'コマンドパレット');

    const head = create('div', 'shell-command-head');
    const input = create('input', 'shell-command-input');
    input.type = 'search';
    input.placeholder = '移動や操作を検索…';
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.setAttribute('role', 'searchbox');
    input.setAttribute('aria-label', 'コマンドを検索');
    input.setAttribute('aria-controls', 'shell-command-list');
    head.appendChild(input);

    const list = create('div', 'shell-command-list');
    list.id = 'shell-command-list';
    list.setAttribute('role', 'listbox');
    list.setAttribute('aria-label', 'コマンド一覧');

    const empty = create('p', 'shell-command-empty', '一致する操作がありません');
    empty.hidden = true;

    const foot = create('footer', 'shell-command-foot');
    [['↑↓', '選択'], ['Enter', '実行'], ['Esc', '閉じる']].forEach(([key, label]) => {
      const item = create('span');
      const kbd = create('kbd', '', key);
      item.append(kbd, document.createTextNode(` ${label}`));
      foot.appendChild(item);
    });

    const options = commands.map((command, index) => {
      const button = create('button', 'shell-command-option');
      button.type = 'button';
      button.id = `shell-command-${command.id}`;
      button.setAttribute('role', 'option');
      button.setAttribute('aria-selected', 'false');
      button.dataset.index = String(index);

      const icon = create('span', 'shell-command-icon', command.icon);
      icon.setAttribute('aria-hidden', 'true');
      const copy = create('span', 'shell-command-copy');
      copy.append(create('strong', '', command.label), create('small', '', command.detail));
      const shortcut = create('span', 'shell-command-shortcut', command.shortcut);
      button.append(icon, copy, shortcut);
      list.appendChild(button);
      return { command, button };
    });

    list.appendChild(empty);
    dialog.append(head, list, foot);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);

    return { backdrop, dialog, input, list, empty, options, selected: -1, restoreFocus: null };
  }

  function visibleOptions(state) {
    return state.options.filter(({ button }) => !button.hidden);
  }

  function selectOption(state, nextIndex, keepInputFocus = false) {
    const visible = visibleOptions(state);
    state.options.forEach(({ button }) => button.setAttribute('aria-selected', 'false'));
    if (!visible.length) {
      state.selected = -1;
      state.input.removeAttribute('aria-activedescendant');
      return;
    }
    const normalized = (nextIndex + visible.length) % visible.length;
    const selected = visible[normalized];
    state.selected = state.options.indexOf(selected);
    selected.button.setAttribute('aria-selected', 'true');
    state.input.setAttribute('aria-activedescendant', selected.button.id);
    selected.button.scrollIntoView({ block: 'nearest' });
    if (keepInputFocus) state.input.focus();
  }

  function filterOptions(state) {
    const query = state.input.value.trim().toLocaleLowerCase('ja');
    state.options.forEach(({ command, button }) => {
      const haystack = `${command.label} ${command.detail}`.toLocaleLowerCase('ja');
      button.hidden = Boolean(query && !haystack.includes(query));
    });
    const visible = visibleOptions(state);
    state.empty.hidden = visible.length > 0;
    selectOption(state, 0);
  }

  function openPalette(state) {
    state.restoreFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    state.input.value = '';
    state.backdrop.hidden = false;
    state.backdrop.setAttribute('aria-hidden', 'false');
    document.body.classList.add('shell-command-open');
    filterOptions(state);
    requestAnimationFrame(() => state.input.focus());
  }

  function closePalette(state) {
    if (state.backdrop.hidden) return;
    state.backdrop.hidden = true;
    state.backdrop.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('shell-command-open');
    const target = state.restoreFocus;
    state.restoreFocus = null;
    if (target?.isConnected) target.focus();
  }

  function runOption(state, item) {
    closePalette(state);
    item?.command.run();
  }

  function injectTrigger(open) {
    const actions = document.querySelector('.site-actions');
    if (!actions) return;
    const button = create('button', 'shell-command-trigger');
    button.type = 'button';
    button.title = '操作を検索（Ctrl+K）';
    button.setAttribute('aria-label', '操作を検索してコマンドを開く');
    const label = create('span', '', '操作を検索');
    const shortcut = create('kbd', '', '⌘K');
    shortcut.setAttribute('aria-label', 'Command または Control と K');
    button.append(label, shortcut);
    button.addEventListener('click', open);
    actions.insertBefore(button, actions.firstChild);
  }

  function focusPrimaryInput() {
    const target = document.querySelector('#message-input, #task-text-input');
    if (!target) return false;
    target.focus();
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return true;
  }

  function init() {
    if (!document.body) return;
    const state = buildPalette([...baseCommands(), ...contextualCommands()]);
    const open = () => openPalette(state);
    injectTrigger(open);

    state.input.addEventListener('input', () => filterOptions(state));
    state.options.forEach((item) => {
      item.button.addEventListener('click', () => runOption(state, item));
      item.button.addEventListener('pointerenter', () => {
        const visible = visibleOptions(state);
        selectOption(state, visible.indexOf(item));
      });
    });

    state.dialog.addEventListener('keydown', (event) => {
      const visible = visibleOptions(state);
      const current = state.options[state.selected];
      const currentVisible = Math.max(0, visible.indexOf(current));
      if (event.key === 'Tab') {
        event.preventDefault();
        state.input.focus();
      } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        selectOption(state, currentVisible + 1, true);
      } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        selectOption(state, currentVisible - 1, true);
      } else if (event.key === 'Enter') {
        event.preventDefault();
        runOption(state, state.options[state.selected] || visible[0]);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        closePalette(state);
      }
    });

    state.backdrop.addEventListener('pointerdown', (event) => {
      if (event.target === state.backdrop) closePalette(state);
    });

    document.addEventListener('keydown', (event) => {
      const key = event.key.toLocaleLowerCase();
      // 設定ダイアログなどシェル外のモーダルが開いているときは、
      // グローバルショートカットでフォーカスを奪ったりパレットを開かない。
      if (settingsModalOpen() && state.backdrop.hidden) {
        return;
      }
      if ((event.ctrlKey || event.metaKey) && key === 'k') {
        event.preventDefault();
        state.backdrop.hidden ? openPalette(state) : closePalette(state);
        return;
      }
      if (event.altKey && ROUTES[event.key]) {
        event.preventDefault();
        navigate(ROUTES[event.key]);
        return;
      }
      if (event.key === '/' && !isEditable(event.target) && state.backdrop.hidden) {
        if (focusPrimaryInput()) event.preventDefault();
      }
      if (event.key === 'Escape' && !state.backdrop.hidden) {
        event.preventDefault();
        closePalette(state);
      }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
