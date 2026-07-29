// ============================================
//  subpc_living Web UI — Frontend JS
//  WebSocket ストリーミングチャット + TTS + STT
// ============================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// --- 状態 ---
let ws = null;
let isStreaming = false;
let sessionId = null;
let currentAudio = null;
let sttAvailable = false;
let micUnavailableReason = '';
let secureWebUrl = '';

const SESSION_STORAGE_KEY = 'subpc_chat_session_id';

// --- 音声録音状態 ---
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let recordingStream = null;

// --- DOM要素 ---
const chatArea = $('#chat-area');
const messageInput = $('#message-input');
const sendBtn = $('#send-btn');
const statusDot = $('#status-dot');
const ttsToggle = $('#tts-toggle');
const appShell = $('.app');
const settingsPanel = $('#settings-panel');
const voiceSelect = $('#voice-select');
const micBtn = $('#mic-btn');
const sttStatus = $('#stt-status');
const growthPanel = $('#growth-panel');
const growthProgress = growthPanel?.querySelector('.growth-progress');
let lastGrowthPoints = null;
const gameHub = $('#game-hub');
const gameToggle = $('#game-toggle');
const gameDetails = $('#game-details');
const GAME_PANEL_KEY = 'subpc_game_panel_open';
let lastClaimableMissions = null;
let settingsRestoreFocus = null;
const runtimeStatus = { ollama: null, stt: null, tts: null, rag: null, websocket: 'connecting' };

function formatCount(value) {
  return Number(value || 0).toLocaleString('ja-JP');
}

function statusWord(value) {
  if (value === true) return 'ONLINE';
  if (value === false) return 'OFFLINE';
  return 'CHECKING';
}

function updateInstrumentStatus(next = {}) {
  Object.assign(runtimeStatus, next);
  const labels = {
    'instrument-ollama': `LOCAL · OLLAMA · ${statusWord(runtimeStatus.ollama)}`,
    'instrument-stt': `VOICE · STT · ${statusWord(runtimeStatus.stt)}`,
    'instrument-tts': `VOICE · TTS · ${statusWord(runtimeStatus.tts)}`,
    'instrument-rag': `MEMORY · RAG · ${statusWord(runtimeStatus.rag)}`,
  };
  Object.entries(labels).forEach(([id, text]) => {
    const node = document.getElementById(id);
    if (node) node.textContent = text;
  });
  const caption = document.getElementById('instrument-caption');
  if (caption) {
    caption.textContent = runtimeStatus.websocket === 'connected'
      ? 'CONNECTED'
      : runtimeStatus.websocket === 'disconnected' ? 'DISCONNECTED' : 'CONNECTING';
  }
  const wsStatus = document.getElementById('ws-status');
  if (wsStatus) {
    wsStatus.textContent = runtimeStatus.websocket === 'connected'
      ? 'つながっています'
      : runtimeStatus.websocket === 'disconnected' ? '接続がきれました' : 'つないでいます…';
  }
}

function createWelcome(title, description) {
  const welcome = document.createElement('div');
  welcome.className = 'welcome';
  const apparatus = document.createElement('div');
  apparatus.className = 'signal-apparatus';
  apparatus.setAttribute('role', 'group');
  apparatus.setAttribute('aria-label', 'ローカル実行環境の接続情報');
  const labelData = [
    ['one', 'instrument-ollama'],
    ['two', 'instrument-stt'],
    ['three', 'instrument-tts'],
    ['four', 'instrument-rag'],
  ];
  labelData.forEach(([className, id]) => {
    const label = document.createElement('span');
    label.className = `signal-label ${className}`;
    label.id = id;
    apparatus.appendChild(label);
  });
  const readout = document.createElement('span');
  readout.className = 'signal-readout';
  readout.setAttribute('aria-hidden', 'true');
  const caption = document.createElement('span');
  caption.className = 'signal-caption';
  caption.id = 'instrument-caption';
  apparatus.append(readout, caption);
  const heading = document.createElement('h2');
  heading.textContent = title;
  const copy = document.createElement('p');
  copy.textContent = description;
  const link = document.createElement('a');
  link.className = 'welcome-link';
  link.href = '/tasks';
  link.textContent = 'やることを見る →';
  welcome.append(apparatus, heading, copy, link);
  return welcome;
}

async function loadGrowth({ animate = false } = {}) {
  if (!growthPanel) return;
  try {
    const resp = await fetch('/api/growth?days=14', { cache: 'no-store' });
    if (!resp.ok) throw new Error(`growth ${resp.status}`);
    const data = await resp.json();
    if (!data.enabled) throw new Error('growth disabled');
    growthPanel.classList.remove('growth-unavailable');

    const points = Number(data.growth_points || 0);
    const previous = lastGrowthPoints;
    lastGrowthPoints = points;
    $('#growth-level').textContent = `Lv.${data.level || 1}`;
    $('#growth-points').textContent = formatCount(points);
    $('#growth-today').textContent = `+${formatCount(data.today_points)}`;
    $('#growth-streak').textContent = formatCount(data.streak_days);
    $('#growth-memory').textContent = formatCount(data.asset_counts?.retrievable_memories);
    $('#growth-corrections').textContent = formatCount(data.asset_counts?.correction_candidates);
    growthPanel.title = data.metric_note || growthPanel.title;

    const progress = Math.max(0, Math.min(100, Number(data.level_progress || 0)));
    $('#growth-progress-bar').style.width = `${progress}%`;
    if (growthProgress) growthProgress.setAttribute('aria-valuenow', String(progress));

    const daily = Array.isArray(data.daily) ? data.daily : [];
    const maxPoints = Math.max(1, ...daily.map((day) => Number(day.points || 0)));
    const chart = $('#growth-chart');
    chart.replaceChildren(...daily.map((day) => {
      const bar = document.createElement('span');
      const value = Number(day.points || 0);
      bar.style.height = `${Math.max(value > 0 ? 18 : 4, Math.round(value / maxPoints * 100))}%`;
      bar.className = value > 0 ? 'active' : '';
      bar.title = `${day.date}：${value} pt・${day.turns || 0}往復`;
      return bar;
    }));

    if (animate && previous !== null && points > previous) {
      const delta = points - previous;
      const badge = $('#growth-delta');
      badge.textContent = `+${formatCount(delta)}`;
      badge.hidden = false;
      growthPanel.classList.remove('growth-pulse');
      void growthPanel.offsetWidth;
      growthPanel.classList.add('growth-pulse');
      setTimeout(() => { badge.hidden = true; }, 2400);
    }
  } catch (e) {
    growthPanel.classList.add('growth-unavailable');
    console.warn('[Growth] Fetch failed:', e);
  }
}

function setGameOpen(open) {
  if (!gameDetails || !gameToggle) return;
  gameDetails.hidden = !open;
  gameToggle.setAttribute('aria-expanded', String(open));
  $('#game-toggle-icon').textContent = open ? '−' : '＋';
  try { localStorage.setItem(GAME_PANEL_KEY, open ? 'open' : 'closed'); } catch (e) {}
}

function missionCard(mission) {
  const card = document.createElement('article');
  card.className = `mission-card${mission.complete ? ' complete' : ''}${mission.claimed ? ' claimed' : ''}`;

  const name = document.createElement('strong');
  name.textContent = mission.name;
  const detail = document.createElement('span');
  detail.className = 'mission-detail';
  detail.textContent = mission.detail;

  const progress = document.createElement('div');
  progress.className = 'mission-progress';
  progress.setAttribute('aria-label', `${mission.current}/${mission.target}`);
  const progressBar = document.createElement('span');
  progressBar.style.width = `${Math.min(100, mission.current / mission.target * 100)}%`;
  progress.appendChild(progressBar);

  const action = document.createElement('button');
  action.type = 'button';
  action.className = mission.complete && !mission.claimed
    ? 'primary-btn compact mission-claim'
    : 'secondary-btn compact mission-claim';
  if (mission.claimed) {
    action.textContent = '受取済み';
    action.disabled = true;
  } else if (mission.complete) {
    action.textContent = `+${mission.reward} pt 受け取る`;
    action.addEventListener('click', () => claimMission(mission.id, action));
  } else {
    action.textContent = `${mission.current} / ${mission.target}`;
    action.disabled = true;
  }

  card.append(name, detail, progress, action);
  return card;
}

function badgeCard(badge) {
  const card = document.createElement('div');
  card.className = `badge-card${badge.unlocked ? ' unlocked' : ' locked'}`;
  card.title = badge.detail;
  const mark = document.createElement('span');
  mark.className = 'badge-mark';
  mark.textContent = badge.unlocked ? badge.mark : '？';
  const name = document.createElement('strong');
  name.textContent = badge.name;
  card.append(mark, name);
  return card;
}

function starterButton(starter) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'starter-btn';
  button.textContent = starter.label;
  button.addEventListener('click', () => {
    messageInput.value = starter.prompt;
    messageInput.dispatchEvent(new Event('input'));
    setGameOpen(false);
    messageInput.focus();
  });
  return button;
}

function renderGame(data, { animate = false } = {}) {
  if (!gameHub || !data?.enabled) return;
  gameHub.classList.remove('game-unavailable');
  $('#game-rank').textContent = `相棒ランク：${data.rank.name}`;
  const claimable = Number(data.claimable_missions || 0);
  $('#game-quest-summary').textContent = claimable > 0
    ? `報酬 ${claimable}個` : `${data.completed_missions || 0} / 3 達成`;
  $('#game-next-rank').textContent = data.rank.next
    ? `次は Lv.${data.rank.next.level}「${data.rank.next.name}」` : '最高ランク';
  $('#game-badge-summary').textContent = `${data.unlocked_badges || 0} / ${data.badges.length}`;
  $('#game-mission-list').replaceChildren(...data.missions.map(missionCard));
  $('#game-badge-list').replaceChildren(...data.badges.map(badgeCard));
  $('#game-starter-list').replaceChildren(...data.starters.map(starterButton));

  if (animate && lastClaimableMissions !== null && claimable > lastClaimableMissions) {
    gameHub.classList.remove('game-ready');
    void gameHub.offsetWidth;
    gameHub.classList.add('game-ready');
  }
  lastClaimableMissions = claimable;
}

async function loadGame({ animate = false } = {}) {
  if (!gameHub) return;
  try {
    const resp = await fetch('/api/game', { cache: 'no-store' });
    if (!resp.ok) throw new Error(`game ${resp.status}`);
    renderGame(await resp.json(), { animate });
  } catch (e) {
    gameHub.classList.add('game-unavailable');
    console.warn('[Game] Fetch failed:', e);
  }
}

async function claimMission(missionId, button) {
  button.disabled = true;
  try {
    const resp = await fetch('/api/game/claim', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mission_id: missionId }),
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || `claim ${resp.status}`);
    renderGame(data.state);
    const toast = $('#game-toast');
    toast.textContent = data.claimed_now
      ? `報酬 +${data.reward} pt を受け取りました！` : 'この報酬は受取済みです。';
    gameHub.classList.remove('game-reward');
    void gameHub.offsetWidth;
    gameHub.classList.add('game-reward');
    await loadGrowth({ animate: true });
  } catch (e) {
    $('#game-toast').textContent = '報酬を受け取れませんでした。もう一度お試しください。';
    button.disabled = false;
    console.warn('[Game] Claim failed:', e);
  }
}

// ============================================
//  WebSocket 接続
// ============================================

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/chat`;

  statusDot.className = 'status-dot';
  updateInstrumentStatus({ websocket: 'connecting' });

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusDot.className = 'status-dot connected';
    updateInstrumentStatus({ websocket: 'connected' });
    console.log('[WS] Connected');
  };

  ws.onclose = () => {
    statusDot.className = 'status-dot error';
    updateInstrumentStatus({ websocket: 'disconnected' });
    console.log('[WS] Disconnected, reconnecting in 3s...');
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    statusDot.className = 'status-dot error';
    updateInstrumentStatus({ websocket: 'disconnected' });
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleMessage(data);
  };
}

// ============================================
//  メッセージ処理
// ============================================

function handleMessage(data) {
  switch (data.type) {
    case 'token':
      appendToken(data.content);
      break;

    case 'done':
      finishResponse(data.full_text);
      break;

    case 'audio':
      playAudio(data.data);
      break;

    case 'stt_result':
      handleSTTResult(data);
      break;

    case 'error':
      showError(data.message);
      isStreaming = false;
      setMicState('idle');
      updateUI();
      break;
  }
}

function handleSTTResult(data) {
  if (data.text) {
    // 認識テキストをステータスに表示
    sttStatus.textContent = `🎤 "${data.text}"`;
    setTimeout(() => { sttStatus.textContent = ''; }, 5000);

    // 「🎤 (音声入力)」プレースホルダーを認識テキストに更新
    const userMsgs = chatArea.querySelectorAll('.message.user .message-bubble');
    if (userMsgs.length > 0) {
      const lastBubble = userMsgs[userMsgs.length - 1];
      if (lastBubble.textContent === '🎤 (音声入力)') {
        lastBubble.textContent = `🎤 ${data.text}`;
      }
    }
  } else {
    sttStatus.textContent = data.message || '音声を認識できませんでした';
    setTimeout(() => { sttStatus.textContent = ''; }, 3000);
    setMicState('idle');
    isStreaming = false;
    updateUI();

    // プレースホルダー削除
    const streamingMsg = document.getElementById('streaming-msg');
    if (streamingMsg) streamingMsg.remove();
  }
}

// ============================================
//  チャットUI
// ============================================

function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

  // 手動で上にスクロールしていても、新しい送信では必ず末尾へ追従する
  forceScrollToBottom();

  // ユーザーメッセージ表示
  addMessage('user', text);

  // 入力クリア
  messageInput.value = '';
  messageInput.style.height = 'auto';

  // 送信
  isStreaming = true;
  updateUI();

  ws.send(JSON.stringify({
    type: 'message',
    text: text,
    session_id: sessionId,
    tts: ttsToggle.checked,
  }));

  // AI応答プレースホルダー
  createAssistantBubble();
}

function addMessage(role, text) {
  removeWelcome();

  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  bubble.textContent = text;

  msg.appendChild(bubble);
  chatArea.appendChild(msg);
  scrollToBottom();
}

let currentBubble = null;

function createAssistantBubble() {
  removeWelcome();

  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.id = 'streaming-msg';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor';
  bubble.appendChild(cursor);

  msg.appendChild(bubble);
  chatArea.appendChild(msg);
  currentBubble = bubble;
  // 新しい応答バブルの冒頭は末尾へ追従させる
  forceScrollToBottom();
}

// --- スクロール追従 ---
// 末尾付近にいれば自動追従し、ユーザーが上にスクロールした時は位置を保持する。
const NEAR_BOTTOM_THRESHOLD = 80;
let autoFollow = true;

function isNearBottom() {
  return chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight
    <= NEAR_BOTTOM_THRESHOLD;
}

function trackChatScroll() {
  autoFollow = isNearBottom();
}

function scrollToBottom() {
  if (!autoFollow) return;
  requestAnimationFrame(() => {
    if (!autoFollow) return;
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

function forceScrollToBottom() {
  autoFollow = true;
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

function appendToken(token) {
  if (!currentBubble) return;

  // カーソルの前にテキスト追加
  const cursor = currentBubble.querySelector('.typing-cursor');
  if (cursor) {
    currentBubble.insertBefore(document.createTextNode(token), cursor);
  } else {
    currentBubble.appendChild(document.createTextNode(token));
  }
  // 末尾付近なら追従、ユーザーが上にスクロール中なら位置を保持
  scrollToBottom();
}

function finishResponse(fullText) {
  if (currentBubble) {
    // カーソル削除
    const cursor = currentBubble.querySelector('.typing-cursor');
    if (cursor) cursor.remove();

    // TTS再生ボタン追加
    if (ttsToggle.checked) {
      const playBtn = document.createElement('button');
      playBtn.className = 'tts-play-btn';
      playBtn.textContent = 'もう一度読む';
      playBtn.dataset.text = fullText;
      playBtn.addEventListener('click', () => replayTTS(playBtn));
      currentBubble.appendChild(document.createElement('br'));
      currentBubble.appendChild(playBtn);
    }

    currentBubble = null;
  }

  isStreaming = false;
  setMicState('idle');
  updateUI();
  scrollToBottom();
  loadGrowth({ animate: true });
  loadGame({ animate: true });
}

function showError(message) {
  if (currentBubble) {
    const cursor = currentBubble.querySelector('.typing-cursor');
    if (cursor) cursor.remove();
    currentBubble.style.color = '#ff5555';
    currentBubble.textContent = `エラー: ${message}`;
    currentBubble = null;
  }
}

function removeWelcome() {
  const welcome = $('.welcome');
  if (welcome) welcome.remove();
}

function updateUI() {
  sendBtn.disabled = isStreaming;
  messageInput.disabled = isStreaming;
  if (!isStreaming) {
    messageInput.focus();
  }
}

// ============================================
//  音声再生
// ============================================

function playAudio(base64Data) {
  const byteChars = atob(base64Data);
  const byteArray = new Uint8Array(byteChars.length);
  for (let i = 0; i < byteChars.length; i++) {
    byteArray[i] = byteChars.charCodeAt(i);
  }

  const blob = new Blob([byteArray], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);

  if (currentAudio) {
    currentAudio.pause();
    currentAudio = null;
  }

  currentAudio = new Audio(url);
  currentAudio.play().catch(e => console.warn('[Audio] Play failed:', e));
  currentAudio.onended = () => {
    URL.revokeObjectURL(url);
    currentAudio = null;
  };
}

async function replayTTS(btn) {
  const text = btn.dataset.text;
  if (!text) return;

  btn.classList.add('playing');
  btn.textContent = '読んでいます…';

  try {
    const resp = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) throw new Error('TTS failed');

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);

    if (currentAudio) {
      currentAudio.pause();
    }

    currentAudio = new Audio(url);
    currentAudio.play();
    currentAudio.onended = () => {
      URL.revokeObjectURL(url);
      currentAudio = null;
      btn.classList.remove('playing');
      btn.textContent = 'もう一度読む';
    };
  } catch (e) {
    console.error('[TTS]', e);
    btn.classList.remove('playing');
    btn.textContent = 'もう一度読む';
  }
}

// ============================================
//  音声録音 (STT)
// ============================================

function setMicState(state) {
  micBtn.classList.remove('recording', 'processing', 'disabled');
  sttStatus.classList.remove('actionable');
  switch (state) {
    case 'recording':
      micBtn.classList.add('recording');
      sttStatus.textContent = '聞いています。もう一度押すと止まります';
      break;
    case 'processing':
      micBtn.classList.add('processing');
      sttStatus.textContent = 'ことばにしています…';
      break;
    case 'disabled':
      micBtn.classList.add('disabled');
      sttStatus.textContent = '';
      break;
    case 'idle':
    default:
      sttStatus.textContent = '';
      break;
  }
}

function showMicStatus(message, linkUrl = '') {
  sttStatus.replaceChildren(document.createTextNode(message));
  sttStatus.classList.toggle('actionable', Boolean(linkUrl));
  if (linkUrl) {
    const link = document.createElement('a');
    link.href = linkUrl;
    link.textContent = '安全な接続で開く';
    sttStatus.appendChild(link);
  }
}

function showMicUnavailable() {
  if (micUnavailableReason === 'secure_context') {
    showMicStatus('マイクはHTTPS接続で使えます。', secureWebUrl);
  } else if (micUnavailableReason === 'unsupported_browser') {
    showMicStatus('このブラウザは音声入力に対応していません。');
  } else {
    showMicStatus('いま音声入力は使えません。');
  }
}

function configureMicAvailability(status) {
  sttAvailable = Boolean(status.stt);
  secureWebUrl = typeof status.secure_web_url === 'string' ? status.secure_web_url : '';
  if (!sttAvailable) {
    micUnavailableReason = 'stt_unavailable';
  } else if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    micUnavailableReason = 'secure_context';
  } else if (typeof window.MediaRecorder === 'undefined') {
    micUnavailableReason = 'unsupported_browser';
  } else {
    micUnavailableReason = '';
  }
  if (micUnavailableReason) {
    setMicState('disabled');
    micBtn.title = micUnavailableReason === 'secure_context'
      ? 'HTTPSで開くと音声入力できます'
      : '音声入力は使えません';
  } else {
    setMicState('idle');
    micBtn.title = '音声入力';
  }
  micBtn.setAttribute('aria-label', micBtn.title);
}

async function toggleRecording() {
  if (!sttAvailable || micUnavailableReason) {
    showMicUnavailable();
    return;
  }

  if (isStreaming) return;

  if (isRecording) {
    stopRecording();
  } else {
    await startRecording();
  }
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || typeof window.MediaRecorder === 'undefined') {
    micUnavailableReason = window.isSecureContext ? 'unsupported_browser' : 'secure_context';
    setMicState('disabled');
    showMicUnavailable();
    return;
  }
  try {
    // マイク権限を要求
    recordingStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    });

    audioChunks = [];

    // MediaRecorder の MIME タイプを選択
    const mimeType = getPreferredMimeType();
    const options = mimeType ? { mimeType } : {};

    mediaRecorder = new MediaRecorder(recordingStream, options);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = () => {
      processRecordedAudio();
    };

    mediaRecorder.start();
    isRecording = true;
    setMicState('recording');

    // 振動フィードバック (モバイル)
    if (navigator.vibrate) {
      navigator.vibrate(50);
    }

  } catch (err) {
    console.error('[Mic] Error:', err);
    if (err.name === 'NotAllowedError') {
      showMicStatus('ブラウザのサイト設定でマイクを許可してください。');
    } else if (err.name === 'NotFoundError') {
      showMicStatus('使えるマイクが見つかりません。');
    } else if (err.name === 'NotReadableError') {
      showMicStatus('マイクが別のアプリで使用中です。');
    } else if (err.name === 'SecurityError') {
      micUnavailableReason = 'secure_context';
      setMicState('disabled');
      showMicUnavailable();
    } else {
      showMicStatus(`マイクを開けません: ${err.message}`);
    }
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
  }
  isRecording = false;

  // ストリームを停止
  if (recordingStream) {
    recordingStream.getTracks().forEach(t => t.stop());
    recordingStream = null;
  }

  // 振動フィードバック (モバイル)
  if (navigator.vibrate) {
    navigator.vibrate([30, 30, 30]);
  }
}

function getPreferredMimeType() {
  // ブラウザ対応順にチェック
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ];
  for (const t of types) {
    if (MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return '';
}

function getAudioFormat(mimeType) {
  if (mimeType.includes('webm')) return 'webm';
  if (mimeType.includes('ogg')) return 'ogg';
  if (mimeType.includes('mp4') || mimeType.includes('m4a')) return 'mp4';
  return 'wav';
}

async function processRecordedAudio() {
  if (audioChunks.length === 0) {
    setMicState('idle');
    return;
  }

  setMicState('processing');

  const mimeType = mediaRecorder?.mimeType || 'audio/webm';
  const blob = new Blob(audioChunks, { type: mimeType });
  audioChunks = [];

  // 最小サイズチェック (録音が短すぎないか)
  if (blob.size < 1000) {
    sttStatus.textContent = 'もう少し長く話してみてください';
    setTimeout(() => { sttStatus.textContent = ''; }, 3000);
    setMicState('idle');
    return;
  }

  try {
    // Base64に変換
    const arrayBuffer = await blob.arrayBuffer();
    const base64 = arrayBufferToBase64(arrayBuffer);
    const format = getAudioFormat(mimeType);

    // WebSocketで送信
    if (ws && ws.readyState === WebSocket.OPEN) {
      // ユーザーメッセージのプレースホルダー (音声アイコン表示)
      addMessage('user', '🎤 (音声入力)');

      isStreaming = true;
      updateUI();

      ws.send(JSON.stringify({
        type: 'audio_message',
        data: base64,
        format: format,
        session_id: sessionId,
        tts: ttsToggle.checked,
      }));

      // AI応答プレースホルダー
      createAssistantBubble();
    } else {
      sttStatus.textContent = '接続が切れています';
      setTimeout(() => { sttStatus.textContent = ''; }, 3000);
      setMicState('idle');
    }
  } catch (err) {
    console.error('[STT] Process error:', err);
    sttStatus.textContent = `⚠️ エラー: ${err.message}`;
    setTimeout(() => { sttStatus.textContent = ''; }, 5000);
    setMicState('idle');
  }
}

function arrayBufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.byteLength; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// ============================================
//  設定パネル
// ============================================

// 設定シートが開いている間は root / 背景 のスクロールを停止し、
// シート内にスクロールを閉じ込める。shell-command-open と併存する。
function setSettingsScrollLock(open) {
  document.documentElement.classList.toggle('settings-open', open);
  document.body.classList.toggle('settings-open', open);
}

function openSettings() {
  settingsRestoreFocus = document.activeElement instanceof HTMLElement && document.activeElement !== document.body
    ? document.activeElement : $('#settings-btn');
  if (appShell) appShell.inert = true;
  settingsPanel.classList.add('open');
  settingsPanel.setAttribute('aria-hidden', 'false');
  setSettingsScrollLock(true);
  requestAnimationFrame(() => {
    if (!settingsPanel.classList.contains('open')) return;
    const firstControl = settingsPanel.querySelector('select, input, textarea, button');
    firstControl?.focus();
  });
}

function closeSettings() {
  settingsPanel.classList.remove('open');
  settingsPanel.setAttribute('aria-hidden', 'true');
  setSettingsScrollLock(false);
  if (appShell) appShell.inert = false;
  const restoreTarget = settingsRestoreFocus;
  settingsRestoreFocus = null;
  if (restoreTarget?.isConnected) restoreTarget.focus();
}

function handleSettingsKeydown(event) {
  if (event.key === 'Escape') {
    event.preventDefault();
    closeSettings();
    return;
  }
  if (event.key !== 'Tab') return;
  const focusable = [...settingsPanel.querySelectorAll('button, select, input, textarea, a[href]')]
    .filter((node) => !node.disabled && node.getClientRects().length > 0);
  if (!focusable.length) {
    event.preventDefault();
    return;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

async function changeVoice() {
  const voice = voiceSelect.value;
  try {
    await fetch('/api/tts/voice', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice }),
    });
  } catch (e) {
    console.error('[Voice]', e);
  }
}

function saveSessionId() {
  try { localStorage.setItem(SESSION_STORAGE_KEY, sessionId); } catch (e) {}
}

function newSession() {
  if (isStreaming) return;
  sessionId = `web_${Date.now()}`;
  saveSessionId();
  currentBubble = null;
  chatArea.replaceChildren(createWelcome('新しい話をしよう！', 'ここからは別の話題です。'));
  updateInstrumentStatus();
}

// ============================================
//  初期化
// ============================================

async function init() {
  await Promise.all([loadGrowth(), loadGame()]);

  // 状態取得
  try {
    const resp = await fetch('/api/status');
    const status = await resp.json();

    // STTサーバーと、ブラウザ側のHTTPS/録音APIを別々に確認する
    configureMicAvailability(status);
    updateInstrumentStatus({
      ollama: status.ollama,
      stt: status.stt,
      tts: status.tts,
      rag: status.rag,
    });

    if (status.tts_voices && voiceSelect) {
      voiceSelect.innerHTML = '';
      for (const [key, desc] of Object.entries(status.tts_voices)) {
        const opt = document.createElement('option');
        opt.value = key;
        opt.textContent = `${key} — ${desc}`;
        if (key === status.tts_voice) opt.selected = true;
        voiceSelect.appendChild(opt);
      }
    }
  } catch (e) {
    console.warn('[Init] Status fetch failed:', e);
  }

  // セッション復元: localStorage に保存IDがあればそれを、無ければ最新履歴を引継ぎ
  let savedId = null;
  try { savedId = localStorage.getItem(SESSION_STORAGE_KEY); } catch (e) {}

  try {
    const resumeUrl = savedId
      ? `/api/chat/resume?session_id=${encodeURIComponent(savedId)}`
      : '/api/chat/resume';
    const resp = await fetch(resumeUrl, { cache: 'no-store' });
    if (resp.ok) {
      const data = await resp.json();
      if (data.session_id) {
        sessionId = data.session_id;
        try { localStorage.setItem(SESSION_STORAGE_KEY, sessionId); } catch (e) {}
      }
      if (data.messages && data.messages.length > 0) {
        for (const m of data.messages) {
          if (m.role === 'user' || m.role === 'assistant') {
            addMessage(m.role, m.content);
          }
        }
      }
    } else if (savedId && resp.status === 404) {
      // 保存IDはあるが履歴未作成 → 空セッションを維持し最新へは勝手に戻らない
      sessionId = savedId;
    }
  } catch (e) {
    console.warn('[Init] Resume failed:', e);
  }

  if (!sessionId) {
    sessionId = `web_${Date.now()}`;
    try { localStorage.setItem(SESSION_STORAGE_KEY, sessionId); } catch (e) {}
  }

  // WebSocket接続
  connect();

  // イベント
  sendBtn.addEventListener('click', sendMessage);
  micBtn.addEventListener('click', toggleRecording);

  // 手動スクロール位置の受動追跡（末尾付近かどうかで自動追従を切り替える）
  chatArea.addEventListener('scroll', trackChatScroll, { passive: true });

  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendMessage();
    }
  });

  // テキストエリア自動リサイズ
  messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
  });

  // 設定パネル
  $('#settings-btn').addEventListener('click', openSettings);
  $('#settings-close').addEventListener('click', closeSettings);
  $('#new-session-btn').addEventListener('click', newSession);
  gameToggle?.addEventListener('click', () => setGameOpen(gameDetails.hidden));
  let savedGamePanel = 'closed';
  try { savedGamePanel = localStorage.getItem(GAME_PANEL_KEY) || 'closed'; } catch (e) {}
  setGameOpen(savedGamePanel === 'open');
  voiceSelect.addEventListener('change', changeVoice);

  settingsPanel.addEventListener('keydown', handleSettingsKeydown);
  settingsPanel.addEventListener('click', (e) => {
    if (e.target === settingsPanel) closeSettings();
  });

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/service-worker.js').catch((e) => {
      console.warn('[PWA] Service worker registration failed:', e);
    });
  }

  // 「やること」の最初の一歩を、現在の会話へ持ち込む。自動送信はせず確認できる状態にする。
  const prompt = new URLSearchParams(window.location.search).get('prompt');
  if (prompt) {
    messageInput.value = prompt.slice(0, 1000);
    messageInput.dispatchEvent(new Event('input'));
    window.history.replaceState({}, '', window.location.pathname);
  }

  messageInput.focus();
}

document.addEventListener('DOMContentLoaded', init);
