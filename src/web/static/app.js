// ============================================
//  subpc_living Web UI — Frontend JS
//  WebSocket ストリーミングチャット + TTS
// ============================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// --- 状態 ---
let ws = null;
let isStreaming = false;
let sessionId = `web_${Date.now()}`;
let currentAudio = null;

// --- DOM要素 ---
const chatArea = $('#chat-area');
const messageInput = $('#message-input');
const sendBtn = $('#send-btn');
const statusDot = $('#status-dot');
const ttsToggle = $('#tts-toggle');
const settingsPanel = $('#settings-panel');
const voiceSelect = $('#voice-select');

// ============================================
//  WebSocket 接続
// ============================================

function connect() {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${location.host}/ws/chat`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    statusDot.className = 'status-dot connected';
    console.log('[WS] Connected');
  };

  ws.onclose = () => {
    statusDot.className = 'status-dot error';
    console.log('[WS] Disconnected, reconnecting in 3s...');
    setTimeout(connect, 3000);
  };

  ws.onerror = () => {
    statusDot.className = 'status-dot error';
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

    case 'error':
      showError(data.message);
      isStreaming = false;
      updateUI();
      break;
  }
}

// ============================================
//  チャットUI
// ============================================

function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isStreaming || !ws || ws.readyState !== WebSocket.OPEN) return;

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
  scrollToBottom();
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
      playBtn.innerHTML = '🔊 再生';
      playBtn.dataset.text = fullText;
      playBtn.addEventListener('click', () => replayTTS(playBtn));
      currentBubble.appendChild(document.createElement('br'));
      currentBubble.appendChild(playBtn);
    }

    currentBubble = null;
  }

  isStreaming = false;
  updateUI();
  scrollToBottom();
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

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
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
  btn.innerHTML = '🔊 再生中...';

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
      btn.innerHTML = '🔊 再生';
    };
  } catch (e) {
    console.error('[TTS]', e);
    btn.classList.remove('playing');
    btn.innerHTML = '🔊 再生';
  }
}

// ============================================
//  設定パネル
// ============================================

function openSettings() {
  settingsPanel.classList.add('open');
}

function closeSettings() {
  settingsPanel.classList.remove('open');
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

function newSession() {
  sessionId = `web_${Date.now()}`;
  chatArea.innerHTML = `
    <div class="welcome">
      <h2>💬 subpc_living</h2>
      <p>パーソナルAIとチャットできます。<br>メッセージを入力してください。</p>
    </div>
  `;
}

// ============================================
//  初期化
// ============================================

async function init() {
  // 状態取得
  try {
    const resp = await fetch('/api/status');
    const status = await resp.json();

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

  // WebSocket接続
  connect();

  // イベント
  sendBtn.addEventListener('click', sendMessage);

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
  voiceSelect.addEventListener('change', changeVoice);

  settingsPanel.addEventListener('click', (e) => {
    if (e.target === settingsPanel) closeSettings();
  });

  messageInput.focus();
}

document.addEventListener('DOMContentLoaded', init);
