// ============================================
//  subpc_living Web UI — Frontend JS
//  WebSocket ストリーミングチャット + TTS + STT
// ============================================

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// --- 状態 ---
let ws = null;
let isStreaming = false;
let sessionId = `web_${Date.now()}`;
let currentAudio = null;
let sttAvailable = false;

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
const settingsPanel = $('#settings-panel');
const voiceSelect = $('#voice-select');
const micBtn = $('#mic-btn');
const sttStatus = $('#stt-status');

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
  setMicState('idle');
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
//  音声録音 (STT)
// ============================================

function setMicState(state) {
  micBtn.classList.remove('recording', 'processing', 'disabled');
  switch (state) {
    case 'recording':
      micBtn.classList.add('recording');
      sttStatus.textContent = '🎤 録音中... タップで停止';
      break;
    case 'processing':
      micBtn.classList.add('processing');
      sttStatus.textContent = '⏳ 音声認識中...';
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

async function toggleRecording() {
  if (!sttAvailable) {
    sttStatus.textContent = '⚠️ STTが利用できません';
    setTimeout(() => { sttStatus.textContent = ''; }, 3000);
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
      sttStatus.textContent = '⚠️ マイクの使用が許可されていません';
    } else {
      sttStatus.textContent = `⚠️ マイクエラー: ${err.message}`;
    }
    setTimeout(() => { sttStatus.textContent = ''; }, 5000);
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
    sttStatus.textContent = '⚠️ 録音が短すぎます';
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
      sttStatus.textContent = '⚠️ 接続が切れています';
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
      <div class="welcome-orb">✦</div>
      <h2>subpc_living</h2>
      <p>会話、音声入力、読み上げ</p>
      <a class="welcome-link" href="/tasks">タスクを見る →</a>
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

    // STT 利用可能かチェック
    sttAvailable = !!status.stt;
    if (!sttAvailable) {
      setMicState('disabled');
    }

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
  micBtn.addEventListener('click', toggleRecording);

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

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/service-worker.js').catch((e) => {
      console.warn('[PWA] Service worker registration failed:', e);
    });
  }

  messageInput.focus();
}

document.addEventListener('DOMContentLoaded', init);
