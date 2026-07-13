#!/bin/bash
# =============================================================================
# Phase 3: 音声対話 検証スクリプト
# STT / TTS (kokoro-onnx) / Audio I/O / VAD (Energy + Silero) / パイプラインの検証
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 3: 音声対話 検証"
echo "=========================================="

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    echo -n "  ${name}... "
    if eval "${cmd}" > /dev/null 2>&1; then
        echo "✅ OK"
        ((PASS++))
    else
        echo "❌ FAIL"
        ((FAIL++))
    fi
}

# 仮想環境の有効化
source "${VENV_DIR}/bin/activate"

# --- Python パッケージ ---
echo ""
echo "[Pythonパッケージ]"
check "faster-whisper" "python3 -c 'import faster_whisper'"
check "sounddevice" "python3 -c 'import sounddevice'"
check "numpy" "python3 -c 'import numpy'"
check "kokoro-onnx" "python3 -c 'import kokoro_onnx'"
check "misaki" "python3 -c 'import misaki'"

# --- システムパッケージ ---
echo ""
echo "[システムパッケージ]"
check "portaudio" "dpkg -l portaudio19-dev"
check "ffmpeg" "command -v ffmpeg"
check "libsndfile" "dpkg -l libsndfile1"
check "espeak-ng" "command -v espeak-ng"

# --- kokoro-onnx TTS ---
echo ""
echo "[kokoro-onnx TTS]"
check "モデルファイル (.onnx)" "[ -f '${PROJECT_ROOT}/models/tts/kokoro/kokoro-v1.0.onnx' ]"
check "ボイスファイル (.bin)" "[ -f '${PROJECT_ROOT}/models/tts/kokoro/voices-v1.0.bin' ]"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/audio/__init__.py" "[ -f '${PROJECT_ROOT}/src/audio/__init__.py' ]"
check "src/audio/stt.py" "[ -f '${PROJECT_ROOT}/src/audio/stt.py' ]"
check "src/audio/tts.py" "[ -f '${PROJECT_ROOT}/src/audio/tts.py' ]"
check "src/audio/vad.py" "[ -f '${PROJECT_ROOT}/src/audio/vad.py' ]"
check "src/audio/audio_io.py" "[ -f '${PROJECT_ROOT}/src/audio/audio_io.py' ]"
check "src/audio/pipeline.py" "[ -f '${PROJECT_ROOT}/src/audio/pipeline.py' ]"
check "src/audio/main.py" "[ -f '${PROJECT_ROOT}/src/audio/main.py' ]"

# --- オーディオデバイス ---
echo ""
echo "[オーディオデバイス]"
check "入力デバイス存在" "python3 -c \"
import sounddevice as sd
devs = sd.query_devices()
has_input = any(d['max_input_channels'] > 0 for d in devs)
assert has_input, 'No input device'
\""
check "出力デバイス存在" "python3 -c \"
import sounddevice as sd
devs = sd.query_devices()
has_output = any(d['max_output_channels'] > 0 for d in devs)
assert has_output, 'No output device'
\""

# --- TTS テスト ---
echo ""
echo "[TTS合成テスト]"
echo -n "  kokoro-onnx 音声合成... "
TTS_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.tts import KokoroTTS
tts = KokoroTTS(
    models_dir='${PROJECT_ROOT}/models/tts/kokoro',
)
wav = tts.synthesize('こんにちは、テストです')
print(f'OK: {len(wav)} bytes')
" 2>&1)
if echo "$TTS_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $TTS_RESULT"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $TTS_RESULT"
    ((FAIL++))
fi

# --- STT テスト (モデルロードのみ — 初回DLがあるため時間がかかります) ---
echo ""
echo "[STTモデルテスト]"
echo -n "  Whisperモデルロード (初回はDLで数分かかります)... "
STT_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.stt import WhisperSTT
stt = WhisperSTT(model_size='small', device='cpu', compute_type='int8')
stt.load()
print('OK: model loaded')
" 2>&1)
if echo "$STT_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $STT_RESULT"
    ((FAIL++))
fi

# --- VAD テスト ---
echo ""
echo "[VADテスト]"
check "Energy VAD初期化" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.vad import EnergyVAD
import numpy as np
vad = EnergyVAD()
# 無音フレーム → None
frame = np.zeros(vad.frame_size, dtype=np.float32)
result = vad.process_frame(frame)
assert result is None
print('OK')
\""

# Silero VAD テスト (torch が利用可能な場合のみ)
echo -n "  Silero VAD (torch依存)... "
SILERO_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
try:
    import torch
    from src.audio.vad import SileroVAD
    import numpy as np
    vad = SileroVAD()
    frame = np.zeros(vad.frame_size, dtype=np.float32)
    result = vad.process_frame(frame)
    assert result is None
    print('OK')
except ImportError:
    print('SKIP: torch未インストール')
" 2>&1)
if echo "$SILERO_RESULT" | grep -q "OK"; then
    echo "✅ OK"
    ((PASS++))
elif echo "$SILERO_RESULT" | grep -q "SKIP"; then
    echo "⏭️  SKIP (torch未インストール — Energy VADにフォールバック)"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $SILERO_RESULT"
    ((FAIL++))
fi

check "VADファクトリ (create_vad)" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.vad import create_vad
vad = create_vad(vad_type='auto')
print(f'OK: {type(vad).__name__}')
\""

# --- ストリーミングTTS テスト ---
echo ""
echo "[ストリーミングTTSテスト]"
check "文分割ロジック" "python3 -c \"
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.audio.pipeline import VoicePipeline
result = VoicePipeline._split_sentences('こんにちは。今日はいい天気ですね！明日はどうでしょう？')
assert len(result) == 3, f'Expected 3 sentences, got {len(result)}: {result}'
print(f'OK: {result}')
\""

# --- 結果サマリー ---
echo ""
echo "=========================================="
echo " 結果: ✅ ${PASS} 成功 / ❌ ${FAIL} 失敗"
echo "=========================================="

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "🎉 Phase 3 検証 すべてパス!"
    echo ""
    echo "音声対話を開始するには:"
    echo "  source ${VENV_DIR}/bin/activate"
    echo "  python ${PROJECT_ROOT}/src/audio/main.py"
    echo ""
    echo "テキスト→音声モード (マイクなし):"
    echo "  python ${PROJECT_ROOT}/src/audio/main.py --text-mode"
    echo ""
    echo "VADオプション:"
    echo "  --vad auto     Silero VAD優先、なければEnergy VAD (デフォルト)"
    echo "  --vad silero   Silero VADを強制使用"
    echo "  --vad energy   Energy VADを強制使用"
    echo ""
    echo "ストリーミングTTS無効化:"
    echo "  --no-streaming-tts   全文完了後に音声合成"
    exit 0
else
    echo ""
    echo "⚠️  ${FAIL}件の失敗があります。上記を確認してください。"
    exit 1
fi
