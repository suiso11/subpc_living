#!/bin/bash
# =============================================================================
# Phase 3: 音声対話セットアップスクリプト
# faster-whisper (STT) + kokoro-onnx (TTS) + sounddevice (Audio I/O)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 3: 音声対話セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. システムパッケージ ---
echo ""
echo "[1/4] システムパッケージの確認..."

MISSING_PKGS=""
dpkg -l portaudio19-dev > /dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} portaudio19-dev"
dpkg -l ffmpeg > /dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} ffmpeg"
dpkg -l libsndfile1 > /dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} libsndfile1"
dpkg -l espeak-ng > /dev/null 2>&1 || MISSING_PKGS="${MISSING_PKGS} espeak-ng"

if [ -n "$MISSING_PKGS" ]; then
    echo "不足パッケージをインストール:${MISSING_PKGS}"
    sudo apt install -y ${MISSING_PKGS}
else
    echo "✅ システムパッケージOK"
fi

# --- 2. Python パッケージ ---
echo ""
echo "[2/4] Python パッケージのインストール..."
pip install --upgrade pip --quiet
pip install -r "${PROJECT_ROOT}/requirements.txt" --quiet
echo "✅ Python パッケージインストール完了"

pip list --format=columns 2>/dev/null | grep -iE "faster-whisper|sounddevice|numpy|httpx|ctranslate2|kokoro|misaki"

# --- 3. kokoro-onnx モデルのダウンロード ---
echo ""
echo "[3/4] kokoro-onnx TTS モデルの確認..."
KOKORO_DIR="${PROJECT_ROOT}/models/tts/kokoro"
MODEL_FILE="${KOKORO_DIR}/kokoro-v1.0.onnx"
VOICES_FILE="${KOKORO_DIR}/voices-v1.0.bin"

if [ -f "$MODEL_FILE" ] && [ -f "$VOICES_FILE" ]; then
    echo "✅ kokoro-onnx モデルは既にダウンロード済み"
else
    mkdir -p "$KOKORO_DIR"
    echo "HuggingFaceからモデルをダウンロード中..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('fastrtc/kokoro-onnx', 'kokoro-v1.0.onnx', local_dir='${KOKORO_DIR}')
hf_hub_download('fastrtc/kokoro-onnx', 'voices-v1.0.bin', local_dir='${KOKORO_DIR}')
print('✅ kokoro-onnx モデルダウンロード完了')
"
fi

# モデルサイズ確認
echo "  model: $(du -h "$MODEL_FILE" | cut -f1)"
echo "  voices: $(du -h "$VOICES_FILE" | cut -f1)"

# --- 4. ディレクトリ作成 ---
echo ""
echo "[4/4] 追加ディレクトリの作成..."
mkdir -p "${PROJECT_ROOT}/data/chat_history"
mkdir -p "${PROJECT_ROOT}/models/stt"  # faster-whisperのキャッシュ(自動DL)
echo "✅ ディレクトリ作成完了"

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 3 セットアップ完了!"
echo "=========================================="
echo ""
echo "音声対話を開始するには:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python ${PROJECT_ROOT}/src/audio/main.py"
echo ""
echo "テキスト→音声モード (マイクなし):"
echo "  python ${PROJECT_ROOT}/src/audio/main.py --text-mode"
echo ""
echo "※ 初回実行時にWhisperモデル (small, ~500MB) がダウンロードされます"
echo ""
