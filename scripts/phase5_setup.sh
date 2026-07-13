#!/bin/bash
# =============================================================================
# Phase 5: 映像入力セットアップスクリプト
# OpenCV + emotion-ferplus ONNX モデル
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 5: 映像入力 セットアップ"
echo "=========================================="

# --- 0. 仮想環境の確認 ---
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ 仮想環境が見つかりません。先に phase2_setup.sh を実行してください。"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

# --- 1. Python パッケージ ---
echo ""
echo "[1/3] Python パッケージのインストール..."
pip install --upgrade pip --quiet
pip install opencv-python-headless --quiet
echo "✅ Python パッケージインストール完了"

pip list --format=columns 2>/dev/null | grep -iE "opencv|onnxruntime"

# --- 2. 感情推定モデルのダウンロード ---
echo ""
echo "[2/3] 感情推定 ONNX モデルのダウンロード..."
mkdir -p "${PROJECT_ROOT}/models/vision"

EMOTION_MODEL="${PROJECT_ROOT}/models/vision/emotion-ferplus-8.onnx"

if [ -f "$EMOTION_MODEL" ]; then
    FILE_SIZE=$(stat -c%s "$EMOTION_MODEL" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -gt 1000000 ]; then
        echo "  モデルは既にダウンロード済みです ($(du -h "$EMOTION_MODEL" | cut -f1))"
    else
        echo "  既存ファイルが不完全です。再ダウンロードします..."
        rm -f "$EMOTION_MODEL"
    fi
fi

if [ ! -f "$EMOTION_MODEL" ]; then
    echo "  ダウンロード中: emotion-ferplus-8.onnx (~34MB)"
    echo "  ソース: ONNX Model Zoo (GitHub)"

    # Git LFS対応: curl -L でリダイレクトを追従
    curl -L -o "$EMOTION_MODEL" \
        "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" \
        2>/dev/null

    # ファイルサイズチェック
    FILE_SIZE=$(stat -c%s "$EMOTION_MODEL" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
        echo "⚠️  ダウンロードが不完全な可能性があります (${FILE_SIZE} bytes)"
        echo "  Python フォールバックを試行中..."

        python3 -c "
import urllib.request
import os

url = 'https://github.com/onnx/models/raw/refs/heads/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx'
dest = '${EMOTION_MODEL}'
try:
    urllib.request.urlretrieve(url, dest)
    size = os.path.getsize(dest)
    if size > 1000000:
        print(f'✅ ダウンロード成功 ({size / 1024 / 1024:.1f}MB)')
    else:
        print(f'⚠️  ファイルサイズが小さすぎます ({size} bytes)')
        print('  感情推定は利用できませんが、顔検出は動作します')
except Exception as e:
    print(f'⚠️  ダウンロード失敗: {e}')
    print('  感情推定は利用できませんが、顔検出は動作します')
"
    else
        echo "✅ 感情モデルダウンロード完了 ($(du -h "$EMOTION_MODEL" | cut -f1))"
    fi
fi

# モデル検証
echo ""
echo "  モデル検証中..."
python3 -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('${EMOTION_MODEL}', providers=['CPUExecutionProvider'])
input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]
print(f'  入力: {input_info.name} shape={input_info.shape} type={input_info.type}')
print(f'  出力: {output_info.name} shape={output_info.shape} type={output_info.type}')

# テスト推論
dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)
result = session.run(None, {input_info.name: dummy})
print(f'  テスト推論OK: 出力shape={result[0].shape}')
print('✅ 感情推定モデル検証完了')
" 2>&1 || echo "⚠️  モデル検証失敗 (顔検出のみで動作します)"

# --- 3. カメラデバイスの確認 ---
echo ""
echo "[3/3] カメラデバイスの確認..."

# Linux のビデオデバイスを列挙
if ls /dev/video* 2>/dev/null 1>&2; then
    echo "  検出されたビデオデバイス:"
    for dev in /dev/video*; do
        echo "    $dev"
    done

    # OpenCV でカメラアクセステスト
    python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        print(f'  ✅ カメラ /dev/video0 アクセスOK ({w}x{h})')
    else:
        print('  ⚠️  カメラは開けますがフレーム取得に失敗しました')
    cap.release()
else:
    print('  ⚠️  カメラを開けません (権限を確認してください)')
" 2>&1 || echo "  ⚠️  カメラアクセステスト失敗"
else
    echo "  ⚠️  ビデオデバイスが見つかりません"
    echo "  USBカメラを接続していない場合は正常です"
    echo "  カメラなしでもシステムは動作します (Visionは自動スキップ)"
fi

# --- 完了 ---
echo ""
echo "=========================================="
echo " ✅ Phase 5 セットアップ完了!"
echo "=========================================="
echo ""
echo "確認コマンド:"
echo "  bash scripts/phase5_verify.sh"
echo ""
echo "使い方:"
echo "  # 映像入力付きで音声対話"
echo "  python -m src.audio.main"
echo ""
echo "  # 映像入力を無効化"
echo "  python -m src.audio.main --no-vision"
echo ""
echo "  # カメラデバイスを指定"
echo "  python -m src.audio.main --camera-id 1"
