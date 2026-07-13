#!/bin/bash
# =============================================================================
# Phase 5: 映像入力 検証スクリプト
# OpenCV / 顔検出 / 感情推定 / VisionContext の検証
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "=========================================="
echo " Phase 5: 映像入力 検証"
echo "=========================================="

PASS=0
FAIL=0
SKIP=0

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

skip() {
    local name="$1"
    local reason="$2"
    echo "  ${name}... ⏭️  SKIP (${reason})"
    ((SKIP++))
}

# 仮想環境の有効化
source "${VENV_DIR}/bin/activate"

# --- Python パッケージ ---
echo ""
echo "[Pythonパッケージ]"
check "opencv (cv2)" "python3 -c 'import cv2; print(cv2.__version__)'"
check "onnxruntime" "python3 -c 'import onnxruntime; print(onnxruntime.__version__)'"
check "numpy" "python3 -c 'import numpy; print(numpy.__version__)'"

# --- プロジェクト構成 ---
echo ""
echo "[プロジェクト構成]"
check "src/vision/__init__.py" "[ -f '${PROJECT_ROOT}/src/vision/__init__.py' ]"
check "src/vision/camera.py" "[ -f '${PROJECT_ROOT}/src/vision/camera.py' ]"
check "src/vision/detector.py" "[ -f '${PROJECT_ROOT}/src/vision/detector.py' ]"
check "src/vision/context.py" "[ -f '${PROJECT_ROOT}/src/vision/context.py' ]"
check "models/vision ディレクトリ" "[ -d '${PROJECT_ROOT}/models/vision' ]"

# --- 感情推定モデル ---
echo ""
echo "[感情推定モデル]"
EMOTION_MODEL="${PROJECT_ROOT}/models/vision/emotion-ferplus-8.onnx"
HAS_EMOTION_MODEL=false
if [ -f "$EMOTION_MODEL" ]; then
    FILE_SIZE=$(stat -c%s "$EMOTION_MODEL" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -gt 1000000 ]; then
        HAS_EMOTION_MODEL=true
        check "emotion-ferplus-8.onnx ($(du -h "$EMOTION_MODEL" | cut -f1))" "true"
    else
        skip "emotion-ferplus-8.onnx" "ファイルサイズ不正 (${FILE_SIZE} bytes)"
    fi
else
    skip "emotion-ferplus-8.onnx" "ファイルなし"
fi

# --- 顔検出テスト ---
echo ""
echo "[顔検出テスト]"
echo -n "  OpenCV Haar Cascade ロード... "
FACE_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.detector import FaceDetector
import cv2
import numpy as np

# Haar Cascade ロード
detector = FaceDetector()

# テスト画像を作成 (顔のダミー: グレー背景に楕円)
frame = np.zeros((480, 640, 3), dtype=np.uint8) + 128
# 顔ダミー (明るい楕円 + 目の暗い点)
cv2.ellipse(frame, (320, 200), (80, 100), 0, 0, 360, (200, 180, 170), -1)
cv2.circle(frame, (295, 175), 8, (50, 50, 50), -1)  # 左目
cv2.circle(frame, (345, 175), 8, (50, 50, 50), -1)  # 右目
cv2.ellipse(frame, (320, 220), (25, 10), 0, 0, 360, (100, 80, 80), -1)  # 口

faces = detector.detect(frame)
print(f'OK: Haar Cascade loaded, detected {len(faces)} faces in test image')
" 2>&1)
if echo "$FACE_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $FACE_RESULT" | grep "OK:" | head -1
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $FACE_RESULT" | tail -3
    ((FAIL++))
fi

# --- 感情推定テスト ---
echo ""
echo "[感情推定テスト]"
if [ "$HAS_EMOTION_MODEL" = true ]; then
    echo -n "  EmotionDetector ONNX 推論... "
    EMO_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.detector import EmotionDetector, EMOTION_LABELS
import numpy as np
import cv2

detector = EmotionDetector('${EMOTION_MODEL}')

# ダミー顔画像 (100x100 BGR)
face_img = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
emotion, confidence, scores = detector.detect(face_img)
print(f'OK: emotion={emotion}, confidence={confidence:.3f}')
print(f'  scores: { {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:3]} }')
assert emotion in EMOTION_LABELS, f'Unknown emotion: {emotion}'
assert 0 <= confidence <= 1, f'Invalid confidence: {confidence}'
" 2>&1)
    if echo "$EMO_RESULT" | grep -q "OK:"; then
        echo "✅ OK"
        echo "    $EMO_RESULT" | grep "OK:\|scores:" | head -2
        ((PASS++))
    else
        echo "❌ FAIL"
        echo "    $EMO_RESULT" | tail -3
        ((FAIL++))
    fi
else
    skip "EmotionDetector" "感情モデルなし"
fi

# --- VisionAnalyzer テスト ---
echo ""
echo "[VisionAnalyzer テスト]"
echo -n "  VisionAnalyzer 統合テスト... "
VA_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.detector import VisionAnalyzer, VisionResult
import numpy as np
import cv2

# 感情モデルパス (あれば使う)
import os
emo_path = '${EMOTION_MODEL}'
if not os.path.exists(emo_path) or os.path.getsize(emo_path) < 1000000:
    emo_path = None

analyzer = VisionAnalyzer(emotion_model_path=emo_path)
has_emo = 'あり' if analyzer.has_emotion else 'なし'

# ダミーフレーム
frame = np.zeros((480, 640, 3), dtype=np.uint8) + 128
result = analyzer.analyze(frame)
assert isinstance(result, VisionResult)
assert isinstance(result.person_count, int)
assert isinstance(result.user_present, bool)
print(f'OK: 感情推定={has_emo}, person_count={result.person_count}, user_present={result.user_present}')
" 2>&1)
if echo "$VA_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $VA_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $VA_RESULT" | tail -3
    ((FAIL++))
fi

# --- VisionContext テスト (カメラなし) ---
echo ""
echo "[VisionContext テスト]"
echo -n "  VisionContext インスタンス化... "
VC_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.context import VisionContext, VisionState
import os

emo_path = '${EMOTION_MODEL}'
if not os.path.exists(emo_path) or os.path.getsize(emo_path) < 1000000:
    emo_path = None

# カメラなしでインスタンス化（start()は呼ばない）
vc = VisionContext(camera_id=99, emotion_model_path=emo_path)
state = vc.get_state()
assert isinstance(state, VisionState)
assert state.user_present == False

# コンテキストテキスト（非稼働時は空）
ctx = vc.get_context_text()
assert ctx == ''

# ステータス
status = vc.get_status()
assert 'running' in status
assert status['running'] == False

print(f'OK: state=VisionState, context_text=\"\", running=False')
" 2>&1)
if echo "$VC_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $VC_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $VC_RESULT" | tail -3
    ((FAIL++))
fi

echo -n "  VisionContext コンテキスト生成テスト... "
VC2_RESULT=$(python3 -c "
import sys, time; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.context import VisionContext, VisionState
from src.vision.detector import VisionResult, FaceInfo

# VisionContext のインスタンスを作り、手動で状態を設定
import os
emo_path = '${EMOTION_MODEL}'
if not os.path.exists(emo_path) or os.path.getsize(emo_path) < 1000000:
    emo_path = None

vc = VisionContext(camera_id=99, emotion_model_path=emo_path)
# 稼働中を偽装（テスト用）
vc._running = True
vc.camera._running = True
vc.camera._cap = True  # 非None

# ユーザー在席状態をシミュレート
vc._state.user_present = True
vc._state.person_count = 1
vc._state.dominant_emotion = 'happiness'
vc._state.dominant_emotion_ja = '嬉しそう'
vc._state.emotion_streak = 5

ctx = vc.get_context_text()
assert '映像情報' in ctx
assert 'カメラの前にいます' in ctx
assert '嬉しそう' in ctx
assert 'この表情がしばらく続いています' in ctx
print(f'OK: context contains emotion info')
print(f'  context: {ctx[:100]}...')
" 2>&1)
if echo "$VC2_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $VC2_RESULT" | grep "OK:"
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $VC2_RESULT" | tail -5
    ((FAIL++))
fi

# --- ChatSession + Vision 統合テスト ---
echo ""
echo "[ChatSession + Vision 統合テスト]"
echo -n "  build_messages に Vision コンテキスト注入... "
CS_RESULT=$(python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
import os
from src.chat.session import ChatSession
from src.vision.context import VisionContext

emo_path = '${PROJECT_ROOT}/models/vision/emotion-ferplus-8.onnx'
if not os.path.exists(emo_path) or os.path.getsize(emo_path) < 1000000:
    emo_path = None

# VisionContext を偽装
vc = VisionContext(camera_id=99, emotion_model_path=emo_path)
vc._running = True
vc.camera._running = True
vc.camera._cap = True
vc._state.user_present = True
vc._state.person_count = 1
vc._state.dominant_emotion = 'sadness'
vc._state.dominant_emotion_ja = '悲しそう'

# ChatSession に vision_context を渡す
session = ChatSession(
    system_prompt='あなたはテストAIです。',
    vision_context=vc,
)
session.add_user_message('元気？')
messages = session.build_messages()

# system promptに映像コンテキストが含まれているか
system_msg = messages[0]
assert system_msg['role'] == 'system'
assert '映像情報' in system_msg['content']
assert '悲しそう' in system_msg['content']
print(f'OK: system prompt にVisionコンテキストが注入されている')
print(f'  system_prompt length: {len(system_msg[\"content\"])} chars')
" 2>&1)
if echo "$CS_RESULT" | grep -q "OK:"; then
    echo "✅ OK"
    echo "    $CS_RESULT" | grep "OK:\|system_prompt" | head -2
    ((PASS++))
else
    echo "❌ FAIL"
    echo "    $CS_RESULT" | tail -5
    ((FAIL++))
fi

# --- カメラ実機テスト (オプション) ---
echo ""
echo "[カメラ実機テスト]"
if ls /dev/video* 2>/dev/null | grep -q "video0"; then
    echo -n "  /dev/video0 アクセス... "
    CAM_RESULT=$(timeout 10 python3 -c "
import sys; sys.path.insert(0, '${PROJECT_ROOT}')
from src.vision.camera import CameraCapture
import time

cam = CameraCapture(device_id=0, width=640, height=480, fps=15)
if cam.start():
    time.sleep(1.0)
    frame = cam.get_frame()
    cam.stop()
    if frame is not None:
        h, w = frame.shape[:2]
        print(f'OK: camera={w}x{h}, frame_count={cam.frame_count}')
    else:
        print('FAIL: camera started but no frame')
else:
    print('FAIL: cannot open camera')
" 2>&1)
    if echo "$CAM_RESULT" | grep -q "OK:"; then
        echo "✅ OK"
        echo "    $CAM_RESULT" | grep "OK:"
        ((PASS++))
    else
        echo "❌ FAIL"
        echo "    $CAM_RESULT" | tail -3
        ((FAIL++))
    fi
else
    skip "カメラアクセス" "ビデオデバイスなし"
fi

# --- 結果 ---
echo ""
echo "=========================================="
TOTAL=$((PASS + FAIL))
echo " 結果: ${PASS}/${TOTAL} passed (SKIP: ${SKIP})"
if [ "$FAIL" -eq 0 ]; then
    echo " ✅ すべてのテストがパスしました！"
else
    echo " ⚠️  ${FAIL}件のテストが失敗しました"
fi
echo "=========================================="
exit $FAIL
