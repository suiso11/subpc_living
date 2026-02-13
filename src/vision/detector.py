"""
顔検出 + 感情推定
- 顔検出: OpenCV Haar Cascade (軽量、CPU向け)
- 感情推定: emotion-ferplus ONNX モデル (onnxruntime)
- Phase 9: onnxruntime-gpu 利用時は CUDAExecutionProvider を自動選択
"""
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# --- 感情ラベル定義 (emotion-ferplus) ---

EMOTION_LABELS = [
    "neutral",      # ニュートラル
    "happiness",    # 嬉しい
    "surprise",     # 驚き
    "sadness",      # 悲しい
    "anger",        # 怒り
    "disgust",      # 嫌悪
    "fear",         # 恐怖
    "contempt",     # 軽蔑
]

EMOTION_JA = {
    "neutral":   "普通",
    "happiness": "嬉しそう",
    "surprise":  "驚いている",
    "sadness":   "悲しそう",
    "anger":     "怒っている",
    "disgust":   "嫌そう",
    "fear":      "怖がっている",
    "contempt":  "冷めている",
}


# --- データ構造 ---

@dataclass
class FaceInfo:
    """検出された顔の情報"""
    x: int
    y: int
    w: int
    h: int
    emotion: str = "unknown"
    emotion_ja: str = "不明"
    emotion_confidence: float = 0.0
    emotion_scores: dict = field(default_factory=dict)


@dataclass
class VisionResult:
    """1フレーム分の解析結果"""
    person_count: int = 0
    faces: list[FaceInfo] = field(default_factory=list)
    dominant_emotion: str = "unknown"
    dominant_emotion_ja: str = "不明"
    user_present: bool = False
    timestamp: float = 0.0


# --- 顔検出 ---

class FaceDetector:
    """OpenCV Haar Cascade による顔検出"""

    def __init__(self):
        if not HAS_CV2:
            raise RuntimeError("opencv-python がインストールされていません")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Haar Cascade ロード失敗: {cascade_path}")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        顔を検出して矩形リストを返す

        Args:
            frame: BGR画像 (OpenCV形式)

        Returns:
            [(x, y, w, h), ...] 検出された顔の矩形リスト
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # equalizeHist でコントラスト改善（照明変化に強くなる）
        gray = cv2.equalizeHist(gray)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


# --- 感情推定 ---

def _detect_onnx_providers() -> list[str]:
    """利用可能な ONNX Runtime プロバイダーを検出する (Phase 9)"""
    if not HAS_ORT:
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class EmotionDetector:
    """
    emotion-ferplus ONNX モデルによる感情推定

    モデル入力: float32 [1, 1, 64, 64] (grayscale)
    モデル出力: float32 [1, 8] (8感情のスコア)
    """

    def __init__(self, model_path: str, providers: Optional[list[str]] = None):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime がインストールされていません")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"感情モデルが見つかりません: {model_path}")

        # providers が未指定の場合は自動検出
        if providers is None:
            providers = _detect_onnx_providers()

        self._providers = providers
        self._session = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        # 入力形状を確認
        input_shape = self._session.get_inputs()[0].shape
        self._input_h = input_shape[2] if len(input_shape) == 4 else 64
        self._input_w = input_shape[3] if len(input_shape) == 4 else 64
        provider_str = ", ".join(providers)
        print(f"  感情推定: emotion-ferplus ONNX ({provider_str})")

    def detect(self, face_image: np.ndarray) -> tuple[str, float, dict]:
        """
        顔画像から感情を推定

        Args:
            face_image: BGR顔画像 (OpenCV形式、任意サイズ)

        Returns:
            (emotion_label, confidence, all_scores_dict)
        """
        if not HAS_CV2:
            raise RuntimeError("opencv-python がインストールされていません")

        # 前処理: BGR → グレースケール → リサイズ → 正規化 → reshape
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self._input_w, self._input_h))
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.reshape(1, 1, self._input_h, self._input_w)

        # 推論
        outputs = self._session.run(None, {self._input_name: input_tensor})
        scores = outputs[0][0]

        # softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 結果
        top_idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        all_scores = {
            EMOTION_LABELS[i]: round(float(probs[i]), 4)
            for i in range(len(EMOTION_LABELS))
        }

        return emotion, confidence, all_scores


# --- 統合アナライザー ---

class VisionAnalyzer:
    """顔検出 + 感情推定を統合した映像解析"""

    def __init__(self, emotion_model_path: Optional[str] = None):
        """
        Args:
            emotion_model_path: emotion-ferplus ONNX モデルのパス
                                None または存在しない場合は顔検出のみ
        """
        self.face_detector = FaceDetector()
        self.emotion_detector: Optional[EmotionDetector] = None

        if emotion_model_path and Path(emotion_model_path).exists():
            try:
                self.emotion_detector = EmotionDetector(emotion_model_path)
            except Exception as e:
                print(f"⚠️  感情モデル ロード失敗 (顔検出のみ使用): {e}")

    @property
    def has_emotion(self) -> bool:
        """感情検出が有効かどうか"""
        return self.emotion_detector is not None

    def analyze(self, frame: np.ndarray) -> VisionResult:
        """
        フレームを解析して VisionResult を返す

        Args:
            frame: BGR画像 (OpenCV形式)

        Returns:
            VisionResult (顔情報、感情、人数など)
        """
        import time
        result = VisionResult(timestamp=time.time())

        # 顔検出
        face_rects = self.face_detector.detect(frame)
        result.person_count = len(face_rects)
        result.user_present = result.person_count > 0

        if not face_rects:
            return result

        # 各顔の感情を推定
        for (x, y, w, h) in face_rects:
            face_info = FaceInfo(x=x, y=y, w=w, h=h)

            if self.emotion_detector is not None:
                # 顔領域を切り出し（境界チェック）
                fh, fw = frame.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(fw, x + w)
                y2 = min(fh, y + h)
                face_img = frame[y1:y2, x1:x2]

                if face_img.size > 0:
                    try:
                        emotion, confidence, scores = self.emotion_detector.detect(face_img)
                        face_info.emotion = emotion
                        face_info.emotion_ja = EMOTION_JA.get(emotion, "不明")
                        face_info.emotion_confidence = confidence
                        face_info.emotion_scores = scores
                    except Exception:
                        pass  # 感情推定失敗は無視

            result.faces.append(face_info)

        # 最も大きい顔（= 最も近い人 = メインユーザー）の感情を dominant とする
        if result.faces:
            largest = max(result.faces, key=lambda f: f.w * f.h)
            result.dominant_emotion = largest.emotion
            result.dominant_emotion_ja = largest.emotion_ja

        return result
