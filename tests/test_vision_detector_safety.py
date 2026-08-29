"""
Vision 検出器のログ安全性を検証する。

モデルロード・推論のログが固定文言 + 例外型名のみに限定され、
生の例外本文・モデルパス・プロバイダー・デバイス等の可変情報を含まず、
かつ CP932 でエンコード可能 (ASCII 安全) であることを検証する。

実カメラ・ONNX・OpenCV・モデル・ファイル・ネットワークは使わず、全て mock で代替する。
"""
from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.vision import detector


class FakeInput:
    name = "input"
    shape = [1, 1, 64, 64]


class FakeSession:
    def __init__(self, *args, **kwargs):
        self._inputs = [FakeInput()]

    def get_inputs(self):
        return self._inputs


class _StubFaceDetectorEmpty:
    def __init__(self):
        pass

    def detect(self, frame):
        return []


class _StubFaceDetectorRects:
    def __init__(self):
        pass

    def detect(self, frame):
        return [(10, 10, 40, 40)]


class _StubEmotion:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, face_image):
        raise RuntimeError("secret inference error detail")


class _BoomDetector:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("secret raw load error detail")


def _assert_cp932_safe(testcase, lines):
    for line in lines:
        line.encode("cp932")  # CP932 でエンコード可能でなければ例外が上がる


class VisionDetectorSafetyTest(unittest.TestCase):
    def test_emotion_detector_load_logs_fixed_message_only(self):
        with (
            patch.object(detector, "HAS_ORT", True),
            patch.object(detector, "ort") as ort_mock,
            patch.object(
                detector, "_detect_onnx_providers", return_value=["CPUExecutionProvider"]
            ),
            patch.object(Path, "exists", return_value=True),
            self.assertLogs(detector.__name__, level=logging.INFO) as cm,
        ):
            ort_mock.InferenceSession = FakeSession
            detector.EmotionDetector("secret/model.onnx")

        joined = "\n".join(cm.output)
        self.assertIn("emotion model loaded", joined)
        # プロバイダー名・デバイス・モデルパスの可変情報は含まれない
        self.assertNotIn("CPUExecutionProvider", joined)
        self.assertNotIn("CUDA", joined)
        self.assertNotIn("secret/model.onnx", joined)
        _assert_cp932_safe(self, cm.output)

    def test_vision_analyzer_load_failure_logs_type_only(self):
        with (
            patch.object(detector, "FaceDetector", _StubFaceDetectorEmpty),
            patch.object(detector, "EmotionDetector", _BoomDetector),
            patch.object(Path, "exists", return_value=True),
            self.assertLogs(detector.__name__, level=logging.WARNING) as cm,
        ):
            analyzer = detector.VisionAnalyzer(emotion_model_path="secret/model.onnx")

        self.assertFalse(analyzer.has_emotion)
        joined = "\n".join(cm.output)
        self.assertIn("emotion model load failed", joined)
        # 例外型名のみ (type-only)。生の例外本文・モデルパスは含まれない
        self.assertIn("RuntimeError", joined)
        self.assertNotIn("secret raw load error detail", joined)
        self.assertNotIn("secret/model.onnx", joined)
        _assert_cp932_safe(self, cm.output)

    def test_vision_analyzer_inference_failure_logs_type_only_once(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch.object(detector, "FaceDetector", _StubFaceDetectorRects),
            patch.object(detector, "EmotionDetector", _StubEmotion),
            patch.object(Path, "exists", return_value=True),
            self.assertLogs(detector.__name__, level=logging.WARNING) as cm,
        ):
            analyzer = detector.VisionAnalyzer(emotion_model_path="dummy.onnx")
            analyzer.analyze(frame)
            analyzer.analyze(frame)

        joined = "\n".join(cm.output)
        self.assertIn("emotion inference failed", joined)
        self.assertIn("RuntimeError", joined)
        self.assertNotIn("secret inference error detail", joined)
        # 初回のみ警告 (以降サイレント)
        self.assertEqual(joined.count("emotion inference failed"), 1)
        _assert_cp932_safe(self, cm.output)


if __name__ == "__main__":
    unittest.main()
