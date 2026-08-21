from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication

    from src.desktop.bridge import DesktopBridge
    from src.desktop.config import DesktopSettings
except ImportError:
    QApplication = None
    DesktopBridge = None
    DesktopSettings = None


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class AvatarModelPropertyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_bridge(self) -> DesktopBridge:
        return DesktopBridge(DesktopSettings(), offline=True)

    def test_avatar_model_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vrm_dir = root / "models" / "vrm"
            vrm_dir.mkdir(parents=True)
            with patch(
                "src.desktop.bridge.resolve_avatar_model",
                return_value=None,
            ):
                bridge = self._make_bridge()
                result = bridge.avatarModel
                self.assertFalse(result["exists"])
                self.assertEqual(result["path"], "")
                bridge.shutdown()

    def test_avatar_model_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vrm_dir = root / "models" / "vrm"
            vrm_dir.mkdir(parents=True)
            model = vrm_dir / "test.glb"
            model.write_bytes(b"glTF")
            with patch(
                "src.desktop.bridge.resolve_avatar_model",
                return_value=model,
            ):
                bridge = self._make_bridge()
                result = bridge.avatarModel
                self.assertTrue(result["exists"])
                self.assertEqual(result["path"], str(model))
                bridge.shutdown()

    def test_avatar_model_caches(self) -> None:
        call_count = 0

        def mock_resolve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Path("/tmp/model.vrm")

        with patch("src.desktop.bridge.resolve_avatar_model", side_effect=mock_resolve):
            bridge = self._make_bridge()
            _ = bridge.avatarModel
            _ = bridge.avatarModel
            _ = bridge.avatarModel
            self.assertEqual(call_count, 1)
            bridge.shutdown()

    def test_refresh_recomputes(self) -> None:
        call_count = 0

        def mock_resolve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return Path("/tmp/model.vrm")

        with patch("src.desktop.bridge.resolve_avatar_model", side_effect=mock_resolve):
            bridge = self._make_bridge()
            _ = bridge.avatarModel
            self.assertEqual(call_count, 1)
            bridge.refreshAvatarModel()
            self.assertEqual(call_count, 2)
            bridge.shutdown()

    def test_exception_returns_safe_default(self) -> None:
        with patch(
            "src.desktop.bridge.resolve_avatar_model",
            side_effect=RuntimeError("boom"),
        ):
            bridge = self._make_bridge()
            result = bridge.avatarModel
            self.assertFalse(result["exists"])
            self.assertEqual(result["path"], "")
            bridge.shutdown()


if __name__ == "__main__":
    unittest.main()
