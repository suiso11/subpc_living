from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.desktop.api import DesktopApi

try:
    from PySide6.QtWidgets import QApplication

    from src.desktop.bridge import DesktopBridge
    from src.desktop.config import DesktopSettings
except ImportError:
    QApplication = None
    DesktopBridge = None
    DesktopSettings = None


class CompanionStatePayloadTest(unittest.TestCase):
    def test_disabled_returns_enabled_false(self) -> None:
        api = DesktopApi("http://test")
        try:
            self.assertEqual(api.companion_state(None), {"enabled": False})
        finally:
            api.close()

    def test_payload_is_privacy_safe(self) -> None:
        runtime = SimpleNamespace(
            status=SimpleNamespace(
                running=True,
                last_state=SimpleNamespace(
                    activity_mode="focused",
                    present=True,
                    focused_since=123.0,
                    interruptible=False,
                    display_state="focused",
                    updated_at=456.0,
                ),
                last_update_at=456.0,
                failure_count=2,
                consecutive_failures=1,
                last_error_type="ValueError",
            )
        )
        api = DesktopApi("http://test")
        try:
            payload = api.companion_state(runtime)
        finally:
            api.close()

        self.assertEqual(
            set(payload),
            {
                "enabled",
                "running",
                "last_update_at",
                "failure_count",
                "consecutive_failures",
                "last_error_type",
                "state",
            },
        )
        self.assertEqual(
            set(payload["state"]),
            {
                "activity_mode",
                "present",
                "focused_since",
                "interruptible",
                "display_state",
                "updated_at",
            },
        )
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["running"])
        self.assertEqual(payload["state"]["activity_mode"], "focused")

        blob = json.dumps(payload, sort_keys=True)
        for forbidden in (
            "process",
            "pid",
            "window",
            "title",
            "app_category",
            "classifier",
            "error_text",
            "traceback",
            "sample",
            "event",
        ):
            self.assertNotIn(forbidden, blob)


@unittest.skipIf(DesktopBridge is None, "PySide6 is not installed")
class DesktopBridgeCompanionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_runtime(self) -> SimpleNamespace:
        runtime = SimpleNamespace(
            stopped=False,
            status=SimpleNamespace(
                running=True,
                last_state=None,
                last_update_at=None,
                failure_count=0,
                consecutive_failures=0,
                last_error_type=None,
            ),
        )

        def stop() -> None:
            runtime.stopped = True

        runtime.stop = stop
        return runtime

    def test_bridge_starts_runtime_and_stops_on_shutdown(self) -> None:
        runtime = self._make_runtime()
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=runtime,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        self.assertIs(bridge._activity_runtime, runtime)
        payload = bridge.companionState
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["running"])
        self.assertIsNone(payload["state"])
        self.assertTrue(bridge.companion_timer.isActive())

        bridge.shutdown()
        self.assertTrue(runtime.stopped)

    def test_disabled_runtime_is_hidden_and_stops_nothing(self) -> None:
        with patch(
            "src.desktop.bridge.create_activity_runtime_from_env",
            return_value=None,
        ):
            bridge = DesktopBridge(DesktopSettings(), offline=True)

        self.assertIsNone(bridge._activity_runtime)
        self.assertEqual(bridge.companionState, {"enabled": False})
        self.assertFalse(bridge.companion_timer.isActive())
        bridge.shutdown()


if __name__ == "__main__":
    unittest.main()
