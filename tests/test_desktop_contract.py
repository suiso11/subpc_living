from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "src" / "desktop"
QML = DESKTOP / "qml"


class DesktopContractTest(unittest.TestCase):
    def test_native_qml_has_all_primary_surfaces(self) -> None:
        expected = {
            "Main.qml",
            "ChatPage.qml",
            "TasksPage.qml",
            "CalendarPage.qml",
            "LogsPage.qml",
            "AchievementsPage.qml",
            "CommandPalette.qml",
            "Theme.qml",
            "BuddyButton.qml",
            "BuddyComboBox.qml",
        }
        self.assertTrue(expected.issubset({path.name for path in QML.glob("*.qml")}))

    def test_main_shell_has_native_navigation_and_shortcuts(self) -> None:
        source = (QML / "Main.qml").read_text(encoding="utf-8")
        for label in ("話す", "やること", "記録", "実績"):
            self.assertIn(label, source)
        for shortcut in ("Ctrl+K", "Alt+1", "Alt+2", "Alt+3", "Alt+4"):
            self.assertIn(shortcut, source)
        self.assertIn("Qt.FramelessWindowHint", source)
        self.assertNotIn("WebEngineView", source)
        self.assertNotIn("WebView", source)

    def test_bridge_uses_api_websocket_and_native_audio(self) -> None:
        bridge = (DESKTOP / "bridge.py").read_text(encoding="utf-8")
        api = (DESKTOP / "api.py").read_text(encoding="utf-8")
        self.assertIn("QWebSocket", bridge)
        self.assertIn("NativeAudioRecorder", bridge)
        self.assertIn("nativeNotification.emit", bridge)
        self.assertIn('"/api/tasks"', api)
        self.assertIn('"/api/game"', api)
        self.assertIn('"/api/chat/resume"', api)
        self.assertIn("/ws/chat", api)
        self.assertNotIn("from src.web import server", bridge)

    def test_native_client_keeps_web_feature_parity_routes_and_actions(self) -> None:
        api = (DESKTOP / "api.py").read_text(encoding="utf-8")
        bridge = (DESKTOP / "bridge.py").read_text(encoding="utf-8")
        qml = "\n".join(path.read_text(encoding="utf-8") for path in QML.glob("*.qml"))
        for route in (
            "/api/growth",
            "/api/tasks/preview",
            "/snooze",
            "/api/calendar/events",
            "/api/tts",
            "/api/tts/voice",
            "/api/logs/files",
        ):
            self.assertIn(route, api)
        for action in (
            "newSession",
            "replayText",
            "previewTask",
            "snoozeTask",
            "loadCalendar",
            "createCalendarEvent",
            "loadLogFiles",
            "deleteHistory",
            "setTtsVoice",
        ):
            self.assertIn(action, bridge)
            self.assertIn(action, qml)
        for label in ("新しい会話", "予定表", "30分後", "アプリ", "解除済み"):
            self.assertIn(label, qml)

    def test_windows_distribution_files_are_present(self) -> None:
        requirements = (ROOT / "requirements-desktop.txt").read_text(encoding="utf-8")
        script = (ROOT / "scripts" / "build_windows_desktop.ps1").read_text(encoding="utf-8")
        spec = (ROOT / "subpc-desktop.spec").read_text(encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" / "windows-desktop.yml").read_text(encoding="utf-8")
        self.assertIn("PySide6", requirements)
        self.assertIn("pyinstaller", script.lower())
        self.assertIn("Start-Process", script)
        self.assertIn("SUBPC-BUDDY", spec)
        self.assertIn("--smoke-test", workflow)
        icon = ROOT / "src" / "desktop" / "assets" / "app-icon.ico"
        self.assertTrue(icon.is_file())
        self.assertEqual(icon.read_bytes()[:4], b"\x00\x00\x01\x00")
        self.assertIn("app-icon.ico", spec)
        self.assertIn("windows-version-info.txt", spec)


if __name__ == "__main__":
    unittest.main()
