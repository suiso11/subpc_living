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


    def test_overlay_qml_has_required_flags_and_objectnames(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn("Qt.FramelessWindowHint", source)
        self.assertIn("Qt.WindowStaysOnTopHint", source)
        self.assertIn("Qt.Tool", source)
        self.assertIn('color: "transparent"', source)
        for name in (
            "overlayRoot",
            "overlayAvatar",
            "overlayHud",
            "overlayStateLabel",
            "overlayProvenance",
            "overlayContent",
            "overlayStatusBanner",
            "overlayRecentMessages",
            "overlayStarterChips",
            "overlayComposer",
            "overlaySendButton",
        ):
            self.assertIn(f'objectName: "{name}"', source)

    def test_overlay_qml_provenance_labels(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        for label in ("出所", "取得", "保存"):
            self.assertIn(label, source)

    def test_overlay_qml_no_vrm_references(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8").lower()
        self.assertNotIn("vrm", source)
        self.assertNotIn(".vrm", source)

    def test_overlay_qml_action_buttons(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        for label in ("閉じる", "本体を開く", "停止"):
            self.assertIn(label, source)

    def test_overlay_qml_state_labels_japanese(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        for label in ("待機中", "作業中", "会話中", "離席中", "予定が近づいています", "エラー"):
            self.assertIn(label, source)

    def test_overlay_qml_send_message_contract(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertEqual(source.count("sendMessage("), 1)
        self.assertIn("composerField.text.trim()", source)
        self.assertIn("composerField.clear()", source)
        self.assertIn("onAccepted: overlayRoot.send()", source)
        self.assertIn("enabled: overlayRoot.canSend", source)
        self.assertIn("property bool canSend: overlayRoot.connected && !overlayRoot.loading", source)

    def test_overlay_qml_escape_collapses_and_expand_focuses_composer(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn("Keys.onEscapePressed", source)
        self.assertIn("forceActiveFocus", source)
        self.assertIn("onExpandedChanged", source)
        self.assertIn("overlayRoot.expanded = false", source)

    def test_overlay_qml_status_banner_has_no_secrets(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn('objectName: "overlayStatusBanner"', source)
        for label in ("オフライン", "接続済み", "読み込み中…"):
            self.assertIn(label, source)
        self.assertNotIn("statusText", source)
        self.assertNotIn("serverUrl", source)

    def test_overlay_qml_hides_avatar_when_expanded(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn("active: overlayRoot.hasAvatar3D && !overlayRoot.expanded", source)
        self.assertIn("visible: !overlayRoot.hasAvatar3D && !overlayRoot.expanded", source)

    def test_overlay_qml_message_list_is_bounded(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn("result.length < 4", source)
        self.assertIn("maximumLineCount: 2", source)
        self.assertIn("Math.min(52, messageText.implicitHeight + 12)", source)
        self.assertIn("clip: true", source)

    def test_overlay_qml_no_new_endpoint_strings(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertNotIn("/api/", source)
        self.assertNotIn("/ws/", source)
        self.assertNotIn("http://", source)
        self.assertNotIn("https://", source)

    def test_overlay_qml_summary_uses_allowlisted_counts_only(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        for name in (
            "overlayTodaySummary",
            "overlaySummaryGrowth",
            "overlaySummaryTasks",
            "overlaySummaryCalendar",
            "overlaySummaryGrowthValue",
            "overlaySummaryTasksValue",
            "overlaySummaryCalendarValue",
        ):
            self.assertIn(f'objectName: "{name}"', source)
        self.assertIn("today_points", source)
        self.assertIn("streak_days", source)
        self.assertIn("Accessible.description", source)
        summary_block = source[source.index("overlayTodaySummary") : source.index("overlayRecentMessages")]
        for label in ("今日の成長", "タスク", "予定"):
            self.assertIn(label, summary_block)
        self.assertNotIn("modelData", summary_block)
        self.assertNotIn("statusText", summary_block)
        self.assertNotIn("monitor", summary_block)
        self.assertNotIn("snapshot", summary_block)
        self.assertNotIn("context", summary_block)
        self.assertNotIn("process", summary_block)
        self.assertNotIn("sensor", summary_block)
        self.assertNotIn("path", summary_block)
        self.assertNotIn("model", summary_block)

    def test_overlay_qml_click_through_control_contract(self) -> None:
        source = (QML / "Overlay.qml").read_text(encoding="utf-8")
        self.assertIn('objectName: "overlayClickThrough"', source)
        self.assertIn('objectName: "overlayClickThroughHint"', source)
        self.assertIn("overlayBridge.overlayClickThrough", source)
        self.assertEqual(source.count("setOverlayClickThrough("), 1)
        self.assertIn("Ctrl+Alt+Space で解除", source)
        self.assertIn("クリックスルー", source)
        self.assertIn("overlayRoot.collapse()", source)
        self.assertIn("Accessible.description", source)
        for detail in ("hwnd", "winId", "HOTKEY_ID", "apply_click_through", "WS_EX"):
            self.assertNotIn(detail, source)
        for label in ("閉じる", "本体を開く", "停止"):
            self.assertIn(label, source)
        self.assertEqual(source.count("sendMessage("), 1)


class DesktopClickThroughContractTest(unittest.TestCase):
    """Static wiring contract for the overlay click-through bridge."""

    def test_app_wires_overlay_click_through_controller(self) -> None:
        source = (DESKTOP / "app.py").read_text(encoding="utf-8")
        self.assertIn("OverlayClickThroughController(bridge)", source)
        self.assertIn("overlayClickThroughRequested.connect(self._on_requested)", source)
        self.assertIn("def set_overlay_window", source)
        self.assertIn("apply_default_click_through", source)
        self.assertIn("setOverlayClickThrough(False)", source)
        self.assertIn("restore_interaction", source)

    def test_hotkey_restores_interaction_before_toggle(self) -> None:
        source = (DESKTOP / "app.py").read_text(encoding="utf-8")
        handler = source[
            source.index("def _on_hotkey_activated") : source.index("def _show_window")
        ]
        self.assertLess(
            handler.index("restore_interaction()"),
            handler.index("_toggle_window(window)"),
        )
        self.assertIn(
            "hotkey.activated.connect(lambda: _on_hotkey_activated(click_through, window))",
            source,
        )

    def test_overlay_failure_and_shutdown_disconnect_controller(self) -> None:
        source = (DESKTOP / "app.py").read_text(encoding="utf-8")
        self.assertIn("click_through.disconnect()", source)
        self.assertIn("click_through.set_overlay_window(None)", source)
        self.assertIn("app.aboutToQuit.connect(click_through.disconnect)", source)
        self.assertIn("QTimer.singleShot(250, click_through.apply_default_click_through)", source)


if __name__ == "__main__":
    unittest.main()
