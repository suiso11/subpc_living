from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "web" / "static"


class WebShellUiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.js = (STATIC / "shell-ui.js").read_text(encoding="utf-8")

    def test_key_handling(self) -> None:
        js = self.js
        self.assertIn("event.key", js)
        self.assertIn("'k'", js)
        # Ctrl/Cmd+K トグル
        self.assertTrue(re.search(r"event\.(ctrl|meta)Key", js))
        self.assertIn("Escape", js)
        self.assertIn("ArrowUp", js)
        self.assertIn("ArrowDown", js)
        self.assertIn("Enter", js)

    def test_alt_number_routes(self) -> None:
        js = self.js
        self.assertIn("altKey", js)
        pairs = [
            ("1", "/"),
            ("2", "/tasks"),
            ("3", "/logs"),
            ("4", "/achievements"),
        ]
        for n, route in pairs:
            self.assertIn(n, js)
            self.assertIn(route, js)

    def test_all_routes_present(self) -> None:
        js = self.js
        for route in ("/", "/tasks", "/logs", "/achievements"):
            self.assertIn(route, js)

    def test_forward_slash_focus_targets(self) -> None:
        js = self.js
        self.assertIn('#message-input', js)
        self.assertIn('#task-text-input', js)

    def test_accessibility_attributes(self) -> None:
        js = self.js
        for token in (
            "setAttribute('role', 'dialog')",
            "setAttribute('role', 'listbox')",
            "setAttribute('role', 'option')",
            'aria-selected',
            'aria-label',
        ):
            self.assertIn(token, js)

    def test_no_inner_html_assignment(self) -> None:
        js = self.js
        self.assertNotRegex(js, r"innerHTML\s*=", msg="innerHTML への直接代入は禁止")
        self.assertNotRegex(js, r"insertAdjacentHTML\s*\(", msg="insertAdjacentHTML は禁止")

    def test_common_command_labels(self) -> None:
        js = self.js
        for label in ("話す", "やること", "記録", "実績", "タスク追加", "操作を検索"):
            self.assertIn(label, js)
        self.assertIn("⌘K", js)
        self.assertIn("shell-command-trigger", js)

    def test_page_specific_command_labels(self) -> None:
        js = self.js
        for label in (
            "タスクを読み直す",
            "表示をリストにする",
            "表示をカレンダーにする",
            "記録を読み直す",
            "入力欄に集中",
        ):
            self.assertIn(label, js)

    def test_only_text_content_no_user_input_into_html(self) -> None:
        js = self.js
        # DOM API のみで構築
        self.assertIn("createElement", js)
        self.assertIn("textContent", js)
        self.assertIn("appendChild", js)
        self.assertIn("setAttribute", js)

    def test_no_global_pollution(self) -> None:
        js = self.js
        # IIFE で囲まれている
        self.assertTrue(js.strip().startswith("(() =>"))
        self.assertTrue(js.strip().endswith("})();"))

    def test_dialog_hidden_initial(self) -> None:
        js = self.js
        self.assertIn("hidden", js)

    def test_arrow_selection_keeps_searchbox_focus(self) -> None:
        js = self.js
        self.assertIn("function selectOption(state, nextIndex, keepInputFocus = false)", js)
        self.assertIn("selectOption(state, currentVisible + 1, true)", js)
        self.assertIn("selectOption(state, currentVisible - 1, true)", js)
        self.assertIn("if (keepInputFocus) state.input.focus();", js)
        self.assertNotIn("selected.button.focus()", js)

    def test_tab_is_trapped_in_search_dialog(self) -> None:
        js = self.js
        self.assertIn("if (event.key === 'Tab')", js)
        self.assertIn("event.preventDefault();", js)
        self.assertIn("state.input.focus();", js)

    def test_settings_modal_guard_helper_exists(self) -> None:
        js = self.js
        self.assertIn("function settingsModalOpen()", js)
        self.assertIn("#settings-panel", js)
        self.assertIn("aria-hidden", js)
        self.assertIn("classList.contains('open')", js)

    def test_settings_modal_guard_blocks_global_shortcuts(self) -> None:
        js = self.js
        # グローバル keydown の冒頭で設定モーダルを検知し early return する
        self.assertIn("settingsModalOpen() && state.backdrop.hidden", js)
        # 設定が閉じている通常時はショートカットが変わらない
        self.assertIn("(event.ctrlKey || event.metaKey) && key === 'k'", js)
        self.assertIn("event.altKey && ROUTES[event.key]", js)
        self.assertIn("event.key === '/' && !isEditable(event.target)", js)

    def test_settings_modal_guard_excludes_shell_palette(self) -> None:
        js = self.js
        # パレット自体は誤判定しない：settingsModalOpen は設定パネルだけを見る
        self.assertRegex(js, r"document\.querySelector\('#settings-panel'\)")
        # open 判定後に通常のハンドラへ進む
        self.assertRegex(js, r"if \(settingsModalOpen\(\) && state\.backdrop\.hidden\)")


if __name__ == "__main__":
    unittest.main()
