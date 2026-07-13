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
        for label in ("話す", "やること", "記録", "実績", "タスク追加"):
            self.assertIn(label, js)

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


if __name__ == "__main__":
    unittest.main()
