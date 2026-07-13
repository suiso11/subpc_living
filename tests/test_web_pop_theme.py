from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "web" / "static"


class WebShellThemeTest(unittest.TestCase):
    def test_all_pages_load_shell_theme_and_use_plain_japanese_nav(self) -> None:
        expected_active = {
            "index.html": "話す",
            "tasks.html": "やること",
            "logs.html": "記録",
            "achievements.html": "実績",
        }
        for filename, label in expected_active.items():
            html = (STATIC / filename).read_text(encoding="utf-8")
            self.assertIn('/static/pop-theme.css', html)
            self.assertIn('/static/shell-theme.css', html)
            self.assertLess(html.index('/static/pop-theme.css'), html.index('/static/shell-theme.css'))
            self.assertIn(f'class="top-nav-link active" href=', html)
            self.assertIn(f'>{label}</a>', html)
            self.assertIn('content="#120B0A"', html)
            self.assertEqual(html.count('class="top-nav-link'), 4)
            self.assertIn('/static/shell-ui.js', html)

    def test_shell_theme_has_context_panels_and_restrained_accent_tokens(self) -> None:
        css = (STATIC / "shell-theme.css").read_text(encoding="utf-8")
        for token in (
            "--shell-bg: #120b0a",
            "--shell-panel:",
            "--shell-accent: #ffafa1",
            ".site-header",
            ".task-workspace",
            ".shell-command-dialog",
            "backdrop-filter",
            "prefers-reduced-motion",
        ):
            self.assertIn(token, css)
        self.assertIn("linear-gradient(", css)
        self.assertIn("radial-gradient(", css)

    def test_service_worker_caches_theme_revision(self) -> None:
        worker = (STATIC / "service-worker.js").read_text(encoding="utf-8")
        self.assertIn("subpc-living-v22", worker)
        self.assertIn("/static/shell-theme.css", worker)
        self.assertIn("/static/shell-ui.js", worker)
        self.assertIn("/achievements", worker)
        self.assertIn("/static/achievements.js", worker)

    def test_achievements_page_keeps_primary_navigation_and_conditions(self) -> None:
        html = (STATIC / "achievements.html").read_text(encoding="utf-8")
        self.assertIn('/static/shell-theme.css', html)
        self.assertIn('id="achievement-grid"', html)
        self.assertIn('未解除でも条件を確認できます', html)
        self.assertEqual(html.count('class="top-nav-link'), 4)
        self.assertIn('class="top-nav-link active" href="/achievements"', html)

    def test_shared_header_and_refresh_action_placement(self) -> None:
        chat = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="header site-header"', chat)
        self.assertIn('class="header-actions site-actions"', chat)

        for filename, refresh_id in (
            ("tasks.html", "task-refresh-btn"),
            ("logs.html", "log-refresh-btn"),
        ):
            html = (STATIC / filename).read_text(encoding="utf-8")
            self.assertIn('class="task-header site-header"', html)
            self.assertLess(html.index(f'id="{refresh_id}"'), html.index("<main"))
            self.assertIn('class="page-heading"', html)


if __name__ == "__main__":
    unittest.main()
