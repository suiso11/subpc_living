from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.web import server


ROOT = Path(__file__).resolve().parents[1]


class WebMicTest(unittest.TestCase):
    def tearDown(self) -> None:
        server.get_secure_web_url.cache_clear()

    def test_configured_https_url_is_exposed_without_trailing_slash(self) -> None:
        server.get_secure_web_url.cache_clear()
        with patch.dict(
            os.environ,
            {"WEB_PUBLIC_HTTPS_URL": "https://buddy.example.test/"},
        ):
            self.assertEqual(
                server.get_secure_web_url(),
                "https://buddy.example.test",
            )

    def test_frontend_guards_insecure_context_and_links_to_https(self) -> None:
        js = (ROOT / "src/web/static/app.js").read_text(encoding="utf-8")
        self.assertIn("window.isSecureContext", js)
        self.assertIn("navigator.mediaDevices?.getUserMedia", js)
        self.assertIn("status.secure_web_url", js)
        self.assertIn("安全な接続で開く", js)
        self.assertIn("サイト設定でマイクを許可", js)


if __name__ == "__main__":
    unittest.main()
