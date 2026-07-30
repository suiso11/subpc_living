from __future__ import annotations

import asyncio
import json
import struct
import unittest
from pathlib import Path

from src.web import server


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "web" / "static"


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"not a PNG: {path}")
    return struct.unpack(">II", data[16:24])


class AndroidPwaTest(unittest.TestCase):
    def test_manifest_has_android_install_identity_and_icons(self) -> None:
        manifest = json.loads((STATIC / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["id"], "/")
        self.assertEqual(manifest["start_url"], "/?source=pwa")
        self.assertEqual(manifest["scope"], "/")
        self.assertEqual(manifest["display"], "standalone")
        self.assertIn("standalone", manifest["display_override"])
        self.assertFalse(manifest["prefer_related_applications"])

        icons = {(icon["sizes"], icon["purpose"]): icon for icon in manifest["icons"]}
        self.assertIn(("192x192", "any"), icons)
        self.assertIn(("512x512", "any"), icons)
        self.assertIn(("512x512", "maskable"), icons)
        self.assertNotIn("any maskable", {icon["purpose"] for icon in manifest["icons"]})
        for icon in manifest["icons"]:
            path = STATIC / Path(icon["src"]).name
            self.assertTrue(path.is_file(), path)
            expected = int(icon["sizes"].split("x", 1)[0])
            self.assertEqual(png_size(path), (expected, expected))

    def test_install_ui_uses_android_prompt_with_manual_fallback(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        css = (STATIC / "style.css").read_text(encoding="utf-8")
        for token in (
            'id="pwa-install-card"', 'id="pwa-install-status"',
            'id="pwa-install-btn"', "この端末に追加",
            'role="status"', 'aria-live="polite"', 'aria-atomic="true"',
        ):
            self.assertIn(token, html)
        for token in (
            "beforeinstallprompt", "event.preventDefault()", "deferredInstallPrompt",
            "deferredInstallPrompt.prompt()", "deferredInstallPrompt.userChoice",
            "appinstalled", "(display-mode: standalone)", "Chromeの︙メニュー",
            "!pwaInstallButton.hidden",
            "settingsPanel?.classList.contains('open')",
            "$('#settings-close')?.focus({ preventScroll: true })",
            "pwaInstallButton.disabled) return",
        ):
            self.assertIn(token, js)
        for token in (
            ".pwa-install-card", ".pwa-install-btn:active",
            ".pwa-install-btn:disabled", 'data-state="success"',
            'data-state="error"',
        ):
            self.assertIn(token, css)

    def test_worker_is_served_and_registered_at_root_scope(self) -> None:
        response = asyncio.run(server.service_worker())
        self.assertEqual(Path(response.path), STATIC / "service-worker.js")
        self.assertEqual(response.media_type, "application/javascript")
        self.assertEqual(response.headers["service-worker-allowed"], "/")
        self.assertEqual(response.headers["cache-control"], "no-cache")
        shell_js = (STATIC / "shell-ui.js").read_text(encoding="utf-8")
        self.assertIn("navigator.serviceWorker.register('/service-worker.js', { scope: '/' })", shell_js)
        for entry in ("app.js", "tasks.js", "logs.js", "achievements.js"):
            entry_js = (STATIC / entry).read_text(encoding="utf-8")
            self.assertNotIn("serviceWorker.register", entry_js)
            self.assertNotIn("register('/static/service-worker.js'", entry_js)

    def test_service_worker_v30_precaches_android_assets(self) -> None:
        worker = (STATIC / "service-worker.js").read_text(encoding="utf-8")
        self.assertIn("subpc-living-v30", worker)
        self.assertIn("'/?source=pwa'", worker)
        for name in ("icon-192.png", "icon-512.png", "icon-maskable-512.png", "manifest.json"):
            self.assertIn(f"/static/{name}", worker)
        self.assertIn("request.mode === 'navigate' && url.search", worker)
        self.assertIn("fetch(request).catch", worker)
        branch_start = worker.index("request.mode === 'navigate' && url.search")
        branch_end = worker.index("  event.respondWith(\n    caches.match(request)", branch_start)
        query_branch = worker[branch_start:branch_end]
        self.assertNotIn("cache.put", query_branch)
        self.assertIn("caches.match('/')", query_branch)

    def test_usage_documents_private_android_install(self) -> None:
        usage = (ROOT / "usage.md").read_text(encoding="utf-8")
        for token in (
            "Androidへアプリとして追加", "AndroidのTailscale", "この端末に追加",
            "ホーム画面に追加", "一般公開せず利用できる",
        ):
            self.assertIn(token, usage)


if __name__ == "__main__":
    unittest.main()
