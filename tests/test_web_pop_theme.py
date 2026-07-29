from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "web" / "static"


class HallmarkWebThemeTest(unittest.TestCase):
    def test_all_pages_load_hallmark_runtime_and_routes(self) -> None:
        expected_active = {
            "index.html": "話す",
            "tasks.html": "やること",
            "logs.html": "記録",
            "achievements.html": "実績",
        }
        for filename, label in expected_active.items():
            html = (STATIC / filename).read_text(encoding="utf-8")
            self.assertIn('/static/tokens.css', html)
            self.assertIn('/static/hallmark-theme.css', html)
            self.assertNotIn('/static/pop-theme.css', html)
            self.assertNotIn('/static/shell-theme.css', html)
            self.assertIn(f'class="top-nav-link active" href=', html)
            self.assertIn(f'>{label}</a>', html)
            self.assertIn('content="#171326"', html)
            self.assertEqual(html.count('class="top-nav-link'), 4)
            self.assertIn('/static/shell-ui.js', html)
            self.assertIn('class="site-footer"', html)
            self.assertNotIn('fonts.googleapis.com', html)
            self.assertNotIn('fonts.gstatic.com', html)
            self.assertIn('width=device-width, initial-scale=1.0, viewport-fit=cover', html)
            self.assertNotIn('maximum-scale', html)
            self.assertNotIn('user-scalable', html)

    def test_hallmark_stamp_tokens_and_instrument(self) -> None:
        css = (STATIC / "hallmark-theme.css").read_text(encoding="utf-8")
        self.assertTrue(css.startswith('/* Hallmark · genre: atmospheric · macrostructure: Workbench · theme: Lumen / Night Foundry · enrichment: Tier-A signal instrument · nav: N13 · footer: Ft2 */'))
        for token in (
            'genre: atmospheric',
            'macrostructure: Workbench',
            'theme: Lumen / Night Foundry',
            'enrichment: Tier-A signal instrument',
            'nav: N13',
            'footer: Ft2',
            '.shell-command-dialog',
            'command-crossfade',
        ):
            self.assertIn(token, css)
        tokens = (STATIC / "tokens.css").read_text(encoding="utf-8")
        portable = (ROOT / "tokens.css").read_text(encoding="utf-8")
        self.assertEqual(tokens, portable)
        for token in ('--color-paper-0: oklch', '--color-brass:', '--color-coral:', '--font-display:', '--space-4:', '--text-primary:', '--ease-standard:', '--dur-base:', '--rule-thin:', '--radius-lg:', '--color-grid-line:', '--color-overlay:', '--color-on-brass-muted:'):
            self.assertIn(token, tokens)
        self.assertIn('265)', tokens)
        self.assertIn('50)', tokens)
        self.assertIn('18)', tokens)

    def test_mobile_command_trigger_collapses_without_losing_label(self) -> None:
        css = (STATIC / "hallmark-theme.css").read_text(encoding="utf-8")
        js = (STATIC / "shell-ui.js").read_text(encoding="utf-8")
        self.assertIn('@media (max-width: 700px)', css)
        self.assertIn('.shell-command-trigger span { display: none; }', css)
        self.assertIn("aria-label', '操作を検索してコマンドを開く'", js)

    def test_runtime_css_is_scrollbar_safe(self) -> None:
        css = (STATIC / "style.css").read_text(encoding="utf-8")
        self.assertNotIn('100vw', css)

    def test_hallmark_log_is_newest_first_array(self) -> None:
        log = json.loads((ROOT / ".hallmark" / "log.json").read_text(encoding="utf-8"))
        self.assertIsInstance(log, list)
        self.assertGreater(len(log), 0)
        required = {"date", "build", "macrostructure", "theme", "drop", "enrichment", "summary"}
        dates: list[str] = []
        for entry in log:
            self.assertIsInstance(entry, dict)
            self.assertTrue(required.issubset(entry.keys()), f"missing fields: {required - entry.keys()}")
            self.assertRegex(entry["date"], r"^\d{4}-\d{2}-\d{2}$")
            dates.append(entry["date"])
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_service_worker_caches_hallmark_revision(self) -> None:
        worker = (STATIC / "service-worker.js").read_text(encoding="utf-8")
        self.assertIn("subpc-living-v27", worker)
        self.assertIn("/static/tokens.css", worker)
        self.assertIn("/static/hallmark-theme.css", worker)
        for asset in ("geist-latin.woff2", "instrument-serif-latin.woff2", "jetbrains-mono-latin.woff2", "LICENSES.txt"):
            self.assertIn(f"/static/fonts/{asset}", worker)
            self.assertTrue((STATIC / "fonts" / asset).is_file())
        self.assertNotIn("/static/pop-theme.css", worker)
        self.assertNotIn("/static/shell-theme.css", worker)
        self.assertIn("/static/shell-ui.js", worker)
        self.assertIn("/achievements", worker)

    def test_chat_workbench_keeps_hooks_and_factual_instrument_labels(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        for hook in ('id="chat-area"', 'id="message-input"', 'id="send-btn"', 'id="growth-panel"', 'id="game-hub"'):
            self.assertIn(hook, html)
        for label in ('LOCAL · OLLAMA · CHECKING', 'VOICE · STT · CHECKING', 'VOICE · TTS · CHECKING', 'MEMORY · RAG · CHECKING', 'CONNECTING'):
            self.assertIn(label, html)
        for hook in ('instrument-ollama', 'instrument-stt', 'instrument-tts', 'instrument-rag', 'instrument-caption'):
            self.assertIn(f'id="{hook}"', html)
        self.assertNotIn('READY / PRIVATE', html)
        self.assertNotIn('welcome-orb', html)
        self.assertIn('class="workbench"', html)
        self.assertIn('class="signal-apparatus" role="group"', html)
        self.assertNotIn('class="signal-apparatus" role="img"', html)

    def test_runtime_status_uses_text_content_and_rebuilds_neutral_apparatus(self) -> None:
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        for token in ("updateInstrumentStatus", "status.ollama", "status.stt", "status.tts", "status.rag", "websocket: 'connected'", "websocket: 'disconnected'", "textContent"):
            self.assertIn(token, app)
        self.assertIn("chatArea.replaceChildren(createWelcome", app)
        self.assertNotIn("READY / PRIVATE", app)

    def test_settings_voice_label_is_associated_and_ws_status_lifecycle(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn('for="voice-select"', html)
        self.assertIn('id="voice-select"', html)
        self.assertIn('id="ws-status"', html)
        for token in (
            "websocket: 'connected'",
            "websocket: 'disconnected'",
            "websocket: 'connecting'",
            "'つながっています'",
            "'接続がきれました'",
            "'つないでいます…'",
            "document.getElementById('ws-status')",
        ):
            self.assertIn(token, app)
        self.assertNotIn("'つないでいます...'", app)
        self.assertIn("updateInstrumentStatus({ websocket: 'connecting' })", app)
        self.assertNotIn('>つないでいます...<', html)
        self.assertIn('つないでいます…', html)
        self.assertNotIn('<label>つながり</label>', html)
        self.assertIn('class="conn-label"', html)

    def test_settings_modal_keyboard_contract(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn('role="dialog"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn('aria-labelledby="settings-title"', html)
        self.assertIn('id="settings-title"', html)
        for token in ('settingsRestoreFocus', "settingsPanel.setAttribute('aria-hidden', 'false')", "firstControl?.focus()", 'handleSettingsKeydown', "event.key === 'Escape'", "event.key !== 'Tab'", 'event.shiftKey', 'restoreTarget.focus()', 'settingsPanel.addEventListener(\'keydown\', handleSettingsKeydown)', 'e.target === settingsPanel', 'closeSettings()', 'appShell.inert = true', 'appShell.inert = false', "const appShell = $('.app');"):
            self.assertIn(token, app)

    def test_runtime_css_uses_named_tokens_and_mobile_order(self) -> None:
        css = (STATIC / "style.css").read_text(encoding="utf-8")
        self.assertIn("@font-face", css)
        self.assertIn('font-display: swap', css)
        self.assertIn('.task-edit-form { width: 100%; }', css)
        self.assertIn('.edit-grid > .due-quick, .edit-grid > .edit-actions { grid-column: 1 / -1; }', css)
        self.assertIn('.edit-grid { grid-template-columns: 1fr; }', css)
        self.assertIn('.app { height: 100dvh;', css)
        self.assertIn('.workbench { width:', css)
        self.assertIn('overflow: hidden; display: grid;', css)
        self.assertIn('.conversation-pane { min-width: 0; min-height: 0; overflow: hidden;', css)
        self.assertIn('.chat-area { flex: 1; min-height: 0;', css)
        self.assertIn('.conversation-pane { order: 0;', css)
        self.assertIn('.support-rail { order: 1;', css)
        self.assertIn('.app { height: auto; min-height: 100dvh; overflow: visible;', css)
        self.assertIn('.chat-area { min-height: 0; overflow-y: auto;', css)
        self.assertIn('var(--color-brass-ink); background: var(--brass-bright)', css)
        self.assertIn('.cal-toolbar, .cal-day-head { flex-wrap: wrap;', css)
        self.assertIn('.cal-cell.out { color: var(--text-faint); opacity: .5; }', css)
        for selector in ('.cal-dot.high', '.cal-dot.overdue', '.cal-dot.low', '.cal-dot.gcal'):
            self.assertIn(selector, css)

    def test_achievements_page_keeps_primary_navigation_and_conditions(self) -> None:
        html = (STATIC / "achievements.html").read_text(encoding="utf-8")
        self.assertIn('id="achievement-grid"', html)
        self.assertIn('未解除でも条件を確認できます', html)
        self.assertEqual(html.count('class="top-nav-link'), 4)
        self.assertIn('class="top-nav-link active" href="/achievements"', html)

    def test_shared_header_and_refresh_action_placement(self) -> None:
        chat = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="header site-header"', chat)
        self.assertIn('class="header-actions site-actions"', chat)
        for filename, refresh_id in (("tasks.html", "task-refresh-btn"), ("logs.html", "log-refresh-btn")):
            html = (STATIC / filename).read_text(encoding="utf-8")
            self.assertIn('class="task-header site-header"', html)
            self.assertLess(html.index(f'id="{refresh_id}"'), html.index("<main"))
            self.assertIn('class="page-heading"', html)

    def test_chat_streaming_preserves_manual_scroll_position(self) -> None:
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        for token in (
            "NEAR_BOTTOM_THRESHOLD",
            "let autoFollow",
            "isNearBottom()",
            "if (!autoFollow) return;",
            "forceScrollToBottom",
            "autoFollow = true",
            "trackChatScroll",
        ):
            self.assertIn(token, app)
        # passive scroll tracking only (no force on every token)
        self.assertIn("chatArea.addEventListener('scroll', trackChatScroll, { passive: true })", app)
        # scrollToBottom has both a pre-schedule guard and a re-check inside requestAnimationFrame
        # that precedes the scrollTop assignment; forceScrollToBottom must stay unconditional.
        scroll_fn_idx = app.index("function scrollToBottom")
        scroll_fn_end = app.index("\n}", scroll_fn_idx)
        scroll_fn = app[scroll_fn_idx:scroll_fn_end]
        self.assertEqual(scroll_fn.count("if (!autoFollow) return;"), 2)
        raf_idx = scroll_fn.index("requestAnimationFrame")
        assign_idx = scroll_fn.index("chatArea.scrollTop", raf_idx)
        inner_guard_idx = scroll_fn.index("if (!autoFollow) return;", raf_idx)
        self.assertLess(raf_idx, inner_guard_idx)
        self.assertLess(inner_guard_idx, assign_idx)
        force_fn_idx = app.index("function forceScrollToBottom")
        force_fn_end = app.index("\n}", force_fn_idx)
        force_fn = app[force_fn_idx:force_fn_end]
        self.assertNotIn("if (!autoFollow) return;", force_fn)
        # appendToken must use passive scrollToBottom, not forceScrollToBottom
        append_idx = app.index("function appendToken")
        self.assertIn("scrollToBottom();", app[append_idx:append_idx + 400])
        self.assertNotIn("forceScrollToBottom();", app[append_idx:append_idx + 400])
        # send path re-enables autoFollow before dispatching
        send_idx = app.index("function sendMessage")
        force_idx = app.index("forceScrollToBottom();", send_idx)
        ws_send_idx = app.index("ws.send(JSON.stringify({", send_idx)
        self.assertLess(force_idx, ws_send_idx)
        # new assistant bubble re-arms follow
        bubble_idx = app.index("function createAssistantBubble")
        self.assertIn("forceScrollToBottom();", app[bubble_idx:bubble_idx + 800])
        # finished response keeps passive behavior so manual scroll-up is preserved
        done_idx = app.index("function finishResponse")
        self.assertIn("scrollToBottom();", app[done_idx:done_idx + 800])
        self.assertNotIn("forceScrollToBottom", app[done_idx:done_idx + 800])

    def test_settings_modal_locks_root_scroll_and_restores_on_close(self) -> None:
        css = (STATIC / "style.css").read_text(encoding="utf-8")
        app = (STATIC / "app.js").read_text(encoding="utf-8")
        # dedicated lock class coexists with shell-command-open, applied to both html and body
        self.assertIn('.settings-open { overflow: hidden !important; }', css)
        self.assertIn('overscroll-behavior: contain', css)
        for token in (
            "document.documentElement.classList.toggle('settings-open'",
            "document.body.classList.toggle('settings-open'",
            "setSettingsScrollLock(true)",
            "setSettingsScrollLock(false)",
        ):
            self.assertIn(token, app)
        # open must arm the lock and close must always disarm it
        open_idx = app.index("function openSettings")
        self.assertIn("setSettingsScrollLock(true)", app[open_idx:open_idx + 600])
        close_idx = app.index("function closeSettings")
        self.assertIn("setSettingsScrollLock(false)", app[close_idx:close_idx + 600])
        # shell-command-open must remain present (coexistence contract)
        self.assertIn('shell-command-open', (STATIC / "hallmark-theme.css").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
