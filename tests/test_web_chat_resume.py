from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi.responses import JSONResponse

from src.web import server


class WebChatResumeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.history_dir = Path(self.tmp.name)
        self.original_config = server.config
        self.original_sessions = server.sessions
        server.config = SimpleNamespace(
            system_prompt="現在のプロンプト",
            max_history_turns=20,
            history_dir=str(self.history_dir),
            emotion_tag_enabled=False,
        )
        server.sessions = {}

    def tearDown(self) -> None:
        server.sessions = self.original_sessions
        server.config = self.original_config
        self.tmp.cleanup()

    def test_session_is_saved_under_browser_id_and_restored_after_memory_clear(self) -> None:
        session = server.get_or_create_session("web_keep")
        session.add_user_message("前の話")
        session.add_assistant_message("覚えています")
        saved = session.save()
        self.assertEqual(saved.name, "session_web_keep.json")

        server.sessions.clear()  # Webサービス再起動相当
        restored = server.get_or_create_session("web_keep")
        self.assertEqual(restored.messages[-2]["content"], "前の話")
        self.assertEqual(restored.messages[-1]["content"], "覚えています")
        self.assertEqual(restored.system_prompt, "現在のプロンプト")

    def test_resume_with_explicit_new_id_does_not_fall_back_to_latest(self) -> None:
        old = server.get_or_create_session("web_old")
        old.add_user_message("古い会話")
        old.add_assistant_message("古い返答")
        old.save()
        server.sessions.clear()

        result = asyncio.run(server.chat_resume("web_new"))
        self.assertEqual(result["session_id"], "web_new")
        self.assertEqual(result["messages"], [])

    def test_resume_without_id_uses_latest_valid_history(self) -> None:
        old = server.get_or_create_session("web_old")
        old.add_user_message("続けたい話")
        old.add_assistant_message("続き")
        old.save()
        server.sessions.clear()

        result = asyncio.run(server.chat_resume())
        self.assertEqual(result["session_id"], "web_old")
        self.assertEqual([m["content"] for m in result["messages"]], ["続けたい話", "続き"])

    def test_resume_rejects_unsafe_id(self) -> None:
        result = asyncio.run(server.chat_resume("../secret"))
        self.assertIsInstance(result, JSONResponse)
        self.assertEqual(result.status_code, 400)
        self.assertEqual(json.loads(result.body)["error"], "invalid session_id")

    def test_new_session_ids_differ_under_same_timestamp(self) -> None:
        fixed_ts = 1700000000.123
        with mock.patch.object(server.time, "time", return_value=fixed_ts):
            a = server._new_web_session_id()
            b = server._new_web_session_id()
        self.assertNotEqual(a, b)

    def test_new_session_id_format_is_valid(self) -> None:
        with mock.patch.object(server.time, "time", return_value=1700000000.123):
            sid = server._new_web_session_id()
        self.assertRegex(sid, r"^web_\d+_[0-9a-f]{8}$")
        self.assertTrue(server.history_admin.is_safe_session_id(sid))
        self.assertEqual(sid.split("_")[1], "1700000000123")

    def test_chat_resume_without_history_uses_unique_new_id(self) -> None:
        with mock.patch.object(server.time, "time", return_value=1700000000.123):
            first = asyncio.run(server.chat_resume())
            second = asyncio.run(server.chat_resume())
        self.assertEqual(first["messages"], [])
        self.assertEqual(second["messages"], [])
        self.assertNotEqual(first["session_id"], second["session_id"])
        for result in (first, second):
            self.assertRegex(
                result["session_id"], r"^web_\d+_[0-9a-f]{8}$"
            )
            self.assertTrue(
                server.history_admin.is_safe_session_id(result["session_id"])
            )


class WebChatResumeRouteTest(unittest.TestCase):
    """/api/chat/resume ルーティング回帰テスト (TestClient 経由)。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.history_dir = Path(self.tmp.name)
        self.original_config = server.config
        self.original_sessions = server.sessions
        server.config = SimpleNamespace(
            system_prompt="現在のプロンプト",
            max_history_turns=20,
            history_dir=str(self.history_dir),
            emotion_tag_enabled=False,
        )
        server.sessions = {}
        from starlette.testclient import TestClient

        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        self.client.close()
        server.sessions = self.original_sessions
        server.config = self.original_config
        self.tmp.cleanup()

    def test_get_returns_session_object_with_messages(self) -> None:
        resp = self.client.get("/api/chat/resume")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body, dict)
        self.assertIn("session_id", body)
        self.assertIn("messages", body)
        self.assertIsInstance(body["messages"], list)
        self.assertTrue(server.history_admin.is_safe_session_id(body["session_id"]))

    def test_query_session_id_is_honored(self) -> None:
        session = server.get_or_create_session("web_keep")
        session.add_user_message("続けたい話")
        session.add_assistant_message("続き")
        session.save()

        resp = self.client.get("/api/chat/resume", params={"session_id": "web_keep"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["session_id"], "web_keep")
        self.assertEqual([m["content"] for m in body["messages"]], ["続けたい話", "続き"])

    def test_query_session_id_is_validated(self) -> None:
        resp = self.client.get(
            "/api/chat/resume", params={"session_id": "../secret"}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "invalid session_id")

    def test_route_is_bound_to_chat_resume_not_helper(self) -> None:
        route = next(
            r for r in server.app.routes if getattr(r, "path", None) == "/api/chat/resume"
        )
        self.assertEqual(route.endpoint, server.chat_resume)
        self.assertNotEqual(route.endpoint, server._new_web_session_id)

    def test_helper_is_plain_callable_not_an_endpoint(self) -> None:
        endpoints = {
            getattr(r, "endpoint", None) for r in server.app.routes
        }
        self.assertNotIn(server._new_web_session_id, endpoints)
        sid = server._new_web_session_id()
        self.assertRegex(sid, r"^web_\d+_[0-9a-f]{8}$")


class WebClientSessionIdStaticTest(unittest.TestCase):
    """app.js のクライアント側セッションID生成を静的検査する回帰テスト。

    JS はこのPythonスイートから直接実行できないため、ソースを文字列検査し、
    「ミリ秒のみの web_${Date.now()} リテラルが残っていない」ことと、
    「衝突耐性ヘルパーが存在しサーバー安全ID検証に適合する形式を生成する」こと
    を保証する。
    """

    APP_JS = Path(__file__).resolve().parents[1] / "src" / "web" / "static" / "app.js"

    def setUp(self) -> None:
        self.source = self.APP_JS.read_text(encoding="utf-8")

    def test_no_millisecond_only_session_id_literal_remains(self) -> None:
        # ミリ秒のみの旧形式リテラル (テンプレート直閉じ) が残っていないこと。
        # newWebSessionId 内部の `web_${Date.now()}_${hex}` はサフィックス付きなので対象外。
        self.assertNotIn("`web_${Date.now()}`", self.source)
        self.assertNotIn("sessionId = `web_${Date.now()}`", self.source)

    def test_client_uses_new_session_id_helper_at_all_creation_sites(self) -> None:
        self.assertEqual(self.source.count("= newWebSessionId();"), 2)

    def test_helper_is_defined_and_prefers_crypto_getrandomvalues(self) -> None:
        self.assertIn("function newWebSessionId()", self.source)
        self.assertIn("crypto.getRandomValues(bytes)", self.source)
        self.assertIn("return `web_${Date.now()}_${hex}`;", self.source)

    def test_helper_has_non_secure_fallback(self) -> None:
        self.assertIn("Math.random()", self.source)

    def test_generated_format_matches_server_safe_validation(self) -> None:
        sid = "web_" + "1700000000123" + "_" + "ab" * 8
        self.assertRegex(sid, r"^web_\d+_[0-9a-f]{16}$")
        self.assertTrue(server.history_admin.is_safe_session_id(sid))
        self.assertLessEqual(len(sid), 128)


if __name__ == "__main__":
    unittest.main()
