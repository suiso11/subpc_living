from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.tasks.store import TaskStore
from src.web import server


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return json.dumps(self.payload, ensure_ascii=False)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


class WebTaskCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(
            str(Path(self.tmp.name) / "tasks.db"), timezone_name="Asia/Tokyo"
        ).initialize()
        self.original_store = server.task_store
        self.original_llm = server.llm
        self.original_config = server.config
        server.task_store = self.store
        server.config = SimpleNamespace(num_ctx=4096)

    def tearDown(self) -> None:
        server.task_store = self.original_store
        server.llm = self.original_llm
        server.config = self.original_config
        self.store.close()
        self.tmp.cleanup()

    def test_extracts_multiple_candidates_after_response_contract(self) -> None:
        server.llm = _FakeLLM({
            "tasks": [
                {"is_task": True, "title": "レポートを直す", "due": None, "priority": "high"},
                {"is_task": True, "title": "先生へメールする", "due": None, "priority": "normal"},
            ]
        })
        ws = _FakeWebSocket()
        with patch.dict("os.environ", {"TASKS_CHAT_EXTRACTION_ENABLED": "true"}):
            asyncio.run(server._offer_task_candidates(
                ws,
                user_text="レポートを直して先生へメールしないと",
                session_id="web-test",
                turn=2,
            ))
        self.assertEqual([m["type"] for m in ws.sent], ["task_candidate", "task_candidate"])
        self.assertEqual(len(self.store.list_candidates("pending")), 2)
        call = server.llm.calls[0]
        self.assertEqual(call["temperature"], 0.0)
        self.assertLessEqual(call["num_predict"], 256)
        self.assertEqual(call["timeout"], 15.0)

    def test_existing_pending_candidate_is_not_offered_twice(self) -> None:
        server.llm = _FakeLLM({
            "tasks": [{"is_task": True, "title": "掃除する", "due": None, "priority": "normal"}]
        })
        first = _FakeWebSocket()
        second = _FakeWebSocket()
        with patch.dict("os.environ", {"TASKS_CHAT_EXTRACTION_ENABLED": "true"}):
            asyncio.run(server._offer_task_candidates(
                first, user_text="掃除しないと", session_id="web-test", turn=2
            ))
            asyncio.run(server._offer_task_candidates(
                second, user_text="掃除しないと", session_id="web-test", turn=2
            ))
        self.assertEqual(len(first.sent), 1)
        self.assertEqual(second.sent, [])
        self.assertEqual(len(self.store.list_candidates("pending")), 1)

    def test_feature_flag_disables_extraction(self) -> None:
        server.llm = _FakeLLM({
            "tasks": [{"is_task": True, "title": "掃除する", "due": None, "priority": "normal"}]
        })
        ws = _FakeWebSocket()
        with patch.dict("os.environ", {"TASKS_CHAT_EXTRACTION_ENABLED": "false"}):
            asyncio.run(server._offer_task_candidates(
                ws, user_text="掃除しないと", session_id="web-test", turn=2
            ))
        self.assertEqual(ws.sent, [])
        self.assertEqual(server.llm.calls, [])

    def test_sensitive_user_text_never_reaches_extractor_model(self) -> None:
        server.llm = _FakeLLM({"tasks": []})
        sample = "API token sk-" + "a" * 24
        with patch.dict("os.environ", {"TASKS_CHAT_EXTRACTION_ENABLED": "true"}):
            self.assertEqual(server._extract_task_candidates(sample), [])
        self.assertEqual(server.llm.calls, [])

    def test_date_only_granularity_reaches_candidate_and_task(self) -> None:
        server.llm = _FakeLLM({
            "tasks": [{
                "is_task": True,
                "title": "提出する",
                "due": "2026-08-03",
                "priority": "high",
            }]
        })
        ws = _FakeWebSocket()
        with patch.dict("os.environ", {"TASKS_CHAT_EXTRACTION_ENABLED": "true"}):
            asyncio.run(server._offer_task_candidates(
                ws, user_text="来週までに提出する", session_id="web-test", turn=2
            ))
        candidate = self.store.list_candidates("pending")[0]
        self.assertEqual(candidate["due_granularity"], "date")
        self.assertEqual(ws.sent[0]["candidate"]["due_granularity"], "date")
        accepted = asyncio.run(server.task_candidate_accept(candidate["id"]))
        self.assertEqual(accepted["task"]["due_granularity"], "date")

    def test_background_boundary_swallows_failure_and_timeout(self) -> None:
        async def raises(*args, **kwargs):
            raise RuntimeError("sample failure")

        async def hangs(*args, **kwargs):
            await asyncio.sleep(1)

        ws = _FakeWebSocket()
        with patch.object(server, "_offer_task_candidates", raises):
            asyncio.run(server._run_task_candidate_offer(
                ws, user_text="x", session_id="web-test", turn=2
            ))
        with patch.object(server, "_offer_task_candidates", hangs), \
             patch.object(server, "_extraction_timeout_seconds", return_value=-1.999):
            asyncio.run(server._run_task_candidate_offer(
                ws, user_text="x", session_id="web-test", turn=2
            ))

    def test_list_accept_and_dismiss_api(self) -> None:
        accept_id = self.store.create_candidate(title="提出する", source="chat")
        dismiss_id = self.store.create_candidate(title="片付ける", source="chat")

        listed = asyncio.run(server.task_candidates_list())
        self.assertEqual(len(listed["candidates"]), 2)

        accepted = asyncio.run(server.task_candidate_accept(accept_id))
        self.assertTrue(accepted["ok"])
        self.assertTrue(accepted["created"])
        self.assertEqual(accepted["task"]["title"], "提出する")
        repeated = asyncio.run(server.task_candidate_accept(accept_id))
        self.assertFalse(repeated["created"])

        dismissed = asyncio.run(server.task_candidate_dismiss(dismiss_id))
        self.assertTrue(dismissed["ok"])
        self.assertTrue(dismissed["changed"])
        repeated_dismiss = asyncio.run(server.task_candidate_dismiss(dismiss_id))
        self.assertFalse(repeated_dismiss["changed"])
        self.assertEqual(asyncio.run(server.task_candidates_list())["candidates"], [])

    def test_missing_and_conflicting_candidate_statuses(self) -> None:
        missing = asyncio.run(server.task_candidate_accept(9999))
        self.assertEqual(missing.status_code, 404)
        candidate_id = self.store.create_candidate(title="見送る候補")
        asyncio.run(server.task_candidate_dismiss(candidate_id))
        conflict = asyncio.run(server.task_candidate_accept(candidate_id))
        self.assertEqual(conflict.status_code, 409)

    def test_static_ui_and_done_order_contract(self) -> None:
        root = Path(server.PROJECT_ROOT)
        app = (root / "src/web/static/app.js").read_text(encoding="utf-8")
        tasks_html = (root / "src/web/static/tasks.html").read_text(encoding="utf-8")
        tasks_js = (root / "src/web/static/tasks.js").read_text(encoding="utf-8")
        style = (root / "src/web/static/style.css").read_text(encoding="utf-8")
        source = Path(server.__file__).read_text(encoding="utf-8")
        for token in ("case 'task_candidate'", "renderCandidateCard", "textContent", "/api/tasks/candidates/"):
            self.assertIn(token, app)
        self.assertIn('id="task-inbox"', tasks_html)
        for token in ("loadInbox", "acceptInboxCandidate", "dismissInboxCandidate", "inboxState", "taskInboxRetry"):
            self.assertIn(token, tasks_js)
        self.assertIn('role="status"', tasks_html)
        self.assertIn('id="task-inbox-retry"', tasks_html)
        for token in (".candidate-card", ".task-inbox-row", "@media (max-width: 700px)"):
            self.assertIn(token, style)
        done_pos = source.index('"type": "done"', source.index("async def websocket_chat"))
        offer_pos = source.index("_launch_task_candidate_offer(", done_pos)
        tts_pos = source.index("# TTS", offer_pos)
        self.assertLess(done_pos, offer_pos)
        self.assertLess(offer_pos, tts_pos)
        websocket_body = source[source.index("async def websocket_chat"):]
        self.assertNotIn("await _offer_task_candidates", websocket_body)


if __name__ == "__main__":
    unittest.main()
