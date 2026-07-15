from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from src.tasks.store import TaskStore
from src.web import server


class _Request:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class WebTaskStepsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = TaskStore(str(Path(self.tmp.name) / "tasks.db")).initialize()
        self.original_store = server.task_store
        server.task_store = self.store

    def tearDown(self) -> None:
        server.task_store = self.original_store
        self.store.close()
        self.tmp.cleanup()

    def test_step_endpoint_updates_and_returns_check_state(self) -> None:
        task_id = self.store.add("実験レポートを書く")
        result = asyncio.run(server.tasks_step_done(task_id, 0, _Request({"done": True})))
        self.assertTrue(result["task"]["step_done"][0])
        self.assertTrue(self.store.get(task_id)["step_done"][0])

    def test_step_endpoint_requires_boolean(self) -> None:
        task_id = self.store.add("実験レポートを書く")
        response = asyncio.run(server.tasks_step_done(task_id, 0, _Request({"done": 1})))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.body)["error"], "done must be boolean")

    def test_task_page_renders_step_checkboxes_and_calls_endpoint(self) -> None:
        static = Path(server.__file__).parent / "static"
        js = (static / "tasks.js").read_text(encoding="utf-8")
        css = (static / "style.css").read_text(encoding="utf-8")
        self.assertIn('data-action="step-done"', js)
        self.assertIn("/steps/${stepIndex}", js)
        self.assertIn("task-step-check input:checked + span", css)


if __name__ == "__main__":
    unittest.main()
