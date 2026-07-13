"""
remote モード (RemoteScreenContext) と ingest エンドポイントの検証。

- RemoteScreenContext: 鮮度判定 / ファイル無し / 壊れた JSON
- POST /api/screen/ingest: 認証 (未設定403・不一致403・一致200) /
  マジックバイト検証 / サイズ上限
- ファクトリ create_screen_context のモード切替
- scripts/screen_agent.py の純粋ロジック (should_send / image_hash)

実 Ollama は呼ばない (describer をモック)。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.screen.remote import RemoteScreenContext
from src.screen import create_screen_context
from src.screen.context import ScreenContext


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# --------------------------- RemoteScreenContext ---------------------------

class RemoteScreenContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.latest = Path(self.tmp.name) / "latest.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _ctx(self, **kwargs) -> RemoteScreenContext:
        return RemoteScreenContext(latest_path=self.latest, **kwargs)

    def test_context_text_empty_when_not_running(self):
        _write_json(self.latest, {"description": "作業中", "captured_at": time.time()})
        ctx = self._ctx()
        # start していない (_running=False) → 空
        self.assertEqual(ctx.get_context_text(), "")

    def test_fresh_description_rendered(self):
        _write_json(self.latest, {
            "description": "VSCodeでコードを書いています。",
            "captured_at": time.time(),
            "described_at": time.time(),
            "source": "remote",
        })
        ctx = self._ctx()
        ctx._running = True
        self.assertTrue(ctx._read_once())
        text = ctx.get_context_text()
        self.assertIn("画面情報", text)
        self.assertIn("メインPC", text)
        self.assertIn("VSCodeでコードを書いています。", text)
        self.assertTrue(text.startswith("\n"))

    def test_missing_file_gives_empty(self):
        ctx = self._ctx()
        ctx._running = True
        # ファイルが存在しない
        self.assertFalse(ctx._read_once())
        self.assertEqual(ctx.get_context_text(), "")

    def test_broken_json_keeps_empty(self):
        self.latest.write_text("{ this is not valid json ", encoding="utf-8")
        ctx = self._ctx()
        ctx._running = True
        self.assertFalse(ctx._read_once())
        self.assertEqual(ctx.get_context_text(), "")

    def test_stale_description_gives_empty(self):
        _write_json(self.latest, {
            "description": "古い作業",
            "captured_at": time.time() - 11 * 60,  # 11分前
        })
        ctx = self._ctx(stale_after=600.0)
        ctx._running = True
        ctx._read_once()
        self.assertEqual(ctx.get_context_text(), "")

    def test_within_stale_window_rendered(self):
        _write_json(self.latest, {
            "description": "まだ有効",
            "captured_at": time.time() - 9 * 60,  # 9分前
        })
        ctx = self._ctx(stale_after=600.0)
        ctx._running = True
        ctx._read_once()
        text = ctx.get_context_text()
        self.assertIn("画面情報", text)
        self.assertIn("(9分前時点)", text)

    def test_get_status_fields(self):
        _write_json(self.latest, {"description": "x", "captured_at": time.time()})
        ctx = self._ctx()
        ctx._running = True
        ctx._read_once()
        status = ctx.get_status()
        self.assertEqual(status["mode"], "remote")
        self.assertTrue(status["running"])
        self.assertEqual(status["source"], "remote")
        self.assertIsNotNone(status["age_seconds"])


# --------------------------- ファクトリ ---------------------------

class FactoryTest(unittest.TestCase):
    def test_default_is_local(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCREEN_CONTEXT_MODE", None)
            ctx = create_screen_context()
            self.assertIsInstance(ctx, ScreenContext)

    def test_explicit_remote(self):
        ctx = create_screen_context(mode="remote")
        self.assertIsInstance(ctx, RemoteScreenContext)

    def test_env_remote(self):
        with mock.patch.dict(os.environ, {"SCREEN_CONTEXT_MODE": "remote"}):
            ctx = create_screen_context()
            self.assertIsInstance(ctx, RemoteScreenContext)

    def test_remote_ignores_local_kwargs(self):
        # local 向け kwargs (base_url/model/analysis_interval) を渡しても remote は無視
        ctx = create_screen_context(
            mode="remote",
            analysis_interval=90.0,
            base_url="http://x:1",
            model="foo",
            stale_after=123.0,
        )
        self.assertIsInstance(ctx, RemoteScreenContext)
        self.assertEqual(ctx.stale_after, 123.0)


# --------------------------- ingest エンドポイント ---------------------------

class IngestEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from src.web import server
        cls.server = server
        cls.TestClient = TestClient

    def setUp(self):
        # latest.jpg / latest.json をテンポラリに向ける
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.server.SCREEN_DIR = d
        self.server.SCREEN_LATEST_JPG = d / "latest.jpg"
        self.server.SCREEN_LATEST_JSON = d / "latest.json"
        self.server._ingest_describing = False

        class _FakeDescriber:
            model = "fake-vlm"

            def describe(self, jpeg):
                return "メインPCでブラウザを見ています。"

        self.server.screen_ingest_describer = _FakeDescriber()

    def tearDown(self):
        self.server.screen_ingest_describer = None
        self.tmp.cleanup()

    def _client(self):
        return self.TestClient(self.server.app)

    _JPEG = b"\xff\xd8\xff" + b"\x00" * 200

    def test_token_unset_is_403(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCREEN_INGEST_TOKEN", None)
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "whatever", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)

    def test_token_mismatch_is_403(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "wrong", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 403)

    def test_valid_token_and_jpeg_is_200_and_writes_json(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=self._JPEG,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body["ok"])
            self.assertTrue(body["described"])
            # 画像は即保存されている
            self.assertTrue(self.server.SCREEN_LATEST_JPG.exists())
            # 描写はバックグラウンド → latest.json 生成を待つ
            deadline = time.time() + 5.0
            while time.time() < deadline and not self.server.SCREEN_LATEST_JSON.exists():
                time.sleep(0.05)
            self.assertTrue(self.server.SCREEN_LATEST_JSON.exists())
            data = json.loads(self.server.SCREEN_LATEST_JSON.read_text(encoding="utf-8"))
            self.assertEqual(data["source"], "remote")
            self.assertEqual(data["description"], "メインPCでブラウザを見ています。")
            self.assertIn("captured_at", data)
            self.assertIn("described_at", data)

    def test_bad_magic_bytes_is_400(self):
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=b"not-a-jpeg-at-all",
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 400)

    def test_oversize_is_413(self):
        big = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024 + 1)
        with mock.patch.dict(os.environ, {"SCREEN_INGEST_TOKEN": "secret"}):
            r = self._client().post(
                "/api/screen/ingest",
                content=big,
                headers={"X-Screen-Token": "secret", "Content-Type": "image/jpeg"},
            )
        self.assertEqual(r.status_code, 413)


# --------------------------- screen_agent 純粋ロジック ---------------------------

def _load_screen_agent():
    path = PROJECT_ROOT / "scripts" / "screen_agent.py"
    spec = importlib.util.spec_from_file_location("screen_agent", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ScreenAgentLogicTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = _load_screen_agent()

    def test_image_hash_stable_and_distinct(self):
        a = self.agent
        self.assertEqual(a.image_hash(b"abc"), a.image_hash(b"abc"))
        self.assertNotEqual(a.image_hash(b"abc"), a.image_hash(b"abd"))

    def test_should_send_first_time(self):
        a = self.agent
        self.assertTrue(a.should_send("h1", None, None, now=100.0))

    def test_should_send_when_hash_changed(self):
        a = self.agent
        self.assertTrue(a.should_send("h2", "h1", 100.0, now=110.0, min_resend_interval=600.0))

    def test_skip_when_same_hash_within_interval(self):
        a = self.agent
        self.assertFalse(a.should_send("h1", "h1", 100.0, now=200.0, min_resend_interval=600.0))

    def test_resend_when_same_hash_after_interval(self):
        a = self.agent
        self.assertTrue(a.should_send("h1", "h1", 100.0, now=100.0 + 601.0, min_resend_interval=600.0))

    def test_next_backoff_caps(self):
        a = self.agent
        self.assertEqual(a.next_backoff(200.0), 300.0)  # 400 -> capped at 300
        self.assertEqual(a.next_backoff(5.0), 10.0)


if __name__ == "__main__":
    unittest.main()
