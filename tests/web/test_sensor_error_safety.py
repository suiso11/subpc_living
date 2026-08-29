"""Web センサー (Vision/Screen/ingest/Monitor/Activity/mic) の例外安全テスト。

canary な secret・パス・画像内容を例外に混ぜて注入し、ログ・JSON・status へ
漏れないこと、例外の型名 (allowlist) だけが残ること、リソース解放・ゲートが
機能し続けることを検証する。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.web import server

CANARY_PATH = r"C:\Users\canary\secret\passwd_8231.txt"
CANARY_TOKEN = "super-secret-canary-token-7f3a"
CANARY_EXC = f"boom at {CANARY_PATH} token={CANARY_TOKEN}"


def _raising(fn):
    def _inner(*args, **kwargs):
        raise RuntimeError(CANARY_EXC)
    return _inner


class _RaisingStt:
    def __init__(self) -> None:
        self.loaded = True

    def is_loaded(self) -> bool:
        return True

    def transcribe(self, audio, rate):
        raise RuntimeError(CANARY_EXC)


class _FakeIdle:
    def __init__(self) -> None:
        self.starts = 0
        self.ends = 0

    def notify_inference_start(self, wait_for_gpu: bool = False) -> None:
        self.starts += 1

    def notify_inference_end(self) -> None:
        self.ends += 1


class SensorInitFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_config = server.config

    def tearDown(self) -> None:
        server.config = self.original_config

    def test_vision_init_failure_logs_type_only(self) -> None:
        policy = SimpleNamespace(camera=True)
        with mock.patch.object(
            server, "VisionContext", side_effect=RuntimeError(CANARY_EXC)
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            result = server._init_vision_from_policy(policy)
        self.assertIsNone(result)
        logged = "\n".join(cm.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(CANARY_PATH, logged)
        self.assertNotIn(CANARY_TOKEN, logged)
        self.assertNotIn(CANARY_EXC, logged)

    def test_screen_init_failure_logs_type_only(self) -> None:
        server.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434", model="llava"
        )
        policy = SimpleNamespace(screen_capture=True)
        with mock.patch.object(
            server, "create_screen_context", side_effect=RuntimeError(CANARY_EXC)
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            result = server._init_screen_from_policy(
                policy,
                config=server.config,
                primary_provider_kind="ollama",
            )
        self.assertIsNone(result)
        logged = "\n".join(cm.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(CANARY_PATH, logged)
        self.assertNotIn(CANARY_TOKEN, logged)

    def test_monitor_init_failure_logs_type_only(self) -> None:
        policy = SimpleNamespace(monitor=True)
        with mock.patch.object(
            server, "MonitorContext", side_effect=RuntimeError(CANARY_EXC)
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            result = server._init_monitor_from_policy(policy)
        self.assertIsNone(result)
        logged = "\n".join(cm.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(CANARY_PATH, logged)
        self.assertNotIn(CANARY_TOKEN, logged)

    def test_disabled_policy_skips_construction(self) -> None:
        with mock.patch.object(
            server, "VisionContext", side_effect=AssertionError("must not construct")
        ):
            self.assertIsNone(
                server._init_vision_from_policy(SimpleNamespace(camera=False))
            )
        with mock.patch.object(
            server, "MonitorContext", side_effect=AssertionError("must not construct")
        ):
            self.assertIsNone(
                server._init_monitor_from_policy(SimpleNamespace(monitor=False))
            )


class SensorPartialStartCleanupTest(unittest.TestCase):
    """start が false / 例外で失敗したときの best-effort stop と exactly-once を検証。

    構築済み context を破棄する前に stop() をちょうど1回呼び、成功して保持する
    context は stop() しない (終了時の二重 stop を防ぐ)。cleanup 例外は生の詳細を
    漏らさず型名のみをログする。
    """

    def setUp(self) -> None:
        self.original_config = server.config
        server.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434", model="llava"
        )

    def tearDown(self) -> None:
        server.config = self.original_config

    def _fake(self, *, start_return=True, start_raise=False, stop_raise=False):
        fake = mock.Mock()
        if start_raise:
            fake.start.side_effect = RuntimeError(CANARY_EXC)
        else:
            fake.start.return_value = start_return
        if stop_raise:
            fake.stop.side_effect = RuntimeError(CANARY_EXC)
        return fake

    def _assert_stopped_exactly_once(self, fake) -> None:
        fake.stop.assert_called_once_with()

    def test_vision_start_false_stops_once(self) -> None:
        fake = self._fake(start_return=False)
        with mock.patch.object(server, "VisionContext", return_value=fake):
            self.assertIsNone(server._init_vision_from_policy(SimpleNamespace(camera=True)))
        self._assert_stopped_exactly_once(fake)

    def test_vision_start_raise_stops_once_and_logs_type_only(self) -> None:
        fake = self._fake(start_raise=True)
        with mock.patch.object(
            server, "VisionContext", return_value=fake
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            self.assertIsNone(server._init_vision_from_policy(SimpleNamespace(camera=True)))
        self._assert_stopped_exactly_once(fake)
        self.assertNotIn(CANARY_PATH, "\n".join(cm.output))
        self.assertNotIn(CANARY_TOKEN, "\n".join(cm.output))

    def test_vision_start_true_does_not_stop(self) -> None:
        fake = self._fake(start_return=True)
        fake.get_status.return_value = {"emotion_detection": True}
        with mock.patch.object(
            server, "VisionContext", return_value=fake
        ), mock.patch.object(server.time, "sleep"):
            self.assertIs(
                server._init_vision_from_policy(SimpleNamespace(camera=True)), fake
            )
        fake.stop.assert_not_called()

    def test_screen_start_false_stops_once(self) -> None:
        fake = self._fake(start_return=False)
        with mock.patch.object(server, "create_screen_context", return_value=fake):
            self.assertIsNone(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                )
            )
        self._assert_stopped_exactly_once(fake)

    def test_screen_start_raise_stops_once(self) -> None:
        fake = self._fake(start_raise=True)
        with mock.patch.object(server, "create_screen_context", return_value=fake):
            self.assertIsNone(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                )
            )
        self._assert_stopped_exactly_once(fake)

    def test_screen_start_true_does_not_stop(self) -> None:
        fake = self._fake(start_return=True)
        fake.get_status.return_value = {
            "mode": "local",
            "model": "llava",
            "analysis_interval": 90.0,
        }
        with mock.patch.object(server, "create_screen_context", return_value=fake):
            self.assertIs(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                ),
                fake,
            )
        fake.stop.assert_not_called()

    def test_monitor_start_false_stops_once(self) -> None:
        fake = self._fake(start_return=False)
        with mock.patch.object(server, "MonitorContext", return_value=fake):
            self.assertIsNone(server._init_monitor_from_policy(SimpleNamespace(monitor=True)))
        self._assert_stopped_exactly_once(fake)

    def test_monitor_start_raise_stops_once(self) -> None:
        fake = self._fake(start_raise=True)
        with mock.patch.object(server, "MonitorContext", return_value=fake):
            self.assertIsNone(server._init_monitor_from_policy(SimpleNamespace(monitor=True)))
        self._assert_stopped_exactly_once(fake)

    def test_monitor_start_true_does_not_stop(self) -> None:
        fake = self._fake(start_return=True)
        with mock.patch.object(server, "MonitorContext", return_value=fake):
            self.assertIs(
                server._init_monitor_from_policy(SimpleNamespace(monitor=True)), fake
            )
        fake.stop.assert_not_called()

    def test_stop_raise_does_not_mask_and_logs_type_only(self) -> None:
        for entrypoint in ("vision", "screen", "monitor"):
            with self.subTest(entrypoint=entrypoint):
                fake = self._fake(start_return=False, stop_raise=True)
                with self.assertLogs(server.logger, level=logging.WARNING) as cm:
                    if entrypoint == "vision":
                        with mock.patch.object(server, "VisionContext", return_value=fake):
                            result = server._init_vision_from_policy(SimpleNamespace(camera=True))
                    elif entrypoint == "screen":
                        with mock.patch.object(
                            server, "create_screen_context", return_value=fake
                        ):
                            result = server._init_screen_from_policy(
                                SimpleNamespace(screen_capture=True),
                                config=server.config,
                                primary_provider_kind="ollama",
                            )
                    else:
                        with mock.patch.object(server, "MonitorContext", return_value=fake):
                            result = server._init_monitor_from_policy(SimpleNamespace(monitor=True))
                self.assertIsNone(result)
                logged = "\n".join(cm.output)
                self.assertIn("cleanup failed", logged)
                self.assertIn("RuntimeError", logged)
                self.assertNotIn(CANARY_PATH, logged)
                self.assertNotIn(CANARY_TOKEN, logged)
                self.assertNotIn(CANARY_EXC, logged)


class WebmDecodeFailureTest(unittest.TestCase):
    def test_ffmpeg_failure_raises_fixed_message(self) -> None:
        proc = SimpleNamespace(
            returncode=1,
            stderr=f"error opening {CANARY_PATH}".encode(),
        )
        with mock.patch.object(server.subprocess, "run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                server._decode_webm_bytes(b"\x1a\x45\xdf\xa3fake-webm")
        message = str(ctx.exception)
        self.assertEqual(message, "ffmpeg decode failed")
        self.assertNotIn(CANARY_PATH, message)


class WebTtsFailurePrivacyTest(unittest.TestCase):
    def test_http_tts_failure_uses_fixed_internal_error(self) -> None:
        from starlette.testclient import TestClient

        original_tts = server.tts
        server.tts = SimpleNamespace(
            synthesize=lambda text: (_ for _ in ()).throw(RuntimeError(CANARY_EXC))
        )
        try:
            with self.assertLogs(server.logger, level=logging.WARNING) as cm:
                response = TestClient(server.app).post(
                    "/api/tts", json={"text": "tts transcript canary"}
                )
        finally:
            server.tts = original_tts

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"error": "TTS error", "error_type": "internal_error"}
        )
        diagnostics = "\\n".join(cm.output)
        for secret in (CANARY_EXC, CANARY_PATH, CANARY_TOKEN, "tts transcript canary"):
            self.assertNotIn(secret, response.text)
            self.assertNotIn(secret, diagnostics)

    def test_websocket_tts_failure_keeps_reply_and_uses_fixed_internal_error(self) -> None:
        from starlette.testclient import TestClient

        class Session:
            system_prompt = ""

            def __init__(self) -> None:
                self._messages = []

            @property
            def messages(self):
                return list(self._messages)

            def add_user_message(self, text: str) -> None:
                self._messages.append({"role": "user", "content": text})

            def add_assistant_message(self, text: str, **kwargs) -> None:
                self._messages.append({"role": "assistant", "content": text})

            def save(self) -> None:
                pass

        original = {
            "tts": server.tts,
            "sessions": server.sessions,
        }
        session_id = "canary-session-8231"
        reply = "websocket response canary 8231"
        server.sessions = {session_id: Session()}
        server.tts = SimpleNamespace(
            synthesize=lambda text: (_ for _ in ()).throw(RuntimeError(CANARY_EXC))
        )
        try:
            client = TestClient(server.app)
            with mock.patch.object(
                server, "_try_edit_task_text", return_value=reply
            ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
                with client.websocket_connect("/ws/chat") as ws:
                    ws.send_json({
                        "type": "message",
                        "text": CANARY_EXC,
                        "session_id": session_id,
                        "tts": True,
                    })
                    token = ws.receive_json()
                    done = ws.receive_json()
                    error = ws.receive_json()
        finally:
            server.tts = original["tts"]
            server.sessions = original["sessions"]

        self.assertEqual(token, {"type": "token", "content": reply})
        self.assertEqual(done, {"type": "done", "full_text": reply})
        self.assertEqual(
            error,
            {"type": "error", "message": "TTS error", "error_type": "internal_error"},
        )
        diagnostics = "\\n".join(cm.output)
        for secret in (CANARY_EXC, CANARY_PATH, CANARY_TOKEN, reply):
            self.assertNotIn(secret, json.dumps(error))
            self.assertNotIn(secret, diagnostics)


class WebSessionDiagnosticPrivacyTest(unittest.TestCase):
    def test_session_load_failure_caplog_omits_id_path_and_raw_text(self) -> None:
        session_id = "load-canary-session-8231"
        replacement = SimpleNamespace(session_id=session_id)
        config = SimpleNamespace(max_history_turns=4)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            server, "sessions", {}
        ), mock.patch.object(server, "config", config), mock.patch.object(
            server, "_history_dir_path", return_value=Path(tmp)
        ), mock.patch.object(
            server.history_admin, "session_file_for", return_value=Path(CANARY_PATH)
        ), mock.patch.object(
            server.history_admin, "read_session_by_id", return_value=True
        ), mock.patch.object(
            server.ChatSession,
            "load",
            side_effect=RuntimeError(CANARY_EXC),
        ), mock.patch.object(
            server, "_new_chat_session", return_value=replacement
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            result = server.get_or_create_session(session_id)

        self.assertIs(result, replacement)
        diagnostics = "\\n".join(cm.output)
        for secret in (session_id, CANARY_PATH, CANARY_TOKEN, CANARY_EXC):
            self.assertNotIn(secret, diagnostics)

    def test_session_save_failure_caplog_omits_id_path_and_raw_text(self) -> None:
        session_id = "save-canary-session-8231"
        reply = "save reply canary 8231"
        session = mock.Mock()
        session.save.side_effect = RuntimeError(CANARY_EXC)
        websocket = SimpleNamespace(send_json=mock.AsyncMock())

        with mock.patch.object(server, "get_or_create_session", return_value=session), self.assertLogs(
            server.logger, level=logging.WARNING
        ) as cm:
            asyncio.run(server._send_direct_chat_reply(
                websocket,
                session_id=session_id,
                user_text=CANARY_EXC,
                reply=reply,
                want_tts=False,
            ))

        diagnostics = "\\n".join(cm.output)
        for secret in (session_id, CANARY_PATH, CANARY_TOKEN, CANARY_EXC, reply):
            self.assertNotIn(secret, diagnostics)
        sent = [call.args[0] for call in websocket.send_json.await_args_list]
        self.assertEqual(sent[0], {"type": "token", "content": reply})
        self.assertEqual(sent[1], {"type": "done", "full_text": reply})


class WebSensorEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original = {
            name: getattr(server, name)
            for name in (
                "stt",
                "vision",
                "screen",
                "monitor",
                "sensor_policy",
                "idle_manager",
                "_ingest_active_generation",
                "_ingest_future",
                "_ingest_generation",
                "_ingest_accepting",
                "_ingest_done_events",
            )
        }
        server.stt = None
        server.vision = None
        server.screen = None
        server.monitor = None
        server.sensor_policy = None
        server.idle_manager = None
        server._ingest_active_generation = None
        server._ingest_future = None
        server._ingest_generation = 0
        server._ingest_accepting = True
        server._ingest_done_events = {}
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        # 実行中 ingest worker が残っている場合は bounded に待ち、後続テストへ干渉させない。
        generation = server._ingest_active_generation
        if generation is not None:
            event = server._ingest_done_events.get(generation)
            if event is not None:
                event.wait(2.0)
        for name, value in self.original.items():
            setattr(server, name, value)

    def test_stt_endpoint_never_leaks_exception(self) -> None:
        server.stt = _RaisingStt()
        server.sensor_policy = server.SensorPolicy(microphone=True)
        with mock.patch.object(server, "_decode_wav_bytes", return_value=[0.0]):
            r = self.client.post(
                "/api/stt",
                json={"audio": base64.b64encode(b"fake-wav").decode("ascii")},
            )
        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertEqual(body["error"], "STT error")
        self.assertEqual(body["error_type"], "internal_error")
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_vision_status_never_leaks_exception(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            get_status=_raising(lambda: None),
        )
        r = self.client.get("/api/vision/status")
        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertEqual(body["error"], "Vision status unavailable")
        self.assertEqual(body["error_type"], "internal_error")
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_vision_context_disabled_no_data(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            get_context_text=_raising(lambda: None),
            get_status=_raising(lambda: None),
        )
        r = self.client.get("/api/vision/context")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_vision_snapshot_disabled_no_data(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            camera=SimpleNamespace(get_jpeg=lambda: b"\xff\xd8\xffcanary-jpeg-bytes"),
        )
        r = self.client.get("/api/vision/snapshot")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)
        self.assertNotIn("canary-jpeg-bytes", r.text)
        self.assertNotIn("image/jpeg", r.headers.get("content-type", ""))

    def test_screen_status_never_leaks_exception(self) -> None:
        server.screen = SimpleNamespace(
            get_context_text=_raising(lambda: None),
            get_status=_raising(lambda: None),
        )
        with mock.patch.object(server, "_remove_legacy_latest_jpg"):
            r = self.client.get("/api/screen/status")
        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertEqual(body["error"], "Screen status unavailable")
        self.assertNotIn(CANARY_PATH, r.text)

    def test_screen_context_disabled_no_data(self) -> None:
        server.screen = SimpleNamespace(
            get_context_text=lambda: "canary screen context text",
            get_status=lambda: {},
        )
        r = self.client.get("/api/screen/context")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("canary screen context text", r.text)
        self.assertNotIn(CANARY_PATH, r.text)

    def test_vision_status_allowlist_only(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            get_status=lambda: {
                "running": True,
                "paused": False,
                "stop_pending": False,
                "thread_alive": True,
                "user_present": True,
                "person_count": 2,
                "emotion": "happiness",
                "emotion_ja": "嬉しそう",
                "emotion_detection": True,
                "analysis_interval": 2.0,
                "analysis_count": 10,
            },
        )
        r = self.client.get("/api/vision/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["enabled"])
        for leak in (
            "emotion",
            "emotion_ja",
            "person_count",
            "analysis_interval",
            "analysis_count",
        ):
            self.assertNotIn(leak, body)
        self.assertTrue(body["running"])
        self.assertTrue(body["user_present"])
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_screen_status_allowlist_only(self) -> None:
        server.screen = SimpleNamespace(
            get_status=lambda: {
                "running": True,
                "paused": False,
                "description": "canary screen VLM description",
                "captured_at": 100.0,
                "age_seconds": 5.0,
                "source": "local",
                "model": "canary-vlm-model",
                "consecutive_failures": 0,
                "analysis_count": 1,
            },
        )
        with mock.patch.object(server, "_remove_legacy_latest_jpg"):
            r = self.client.get("/api/screen/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["enabled"])
        self.assertEqual(body["source"], "local")
        for leak in (
            "description",
            "model",
            "context",
            "consecutive_failures",
            "analysis_count",
        ):
            self.assertNotIn(leak, body)
        self.assertTrue(body["running"])
        self.assertNotIn("canary screen VLM description", r.text)
        self.assertNotIn(CANARY_PATH, r.text)

    def test_screen_ingest_status_excludes_vlm_description(self) -> None:
        server.sensor_policy = SimpleNamespace(screen_ingest=True)
        server.screen = None
        fake_json = mock.Mock()
        fake_json.exists.return_value = True
        fake_json.read_text.return_value = json.dumps({
            "description": "canary ingest VLM description",
            "captured_at": time.time() - 3,
            "described_at": time.time(),
            "source": "remote",
        })
        with mock.patch.dict(
            server.os.environ, {"SCREEN_INGEST_TOKEN": "sekret"}, clear=False
        ), mock.patch.object(server, "SCREEN_LATEST_JSON", fake_json), mock.patch.object(
            server, "_remove_legacy_latest_jpg"
        ):
            r = self.client.get("/api/screen/status")
        self.assertEqual(r.status_code, 200)
        ingest = r.json()["ingest"]
        self.assertTrue(ingest["enabled"])
        self.assertTrue(ingest["available"])
        self.assertEqual(ingest["source"], "remote")
        self.assertNotIn("description", ingest)
        self.assertNotIn("canary ingest VLM description", r.text)

    def test_screen_ingest_save_failure_never_leaks_exception(self) -> None:
        server.sensor_policy = SimpleNamespace(screen_ingest=True)
        fake_dir = mock.Mock()
        fake_dir.mkdir.side_effect = RuntimeError(CANARY_PATH)
        with mock.patch.dict(
            server.os.environ, {"SCREEN_INGEST_TOKEN": "sekret"}, clear=False
        ), mock.patch.object(server, "SCREEN_DIR", fake_dir):
            r = self.client.post(
                "/api/screen/ingest",
                headers={"X-Screen-Token": "sekret"},
                content=b"\xff\xd8\xffcanary-jpeg-bytes",
            )
        self.assertEqual(r.status_code, 500)
        body = r.json()
        self.assertEqual(body["error"], "screen ingest save failed")
        self.assertEqual(body["error_type"], "internal_error")
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)
        self.assertNotIn("canary-jpeg-bytes", r.text)

    def test_screen_ingest_gate_forbids_when_disabled(self) -> None:
        server.sensor_policy = SimpleNamespace(screen_ingest=False)
        with mock.patch.dict(
            server.os.environ, {"SCREEN_INGEST_TOKEN": "sekret"}, clear=False
        ):
            r = self.client.post(
                "/api/screen/ingest",
                headers={"X-Screen-Token": "sekret"},
                content=b"\xff\xd8\xffcanary-jpeg-bytes",
            )
        self.assertEqual(r.status_code, 403)

    def test_monitor_status_never_leaks_exception(self) -> None:
        server.monitor = SimpleNamespace(
            is_running=True,
            get_status=_raising(lambda: None),
            get_context_text=_raising(lambda: None),
            get_recent_summary=_raising(lambda minutes=60: None),
        )
        r = self.client.get("/api/monitor/status")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"], "Monitor status unavailable")
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)
        # context / summary は廃止され固定 404 でデータを返さない (例外も露出しない)
        for path in ("/api/monitor/context", "/api/monitor/summary"):
            with self.subTest(path=path):
                r = self.client.get(path)
                self.assertEqual(r.status_code, 404)
                self.assertNotIn(CANARY_PATH, r.text)
                self.assertNotIn(CANARY_TOKEN, r.text)

    def test_monitor_status_allowlist_only(self) -> None:
        server.monitor = SimpleNamespace(
            is_running=True,
            get_status=lambda: {
                "running": True,
                "last_collected": 1234.5,
                "collect_interval": 30.0,
                "record_count": 42,
                "cpu_percent": 55.0,
                "mem_percent": 70.0,
                "mem_used_gb": 8.1,
                "gpu_util_percent": 20.0,
                "gpu_mem_used_mb": 512,
                "gpu_temp_c": 60.0,
                "cpu_temp_c": 70.0,
                "disk_percent": 66.0,
                "process_count": 300,
                "device": "canary-device",
                "path": CANARY_PATH,
                "last_error": CANARY_EXC,
            },
        )
        r = self.client.get("/api/monitor/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["enabled"])
        self.assertEqual(body["source"], "monitor")
        self.assertTrue(body["running"])
        self.assertEqual(body["last_collected"], 1234.5)
        for leak in (
            "collect_interval",
            "record_count",
            "cpu_percent",
            "mem_percent",
            "mem_used_gb",
            "gpu_util_percent",
            "gpu_mem_used_mb",
            "gpu_temp_c",
            "cpu_temp_c",
            "disk_percent",
            "process_count",
            "device",
            "path",
            "last_error",
        ):
            self.assertNotIn(leak, body)
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_monitor_context_disabled_no_data(self) -> None:
        server.monitor = SimpleNamespace(
            is_running=True,
            get_context_text=lambda: "canary monitor context text",
            get_recent_summary=lambda minutes=60: {"cpu_avg": 99.9},
        )
        r = self.client.get("/api/monitor/context")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("canary monitor context text", r.text)
        self.assertNotIn(CANARY_PATH, r.text)

    def test_monitor_summary_disabled_no_data(self) -> None:
        server.monitor = SimpleNamespace(
            is_running=True,
            get_recent_summary=lambda minutes=60: {"cpu_avg": 99.9},
        )
        r = self.client.get("/api/monitor/summary")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("99.9", r.text)
        self.assertNotIn(CANARY_PATH, r.text)

    def test_status_endpoint_monitor_status_is_allowlist(self) -> None:
        server.monitor = SimpleNamespace(
            is_running=True,
            get_status=lambda: {
                "running": True,
                "last_collected": 1234.0,
                "cpu_percent": 55.0,
                "process_count": 300,
                "device": CANARY_PATH,
            },
        )
        server.vision = None
        with mock.patch.object(server, "get_secure_web_url", return_value=""):
            r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        ms = r.json().get("monitor_status")
        self.assertIsNotNone(ms)
        self.assertEqual(ms["source"], "monitor")
        self.assertTrue(ms["running"])
        self.assertEqual(ms["last_collected"], 1234.0)
        for leak in ("cpu_percent", "process_count", "device", "path"):
            self.assertNotIn(leak, ms)
        self.assertNotIn(CANARY_PATH, r.text)

    def test_screen_ingest_status_source_always_remote_canary(self) -> None:
        server.sensor_policy = SimpleNamespace(screen_ingest=True)
        server.screen = None
        fake_json = mock.Mock()
        fake_json.exists.return_value = True
        fake_json.read_text.return_value = json.dumps({
            "description": "x",
            "captured_at": time.time() - 3,
            "described_at": time.time(),
            "source": "evil-tampered-source",
        })
        with mock.patch.object(
            server, "SCREEN_LATEST_JSON", fake_json
        ), mock.patch.object(server, "_remove_legacy_latest_jpg"):
            r = self.client.get("/api/screen/status")
        ingest = r.json()["ingest"]
        self.assertEqual(ingest["source"], "remote")
        self.assertNotIn("evil-tampered-source", r.text)

    def test_status_endpoint_falls_back_when_sensor_status_raises(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            get_status=_raising(lambda: None),
        )
        server.monitor = SimpleNamespace(
            is_running=True,
            get_status=_raising(lambda: None),
        )
        with mock.patch.object(server, "get_secure_web_url", return_value=""):
            r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIsNone(body["vision_status"])
        self.assertIsNone(body["monitor_status"])
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)

    def test_safe_get_status_logs_type_only(self) -> None:
        ctx = SimpleNamespace(get_status=_raising(lambda: None))
        with self.assertLogs(server.logger, level=logging.WARNING) as cm:
            result = server._safe_get_status(ctx)
        self.assertIsNone(result)
        logged = "\n".join(cm.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(CANARY_PATH, logged)
        self.assertNotIn(CANARY_TOKEN, logged)

    def test_status_endpoint_vision_status_is_allowlist(self) -> None:
        server.vision = SimpleNamespace(
            is_running=True,
            get_status=lambda: {
                "running": True,
                "user_present": True,
                "emotion": "happiness",
                "emotion_ja": "嬉しそう",
                "person_count": 2,
            },
        )
        server.monitor = None
        with mock.patch.object(server, "get_secure_web_url", return_value=""):
            r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        vs = r.json().get("vision_status")
        self.assertIsNotNone(vs)
        self.assertTrue(vs["running"])
        self.assertTrue(vs["user_present"])
        for leak in ("emotion", "emotion_ja", "person_count"):
            self.assertNotIn(leak, vs)
        self.assertNotIn(CANARY_PATH, r.text)
        self.assertNotIn(CANARY_TOKEN, r.text)


class DescribeIngestedGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            name: getattr(server, name)
            for name in (
                "_ingest_active_generation",
                "_ingest_future",
                "_ingest_generation",
                "_ingest_accepting",
                "_ingest_done_events",
            )
        }
        server._ingest_active_generation = None
        server._ingest_future = None
        server._ingest_generation = 1
        server._ingest_accepting = True
        server._ingest_done_events = {}

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(server, name, value)

    def test_describe_failure_resets_gate_and_logs_type_only(self) -> None:
        describer = mock.Mock()
        describer.describe.side_effect = RuntimeError(CANARY_EXC)
        with mock.patch.object(
            server, "_get_ingest_describer", return_value=describer
        ), self.assertLogs(server.logger, level=logging.WARNING) as cm:
            server._ingest_active_generation = 1
            server._describe_ingested(b"\xff\xd8\xff", time.time(), 1)
        self.assertIsNone(server._ingest_active_generation)
        logged = "\n".join(cm.output)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(CANARY_PATH, logged)
        self.assertNotIn(CANARY_TOKEN, logged)

    def test_describe_after_shutdown_revoke_writes_nothing_and_cleans(self) -> None:
        # シャットダウン (受付 revoke) 後に worker が遅延完了しても latest.json は書かれず、
        # canary 描写テキストも外部へ出ない (コミット経路に到達しない)。
        server._ingest_active_generation = 1
        server._ingest_accepting = False  # revoke 済み
        server._ingest_generation = 2  # revoke 後の世代
        with mock.patch.object(
            server, "_get_ingest_describer", return_value=mock.Mock()
        ), mock.patch.object(
            server,
            "_commit_ingest_result",
            side_effect=AssertionError("must not commit"),
        ) as commit:
            server._describe_ingested(b"\xff\xd8\xff", time.time(), 1)
        commit.assert_not_called()
        self.assertIsNone(server._ingest_active_generation)
        self.assertIsNone(server._ingest_future)

    def test_commit_barrier_revoke_first_suppresses_write(self) -> None:
        # revoke (accepting=False + 世代前進) が先なら、最終受付確認が失敗して replace は
        # 抑止され、準備済み tmp は破棄される。失敗しても例外・秘密は露出しない。
        server._ingest_accepting = False
        server._ingest_generation = 2
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        target = d / "latest.json"
        with mock.patch.object(server, "SCREEN_LATEST_JSON", target):
            ok = server._commit_ingest_result(1, {"description": "canary"})
        self.assertFalse(ok)
        self.assertFalse(target.exists())
        self.assertFalse((d / "latest.json.tmp").exists())

    def test_commit_barrier_commit_first_writes_then_revoke_follows(self) -> None:
        # 最終受付確認と os.replace が lock 下で原子的に通れば、revoke はコミット完了後に
        # 続き、書き込み済み結果は残る。tmp は replace 後に残らない。
        server._ingest_accepting = True
        server._ingest_generation = 1
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        target = d / "latest.json"
        with mock.patch.object(server, "SCREEN_LATEST_JSON", target):
            ok = server._commit_ingest_result(1, {"description": "canary"})
        self.assertTrue(ok)
        self.assertTrue(target.exists())
        self.assertFalse((d / "latest.json.tmp").exists())
        # コミット後の revoke は書き込み済み結果を破棄しない
        server._ingest_accepting = False
        server._ingest_generation = 2
        self.assertTrue(target.exists())


class WebSocketSttFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original = {
            name: getattr(server, name)
            for name in (
                "stt",
                "idle_manager",
                "sensor_policy",
                "_ingest_active_generation",
            )
        }
        server.stt = _RaisingStt()
        server.sensor_policy = server.SensorPolicy(microphone=True)
        server.idle_manager = _FakeIdle()
        server._ingest_active_generation = None
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(server, name, value)

    def test_audio_message_stt_error_never_leaks_exception(self) -> None:
        audio_b64 = base64.b64encode(b"fake-wav-bytes").decode("ascii")
        with mock.patch.object(server, "_decode_wav_bytes", return_value=[0.0]):
            with self.client.websocket_connect("/ws/chat") as ws:
                ws.send_json({
                    "type": "audio_message",
                    "data": audio_b64,
                    "format": "wav",
                    "session_id": "test123",
                    "tts": False,
                })
                msg = ws.receive_json()
        self.assertEqual(msg["type"], "error")
        self.assertEqual(msg["message"], "STT error")
        self.assertEqual(msg["error_type"], "internal_error")
        self.assertNotIn(CANARY_PATH, json.dumps(msg))
        self.assertNotIn(CANARY_TOKEN, json.dumps(msg))
        self.assertEqual(server.idle_manager.starts, 1)
        self.assertEqual(server.idle_manager.ends, 1)


class _Cp932StrictHandler(logging.Handler):
    """ログメッセージを CP932 へ厳格 (errors='strict') エンコードする handler。

    エンコードに失敗したメッセージを failures に蓄積する。Windows 開発機の
    CP932 ログファイルで UnicodeEncodeError を起こさないことを検証するために使う。
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.failures: list[tuple[str, UnicodeEncodeError]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        msg = record.getMessage()
        try:
            msg.encode("cp932")
        except UnicodeEncodeError as e:
            self.failures.append((msg, e))


class Cp932SensorDiagnosticsTest(unittest.TestCase):
    """センサー診断ログが ASCII-only かつ CP932 厳格エンコード可能であること。

    サブPC 常駐のログは CP932 (Windows-31J) ファイルへ書かれるため、絵文字などの
    非CP932文字が混ざると UnicodeEncodeError を起こす。Vision/Screen/Monitor の
    成功・失敗 (type-only) 診断がすべて ASCII で表現されることを検証する。
    """

    def setUp(self) -> None:
        self.handler = _Cp932StrictHandler()
        server.logger.addHandler(self.handler)
        self.addCleanup(server.logger.removeHandler, self.handler)

    def _assert_cp932_ascii(self) -> None:
        self.assertFalse(
            self.handler.failures,
            f"non-CP932 messages: {[m for m, _ in self.handler.failures]}",
        )
        for record in self.handler.records:
            self.assertTrue(record.getMessage().isascii(), record.getMessage())

    def test_success_diagnostics_are_cp932_ascii(self) -> None:
        fake_vision = SimpleNamespace(
            start=lambda: True,
            get_status=lambda: {"emotion_detection": True},
        )
        with mock.patch.object(
            server, "VisionContext", return_value=fake_vision
        ), mock.patch.object(server.time, "sleep"):
            self.assertIsNotNone(
                server._init_vision_from_policy(SimpleNamespace(camera=True))
            )

        server.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434", model="llava"
        )
        fake_screen_local = SimpleNamespace(
            start=lambda: True,
            get_status=lambda: {
                "mode": "local",
                "model": "llava",
                "analysis_interval": 90.0,
            },
        )
        with mock.patch.object(
            server, "create_screen_context", return_value=fake_screen_local
        ):
            self.assertIsNotNone(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                )
            )

        fake_screen_remote = SimpleNamespace(
            start=lambda: True,
            get_status=lambda: {"mode": "remote"},
        )
        with mock.patch.object(
            server, "create_screen_context", return_value=fake_screen_remote
        ):
            self.assertIsNotNone(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                )
            )

        fake_monitor = SimpleNamespace(start=lambda: True)
        with mock.patch.object(
            server, "MonitorContext", return_value=fake_monitor
        ):
            self.assertIsNotNone(
                server._init_monitor_from_policy(SimpleNamespace(monitor=True))
            )

        self._assert_cp932_ascii()

    def test_failure_diagnostics_are_cp932_ascii_type_only(self) -> None:
        with mock.patch.object(
            server, "VisionContext", side_effect=RuntimeError(CANARY_EXC)
        ):
            self.assertIsNone(server._init_vision_from_policy(SimpleNamespace(camera=True)))

        server.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434", model="llava"
        )
        with mock.patch.object(
            server, "create_screen_context", side_effect=RuntimeError(CANARY_EXC)
        ):
            self.assertIsNone(
                server._init_screen_from_policy(
                    SimpleNamespace(screen_capture=True),
                    config=server.config,
                    primary_provider_kind="ollama",
                )
            )

        with mock.patch.object(
            server, "MonitorContext", side_effect=RuntimeError(CANARY_EXC)
        ):
            self.assertIsNone(server._init_monitor_from_policy(SimpleNamespace(monitor=True)))

        self._assert_cp932_ascii()


class SensorErrorCodeMapperTest(unittest.TestCase):
    """外部向け error code は allowlist の固定値のみで、未知/カスタムは internal_error。"""

    class _CanaryCustomBoom(Exception):
        """canary: allowlist に無いカスタム例外クラス。"""

    def test_known_timeout_maps_to_timeout(self) -> None:
        self.assertEqual(server.sensor_error_code(TimeoutError("x")), "timeout")
        self.assertEqual(
            server.sensor_error_code_from_name("TimeoutExpired"), "timeout"
        )

    def test_known_input_error_maps_to_invalid_input(self) -> None:
        self.assertEqual(server.sensor_error_code(ValueError("x")), "invalid_input")
        self.assertEqual(server.sensor_error_code(TypeError("x")), "invalid_input")

    def test_known_unavailable_maps_to_unavailable(self) -> None:
        self.assertEqual(
            server.sensor_error_code(ConnectionError("x")), "unavailable"
        )
        self.assertEqual(server.sensor_error_code(OSError("x")), "unavailable")

    def test_canary_custom_exception_maps_to_internal_error(self) -> None:
        err = self._CanaryCustomBoom("secret canary detail")
        self.assertEqual(server.sensor_error_code(err), "internal_error")
        self.assertEqual(err.__class__.__name__, "_CanaryCustomBoom")

    def test_unknown_name_maps_to_internal_error(self) -> None:
        self.assertEqual(
            server.sensor_error_code_from_name("_CanaryCustomBoom"), "internal_error"
        )
        self.assertEqual(server.sensor_error_code_from_name(None), "internal_error")
        self.assertEqual(server.sensor_error_code_from_name(""), "internal_error")

    def test_http_payload_never_contains_class_name(self) -> None:
        from starlette.testclient import TestClient

        original_stt = server.stt
        original_policy = server.sensor_policy

        class _CanaryRaisingStt:
            def is_loaded(self) -> bool:
                return True

            def transcribe(self, audio, rate):
                raise SensorErrorCodeMapperTest._CanaryCustomBoom("boom")

        server.stt = _CanaryRaisingStt()
        server.sensor_policy = server.SensorPolicy(microphone=True)
        try:
            with mock.patch.object(server, "_decode_wav_bytes", return_value=[0.0]):
                r = TestClient(server.app).post(
                    "/api/stt",
                    json={"audio": base64.b64encode(b"fake-wav").decode("ascii")},
                )
        finally:
            server.stt = original_stt
            server.sensor_policy = original_policy
        body = r.json()
        self.assertEqual(body["error_type"], "internal_error")
        self.assertNotIn("_CanaryCustomBoom", r.text)


if __name__ == "__main__":
    unittest.main()