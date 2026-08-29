"""Web マイク入力の SensorPolicy.microphone fail-closed gate テスト。

- POST /api/stt : policy が未解決 (None)・false・不正値のとき固定 403 を返し、
  base64 デコード・WAV デコード・STT・一時ファイル・subprocess へ一切到達しない。
  明示 true のみ従来の機能を維持する。
- WebSocket audio_message : 同様に fail closed (固定文言の error のみ送出)。
- /api/status : stt は engine ロード済み かつ policy true のときだけ True。
- static UI : app.js は status.stt でマイクUIを gate し、ブラウザ権限だけで
  サーバー policy を迂回しない。env 名・値は frontend に出さない。
"""
from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.web import server

ROOT = Path(__file__).resolve().parents[2]


def _policy(microphone: bool) -> SimpleNamespace:
    return SimpleNamespace(microphone=microphone)


class _FakeStt:
    def __init__(self, loaded: bool = True, text: str = "canary テキスト") -> None:
        self.loaded = loaded
        self.text = text
        self.model_size = "canary-model"
        self.calls: list = []

    def is_loaded(self) -> bool:
        return self.loaded

    def transcribe(self, audio, rate):
        self.calls.append((audio, rate))
        return self.text


class SttHttpGateTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original = {
            name: getattr(server, name)
            for name in ("stt", "sensor_policy")
        }
        server.stt = None
        server.sensor_policy = None
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(server, name, value)

    def _post(self):
        return self.client.post(
            "/api/stt",
            json={"audio": base64.b64encode(b"canary-audio-bytes").decode("ascii")},
        )

    def _assert_forbidden_no_work(self, r) -> None:
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()["error"], "forbidden")
        self.assertNotIn("canary-audio-bytes", r.text)
        self.assertNotIn("SENSOR_MICROPHONE_ENABLED", r.text)
        self.assertEqual(server.stt.calls, [])

    def test_policy_none_forbids_before_decode_or_stt(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = None
        with mock.patch.object(server, "base64") as b64, mock.patch.object(
            server, "_decode_wav_bytes"
        ):
            r = self._post()
        self._assert_forbidden_no_work(r)
        b64.b64decode.assert_not_called()

    def test_policy_false_forbids_before_decode_or_stt(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = _policy(False)
        with mock.patch.object(server, "_decode_wav_bytes") as dec:
            r = self._post()
        self._assert_forbidden_no_work(r)
        dec.assert_not_called()

    def test_policy_true_preserves_stt_functionality(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = _policy(True)
        with mock.patch.object(server, "_decode_wav_bytes", return_value=[0.0, 1.0]):
            r = self._post()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["text"], "canary テキスト")
        self.assertEqual(len(server.stt.calls), 1)

    def test_policy_true_with_stt_unloaded_reports_unavailable(self) -> None:
        server.stt = None
        server.sensor_policy = _policy(True)
        r = self._post()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"], "STT not available")


class SttWsGateTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original = {
            name: getattr(server, name)
            for name in ("stt", "sensor_policy", "idle_manager")
        }
        server.stt = None
        server.sensor_policy = None
        server.idle_manager = None
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(server, name, value)

    def _send(self, ws):
        ws.send_json({
            "type": "audio_message",
            "data": base64.b64encode(b"canary-audio-bytes").decode("ascii"),
            "format": "wav",
            "session_id": "test123",
            "tts": False,
        })
        return ws.receive_json()

    def test_policy_none_forbids_before_decode_or_stt(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = None
        with mock.patch.object(server, "_decode_wav_bytes") as dec:
            with self.client.websocket_connect("/ws/chat") as ws:
                msg = self._send(ws)
        self.assertEqual(msg["type"], "error")
        self.assertEqual(msg["message"], "マイク入力は許可されていません。")
        self.assertNotIn("canary-audio-bytes", json.dumps(msg, ensure_ascii=False))
        dec.assert_not_called()
        self.assertEqual(server.stt.calls, [])

    def test_policy_false_forbids_before_decode_or_stt(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = _policy(False)
        with mock.patch.object(server, "_decode_wav_bytes") as dec:
            with self.client.websocket_connect("/ws/chat") as ws:
                msg = self._send(ws)
        self.assertEqual(msg["type"], "error")
        self.assertEqual(msg["message"], "マイク入力は許可されていません。")
        dec.assert_not_called()
        self.assertEqual(server.stt.calls, [])

    def test_policy_true_runs_stt(self) -> None:
        server.stt = _FakeStt(text="")
        server.sensor_policy = _policy(True)
        with mock.patch.object(server, "_decode_wav_bytes", return_value=[0.0]):
            with self.client.websocket_connect("/ws/chat") as ws:
                msg = self._send(ws)
        self.assertEqual(msg["type"], "stt_result")
        self.assertEqual(msg["message"], "音声を認識できませんでした")
        self.assertEqual(len(server.stt.calls), 1)


class StatusSttGateTest(unittest.TestCase):
    def setUp(self) -> None:
        from starlette.testclient import TestClient

        self.original = {
            name: getattr(server, name)
            for name in (
                "stt",
                "sensor_policy",
                "vision",
                "monitor",
                "preloader",
                "idle_manager",
                "activity_runtime",
                "config",
                "primary_provider_kind",
                "primary_provider_base_url",
            )
        }
        server.stt = None
        server.sensor_policy = None
        server.vision = None
        server.monitor = None
        server.preloader = None
        server.idle_manager = None
        server.activity_runtime = None
        server.config = None
        server.primary_provider_kind = None
        server.primary_provider_base_url = None
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for name, value in self.original.items():
            setattr(server, name, value)

    def _status(self) -> dict:
        with mock.patch.object(server, "get_secure_web_url", return_value=""):
            r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        return r.json()

    def test_stt_false_when_policy_disabled_even_if_engine_loaded(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = _policy(False)
        self.assertFalse(self._status()["stt"])

    def test_stt_false_when_policy_unresolved(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = None
        self.assertFalse(self._status()["stt"])

    def test_stt_false_when_engine_unavailable_even_if_policy_true(self) -> None:
        server.stt = None
        server.sensor_policy = _policy(True)
        self.assertFalse(self._status()["stt"])

    def test_stt_true_only_when_engine_and_policy_enabled(self) -> None:
        server.stt = _FakeStt()
        server.sensor_policy = _policy(True)
        self.assertTrue(self._status()["stt"])

    def test_status_never_exposes_policy_env_name_even_if_set(self) -> None:
        # env が true でも frozen policy 未解決なら有効化されず、env 名も露出しない。
        server.stt = _FakeStt()
        server.sensor_policy = None
        with mock.patch.dict(
            server.os.environ,
            {"SENSOR_MICROPHONE_ENABLED": "true"},
            clear=False,
        ):
            body = self._status()
        self.assertFalse(body["stt"])
        self.assertNotIn("SENSOR_MICROPHONE_ENABLED", json.dumps(body))
        self.assertNotIn("microphone", json.dumps(body))


class StaticUiGateTest(unittest.TestCase):
    def test_app_js_gates_mic_ui_on_server_stt_status(self) -> None:
        js = (ROOT / "src/web/static/app.js").read_text(encoding="utf-8")
        self.assertIn("sttAvailable = Boolean(status.stt)", js)
        self.assertIn("if (!sttAvailable || micUnavailableReason)", js)
        self.assertIn("setMicState('disabled')", js)
        self.assertNotIn("SENSOR_MICROPHONE_ENABLED", js)
        self.assertNotIn("microphone", js)

    def test_app_js_does_not_treat_browser_permission_as_server_policy(self) -> None:
        js = (ROOT / "src/web/static/app.js").read_text(encoding="utf-8")
        # サーバー gate が無くてもブラウザ権限だけでは録音開始しない。
        self.assertIn("if (!sttAvailable || micUnavailableReason)", js)
        self.assertIn("micBtn.setAttribute('aria-label'", js)


if __name__ == "__main__":
    unittest.main()