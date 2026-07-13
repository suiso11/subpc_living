from __future__ import annotations

import json
import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from src.desktop.api import DesktopApi, DesktopApiError, normalize_server_url
from src.desktop.audio import NativeAudioRecorder
from src.desktop.config import DesktopSettings
from src.desktop.windows import is_autostart_enabled, set_autostart, startup_command


class DesktopApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/api/status":
                return httpx.Response(200, json={"model": "test-model", "tasks": True})
            if request.url.path == "/api/tasks" and request.method == "GET":
                return httpx.Response(200, json={"tasks": [{"id": 1, "title": "一歩"}]})
            if request.url.path == "/api/tasks" and request.method == "POST":
                return httpx.Response(201, json={"task": {"id": 2, "title": "追加"}})
            if request.url.path == "/api/game":
                return httpx.Response(200, json={"enabled": True, "badges": []})
            if request.url.path == "/api/chat/resume":
                return httpx.Response(200, json={"session_id": "desktop_1", "messages": []})
            return httpx.Response(200, json={"ok": True})

        self.api = DesktopApi(
            "https://buddy.example/",
            transport=httpx.MockTransport(handler),
        )

    def tearDown(self) -> None:
        self.api.close()

    def test_normalizes_http_and_websocket_addresses(self) -> None:
        address = normalize_server_url("buddy.local:8000/")
        self.assertEqual(address.http, "http://buddy.local:8000")
        self.assertEqual(address.websocket, "ws://buddy.local:8000")
        self.assertEqual(self.api.websocket_url, "wss://buddy.example/ws/chat")

    def test_tasks_use_existing_backend_contract(self) -> None:
        tasks = self.api.tasks()
        self.assertEqual(tasks["tasks"][0]["title"], "一歩")
        self.assertEqual(self.requests[-1].url.params["status"], "open")
        self.assertEqual(self.requests[-1].url.params["limit"], "200")

        added = self.api.add_task("明日18時 資料を見る", "high", "確認")
        self.assertEqual(added["task"]["id"], 2)
        payload = json.loads(self.requests[-1].content)
        self.assertEqual(payload["text"], "明日18時 資料を見る")
        self.assertEqual(payload["priority"], "high")

    def test_chat_game_and_status_reuse_web_service(self) -> None:
        self.assertEqual(self.api.status()["model"], "test-model")
        self.assertEqual(self.api.resume_chat()["session_id"], "desktop_1")
        self.assertTrue(self.api.game()["enabled"])

    def test_api_error_exposes_backend_message(self) -> None:
        api = DesktopApi(
            "http://test",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"error": "title is required"})
            ),
        )
        try:
            with self.assertRaisesRegex(DesktopApiError, "title is required"):
                api.add_task("")
        finally:
            api.close()


class DesktopSettingsTest(unittest.TestCase):
    def test_settings_round_trip_and_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "desktop.json"
            DesktopSettings(
                server_url="http://subpc:8000",
                session_id="desktop_keep",
                tts_enabled=True,
            ).save(path)
            loaded = DesktopSettings.load(path)
            self.assertEqual(loaded.session_id, "desktop_keep")
            self.assertTrue(loaded.tts_enabled)

            old = os.environ.get("SUBPC_DESKTOP_SERVER_URL")
            os.environ["SUBPC_DESKTOP_SERVER_URL"] = "http://tailscale:8000"
            try:
                overridden = DesktopSettings.load(path)
            finally:
                if old is None:
                    os.environ.pop("SUBPC_DESKTOP_SERVER_URL", None)
                else:
                    os.environ["SUBPC_DESKTOP_SERVER_URL"] = old
            self.assertEqual(overridden.server_url, "http://tailscale:8000")

    def test_startup_command_uses_absolute_launcher(self) -> None:
        command = startup_command()
        launcher = Path(__file__).resolve().parents[1] / "scripts" / "run_desktop.py"
        self.assertIn(str(launcher), command)
        self.assertIn("--hidden", command)

    @unittest.skipUnless(os.name == "nt", "Windows registry integration")
    def test_autostart_registry_round_trip(self) -> None:
        test_run_name = f"SUBPC BUDDY test {os.getpid()}"
        with patch("src.desktop.windows.APP_RUN_NAME", test_run_name):
            try:
                self.assertTrue(set_autostart(True))
                self.assertTrue(is_autostart_enabled())
                self.assertTrue(set_autostart(False))
                self.assertFalse(is_autostart_enabled())
            finally:
                set_autostart(False)


class NativeAudioRecorderTest(unittest.TestCase):
    def test_records_pcm_as_server_compatible_wav(self) -> None:
        import numpy as np

        class FakeStream:
            def __init__(self, *, callback, **kwargs):
                del kwargs
                self.callback = callback

            def start(self):
                self.callback(np.array([[1], [-2], [3]], dtype=np.int16), 3, None, None)

            def stop(self):
                pass

            def close(self):
                pass

        fake = SimpleNamespace(InputStream=FakeStream)
        with patch.dict(sys.modules, {"sounddevice": fake}):
            recorder = NativeAudioRecorder(sample_rate=16000)
            recorder.start()
            payload = recorder.stop()

        with wave.open(io.BytesIO(payload), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getnframes(), 3)

    def test_falls_back_to_device_rate_and_resamples_to_16khz(self) -> None:
        import numpy as np

        class FakeStream:
            def __init__(self, *, samplerate, callback, **kwargs):
                del kwargs
                self.samplerate = samplerate
                self.callback = callback

            def start(self):
                if self.samplerate == 16000:
                    raise RuntimeError("Invalid sample rate")
                self.callback(np.arange(480, dtype=np.int16).reshape(-1, 1), 480, None, None)

            def stop(self):
                pass

            def close(self):
                pass

        fake = SimpleNamespace(
            InputStream=FakeStream,
            query_devices=lambda **kwargs: {"default_samplerate": 48000.0},
        )
        with patch.dict(sys.modules, {"sounddevice": fake}):
            recorder = NativeAudioRecorder(sample_rate=16000)
            recorder.start()
            payload = recorder.stop()

        with wave.open(io.BytesIO(payload), "rb") as wav:
            self.assertEqual(wav.getframerate(), 16000)
            self.assertEqual(wav.getnframes(), 160)


if __name__ == "__main__":
    unittest.main()
