"""HealthChecker の backward-compatible な include_ollama フラグのテスト。"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import httpx

from src.chat.config import ChatConfig
from src.service.healthcheck import HealthChecker, main


class HealthCheckerOllamaFlagTest(unittest.TestCase):
    """include_ollama フラグが既定で従来挙動を保ち、False で /api/tags を避けること。"""

    def _run_all(self, *, include_ollama: bool) -> tuple[dict, object]:
        checker = HealthChecker(ollama_url="http://localhost:11434")
        with patch.object(
            checker,
            "check_ollama",
            return_value={"status": "ok", "models": ["m"]},
        ) as mock_ollama, patch.object(
            checker,
            "check_disk",
            return_value={"status": "ok", "used_percent": 10.0},
        ), patch.object(
            checker,
            "check_memory",
            return_value={"status": "ok", "used_percent": 20.0},
        ):
            result = checker.check_all(include_ollama=include_ollama)
        return result, mock_ollama

    def test_check_all_default_includes_ollama_probe(self) -> None:
        """後方互換: 既定 (include_ollama=True) では従来どおり ollama を検査する。"""
        result, mock_ollama = self._run_all(include_ollama=True)
        self.assertIn("ollama", result["checks"])
        self.assertEqual(result["checks"]["ollama"]["status"], "ok")
        mock_ollama.assert_called_once()

    def test_check_all_include_ollama_false_skips_probe(self) -> None:
        """False では /api/tags 相当の check_ollama を呼ばず、checks から省略する。"""
        result, mock_ollama = self._run_all(include_ollama=False)
        self.assertNotIn("ollama", result["checks"])
        mock_ollama.assert_not_called()

    def test_check_all_include_ollama_false_status_still_ok(self) -> None:
        """ollama を省略しても disk/memory が ok なら全体 status は ok のまま。"""
        result, _ = self._run_all(include_ollama=False)
        self.assertEqual(result["status"], "ok")
        self.assertIn("disk", result["checks"])
        self.assertIn("memory", result["checks"])

    def test_check_ollama_explicit_false_no_network(self) -> None:
        """check_ollama(include_ollama=False) はネットワークに触れず skip を返す。"""
        checker = HealthChecker(ollama_url="http://127.0.0.1:1")
        result = checker.check_ollama(include_ollama=False)
        self.assertEqual(result["status"], "skip")
        self.assertEqual(result["message"], "ollama check disabled")


class HealthCheckerOpenAICompatibleTest(unittest.TestCase):
    """check_openai_compatible の HTTP 分類 (全て MockTransport でモック)。"""

    def _run(self, handler, base_url: str = "http://localhost:8080/v1") -> tuple[dict, object]:
        real = httpx.Client(transport=httpx.MockTransport(handler))
        real.close = Mock(wraps=real.close)
        checker = HealthChecker()
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_openai_compatible(base_url)
        self.assertTrue(real.close.called, "client must be closed on every path")
        return result, real

    def test_ok_returns_model_ids(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            return httpx.Response(200, json={"data": [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]})

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models"], ["m1", "m2", "m3"])

    def test_ok_empty_data_list(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(200, json={"data": []}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models"], [])

    def test_ok_skips_entries_without_id(self) -> None:
        data = [{"id": "m1"}, {"object": "model"}, {"id": 123}]
        result, _ = self._run(lambda request: httpx.Response(200, json={"data": data}))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models"], ["m1", "123"])

    def test_no_trailing_slash_probes_v1_models(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            self.assertEqual(request.url.query, b"")
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "ok")

    def test_single_trailing_slash_probes_v1_models(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            self.assertEqual(request.url.query, b"")
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        result, _ = self._run(handler, base_url="http://localhost:8080/v1/")
        self.assertEqual(result["status"], "ok")

    def test_multiple_trailing_slashes_probe_v1_models(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/v1/models")
            self.assertEqual(request.url.query, b"")
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        result, _ = self._run(handler, base_url="http://localhost:8080/v1///")
        self.assertEqual(result["status"], "ok")

    def test_query_string_is_not_appended(self) -> None:
        def handler(request):
            self.assertEqual(request.url.query, b"")
            self.assertFalse(any("?" in part for part in request.url.path.split("/")))
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "ok")

    def test_404_returns_unknown(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(404))
        self.assertEqual(result["status"], "unknown")

    def test_405_returns_unknown(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(405))
        self.assertEqual(result["status"], "unknown")

    def test_401_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(401))
        self.assertEqual(result["status"], "error")

    def test_403_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(403))
        self.assertEqual(result["status"], "error")

    def test_500_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(500))
        self.assertEqual(result["status"], "error")
        self.assertNotIn("localhost", result["message"])

    def test_200_missing_data_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(200, json={"models": []}))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "malformed model list response")

    def test_200_data_not_list_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(200, json={"data": {}}))
        self.assertEqual(result["status"], "error")

    def test_200_invalid_json_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(200, text="not-json"))
        self.assertEqual(result["status"], "error")

    def test_200_json_array_returns_error(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(200, json=[{"id": "m1"}]))
        self.assertEqual(result["status"], "error")

    def test_timeout_returns_error(self) -> None:
        def handler(request):
            raise httpx.TimeoutException("timed out", request=request)

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "request timed out")

    def test_transport_error_returns_error(self) -> None:
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "connection failed")


class HealthCheckerAuthHeaderTest(unittest.TestCase):
    """api_key オプション: Bearer 送信 / キーなしでヘッダなし / 秘匿情報の非漏洩。"""

    def _run(self, handler, api_key=None) -> tuple[dict, object]:
        real = httpx.Client(transport=httpx.MockTransport(handler))
        real.close = Mock(wraps=real.close)
        checker = HealthChecker()
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_openai_compatible("http://localhost:8080/v1", api_key=api_key)
        self.assertTrue(real.close.called, "client must be closed on every path")
        return result, real

    def test_authenticated_sends_bearer_header(self) -> None:
        def handler(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer sk-test-123")
            return httpx.Response(200, json={"data": [{"id": "m1"}]})

        result, _ = self._run(handler, api_key="sk-test-123")
        self.assertEqual(result["status"], "ok")

    def test_keyless_sends_no_authorization_header(self) -> None:
        def handler(request):
            self.assertNotIn("Authorization", request.headers)
            return httpx.Response(200, json={"data": []})

        result, _ = self._run(handler)
        self.assertEqual(result["status"], "ok")

    def test_blank_key_sends_no_authorization_header(self) -> None:
        def handler(request):
            self.assertNotIn("Authorization", request.headers)
            return httpx.Response(200, json={"data": []})

        result, _ = self._run(handler, api_key="   ")
        self.assertEqual(result["status"], "ok")

    def test_transport_error_does_not_leak_key_or_url(self) -> None:
        def handler(request):
            raise httpx.ConnectError("auth to http://localhost:8080/v1 failed", request=request)

        result, _ = self._run(handler, api_key="sk-super-secret")
        self.assertEqual(result["message"], "connection failed")
        self.assertNotIn("sk-super-secret", str(result))
        self.assertNotIn("localhost", str(result))

    def test_http_error_does_not_leak_key(self) -> None:
        result, _ = self._run(lambda request: httpx.Response(500), api_key="sk-super-secret")
        self.assertNotIn("sk-super-secret", str(result))

    def test_malformed_json_does_not_leak_key(self) -> None:
        result, _ = self._run(
            lambda request: httpx.Response(200, text="not-json"), api_key="sk-super-secret"
        )
        self.assertNotIn("sk-super-secret", str(result))


class HealthCheckerApiKeyPlumbingTest(unittest.TestCase):
    """api_key が check_selected_provider / check_all を通って伝搬する。"""

    def test_selected_provider_passes_api_key(self) -> None:
        checker = HealthChecker()
        with patch.object(
            checker,
            "check_openai_compatible",
            return_value={"status": "ok", "models": []},
        ) as probe:
            checker.check_selected_provider(
                "openai_compatible", "http://localhost:8080/v1", api_key="sk-abc"
            )
        probe.assert_called_once_with("http://localhost:8080/v1", api_key="sk-abc")

    def test_selected_provider_blank_key_passes_none(self) -> None:
        checker = HealthChecker()
        with patch.object(
            checker,
            "check_openai_compatible",
            return_value={"status": "ok", "models": []},
        ) as probe:
            checker.check_selected_provider("openai_compatible", "http://localhost:8080/v1", api_key="")
        probe.assert_called_once_with("http://localhost:8080/v1", api_key="")

    def test_check_all_forwards_api_key(self) -> None:
        checker = HealthChecker()
        with patch.object(checker, "check_selected_provider", return_value={"status": "ok", "models": []}) as dispatch, patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(
                provider_kind="openai_compatible",
                provider_url="http://localhost:8080/v1",
                provider_api_key="sk-abc",
            )
        dispatch.assert_called_once_with(
            "openai_compatible", "http://localhost:8080/v1", api_key="sk-abc"
        )
        self.assertNotIn("sk-abc", str(result))


class HealthCheckerLegacyCleanupTest(unittest.TestCase):
    """check_ollama / check_web の close 保証とメッセージの非秘匿化。"""

    def _patched(self, handler) -> tuple[object, object, object]:
        transport = httpx.MockTransport(handler)
        transport.close = Mock(wraps=transport.close)
        real = httpx.Client(transport=transport)
        checker = HealthChecker()
        return transport, real, checker

    def test_check_ollama_close_on_success(self) -> None:
        transport, real, checker = self._patched(
            lambda request: httpx.Response(200, json={"models": [{"name": "m1"}]})
        )
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_ollama()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(transport.close.called)

    def test_check_web_close_on_success(self) -> None:
        transport, real, checker = self._patched(lambda request: httpx.Response(200, json={"ok": True}))
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_web()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(transport.close.called)

    def test_check_ollama_close_on_json_error(self) -> None:
        transport, real, checker = self._patched(lambda request: httpx.Response(200, text="not-json"))
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_ollama()
        self.assertEqual(result["status"], "error")
        self.assertTrue(transport.close.called)

    def test_check_ollama_close_on_transport_exception(self) -> None:
        def handler(request):
            raise httpx.ConnectError("boom", request=request)

        transport, real, checker = self._patched(handler)
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_ollama()
        self.assertEqual(result["status"], "error")
        self.assertTrue(transport.close.called)

    def test_check_web_close_on_transport_exception(self) -> None:
        def handler(request):
            raise httpx.TimeoutException("timed out", request=request)

        transport, real, checker = self._patched(handler)
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_web()
        self.assertEqual(result["status"], "error")
        self.assertTrue(transport.close.called)

    def test_check_ollama_error_hides_url_and_exception_text(self) -> None:
        def handler(request):
            raise httpx.ConnectError(
                "connect to http://localhost:11434/api/tags refused (secret-token)",
                request=request,
            )

        transport, real, checker = self._patched(handler)
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_ollama()
        self.assertEqual(result["message"], "request failed (ConnectError)")
        self.assertNotIn("http", result["message"])
        self.assertNotIn("secret", result["message"])

    def test_check_web_error_hides_url_and_exception_text(self) -> None:
        def handler(request):
            raise httpx.ConnectError(
                "connect to http://localhost:8000/api/health refused (secret-token)",
                request=request,
            )

        transport, real, checker = self._patched(handler)
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_web()
        self.assertEqual(result["message"], "request failed (ConnectError)")
        self.assertNotIn("http", result["message"])
        self.assertNotIn("secret", result["message"])


class HealthCheckerSelectedProviderTest(unittest.TestCase):
    """check_selected_provider のディスパッチと未知 kind の安全な拒否。"""

    def test_ollama_dispatch_probes_api_tags(self) -> None:
        def handler(request):
            self.assertEqual(request.url.path, "/api/tags")
            return httpx.Response(200, json={"models": [{"name": "m1"}]})

        real = httpx.Client(transport=httpx.MockTransport(handler))
        real.close = Mock(wraps=real.close)
        checker = HealthChecker()
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_selected_provider("ollama", "http://localhost:11434")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["models"], ["m1"])
        self.assertTrue(real.close.called)

    def test_ollama_dispatch_defaults_to_self_url(self) -> None:
        seen: dict = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"models": []})

        real = httpx.Client(transport=httpx.MockTransport(handler))
        checker = HealthChecker(ollama_url="http://ollama.local:11434")
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = checker.check_selected_provider("ollama")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(seen["url"], "http://ollama.local:11434/api/tags")

    def test_openai_compatible_dispatch(self) -> None:
        checker = HealthChecker()
        with patch.object(
            checker,
            "check_openai_compatible",
            return_value={"status": "ok", "models": ["m"]},
        ) as probe:
            result = checker.check_selected_provider("openai_compatible", "http://localhost:8080/v1")
        probe.assert_called_once_with("http://localhost:8080/v1", api_key=None)
        self.assertEqual(result["status"], "ok")

    def test_openai_alias_dispatch(self) -> None:
        checker = HealthChecker()
        with patch.object(
            checker,
            "check_openai_compatible",
            return_value={"status": "ok", "models": []},
        ) as probe:
            checker.check_selected_provider("openai", "http://localhost:8080/v1")
        probe.assert_called_once_with("http://localhost:8080/v1", api_key=None)

    def test_openai_missing_url_returns_error(self) -> None:
        checker = HealthChecker()
        with patch("src.service.healthcheck.httpx.Client", side_effect=AssertionError("no network")):
            result = checker.check_selected_provider("openai_compatible", None)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "provider base url required")

    def test_unknown_kind_returns_error_without_network(self) -> None:
        checker = HealthChecker()
        with patch("src.service.healthcheck.httpx.Client", side_effect=AssertionError("no network")):
            result = checker.check_selected_provider("bogus", "http://localhost:11434")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["message"], "unknown provider kind")


class HealthCheckerUrlNormalizationTest(unittest.TestCase):
    """末尾スラッシュ正規化: no/single/multiple いずれも同じエンドポイントへ。"""

    def _run(self, call, url: str, handler) -> dict:
        transport = httpx.MockTransport(handler)
        transport.close = Mock(wraps=transport.close)
        real = httpx.Client(transport=transport)
        checker = HealthChecker(ollama_url=url, web_url=url)
        with patch("src.service.healthcheck.httpx.Client", return_value=real):
            result = call(checker)
        self.assertTrue(transport.close.called, "client must be closed on every path")
        return result

    def _assert_tags_path(self, request) -> httpx.Response:
        self.assertEqual(request.url.path, "/api/tags")
        self.assertEqual(request.url.query, b"")
        return httpx.Response(200, json={"models": [{"name": "m1"}]})

    def _assert_health_path(self, request) -> httpx.Response:
        self.assertEqual(request.url.path, "/api/health")
        self.assertEqual(request.url.query, b"")
        return httpx.Response(200, json={"ok": True})

    def test_check_ollama_no_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_ollama(), "http://localhost:11434", self._assert_tags_path
        )
        self.assertEqual(result["status"], "ok")

    def test_check_ollama_single_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_ollama(), "http://localhost:11434/", self._assert_tags_path
        )
        self.assertEqual(result["status"], "ok")

    def test_check_ollama_multiple_trailing_slashes(self) -> None:
        result = self._run(
            lambda c: c.check_ollama(), "http://localhost:11434///", self._assert_tags_path
        )
        self.assertEqual(result["status"], "ok")

    def test_probe_ollama_no_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_selected_provider("ollama", "http://localhost:11434"),
            "http://localhost:11434",
            self._assert_tags_path,
        )
        self.assertEqual(result["status"], "ok")

    def test_probe_ollama_single_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_selected_provider("ollama", "http://localhost:11434/"),
            "http://localhost:11434/",
            self._assert_tags_path,
        )
        self.assertEqual(result["status"], "ok")

    def test_probe_ollama_multiple_trailing_slashes(self) -> None:
        result = self._run(
            lambda c: c.check_selected_provider("ollama", "http://localhost:11434///"),
            "http://localhost:11434///",
            self._assert_tags_path,
        )
        self.assertEqual(result["status"], "ok")

    def test_check_web_no_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_web(), "http://localhost:8000", self._assert_health_path
        )
        self.assertEqual(result["status"], "ok")

    def test_check_web_single_trailing_slash(self) -> None:
        result = self._run(
            lambda c: c.check_web(), "http://localhost:8000/", self._assert_health_path
        )
        self.assertEqual(result["status"], "ok")

    def test_check_web_multiple_trailing_slashes(self) -> None:
        result = self._run(
            lambda c: c.check_web(), "http://localhost:8000///", self._assert_health_path
        )
        self.assertEqual(result["status"], "ok")


class HealthCheckerSelectedCheckAllTest(unittest.TestCase):
    """check_all の provider_kind/provider_url 拡張と全体ステータス。"""

    def test_provider_ollama_emits_legacy_key(self) -> None:
        checker = HealthChecker()
        selected = {"status": "ok", "models": ["m"]}
        with patch.object(checker, "check_selected_provider", return_value=selected) as dispatch, patch.object(
            checker, "check_disk", return_value={"status": "ok", "used_percent": 10.0}
        ), patch.object(checker, "check_memory", return_value={"status": "ok", "used_percent": 20.0}):
            result = checker.check_all(provider_kind="ollama", provider_url="http://localhost:11434")
        dispatch.assert_called_once_with("ollama", "http://localhost:11434", api_key=None)
        self.assertIn("ollama", result["checks"])
        self.assertNotIn("local_provider", result["checks"])
        self.assertEqual(result["checks"]["ollama"], selected)
        self.assertEqual(result["status"], "ok")

    def test_provider_openai_emits_local_provider_with_kind(self) -> None:
        checker = HealthChecker()
        selected = {"status": "ok", "models": ["m"]}
        with patch.object(checker, "check_selected_provider", return_value=selected) as dispatch, patch.object(
            checker, "check_disk", return_value={"status": "ok", "used_percent": 10.0}
        ), patch.object(checker, "check_memory", return_value={"status": "ok", "used_percent": 20.0}):
            result = checker.check_all(
                provider_kind="openai_compatible", provider_url="http://localhost:8080/v1"
            )
        dispatch.assert_called_once_with("openai_compatible", "http://localhost:8080/v1", api_key=None)
        self.assertNotIn("ollama", result["checks"])
        lp = result["checks"]["local_provider"]
        self.assertEqual(lp["status"], "ok")
        self.assertEqual(lp["kind"], "openai_compatible")
        self.assertEqual(result["status"], "ok")

    def test_provider_openai_no_url_or_key(self) -> None:
        checker = HealthChecker()
        selected = {"status": "ok", "models": ["m"]}
        with patch.object(checker, "check_selected_provider", return_value=selected), patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(
                provider_kind="openai_compatible", provider_url="http://localhost:8080/v1"
            )
        lp = result["checks"]["local_provider"]
        for forbidden in ("url", "base_url", "api_key", "key", "token"):
            self.assertNotIn(forbidden, lp)
        self.assertEqual(result["status"], "ok")

    def test_provider_unknown_kind_emits_error_and_overall_error(self) -> None:
        checker = HealthChecker()
        selected = {"status": "error", "message": "unknown provider kind"}
        with patch.object(checker, "check_selected_provider", return_value=selected), patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(provider_kind="bogus", provider_url="http://localhost:11434")
        self.assertEqual(result["status"], "error")
        lp = result["checks"]["local_provider"]
        self.assertEqual(lp["status"], "error")
        self.assertEqual(lp["kind"], "bogus")

    def test_overall_unknown_is_degraded(self) -> None:
        checker = HealthChecker()
        selected = {"status": "unknown", "message": "model discovery unsupported"}
        with patch.object(checker, "check_selected_provider", return_value=selected), patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(
                provider_kind="openai_compatible", provider_url="http://localhost:8080/v1"
            )
        self.assertEqual(result["status"], "degraded")

    def test_overall_warning_is_degraded(self) -> None:
        checker = HealthChecker()
        selected = {"status": "warning", "message": "high load"}
        with patch.object(checker, "check_selected_provider", return_value=selected), patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(provider_kind="ollama", provider_url="http://localhost:11434")
        self.assertEqual(result["status"], "degraded")

    def test_overall_error_priority_over_unknown(self) -> None:
        checker = HealthChecker()
        selected = {"status": "unknown", "message": "model discovery unsupported"}
        with patch.object(checker, "check_selected_provider", return_value=selected), patch.object(
            checker, "check_disk", return_value={"status": "error", "message": "disk gone"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all(
                provider_kind="openai_compatible", provider_url="http://localhost:8080/v1"
            )
        self.assertEqual(result["status"], "error")

    def test_default_params_preserve_legacy_ollama(self) -> None:
        checker = HealthChecker()
        with patch.object(checker, "check_ollama", return_value={"status": "ok", "models": ["m"]}) as legacy, patch.object(
            checker, "check_disk", return_value={"status": "ok"}
        ), patch.object(checker, "check_memory", return_value={"status": "ok"}):
            result = checker.check_all()
        legacy.assert_called_once()
        self.assertIn("ollama", result["checks"])
        self.assertNotIn("local_provider", result["checks"])
        self.assertEqual(result["status"], "ok")


class HealthCheckerMainTest(unittest.TestCase):
    """main() の設定ロード・正規化・エラー時の JSON 出力と終了コード。"""

    def _run_main(self, config) -> tuple[int, str, Mock]:
        out = io.StringIO()
        with patch("src.chat.config.ChatConfig.load", return_value=config), patch(
            "src.service.healthcheck.HealthChecker"
        ) as checker_cls:
            checker_cls.return_value.check_all.return_value = {"status": "ok", "checks": {}}
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as cm:
                    main()
        return cm.exception.code, out.getvalue(), checker_cls

    def test_main_valid_ollama(self) -> None:
        config = ChatConfig(local_provider_kind="ollama", ollama_base_url="http://localhost:11434")
        code, out, checker_cls = self._run_main(config)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["status"], "ok")
        checker_cls.assert_called_once_with(ollama_url="http://localhost:11434")
        kwargs = checker_cls.return_value.check_all.call_args.kwargs
        self.assertEqual(kwargs["provider_kind"], "ollama")
        self.assertEqual(kwargs["provider_url"], "http://localhost:11434")
        self.assertIs(kwargs["include_web"], False)

    def test_main_valid_openai(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible", local_base_url="http://localhost:8080/v1"
        )
        code, out, checker_cls = self._run_main(config)
        self.assertEqual(code, 0)
        kwargs = checker_cls.return_value.check_all.call_args.kwargs
        self.assertEqual(kwargs["provider_kind"], "openai_compatible")
        self.assertEqual(kwargs["provider_url"], "http://localhost:8080/v1")

    def test_main_passes_resolved_api_key(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8080/v1",
            local_api_key_env="LOCAL_HEALTH_TEST_TOKEN",
        )
        out = io.StringIO()
        with patch(
            "src.chat.config.ChatConfig.load", return_value=config
        ), patch("os.environ", {"LOCAL_HEALTH_TEST_TOKEN": "sk-from-env"}), patch(
            "src.service.healthcheck.HealthChecker"
        ) as checker_cls:
            checker_cls.return_value.check_all.return_value = {"status": "ok", "checks": {}}
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 0)
        kwargs = checker_cls.return_value.check_all.call_args.kwargs
        self.assertEqual(kwargs["provider_api_key"], "sk-from-env")
        self.assertNotIn("sk-from-env", out.getvalue())

    def test_main_normalizes_empty_kind_to_ollama(self) -> None:
        config = ChatConfig(local_provider_kind="", ollama_base_url="http://localhost:11434")
        code, _, checker_cls = self._run_main(config)
        self.assertEqual(code, 0)
        kwargs = checker_cls.return_value.check_all.call_args.kwargs
        self.assertEqual(kwargs["provider_kind"], "ollama")

    def test_main_unknown_kind_errors_without_url(self) -> None:
        config = ChatConfig(local_provider_kind="bogus")
        code, out, _ = self._run_main(config)
        self.assertEqual(code, 1)
        result = json.loads(out)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("http", out)

    def test_main_non_loopback_openai_errors(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible", local_base_url="http://10.0.0.5:8080/v1"
        )
        code, out, _ = self._run_main(config)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["status"], "error")

    def test_main_rejects_query_string_in_openai_url(self) -> None:
        for url in (
            "http://localhost:8080/v1?key=value",
            "http://127.0.0.1:8080/v1?apikey=sk-health-secret-1",
        ):
            config = ChatConfig(
                local_provider_kind="openai_compatible", local_base_url=url
            )
            with self.subTest(url=url):
                code, out, _ = self._run_main(config)
                self.assertEqual(code, 1)
                result = json.loads(out)
                self.assertEqual(result["status"], "error")
                self.assertNotIn("sk-", out)
                self.assertNotIn("http://", out)

    def test_main_rejects_fragment_in_openai_url(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:8080/v1#frag",
        )
        code, out, _ = self._run_main(config)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["status"], "error")
        self.assertNotIn("frag", out)

    def test_main_rejects_bare_delimiter_in_openai_url(self) -> None:
        for url in (
            "http://localhost:8080/v1?",
            "http://localhost:8080/v1#",
            "http://localhost:8080/v1?#",
            "http://127.0.0.1:8080/v1?",
        ):
            config = ChatConfig(
                local_provider_kind="openai_compatible", local_base_url=url
            )
            with self.subTest(url=url):
                code, out, _ = self._run_main(config)
                self.assertEqual(code, 1)
                result = json.loads(out)
                self.assertEqual(result["status"], "error")
                self.assertNotIn("http://", out)
                self.assertNotIn("localhost", out)

    def test_main_never_echoes_canary_url_parts(self) -> None:
        canary = "canary-health-99"
        for url in (
            f"http://user:{canary}@localhost:8080/v1",
            f"http://{canary}:8080/v1",
            f"http://localhost:8080/v1?key={canary}",
            f"http://localhost:8080/v1?",
            f"http://localhost:8080/v1#{canary}",
            "http://[::1",
        ):
            config = ChatConfig(
                local_provider_kind="openai_compatible", local_base_url=url
            )
            with self.subTest(url=url):
                code, out, _ = self._run_main(config)
                self.assertEqual(code, 1)
                self.assertNotIn(canary, out)
                self.assertNotIn("http://", out)
                self.assertNotIn("IPv6", out)

    def test_main_config_load_failure_hides_details(self) -> None:
        out = io.StringIO()
        with patch("src.chat.config.ChatConfig.load", side_effect=ValueError("bad json")):
            with redirect_stdout(out):
                with self.assertRaises(SystemExit) as cm:
                    main()
        self.assertEqual(cm.exception.code, 1)
        result = json.loads(out.getvalue())
        self.assertEqual(result["status"], "error")
        self.assertNotIn("bad json", out.getvalue())


if __name__ == "__main__":
    unittest.main()