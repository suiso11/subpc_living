from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.assistant.contracts import AssistantGenerationError
from src.assistant.service import AssistantService
from src.chat.config import LOCAL_OPENAI_DEFAULT_BASE_URL, ChatConfig
from src.chat.session import ChatSession
from src.discord_bot.bot import DiscordConsoleState, DiscordLLMProfile
from src.llm.contracts import GenerationOptions
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter
from src.service.healthcheck import HealthChecker


def _profile(
    name: str,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "model-a",
    provider_kind: str = "ollama",
    local_base_url: str = "",
    api_key_env: str = "",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_ctx: int = 4096,
    num_predict: int | None = None,
) -> DiscordLLMProfile:
    if provider_kind == "openai_compatible" and not local_base_url:
        local_base_url = base_url
    return DiscordLLMProfile(
        name=name,
        ollama_base_url=base_url,
        model=model,
        system_prompt="system",
        max_history_turns=10,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        num_predict=num_predict,
        provider_kind=provider_kind,
        local_base_url=local_base_url,
        api_key_env=api_key_env,
    )


def _options(profile: DiscordLLMProfile) -> GenerationOptions:
    return GenerationOptions(
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        repeat_penalty=profile.repeat_penalty,
        num_ctx=profile.num_ctx,
        num_predict=profile.num_predict,
    )


class _RecordingRouter:
    def __init__(self, delegate: StaticRouter) -> None:
        self.delegate = delegate
        self.requests = []

    def route(self, request):
        self.requests.append(request)
        return self.delegate.route(request)


class _FailingProvider(FakeProvider):
    def generate(self, messages, **kwargs):
        raise RuntimeError("generation failed")


class _CountingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1
        super().close()


class DiscordAssistantRouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _session(self) -> ChatSession:
        return ChatSession(
            system_prompt="system",
            history_dir=str(Path(self.temp_dir.name) / "history"),
        )

    @staticmethod
    def _state(default_provider: FakeProvider) -> DiscordConsoleState:
        return DiscordConsoleState(
            config=SimpleNamespace(emotion_tag_enabled=False, num_ctx=8192),
            llm=default_provider,
            provider_registry=ProviderRegistry(),
        )

    def test_different_profiles_route_to_their_registered_providers(self) -> None:
        fast_profile = _profile("fast", model="small")
        strong_profile = _profile("strong", model="large")
        fast = FakeProvider(response="fast response", model="small")
        strong = FakeProvider(response="strong response", model="large")
        state = self._state(fast)
        assert state.provider_registry is not None
        state.provider_registry.register("fast", fast, local=True)
        state.provider_registry.register("strong", strong, local=True)

        fast_text, _ = state.ask_session(
            self._session(), "fast prompt", threading.Lock(), fast_profile
        )
        strong_text, _ = state.ask_session(
            self._session(), "strong prompt", threading.Lock(), strong_profile
        )

        self.assertEqual(fast_text, "fast response")
        self.assertEqual(strong_text, "strong response")
        self.assertEqual(len(fast.calls), 1)
        self.assertEqual(len(strong.calls), 1)

    def test_profiles_with_same_endpoint_get_separate_provider_ids(self) -> None:
        """同一 backend/model でも profile ごとに provider を分け、provider_id が
        registry 経路 (profile 名) と一致することを保証する。"""
        first = _profile("first", base_url="http://ollama", model="shared")
        second = _profile("second", base_url="http://ollama", model="shared")
        first_provider = FakeProvider(model="shared")
        second_provider = FakeProvider(model="shared")
        state = DiscordConsoleState()

        with patch(
            "src.discord_bot.bot.OllamaProvider",
            side_effect=[first_provider, second_provider],
        ) as factory:
            got_first = state._llm_for_profile(first)
            got_second = state._llm_for_profile(second)

        self.assertIs(got_first, first_provider)
        self.assertIs(got_second, second_provider)
        self.assertIsNot(got_first, got_second)
        self.assertEqual(factory.call_count, 2)
        factory.assert_any_call(
            base_url="http://ollama", model="shared", provider_id="first"
        )
        factory.assert_any_call(
            base_url="http://ollama", model="shared", provider_id="second"
        )
        assert state.provider_registry is not None
        self.assertIs(state.provider_registry.get("first").provider, first_provider)
        self.assertIs(state.provider_registry.get("second").provider, second_provider)

    def test_default_profile_is_ollama_backend(self) -> None:
        cfg = ChatConfig()
        profile = DiscordLLMProfile.from_config(cfg, name="default")
        self.assertEqual(profile.provider_kind, "ollama")
        self.assertEqual(profile.backend_base_url, cfg.ollama_base_url)
        self.assertEqual(profile.api_key_env, "")

    def test_openai_profile_uses_default_loopback_base_url(self) -> None:
        cfg = ChatConfig(local_provider_kind="openai_compatible")
        profile = DiscordLLMProfile.from_config(cfg, name="default")
        self.assertEqual(profile.provider_kind, "openai_compatible")
        self.assertEqual(profile.backend_base_url, LOCAL_OPENAI_DEFAULT_BASE_URL)
        self.assertEqual(profile.ollama_base_url, cfg.ollama_base_url)

    def test_channel_profile_override_selects_openai_backend(self) -> None:
        cfg = ChatConfig(
            discord_channel_profiles={
                "fast": {
                    "provider_kind": "openai_compatible",
                    "local_base_url": "http://127.0.0.1:1234/v1",
                }
            }
        )
        state = DiscordConsoleState()
        state.config = cfg
        profiles = state._load_llm_profiles()
        fast = profiles["fast"]
        self.assertEqual(fast.provider_kind, "openai_compatible")
        self.assertEqual(fast.backend_base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(profiles["default"].provider_kind, "ollama")

    def test_provider_kind_blank_variants_normalize_to_ollama(self) -> None:
        """provider_kind の None/空/空白は ChatConfig と同様に ollama へ正規化する。"""
        cfg = ChatConfig()
        for kind in (None, "", "   "):
            with self.subTest(kind=kind):
                profile = DiscordLLMProfile.from_config(
                    cfg, name="default", overrides={"provider_kind": kind}
                )
                self.assertEqual(profile.provider_kind, "ollama")
                self.assertEqual(profile.backend_base_url, cfg.ollama_base_url)
                self.assertEqual(profile.api_key_env, "")

    def test_provider_kind_non_string_override_rejected(self) -> None:
        """profile override の provider_kind が非 str/None 型なら ValueError で拒否される。"""
        cfg = ChatConfig()
        for kind in (123, [1, 2], ["ollama"], True, False, {"ollama": 1}, 0):
            with self.subTest(kind=kind):
                with self.assertRaises(ValueError):
                    DiscordLLMProfile.from_config(
                        cfg, name="default", overrides={"provider_kind": kind}
                    )

    def test_config_non_string_provider_kind_rejected_in_channel_profile(self) -> None:
        """channel profile の provider_kind が非 str/None 型でもロード時に ValueError になる。"""
        cfg = ChatConfig(
            discord_channel_profiles={
                "bad": {"provider_kind": 42},
                "bad_list": {"provider_kind": ["ollama"]},
                "bad_bool": {"provider_kind": True},
            }
        )
        state = DiscordConsoleState()
        state.config = cfg
        for name in ("bad", "bad_list", "bad_bool"):
            with self.subTest(profile=name):
                with self.assertRaises(ValueError):
                    state._load_llm_profiles()

    def test_config_non_string_provider_kind_rejected_directly(self) -> None:
        """ChatConfig の kind が非 str/None 型のまま profile 化しようとすると ValueError。"""
        cfg = ChatConfig(local_provider_kind=["ollama"])
        with self.assertRaises(ValueError):
            DiscordLLMProfile.from_config(cfg, name="default")

    def test_channel_profile_override_blank_provider_kind_is_ollama(self) -> None:
        """channel profile override の provider_kind が空でも Ollama backend のまま動く。"""
        cfg = ChatConfig(
            discord_channel_profiles={
                "none_kind": {"provider_kind": None},
                "empty_kind": {"provider_kind": ""},
                "ws_kind": {"provider_kind": "  "},
            }
        )
        state = DiscordConsoleState()
        state.config = cfg
        profiles = state._load_llm_profiles()
        for name in ("none_kind", "empty_kind", "ws_kind"):
            with self.subTest(profile=name):
                self.assertEqual(profiles[name].provider_kind, "ollama")
                self.assertEqual(profiles[name].backend_base_url, cfg.ollama_base_url)
        self.assertEqual(profiles["default"].provider_kind, "ollama")

    def test_provider_kind_missing_inherits_config_value(self) -> None:
        """override 未指定は config の kind を継承し、None の config も ollama に正規化される。"""
        cfg = ChatConfig(local_provider_kind=None)
        profile = DiscordLLMProfile.from_config(cfg, name="default")
        self.assertEqual(profile.provider_kind, "ollama")

    def test_provider_kind_whitespace_openai_keeps_explicit_kind(self) -> None:
        """明示的に有効な kind は正規化後も保持され、未知の kind は拒否される。"""
        cfg = ChatConfig()
        openai_profile = DiscordLLMProfile.from_config(
            cfg,
            overrides={
                "provider_kind": "  openai_compatible  ",
                "local_base_url": "http://127.0.0.1:1234/v1",
            },
        )
        self.assertEqual(openai_profile.provider_kind, "openai_compatible")
        with self.assertRaises(ValueError):
            DiscordLLMProfile.from_config(
                cfg, overrides={"provider_kind": "azure"}
            )

    def test_openai_profile_rejects_non_loopback_local_base_url(self) -> None:
        cfg = ChatConfig()
        with self.assertRaises(ValueError):
            DiscordLLMProfile.from_config(
                cfg,
                overrides={
                    "provider_kind": "openai_compatible",
                    "local_base_url": "http://192.168.1.5:8080/v1",
                },
            )

    def test_cache_separates_backend_kind_and_auth_env_name(self) -> None:
        ollama = _profile("a", base_url="http://localhost:8080/v1", model="m")
        openai = _profile(
            "b",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        openai_keyed = _profile(
            "c",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.llm_profiles = {"a": ollama, "b": openai, "c": openai_keyed}

        with patch(
            "src.discord_bot.bot.OllamaProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as ollama_factory:
            with patch(
                "src.discord_bot.bot.LocalOpenAICompatibleProvider",
                side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
            ) as openai_factory:
                state._llm_for_profile(ollama)
                state._llm_for_profile(openai)
                state._llm_for_profile(openai_keyed)

        self.assertEqual(ollama_factory.call_count, 1)
        self.assertEqual(openai_factory.call_count, 2)
        assert state.provider_registry is not None
        self.assertIsNot(
            state.provider_registry.get("a").provider,
            state.provider_registry.get("b").provider,
        )
        self.assertIsNot(
            state.provider_registry.get("b").provider,
            state.provider_registry.get("c").provider,
        )

    def test_cache_separates_providers_for_distinct_profiles_same_backend(self) -> None:
        """同一 backend でも profile 名が違えば別 provider を作る (provider_id 一致)。"""
        first = _profile(
            "first",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        second = _profile(
            "second",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.llm_profiles = {"first": first, "second": second}

        with patch(
            "src.discord_bot.bot.LocalOpenAICompatibleProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            first_provider = state._llm_for_profile(first)
            second_provider = state._llm_for_profile(second)

        self.assertIsNot(first_provider, second_provider)
        self.assertEqual(factory.call_count, 2)
        assert state.provider_registry is not None
        self.assertIs(state.provider_registry.get("first").provider, first_provider)
        self.assertIs(state.provider_registry.get("second").provider, second_provider)

    def test_openai_provider_gets_key_resolved_from_env_name_only(self) -> None:
        profile = _profile(
            "gpt",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        captured: dict = {}

        def _factory(**kwargs):
            captured.update(kwargs)
            return FakeProvider(model=kwargs["model"])

        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            with patch(
                "src.discord_bot.bot.LocalOpenAICompatibleProvider",
                side_effect=_factory,
            ):
                state._llm_for_profile(profile)

        self.assertEqual(captured["api_key"], "sekrit-value")
        # キャッシュキーは env 名のみで、キー値を含まない。profile 名も含む
        self.assertIn(
            (
                "openai_compatible",
                "http://localhost:8080/v1",
                "m",
                "LOCAL_DUMMY_KEY",
                "gpt",
            ),
            state._llm_providers,
        )
        # profile は env 名だけを持つ
        self.assertEqual(profile.api_key_env, "LOCAL_DUMMY_KEY")

    def test_openai_provider_without_env_is_keyless(self) -> None:
        profile = _profile(
            "openai",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        state = DiscordConsoleState()
        captured: dict = {}

        def _factory(**kwargs):
            captured.update(kwargs)
            return FakeProvider(model=kwargs["model"])

        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            with patch(
                "src.discord_bot.bot.LocalOpenAICompatibleProvider",
                side_effect=_factory,
            ):
                state._llm_for_profile(profile)

        self.assertIsNone(captured["api_key"])
        self.assertEqual(profile.api_key_env, "")

    def test_openai_provider_registers_local_under_profile_name(self) -> None:
        profile = _profile(
            "openai",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        state = DiscordConsoleState()

        with patch(
            "src.discord_bot.bot.LocalOpenAICompatibleProvider",
            return_value=FakeProvider(model="m"),
        ):
            provider = state._llm_for_profile(profile)

        assert state.provider_registry is not None
        entry = state.provider_registry.get("openai")
        self.assertTrue(entry.local)
        self.assertIs(entry.provider, provider)

    def test_openai_profile_service_routes_to_provider(self) -> None:
        profile = _profile(
            "openai",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        provider = FakeProvider(response="openai ok", model="m")
        state = self._state(FakeProvider(model="m"))
        assert state.provider_registry is not None
        state.provider_registry.register("openai", provider, local=True)

        text, _ = state.ask_session(self._session(), "hi", threading.Lock(), profile)

        self.assertEqual(text, "openai ok")
        self.assertEqual(len(provider.calls), 1)

    def test_status_text_uses_selected_mode_and_skips_ollama_api_tags(self) -> None:
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {
            "default": _profile(
                "default",
                provider_kind="openai_compatible",
                base_url="http://localhost:8080/v1",
                model="model-a",
            ),
        }
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"local_provider": {"status": "ok", "kind": "openai_compatible"}},
        }
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertEqual(kwargs["provider_kind"], "openai_compatible")
        self.assertEqual(kwargs["provider_url"], "http://localhost:8080/v1")
        self.assertFalse(kwargs["include_web"])
        # Ollama /api/tags に繋がる include_ollama は selected mode では使わない
        self.assertNotIn("include_ollama", kwargs)
        self.assertIn("backend: openai_compatible ok", text)
        self.assertNotIn("ollama (legacy)", text)
        self.assertNotIn("/api/tags", text)

    def test_status_text_probes_ollama_and_keeps_legacy_line(self) -> None:
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="ollama",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {
            "default": _profile(
                "default", base_url="http://localhost:11434", model="model-a"
            ),
        }
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"ollama": {"status": "ok", "models": ["model-a"]}},
        }
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertEqual(kwargs["provider_kind"], "ollama")
        self.assertEqual(kwargs["provider_url"], "http://localhost:11434")
        self.assertFalse(kwargs["include_web"])
        self.assertNotIn("include_ollama", kwargs)
        self.assertIn("backend: ollama ok", text)
        self.assertIn("ollama (legacy): ok", text)

    def test_status_text_backend_reachability_from_selected_probe(self) -> None:
        cases = [
            (
                "ollama",
                {"ollama": {"status": "ok"}},
                "backend: ollama ok",
                "ollama (legacy): ok",
            ),
            (
                "ollama",
                {"ollama": {"status": "error"}},
                "backend: ollama error",
                "ollama (legacy): error",
            ),
            (
                "openai_compatible",
                {"local_provider": {"status": "ok"}},
                "backend: openai_compatible ok",
                None,
            ),
            (
                "openai_compatible",
                {"local_provider": {"status": "unknown"}},
                "backend: openai_compatible unknown",
                None,
            ),
            (
                "openai_compatible",
                {"local_provider": {"status": "error"}},
                "backend: openai_compatible error",
                None,
            ),
        ]
        for kind, checks, expected_backend, expected_legacy in cases:
            with self.subTest(kind=kind, checks=checks):
                state = DiscordConsoleState()
                state.config = SimpleNamespace(
                    ollama_base_url="http://localhost:11434",
                    model="model-a",
                    local_provider_kind=kind,
                )
                state.llm = FakeProvider(model="model-a")
                state.llm_profiles = {
                    "default": _profile(
                        "default",
                        provider_kind=kind,
                        base_url="http://localhost:8080/v1",
                        model="model-a",
                    ),
                }
                checker = MagicMock()
                checker.check_all.return_value = {"status": "ok", "checks": checks}
                with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
                    text = state.status_text()

                self.assertIn(expected_backend, text)
                if expected_legacy is None:
                    self.assertNotIn("ollama (legacy)", text)
                else:
                    self.assertIn(expected_legacy, text)

    def test_status_text_reports_unconfigured_when_probe_missing(self) -> None:
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="ollama",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {
            "default": _profile(
                "default", base_url="http://localhost:11434", model="model-a"
            ),
        }
        checker = MagicMock()
        checker.check_all.return_value = {"status": "ok", "checks": {}}
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        self.assertIn("backend: ollama unconfigured", text)
        self.assertIn("ollama (legacy): unconfigured", text)

    def test_status_text_does_not_expose_key_or_url(self) -> None:
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"local_provider": {"status": "ok"}},
        }
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
                text = state.status_text()

        self.assertNotIn("sekrit-value", text)
        self.assertNotIn("http://localhost:8080", text)
        self.assertNotIn("LOCAL_DUMMY_KEY", text)

    def test_status_text_passes_resolved_key_for_configured_env(self) -> None:
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"local_provider": {"status": "ok"}},
        }
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
                text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertEqual(kwargs["provider_api_key"], "sekrit-value")
        self.assertNotIn("sekrit-value", text)
        self.assertNotIn("http://localhost:8080", text)
        self.assertNotIn("LOCAL_DUMMY_KEY", text)

    def test_status_text_unset_env_sends_none(self) -> None:
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="LOCAL_UNSET_KEY",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"local_provider": {"status": "ok"}},
        }
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertIsNone(kwargs["provider_api_key"])

    def test_status_text_blank_env_name_sends_none(self) -> None:
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"local_provider": {"status": "ok"}},
        }
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertIsNone(kwargs["provider_api_key"])

    def test_status_text_ollama_keyless_passes_none(self) -> None:
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="ollama",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {
            "default": _profile(
                "default", base_url="http://localhost:11434", model="model-a"
            ),
        }
        checker = MagicMock()
        checker.check_all.return_value = {
            "status": "ok",
            "checks": {"ollama": {"status": "ok"}},
        }
        with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
            text = state.status_text()

        _, kwargs = checker.check_all.call_args
        self.assertIsNone(kwargs["provider_api_key"])

    def test_status_text_key_rotation_is_read_per_request(self) -> None:
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}

        def _probe() -> str | None:
            checker = MagicMock()
            checker.check_all.return_value = {
                "status": "ok",
                "checks": {"local_provider": {"status": "ok"}},
            }
            with patch("src.discord_bot.bot.HealthChecker", return_value=checker):
                state.status_text()
            return checker.check_all.call_args.kwargs.get("provider_api_key")

        first = _probe()
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "old-key"}, clear=False):
            second = _probe()
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "new-key"}, clear=False):
            third = _probe()
        self.assertIsNone(first)
        self.assertEqual(second, "old-key")
        self.assertEqual(third, "new-key")

    def test_status_text_bearer_reaches_transport_for_configured_env(self) -> None:
        """configured dummy env のとき、実 HealthChecker が Bearer ヘッダを送る。

        ネットワークは偽の httpx クライアントで遮断し、送信された URL と
        ヘッダを記録する (offline)。
        """

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "m"}]}

        class _FakeClient:
            def __init__(self, timeout=None):
                self.requests = []

            def get(self, url, headers=None):
                self.requests.append((url, headers or {}))
                return _FakeResponse()

            def close(self):
                pass

        client = _FakeClient()
        fake_httpx = SimpleNamespace(
            Client=lambda timeout=None: client,
            TimeoutException=RuntimeError,
            RequestError=RuntimeError,
        )
        profile = _profile(
            "default",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="model-a",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()
        state.config = SimpleNamespace(
            ollama_base_url="http://localhost:11434",
            model="model-a",
            local_provider_kind="openai_compatible",
        )
        state.llm = FakeProvider(model="model-a")
        state.llm_profiles = {"default": profile}
        with patch.dict("os.environ", {"LOCAL_DUMMY_KEY": "sekrit-value"}, clear=False):
            with patch("src.service.healthcheck.httpx", fake_httpx):
                text = state.status_text()

        self.assertEqual(len(client.requests), 1)
        url, headers = client.requests[0]
        self.assertEqual(url, "http://localhost:8080/v1/models")
        self.assertEqual(headers.get("Authorization"), "Bearer sekrit-value")
        self.assertNotIn("sekrit-value", text)
        self.assertNotIn("http://localhost:8080", text)
        self.assertNotIn("LOCAL_DUMMY_KEY", text)

    def test_selected_provider_mode_never_probes_openai_api_tags(self) -> None:
        """openai_compatible は /api/tags ではなく /models をプローブする。

        selected-provider モード (provider_kind 指定) では check_all が
        include_ollama を参照せず、checks["ollama"] を作らない。
        """

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {"data": [{"id": "m"}]}

        class _FakeClient:
            def __init__(self, timeout=None):
                self.requests = []

            def get(self, url):
                self.requests.append(url)
                return _FakeResponse()

            def close(self):
                pass

        client = _FakeClient()
        fake_httpx = SimpleNamespace(
            Client=lambda timeout=None: client,
            TimeoutException=RuntimeError,
            RequestError=RuntimeError,
        )
        with patch("src.service.healthcheck.httpx", fake_httpx):
            checker = HealthChecker(ollama_url="http://localhost:11434")
            result = checker.check_all(
                provider_kind="openai_compatible",
                provider_url="http://127.0.0.1:9999/v1",
                include_web=False,
            )

        self.assertEqual(client.requests, ["http://127.0.0.1:9999/v1/models"])
        self.assertNotIn("ollama", result["checks"])
        self.assertIn("local_provider", result["checks"])

    def test_cache_reuses_provider_for_repeated_same_profile(self) -> None:
        """同一 profile への再呼び出しは provider を作り直さず共有する。"""
        profile = _profile(
            "first",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
            api_key_env="LOCAL_DUMMY_KEY",
        )
        state = DiscordConsoleState()

        with patch(
            "src.discord_bot.bot.LocalOpenAICompatibleProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            first_provider = state._llm_for_profile(profile)
            second_provider = state._llm_for_profile(profile)

        self.assertIs(first_provider, second_provider)
        factory.assert_called_once()
        assert state.provider_registry is not None
        self.assertIs(state.provider_registry.get("first").provider, first_provider)

    def test_provider_id_matches_profile_registry_route(self) -> None:
        """openai_compatible でも provider_id が registry 経路 (profile 名) と一致する。"""
        first = _profile(
            "first",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        second = _profile(
            "second",
            provider_kind="openai_compatible",
            base_url="http://localhost:8080/v1",
            model="m",
        )
        state = DiscordConsoleState()

        with patch(
            "src.discord_bot.bot.LocalOpenAICompatibleProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            first_provider = state._llm_for_profile(first)
            second_provider = state._llm_for_profile(second)

        self.assertEqual(factory.call_count, 2)
        factory.assert_any_call(
            model="m",
            base_url="http://localhost:8080/v1",
            provider_id="first",
            api_key=None,
        )
        factory.assert_any_call(
            model="m",
            base_url="http://localhost:8080/v1",
            provider_id="second",
            api_key=None,
        )
        assert state.provider_registry is not None
        self.assertIs(state.provider_registry.get("first").provider, first_provider)
        self.assertIs(state.provider_registry.get("second").provider, second_provider)

    def test_initialize_profile_caches_builds_all_profiles_and_fixes_cache(self) -> None:
        default_profile = _profile("default")
        fast = _profile("fast", base_url="http://ollama", model="shared")
        strong = _profile("strong", base_url="http://ollama", model="shared")
        state = DiscordConsoleState()
        state.llm_profiles = {
            "default": default_profile,
            "fast": fast,
            "strong": strong,
        }

        with patch(
            "src.discord_bot.bot.OllamaProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            state.llm = state._llm_for_profile(default_profile)
            state._initialize_profile_caches()
            self.assertEqual(factory.call_count, 3)
            state._initialize_profile_caches()
            self.assertEqual(factory.call_count, 3)

        self.assertEqual(set(state.assistant_services), {"default", "fast", "strong"})
        assert state.provider_registry is not None
        fast_provider = state.provider_registry.get("fast").provider
        # profile 名で cache を分離するため、fast/strong は別 provider になる
        self.assertIsNot(state.provider_registry.get("strong").provider, fast_provider)
        self.assertIs(state.llm, state.provider_registry.get("default").provider)
        self.assertIs(
            state._service_for_profile(strong), state.assistant_services["strong"]
        )

    def test_parallel_first_use_of_same_profile_builds_service_once(self) -> None:
        profile = _profile("fast", base_url="http://ollama", model="shared")
        state = DiscordConsoleState()
        state.llm_profiles = {"default": _profile("default"), "fast": profile}

        barrier = threading.Barrier(8)
        services: list[AssistantService] = []
        errors: list[Exception] = []

        def _first_use() -> None:
            barrier.wait()
            try:
                services.append(state._service_for_profile(profile))
            except Exception as e:
                errors.append(e)

        with patch(
            "src.discord_bot.bot.OllamaProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            threads = [threading.Thread(target=_first_use) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(services), 8)
        for service in services:
            self.assertIs(service, state.assistant_services["fast"])
        factory.assert_called_once()
        assert state.provider_registry is not None
        self.assertIs(
            state.provider_registry.get("fast").provider,
            state._llm_providers[("ollama", "http://ollama", "shared", "", "fast")],
        )

    def test_parallel_first_use_of_distinct_profiles_creates_separate_providers(self) -> None:
        fast = _profile("fast", base_url="http://ollama", model="shared")
        strong = _profile("strong", base_url="http://ollama", model="shared")
        state = DiscordConsoleState()
        state.llm_profiles = {
            "default": _profile("default"),
            "fast": fast,
            "strong": strong,
        }

        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def _first_use(profile: DiscordLLMProfile) -> None:
            barrier.wait()
            try:
                state._service_for_profile(profile)
            except Exception as e:
                errors.append(e)

        with patch(
            "src.discord_bot.bot.OllamaProvider",
            side_effect=lambda **kwargs: FakeProvider(model=kwargs["model"]),
        ) as factory:
            threads = [
                threading.Thread(
                    target=_first_use,
                    args=(fast if index % 2 == 0 else strong,),
                )
                for index in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        # profile 名で cache を分離するため fast/strong で2つの provider が作られる
        self.assertEqual(factory.call_count, 2)
        assert state.provider_registry is not None
        self.assertIsNot(
            state.provider_registry.get("fast").provider,
            state.provider_registry.get("strong").provider,
        )

    def test_profile_generation_options_reach_each_provider(self) -> None:
        first_profile = _profile(
            "first",
            temperature=0.2,
            top_p=0.8,
            top_k=12,
            repeat_penalty=1.05,
            num_ctx=2048,
            num_predict=111,
        )
        second_profile = _profile(
            "second",
            model="model-b",
            temperature=0.85,
            top_p=0.95,
            top_k=64,
            repeat_penalty=1.2,
            num_ctx=16384,
            num_predict=333,
        )
        first = FakeProvider(model="model-a")
        second = FakeProvider(model="model-b")
        state = self._state(first)
        assert state.provider_registry is not None
        state.provider_registry.register("first", first, local=True)
        state.provider_registry.register("second", second, local=True)

        state.ask_session(self._session(), "one", threading.Lock(), first_profile)
        state.ask_session(self._session(), "two", threading.Lock(), second_profile)

        self.assertEqual(
            first.calls[0]["options"],
            {
                "temperature": 0.2,
                "top_p": 0.8,
                "top_k": 12,
                "repeat_penalty": 1.05,
                "num_ctx": 2048,
                "num_predict": 111,
            },
        )
        self.assertEqual(
            second.calls[0]["options"],
            {
                "temperature": 0.85,
                "top_p": 0.95,
                "top_k": 64,
                "repeat_penalty": 1.2,
                "num_ctx": 16384,
                "num_predict": 333,
            },
        )

    def test_assistant_request_records_discord_route_fields(self) -> None:
        profile = _profile("voice_short")
        provider = FakeProvider(model=profile.model)
        state = self._state(provider)
        assert state.provider_registry is not None
        state.provider_registry.register(profile.name, provider, local=True)
        router = _RecordingRouter(
            StaticRouter(state.provider_registry, default_provider_id=profile.name)
        )
        state.assistant_services[profile.name] = AssistantService(
            state.provider_registry,
            router,
            options=_options(profile),
        )

        session = self._session()
        state.sessions["discord:1:2"] = session
        state.ask_session(session, "hello", threading.Lock(), profile)

        request = router.requests[0]
        self.assertEqual(request.text, "hello")
        self.assertEqual(request.conversation_id, "discord:1:2")
        self.assertEqual(request.channel, "discord")
        self.assertEqual(request.profile, "chat_auto")
        self.assertEqual(request.privacy, "local_only")
        self.assertEqual(request.requested_provider, "voice_short")

    def test_failed_generation_rolls_back_user_message(self) -> None:
        profile = _profile("default")
        provider = _FailingProvider(model=profile.model)
        state = self._state(provider)
        assert state.provider_registry is not None
        state.provider_registry.register(profile.name, provider, local=True)
        session = self._session()

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            state.ask_session(session, "do not retain", threading.Lock(), profile)

        self.assertEqual(session._messages, [])

    def test_provider_error_surfaces_as_the_common_assistant_error(self) -> None:
        """Provider由来の通信エラーは共通例外へ再ラップされる (Phase C の契約)。

        移行前は httpx 例外がそのまま呼び出し元へ抜けていた。Service経由に
        なった後は全Adapterで `AssistantGenerationError` に統一され、
        元の例外は `__cause__` と本文に残る。Discordへ表示される文言が
        変わるのは、この契約に伴う意図した変更。
        """

        class _NetworkErrorProvider(FakeProvider):
            def generate(self, messages, **kwargs):  # type: ignore[override]
                raise ProviderRequestError("default", "generate", "connection refused")

        profile = _profile("default")
        provider = _NetworkErrorProvider(model=profile.model)
        state = self._state(provider)
        assert state.provider_registry is not None
        state.provider_registry.register(profile.name, provider, local=True)
        session = self._session()

        with self.assertRaises(AssistantGenerationError) as caught:
            state.ask_session(session, "do not retain", threading.Lock(), profile)

        self.assertIsInstance(caught.exception.__cause__, ProviderRequestError)
        self.assertIn("connection refused", str(caught.exception))
        self.assertEqual(session._messages, [])

    def test_task_extraction_calls_provider_with_fixed_options(self) -> None:
        provider = FakeProvider(response="{}")
        state = self._state(provider)
        service = MagicMock()
        state.assistant_services["default"] = service

        state.extract_task_from_text("明日までに資料を作る")

        self.assertEqual(provider.calls[0]["options"]["temperature"], 0.0)
        self.assertEqual(provider.calls[0]["options"]["num_predict"], 256)
        service.generate.assert_not_called()

    def test_ask_session_calls_service_respond_with_blocks_and_base_system(self) -> None:
        """ask_session は service.respond を build_blocks + base_system で呼ぶ。"""
        profile = _profile("default")
        provider = FakeProvider(response="ok")
        state = self._state(provider)
        session = self._session()

        mock_service = MagicMock()
        mock_service.respond.return_value = (MagicMock(text="ok"), None)
        state.assistant_services["default"] = mock_service

        state.ask_session(session, "ping", threading.Lock(), profile)

        mock_service.respond.assert_called_once()
        call_args = mock_service.respond.call_args
        # First positional arg is the AssistantRequest
        request = call_args.args[0]
        self.assertEqual(request.text, "ping")
        # Second positional arg is blocks (from session.build_blocks())
        blocks = call_args.args[1]
        self.assertIsInstance(blocks, tuple)
        # Keyword arg base_system must equal session.system_prompt
        self.assertEqual(call_args.kwargs.get("base_system", None), "system")

    def test_registry_closes_shared_provider_once(self) -> None:
        provider = _CountingProvider()
        registry = ProviderRegistry()
        registry.register("first", provider, local=True)
        registry.register("second", provider, local=True)

        registry.close()

        self.assertEqual(provider.close_count, 1)


if __name__ == "__main__":
    unittest.main()
