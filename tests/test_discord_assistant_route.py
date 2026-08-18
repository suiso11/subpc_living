from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.assistant.contracts import AssistantGenerationError
from src.assistant.service import AssistantService
from src.chat.session import ChatSession
from src.discord_bot.bot import DiscordConsoleState, DiscordLLMProfile
from src.llm.contracts import GenerationOptions
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.static import StaticRouter


def _profile(
    name: str,
    *,
    base_url: str = "http://localhost:11434",
    model: str = "model-a",
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    num_ctx: int = 4096,
    num_predict: int | None = None,
) -> DiscordLLMProfile:
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

    def test_profiles_with_same_endpoint_share_provider_instance(self) -> None:
        first = _profile("first", base_url="http://ollama", model="shared")
        second = _profile("second", base_url="http://ollama", model="shared")
        provider = FakeProvider(model="shared")
        state = DiscordConsoleState()

        with patch("src.discord_bot.bot.OllamaProvider", return_value=provider) as factory:
            first_provider = state._llm_for_profile(first)
            second_provider = state._llm_for_profile(second)

        self.assertIs(first_provider, second_provider)
        factory.assert_called_once_with(
            base_url="http://ollama", model="shared", provider_id="first"
        )
        assert state.provider_registry is not None
        self.assertIs(state.provider_registry.get("first").provider, provider)
        self.assertIs(state.provider_registry.get("second").provider, provider)

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

    def test_registry_closes_shared_provider_once(self) -> None:
        provider = _CountingProvider()
        registry = ProviderRegistry()
        registry.register("first", provider, local=True)
        registry.register("second", provider, local=True)

        registry.close()

        self.assertEqual(provider.close_count, 1)


if __name__ == "__main__":
    unittest.main()
