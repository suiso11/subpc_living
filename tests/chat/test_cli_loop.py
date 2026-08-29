from __future__ import annotations

from contextlib import redirect_stdout
import io
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.assistant import AssistantRequest, AssistantService
from src.chat.config import ChatConfig
import src.chat.main as chat_main
from src.chat.main import Color, build_cli_service, run_chat_loop
from src.chat.session import ChatSession
from src.llm import GenerationOptions, ProviderRegistry
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.routing.static import StaticRouter


class InputSequence:
    def __init__(self, *values: str) -> None:
        self._values = iter(values)

    def __call__(self, prompt: str) -> str:
        try:
            return next(self._values)
        except StopIteration as exc:
            raise EOFError from exc


class RecordingRouter:
    def __init__(self, delegate: StaticRouter) -> None:
        self.delegate = delegate
        self.requests: list[AssistantRequest] = []

    def route(self, request: AssistantRequest):
        self.requests.append(request)
        return self.delegate.route(request)


class FailingProvider(FakeProvider):
    def generate(self, messages, **options):
        self.calls.append(
            {"kind": "generate", "messages": list(messages), "options": options}
        )
        raise ProviderRequestError("ollama", "generate", "planned failure")


class FailingStreamProvider(FakeProvider):
    def __init__(self, *, fail_after_token: bool) -> None:
        super().__init__(response="回復後の応答", stream_chunks=("回復後の応答",))
        self.fail_after_token = fail_after_token
        self.failures_remaining = 1

    def generate_stream(self, messages, **options):
        self.calls.append(
            {"kind": "generate_stream", "messages": list(messages), "options": options}
        )
        if self.failures_remaining:
            self.failures_remaining -= 1
            if self.fail_after_token:
                yield "部分出力"
            raise ProviderRequestError(
                "ollama", "generate_stream", "planned failure"
            )
        yield from self.stream_chunks


class FakeStreamResult:
    """Offline StreamResult 風オブジェクト。close() 回数を記録する。"""

    def __init__(self, chunks, text, stats=None) -> None:
        self._chunks = chunks
        self.response = SimpleNamespace(text=text, stats=stats or {})
        self.close_count = 0

    def __iter__(self):
        yield from self._chunks

    def close(self) -> None:
        self.close_count += 1


class FailingStreamResult(FakeStreamResult):
    """反復途中で例外を送出する StreamResult 風オブジェクト。"""

    def __init__(self, chunks, text, exc, stats=None) -> None:
        super().__init__(chunks, text, stats=stats)
        self.exc = exc

    def __iter__(self):
        yield from self._chunks
        raise self.exc


class StartupProvider(FakeProvider):
    def __init__(
        self, *, available: bool, has_model: bool, models: list[str] | None = None
    ) -> None:
        super().__init__(available=available)
        self._has_model = has_model
        self._models = list(models) if models is not None else []
        self.close_calls = 0
        self.has_model_calls = 0
        self.list_calls = 0

    def has_model(self) -> bool:
        self.has_model_calls += 1
        return self._has_model

    def list_models(self) -> list[str]:
        self.list_calls += 1
        return list(self._models)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class CliLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_session(self, config: ChatConfig | None = None) -> ChatSession:
        config = config or ChatConfig()
        return ChatSession(
            system_prompt=config.effective_system_prompt(),
            max_history_turns=config.max_history_turns,
            history_dir=self.temp_dir.name,
        )

    @staticmethod
    def run_loop(config, session, service, *inputs: str) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            run_chat_loop(
                config,
                session,
                service,
                read_input=InputSequence(*inputs),
            )
        return output.getvalue()

    def test_non_stream_outputs_and_records_response(self) -> None:
        config = ChatConfig(stream=False)
        provider = FakeProvider(response="応答本文")
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        output = self.run_loop(config, session, service, "質問")

        self.assertIn("応答本文", output)
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "応答本文"},
            ],
        )

    def test_stream_matches_non_stream_output_and_history(self) -> None:
        non_stream_config = ChatConfig(stream=False)
        stream_config = ChatConfig(stream=True)
        non_stream_provider = FakeProvider(response="同じ応答")
        stream_provider = FakeProvider(
            response="同じ応答", stream_chunks=("同じ", "応答")
        )
        non_stream_service, _ = build_cli_service(
            non_stream_config, provider=non_stream_provider
        )
        stream_service, _ = build_cli_service(
            stream_config, provider=stream_provider
        )
        non_stream_session = self.make_session(non_stream_config)
        stream_session = self.make_session(stream_config)

        non_stream_output = self.run_loop(
            non_stream_config, non_stream_session, non_stream_service, "質問"
        )
        stream_output = self.run_loop(
            stream_config, stream_session, stream_service, "質問"
        )

        self.assertEqual(stream_output, non_stream_output)
        self.assertEqual(stream_session.messages, non_stream_session.messages)

    def test_config_generation_options_are_passed_to_provider(self) -> None:
        config = ChatConfig(
            stream=False,
            temperature=0.23,
            top_p=0.81,
            top_k=17,
            repeat_penalty=1.37,
            num_ctx=3456,
            num_predict=123,
        )
        provider = FakeProvider()
        service, _ = build_cli_service(config, provider=provider)

        self.run_loop(config, self.make_session(config), service, "質問")

        self.assertEqual(
            provider.calls[0]["options"],
            {
                "temperature": 0.23,
                "top_p": 0.81,
                "top_k": 17,
                "repeat_penalty": 1.37,
                "num_ctx": 3456,
                "num_predict": 123,
            },
        )

    def test_request_contains_cli_session_identity(self) -> None:
        config = ChatConfig(stream=False)
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)
        recording_router = RecordingRouter(
            StaticRouter(registry, default_provider_id="ollama")
        )
        service = AssistantService(
            registry, recording_router, options=GenerationOptions()
        )
        session = self.make_session(config)

        self.run_loop(config, session, service, "質問")

        request = recording_router.requests[0]
        self.assertEqual(request.conversation_id, session.session_id)
        self.assertEqual(request.channel, "cli")
        self.assertEqual(request.privacy, "local_only")

    def test_commands_keep_existing_output_and_behavior(self) -> None:
        cases = {
            "/help": (
                f"\n{Color.YELLOW}コマンド一覧:{Color.RESET}\n"
                "  /help     このヘルプを表示\n"
                "  /info     セッション情報を表示\n"
                "  /clear    会話履歴をクリア\n"
                "  /system   システムプロンプトを表示\n"
                "  /save     会話を保存\n"
                "  /model    現在のモデル情報を表示\n"
                "  /quit     終了 (Ctrl+C でも可)\n\n"
            ),
            "/info": "セッション:",
            "/clear": f"{Color.YELLOW}会話履歴をクリアしました。{Color.RESET}\n",
            "/system": "[System Prompt]",
            "/save": "保存しました:",
            "/model": (
                f"\n{Color.DIM}モデル: qwen2.5:7b-instruct-q4_K_M\n"
                "Temperature: 0.7\n"
                "コンテキスト長: 8192\n"
                f"最大履歴ターン: 20{Color.RESET}\n"
            ),
            "/unknown": "不明なコマンド:",
        }
        for command, expected in cases.items():
            with self.subTest(command=command):
                config = ChatConfig(stream=False, system_prompt="テストシステム")
                provider = FakeProvider()
                service, _ = build_cli_service(config, provider=provider)
                session = self.make_session(config)
                session.add_user_message("既存メッセージ")

                output = self.run_loop(config, session, service, command)

                if command in ("/help", "/clear", "/model"):
                    self.assertEqual(output, expected)
                    self.assertIn("\033[", output)
                    self.assertTrue(output.endswith("\n"))
                else:
                    self.assertIn(expected, output)
                if command == "/clear":
                    self.assertEqual(session.turn_count, 0)
                else:
                    self.assertEqual(session.turn_count, 1)
                self.assertEqual(provider.calls, [])

    def test_quit_and_eof_return_without_generation(self) -> None:
        config = ChatConfig(stream=False)
        for inputs in (("/quit",), ()):
            with self.subTest(inputs=inputs):
                provider = FakeProvider()
                service, _ = build_cli_service(config, provider=provider)
                session = self.make_session(config)

                self.run_loop(config, session, service, *inputs)

                self.assertEqual(session.turn_count, 0)
                self.assertEqual(provider.calls, [])

    def test_generation_error_rolls_back_user_message(self) -> None:
        config = ChatConfig(stream=False)
        provider = FailingProvider()
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        output = self.run_loop(config, session, service, "失敗する質問")

        self.assertIn("エラー:", output)
        self.assertEqual(session.messages, [])

    def test_stream_error_before_first_token_rolls_back_and_continues(self) -> None:
        config = ChatConfig(stream=True)
        provider = FailingStreamProvider(fail_after_token=False)
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        output = self.run_loop(config, session, service, "失敗する質問", "次の質問")

        self.assertIn("エラー:", output)
        self.assertIn("回復後の応答", output)
        self.assertEqual(
            session._messages,
            [
                {"role": "user", "content": "次の質問"},
                {"role": "assistant", "content": "回復後の応答"},
            ],
        )

    def test_stream_error_after_token_rolls_back_and_continues(self) -> None:
        config = ChatConfig(stream=True)
        provider = FailingStreamProvider(fail_after_token=True)
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        output = self.run_loop(config, session, service, "失敗する質問", "次の質問")

        self.assertIn("部分出力", output)
        self.assertIn("エラー:", output)
        self.assertIn("回復後の応答", output)
        self.assertEqual(
            session._messages,
            [
                {"role": "user", "content": "次の質問"},
                {"role": "assistant", "content": "回復後の応答"},
            ],
        )

    def test_stats_are_printed_only_for_stream(self) -> None:
        stats = {
            "total_duration": 2_000_000,
            "eval_count": 4,
            "eval_duration": 1_000_000_000,
        }
        outputs = {}
        for stream in (False, True):
            config = ChatConfig(stream=stream)
            provider = FakeProvider(
                response="応答", stream_chunks=("応", "答"), stats=stats
            )
            service, _ = build_cli_service(config, provider=provider)
            outputs[stream] = self.run_loop(
                config, self.make_session(config), service, "質問"
            )

        stats_line = "[4tokens, 2ms, 4.0tok/s]"
        self.assertNotIn(stats_line, outputs[False])
        self.assertIn(stats_line, outputs[True])

    def test_stream_uses_respond_stream_with_blocks_and_base_system(self) -> None:
        config = ChatConfig(stream=True)
        provider = FakeProvider(response="応答", stream_chunks=("応", "答"))
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        with patch.object(type(service), "respond_stream", wraps=service.respond_stream) as mock_rs:
            output = self.run_loop(config, session, service, "質問")
            mock_rs.assert_called_once()
            call_args = mock_rs.call_args
            # first positional arg is request
            self.assertIsInstance(call_args.args[0], AssistantRequest)
            # second positional arg is blocks (from session.build_blocks)
            self.assertIsNotNone(call_args.args[1])
            # keyword arg base_system
            self.assertEqual(call_args.kwargs.get("base_system"), session.system_prompt)

        self.assertIn("応答", output)

    def test_non_stream_uses_respond_with_blocks_and_base_system(self) -> None:
        config = ChatConfig(stream=False)
        provider = FakeProvider(response="応答")
        service, _ = build_cli_service(config, provider=provider)
        session = self.make_session(config)

        with patch.object(type(service), "respond", wraps=service.respond) as mock_r:
            output = self.run_loop(config, session, service, "質問")
            mock_r.assert_called_once()
            call_args = mock_r.call_args
            self.assertIsInstance(call_args.args[0], AssistantRequest)
            self.assertIsNotNone(call_args.args[1])
            self.assertEqual(call_args.kwargs.get("base_system"), session.system_prompt)

        self.assertIn("応答", output)

    def test_startup_check_failure_closes_registry_once(self) -> None:
        for available, has_model in ((False, True), (True, False)):
            with self.subTest(available=available, has_model=has_model):
                config = ChatConfig()
                provider = StartupProvider(
                    available=available, has_model=has_model
                )
                service, registry = build_cli_service(config, provider=provider)
                output = io.StringIO()

                with (
                    patch.object(chat_main.ChatConfig, "load", return_value=config),
                    patch.object(
                        chat_main, "create_web_search_context", return_value=None
                    ),
                    patch.object(
                        chat_main,
                        "build_cli_service",
                        return_value=(service, registry),
                    ),
                    redirect_stdout(output),
                    self.assertRaises(SystemExit) as raised,
                ):
                    chat_main.main()

                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(provider.close_calls, 1)
                # Ollama は厳格に is_available -> has_model の順で判定し、
                # リスト再取得 (list_models 直接呼び) を行わない。
                self.assertEqual(provider.list_calls, 0)
                self.assertEqual(
                    provider.has_model_calls, 0 if not available else 1
                )

    def _run_startup_ok(self, config, registry) -> str:
        """main() を起動チェック成功まで走らせ、標準出力を返す。

        データベース書き込み・対話ループ・Web検索を回避してオフラインで実行する。
        """
        provider = registry.entries()[0].provider
        service = AssistantService(
            registry,
            StaticRouter(registry, default_provider_id=config.resolved_local_provider_id()),
            options=GenerationOptions(),
        )
        output = io.StringIO()
        with (
            patch.object(chat_main.ChatConfig, "load", return_value=config),
            patch.object(chat_main, "create_web_search_context", return_value=None),
            patch.object(
                chat_main,
                "build_cli_service",
                return_value=(service, registry),
            ),
            patch.object(chat_main, "run_chat_loop", return_value=None),
            patch.object(
                chat_main, "GrowthTracker", side_effect=RuntimeError("no db in tests")
            ),
            patch.object(
                chat_main,
                "ChatSession",
                return_value=SimpleNamespace(turn_count=0, save=lambda: None),
            ),
            redirect_stdout(output),
        ):
            chat_main.main()
        return output.getvalue()

    def test_main_startup_preserves_default_ollama_backend(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=True, has_model=True)
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        output = self._run_startup_ok(config, registry)

        self.assertIn("接続OK", output)
        self.assertIn("モデル確認OK", output)
        # Ollama は厳格に has_model で判定し、直接 list_models は呼ばない。
        self.assertEqual(provider.has_model_calls, 1)
        self.assertEqual(provider.list_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_main_startup_uses_resolved_provider_id_for_openai_compatible(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_provider_id="llama-server",
        )
        provider = StartupProvider(
            available=True, has_model=True, models=[config.model]
        )
        registry = ProviderRegistry()
        registry.register("llama-server", provider, local=True)

        output = self._run_startup_ok(config, registry)

        # "ollama" のみを登録していないので、get("ollama") に戻すと
        # UnknownProviderError で失敗するはず。backend中性の文言で起動できること。
        # openai_compatible は is_available() がライフサイクルのみのため「接続OK」とは
        # 表示せず、/models 確認 (list_models 1回) を続行する。
        self.assertIn("モデル確認OK", output)
        self.assertNotIn("接続OK", output)
        self.assertNotIn("ollama", output.lower())
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_empty_discovery_continues_with_warning(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = StartupProvider(available=True, has_model=False, models=[])
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)

        output = self._run_startup_ok(config, registry)

        # /models が未実装 (空) でも起動を失敗させず、中立的な警告で続行する。
        self.assertIn("モデル情報の取得に失敗しました。生成時に確認します。", output)
        self.assertNotIn("接続OK", output)
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_known_missing_model_fails_and_closes(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = StartupProvider(
            available=True, has_model=False, models=["other-model"]
        )
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)
        service = AssistantService(
            registry,
            StaticRouter(registry, default_provider_id="local-openai"),
            options=GenerationOptions(),
        )
        output = io.StringIO()

        with (
            patch.object(chat_main.ChatConfig, "load", return_value=config),
            patch.object(
                chat_main, "create_web_search_context", return_value=None
            ),
            patch.object(
                chat_main,
                "build_cli_service",
                return_value=(service, registry),
            ),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            chat_main.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("モデル 'qwen2.5:7b-instruct-q4_K_M' が見つかりません", output.getvalue())
        self.assertIn("利用可能なモデル: other-model", output.getvalue())
        # 検出済みリスト1回だけで判定し、has_model (内部リスト再取得) は使わない。
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_unavailable_fails_and_closes(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = StartupProvider(available=False, has_model=True)
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)
        service = AssistantService(
            registry,
            StaticRouter(registry, default_provider_id="local-openai"),
            options=GenerationOptions(),
        )
        output = io.StringIO()

        with (
            patch.object(chat_main.ChatConfig, "load", return_value=config),
            patch.object(
                chat_main, "create_web_search_context", return_value=None
            ),
            patch.object(
                chat_main,
                "build_cli_service",
                return_value=(service, registry),
            ),
            redirect_stdout(output),
            self.assertRaises(SystemExit) as raised,
        ):
            chat_main.main()

        self.assertEqual(raised.exception.code, 1)
        self.assertIn("接続できません", output.getvalue())
        self.assertEqual(provider.list_calls, 0)
        self.assertEqual(provider.has_model_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_main_wires_emotion_tag_enabled_into_cli_session(self) -> None:
        config = ChatConfig(emotion_tag_enabled=True)
        provider = StartupProvider(available=True, has_model=True)
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)
        service = AssistantService(
            registry,
            StaticRouter(registry, default_provider_id="ollama"),
            options=GenerationOptions(),
        )
        session_stub = SimpleNamespace(turn_count=0, save=lambda: "saved")
        output = io.StringIO()

        with (
            patch.object(chat_main.ChatConfig, "load", return_value=config),
            patch.object(chat_main, "create_web_search_context", return_value=None),
            patch.object(
                chat_main,
                "build_cli_service",
                return_value=(service, registry),
            ),
            patch.object(chat_main, "run_chat_loop", return_value=None),
            patch.object(
                chat_main, "GrowthTracker", side_effect=RuntimeError("offline")
            ),
            patch.object(
                chat_main, "ChatSession", return_value=session_stub
            ) as chat_session_cls,
            redirect_stdout(output),
        ):
            chat_main.main()

        self.assertIs(chat_session_cls.call_args.kwargs["emotion_tags"], True)


class CliStreamCloseTest(unittest.TestCase):
    """CLI run_chat_loop の StreamResult close / error 挙動。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_session(self) -> ChatSession:
        config = ChatConfig(stream=True)
        return ChatSession(
            system_prompt=config.effective_system_prompt(),
            max_history_turns=config.max_history_turns,
            history_dir=self.temp_dir.name,
        )

    @staticmethod
    def run_loop(config, session, service, *inputs: str) -> str:
        output = io.StringIO()
        with redirect_stdout(output):
            run_chat_loop(
                config,
                session,
                service,
                read_input=InputSequence(*inputs),
            )
        return output.getvalue()

    def test_stream_normal_completion_closes_exactly_once(self) -> None:
        config = ChatConfig(stream=True)
        stream = FakeStreamResult(("応", "答"), text="応答")
        service = Mock()
        service.respond_stream.return_value = stream
        session = self.make_session()

        self.run_loop(config, session, service, "質問")

        self.assertEqual(stream.close_count, 1)
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": "応答"},
            ],
        )

    def test_stream_empty_response_closes_exactly_once(self) -> None:
        config = ChatConfig(stream=True)
        stream = FakeStreamResult((), text="")
        service = Mock()
        service.respond_stream.return_value = stream
        session = self.make_session()

        self.run_loop(config, session, service, "質問")

        self.assertEqual(stream.close_count, 1)
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "質問"},
                {"role": "assistant", "content": ""},
            ],
        )

    def test_stream_generation_error_closes_exactly_once_and_rolls_back(self) -> None:
        config = ChatConfig(stream=True)
        exc = ProviderRequestError("fake", "respond_stream", "planned failure")
        stream = FailingStreamResult(("部分",), text="", exc=exc)
        service = Mock()
        service.respond_stream.return_value = stream
        session = self.make_session()

        output = self.run_loop(config, session, service, "失敗する質問")

        self.assertEqual(stream.close_count, 1)
        self.assertIn("部分", output)
        self.assertIn("エラー:", output)
        self.assertEqual(session.messages, [])

    def test_respond_stream_failure_continues_without_leaking_close(self) -> None:
        config = ChatConfig(stream=True)
        exc = ProviderRequestError("fake", "respond_stream", "start failure")
        good_stream = FakeStreamResult(("回復",), text="回復")
        service = Mock()
        service.respond_stream.side_effect = [exc, good_stream]
        session = self.make_session()

        output = self.run_loop(
            config, session, service, "失敗する質問", "次の質問"
        )

        self.assertIn("エラー:", output)
        self.assertIn("回復", output)
        self.assertEqual(service.respond_stream.call_count, 2)
        self.assertEqual(good_stream.close_count, 1)
        self.assertEqual(
            session.messages,
            [
                {"role": "user", "content": "次の質問"},
                {"role": "assistant", "content": "回復"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
