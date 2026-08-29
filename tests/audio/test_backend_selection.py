"""VoicePipeline のローカルbackend選択 (P0-2.4) のオフライン検証。

実ネットワーク・音声・モデル・センサーを使わず、注入したFakeProviderと
モックRegistryで両backend kind (ollama / openai_compatible) を検証する。
"""
from __future__ import annotations

import io
import unittest
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest import mock

from src.audio import main as audio_main
from src.audio.pipeline import VoicePipeline
from src.chat.config import ChatConfig
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry


def build_pipeline(
    config: ChatConfig,
    registry: ProviderRegistry,
    *,
    enable_screen: bool = False,
    enable_monitor: bool = False,
    monitor_context_factory=None,
) -> VoicePipeline:
    """重い依存 (STT/TTS/VAD/オーディオI/O/センサー等) をモックしてVoicePipelineを作る。

    monitor_context_factory 指定時は MonitorContext をそのFakeで置き換え、
    生成の有無・引数を検証できるようにする。
    """
    service = mock.Mock()
    patches = [
        mock.patch("src.audio.pipeline.WhisperSTT"),
        mock.patch("src.audio.pipeline.create_tts_backend"),
        mock.patch("src.audio.pipeline.create_vad"),
        mock.patch("src.audio.pipeline.AudioRecorder"),
        mock.patch("src.audio.pipeline.AudioPlayer"),
        mock.patch(
            "src.audio.pipeline.build_local_service",
            return_value=(service, registry),
        ),
        mock.patch("src.audio.pipeline.create_web_search_context"),
        mock.patch("src.audio.pipeline.ChatSession"),
        mock.patch("src.audio.pipeline.IdleManager"),
        mock.patch("src.audio.pipeline.GrowthTracker"),
        mock.patch("src.tasks.store.TaskStore"),
        mock.patch("src.tasks.calendar_sync.CalendarContext"),
        mock.patch("src.audio.pipeline.create_screen_context"),
        (
            mock.patch("src.audio.pipeline.MonitorContext", monitor_context_factory)
            if monitor_context_factory is not None
            else mock.patch("src.audio.pipeline.MonitorContext")
        ),
    ]
    with ExitStack() as stack:
        for cm in patches:
            stack.enter_context(cm)
        with redirect_stdout(io.StringIO()):
            pipeline = VoicePipeline(
                chat_config=config,
                enable_rag=False,
                enable_vision=False,
                enable_screen=enable_screen,
                enable_monitor=enable_monitor,
                enable_persona=False,
            )
    return pipeline


class BackendSelectionTest(unittest.TestCase):
    def test_ollama_backend_resolves_ollama_provider(self) -> None:
        config = ChatConfig()
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        pipeline = build_pipeline(config, registry)

        self.assertIs(pipeline.llm, provider)

    def test_openai_compatible_custom_id_resolves_selected_provider(self) -> None:
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_provider_id="llama-server",
        )
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("llama-server", provider, local=True)

        pipeline = build_pipeline(config, registry)

        self.assertIs(pipeline.llm, provider)

    def test_openai_compatible_default_id_resolves_local_openai(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)

        pipeline = build_pipeline(config, registry)

        self.assertIs(pipeline.llm, provider)

    def test_openai_compatible_gates_ollama_only_components(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)

        pipeline = build_pipeline(
            config, registry, enable_screen=True, enable_monitor=True
        )

        # ScreenDescriber は Ollama /api/chat 前提のため作成されない (Ollama URL は使われない)。
        # MonitorContext は backend 非依存のため作成される。
        self.assertIsNone(pipeline.screen_context)
        self.assertIsNotNone(pipeline.monitor_context)

    def test_ollama_backend_creates_ollama_only_components(self) -> None:
        config = ChatConfig()
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        pipeline = build_pipeline(
            config, registry, enable_screen=True, enable_monitor=True
        )

        self.assertIsNotNone(pipeline.screen_context)
        self.assertIsNotNone(pipeline.monitor_context)


class FakeMonitorContext:
    """MonitorContext の生成を検証するためのFake。生成されたインスタンスを記録する。"""

    instances: list[FakeMonitorContext] = []

    def __init__(self, db_path: str = "", collect_interval: float = 0.0) -> None:
        self.db_path = db_path
        self.collect_interval = collect_interval
        FakeMonitorContext.instances.append(self)


class MonitorContextBackendTest(unittest.TestCase):
    """MonitorContext が backend 非依存 (enable_monitor のみで生成) であることの検証。"""

    def setUp(self) -> None:
        FakeMonitorContext.instances.clear()

    def test_ollama_enabled_creates_monitor_context(self) -> None:
        config = ChatConfig()
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        pipeline = build_pipeline(
            config,
            registry,
            enable_monitor=True,
            monitor_context_factory=FakeMonitorContext,
        )

        self.assertIsNotNone(pipeline.monitor_context)
        self.assertIsInstance(pipeline.monitor_context, FakeMonitorContext)
        self.assertEqual(len(FakeMonitorContext.instances), 1)

    def test_ollama_disabled_skips_monitor_context(self) -> None:
        config = ChatConfig()
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        pipeline = build_pipeline(
            config,
            registry,
            enable_monitor=False,
            monitor_context_factory=FakeMonitorContext,
        )

        self.assertIsNone(pipeline.monitor_context)
        self.assertEqual(len(FakeMonitorContext.instances), 0)

    def test_openai_compatible_enabled_creates_monitor_context(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)

        pipeline = build_pipeline(
            config,
            registry,
            enable_monitor=True,
            monitor_context_factory=FakeMonitorContext,
        )

        self.assertIsNotNone(pipeline.monitor_context)
        self.assertIsInstance(pipeline.monitor_context, FakeMonitorContext)
        self.assertEqual(len(FakeMonitorContext.instances), 1)

    def test_openai_compatible_disabled_skips_monitor_context(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = FakeProvider()
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)

        pipeline = build_pipeline(
            config,
            registry,
            enable_monitor=False,
            monitor_context_factory=FakeMonitorContext,
        )

        self.assertIsNone(pipeline.monitor_context)
        self.assertEqual(len(FakeMonitorContext.instances), 0)


class TTSStartupProvider(FakeProvider):
    """run_text_to_speech_mode の起動チェック用。close/has_model/list_models を数える。"""

    def __init__(
        self, *, available: bool, has_model: bool = True, models: list[str] | None = None
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


class StartupProvider(FakeProvider):
    """VoicePipeline.initialize() の起動チェック用。is_available/has_model/list_models を数える。"""

    def __init__(
        self,
        *,
        available: bool,
        has_model: bool = True,
        models: list[str] | None = None,
        model: str = "fake",
    ) -> None:
        super().__init__(available=available, model=model)
        self._has_model = has_model
        self._models = list(models) if models is not None else []
        self.is_available_calls = 0
        self.has_model_calls = 0
        self.list_calls = 0
        self.close_calls = 0

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return super().is_available()

    def has_model(self) -> bool:
        self.has_model_calls += 1
        return self._has_model

    def list_models(self) -> list[str]:
        self.list_calls += 1
        return list(self._models)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


def run_initialize(pipeline: VoicePipeline) -> tuple[bool, str]:
    """initialize() を実行して (結果, 標準出力) を返す。"""
    output = io.StringIO()
    with redirect_stdout(output):
        ok = pipeline.initialize()
    return ok, output.getvalue()


class AudioMainTTSModeStartupTest(unittest.TestCase):
    """run_text_to_speech_mode の起動チェックとRegistry終了 (P0-2) のオフライン検証。"""

    def _run_tts_mode(self, config, provider, *, input_side_effect) -> str:
        registry = ProviderRegistry()
        registry.register(config.resolved_local_provider_id(), provider, local=True)
        service = mock.Mock()
        tts = mock.Mock()
        tts.voice = "jf_alpha"
        args = SimpleNamespace(tts_voice="jf_alpha")
        output = io.StringIO()
        with (
            mock.patch(
                "src.chat.config.ChatConfig.load", return_value=config
            ),
            mock.patch(
                "src.chat.web_search.create_web_search_context", return_value=None
            ),
            mock.patch(
                "src.audio.tts_factory.create_tts_backend", return_value=tts
            ),
            mock.patch("src.audio.audio_io.AudioPlayer"),
            mock.patch(
                "src.assistant.factory.build_local_service",
                return_value=(service, registry),
            ),
            mock.patch(
                "src.chat.session.ChatSession",
                return_value=mock.Mock(turn_count=0, save=lambda: None),
            ),
            mock.patch(
                "src.growth.tracker.GrowthTracker",
                side_effect=RuntimeError("no db in tests"),
            ),
            mock.patch("builtins.input", side_effect=input_side_effect),
            redirect_stdout(output),
        ):
            audio_main.run_text_to_speech_mode(args)
        return output.getvalue()

    def test_unavailable_provider_closes_registry_before_exit(self) -> None:
        config = ChatConfig()
        provider = TTSStartupProvider(available=False)

        with self.assertRaises(SystemExit) as raised:
            self._run_tts_mode(config, provider, input_side_effect=EOFError)

        self.assertEqual(raised.exception.code, 1)
        # 接続不可の早期終了でもRegistryは閉じられる。
        self.assertEqual(provider.close_calls, 1)

    def test_missing_model_closes_registry_before_exit(self) -> None:
        config = ChatConfig()
        provider = TTSStartupProvider(available=True, has_model=False)

        with self.assertRaises(SystemExit) as raised:
            self._run_tts_mode(config, provider, input_side_effect=EOFError)

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(provider.has_model_calls, 1)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_empty_discovery_continues_and_closes(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = TTSStartupProvider(available=True, models=[])
        # 入力直後に Ctrl+C → 終了処理 (registry.close) が走る。
        output = self._run_tts_mode(
            config, provider, input_side_effect=KeyboardInterrupt
        )

        self.assertIn("モデル情報の取得に失敗しました。生成時に確認します。", output)
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_ollama_available_final_cleanup_closes_registry_once(self) -> None:
        config = ChatConfig()
        provider = TTSStartupProvider(available=True, has_model=True)

        output = self._run_tts_mode(
            config, provider, input_side_effect=KeyboardInterrupt
        )

        self.assertIn("終了します", output)
        self.assertEqual(provider.has_model_calls, 1)
        self.assertEqual(provider.close_calls, 1)


class PipelineInitializeBackendTest(unittest.TestCase):
    """VoicePipeline.initialize() の backend別起動チェック (P0-2.4) のオフライン検証。"""

    def test_blank_kind_normalizes_to_ollama(self) -> None:
        config = ChatConfig(local_provider_kind="   ")
        provider = StartupProvider(available=True, has_model=True)
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)

        pipeline = build_pipeline(config, registry)

        self.assertEqual(pipeline.local_provider_kind, "ollama")

    def test_ollama_unavailable_fails_and_closes_registry(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=False)
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)
        pipeline = build_pipeline(config, registry)

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("ローカル推論サーバーに接続できません", output)
        self.assertNotIn("接続OK", output)
        self.assertEqual(provider.is_available_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)
        # 初期化失敗時は組み立て済みRegistryが閉じられる。
        self.assertIsNone(pipeline._provider_registry)
        self.assertEqual(provider.close_calls, 1)

    def test_ollama_known_missing_model_fails_and_closes(self) -> None:
        config = ChatConfig(model="qwen2.5:7b-instruct-q4_K_M")
        provider = StartupProvider(available=True, has_model=False)
        registry = ProviderRegistry()
        registry.register("ollama", provider, local=True)
        pipeline = build_pipeline(config, registry)

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("モデル 'qwen2.5:7b-instruct-q4_K_M' が見つかりません", output)
        self.assertEqual(provider.has_model_calls, 1)
        # Ollama は厳格に has_model で判定し、直接 list_models は呼ばない。
        self.assertEqual(provider.list_calls, 0)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_known_model_no_false_connection_ok(self) -> None:
        config = ChatConfig(
            model="local-model", local_provider_kind="openai_compatible"
        )
        provider = StartupProvider(available=True, models=[config.model])
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)
        pipeline = build_pipeline(config, registry)

        ok, output = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertNotIn("接続OK", output)
        self.assertIn("モデル確認OK", output)
        # is_available はライフサイクルのみのため接続判定に使わない (重複プローブなし)。
        self.assertEqual(provider.is_available_calls, 0)
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.has_model_calls, 0)

    def test_openai_compatible_known_missing_model_fails_and_closes(self) -> None:
        config = ChatConfig(
            model="local-model", local_provider_kind="openai_compatible"
        )
        provider = StartupProvider(available=True, models=["other-model"])
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)
        pipeline = build_pipeline(config, registry)

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("モデル 'local-model' が見つかりません", output)
        self.assertIn("other-model", output)
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.close_calls, 1)

    def test_openai_compatible_chat_only_continues_with_warning(self) -> None:
        config = ChatConfig(
            model="local-model", local_provider_kind="openai_compatible"
        )
        # /models 未実装 (空) の chat-only サーバー: 失敗させず生成時に検証して続行。
        provider = StartupProvider(available=True, models=[])
        registry = ProviderRegistry()
        registry.register("local-openai", provider, local=True)
        pipeline = build_pipeline(config, registry)

        ok, output = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertIn("モデル情報の取得に失敗しました。生成時に確認します。", output)
        self.assertNotIn("接続OK", output)
        self.assertEqual(provider.list_calls, 1)
        self.assertEqual(provider.close_calls, 0)


class InitializeFailureCleanupTest(unittest.TestCase):
    """VoicePipeline.initialize() の致命的失敗時リソース解放 (冪等) のオフライン検証。"""

    def _build(self, config, provider) -> VoicePipeline:
        registry = ProviderRegistry()
        registry.register(config.resolved_local_provider_id(), provider, local=True)
        return build_pipeline(config, registry)

    def test_backend_failure_closes_registry_once(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=False)
        pipeline = self._build(config, provider)

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("ローカル推論サーバーに接続できません", output)
        # 未ロードのコンポーネントは release しない。
        self.assertIsNone(pipeline._provider_registry)
        self.assertEqual(provider.close_calls, 1)
        self.assertTrue(pipeline._init_cleanup_done)
        pipeline.stt.cleanup.assert_not_called()
        pipeline.tts.cleanup.assert_not_called()

    def test_stt_load_failure_cleans_registry(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=True, has_model=True)
        pipeline = self._build(config, provider)
        pipeline.stt.load.side_effect = RuntimeError("stt load failed")

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        # 診断は ASCII / 例外型名のみ (Windows cp932 移植性)。raw 内容は漏らさない。
        self.assertIn("! STT load failed: RuntimeError", output)
        self.assertNotIn("stt load failed", output)
        # STT がロード完了していないため release なし、Registry は一度だけ閉じられる。
        self.assertIsNone(pipeline._provider_registry)
        self.assertEqual(provider.close_calls, 1)
        self.assertTrue(pipeline._init_cleanup_done)
        pipeline.stt.cleanup.assert_not_called()
        pipeline.tts.cleanup.assert_not_called()

    def test_tts_load_failure_releases_loaded_stt_and_registry(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=True, has_model=True)
        pipeline = self._build(config, provider)
        pipeline.tts.load.side_effect = RuntimeError("tts load failed")

        ok, output = run_initialize(pipeline)

        self.assertFalse(ok)
        self.assertIn("TTS ロード失敗", output)
        # ロード済みの STT は release、Registry は一度だけ閉じられる。
        self.assertIsNone(pipeline._provider_registry)
        self.assertEqual(provider.close_calls, 1)
        self.assertTrue(pipeline._init_cleanup_done)
        pipeline.stt.cleanup.assert_called_once()
        pipeline.tts.cleanup.assert_not_called()

    def test_init_failure_cleanup_is_idempotent(self) -> None:
        config = ChatConfig()
        provider = StartupProvider(available=True, has_model=True)
        pipeline = self._build(config, provider)
        pipeline.tts.load.side_effect = RuntimeError("tts load failed")

        ok, _ = run_initialize(pipeline)
        self.assertFalse(ok)
        self.assertEqual(provider.close_calls, 1)

        # 再呼び出ししても二重解放しない。
        pipeline._cleanup_init_failure()
        pipeline._cleanup_init_failure()

        self.assertEqual(provider.close_calls, 1)
        pipeline.stt.cleanup.assert_called_once()
        pipeline.tts.cleanup.assert_not_called()

    def test_success_path_skips_failure_cleanup(self) -> None:
        config = ChatConfig(local_provider_kind="openai_compatible")
        provider = StartupProvider(available=True, models=[])
        pipeline = self._build(config, provider)

        ok, _ = run_initialize(pipeline)

        self.assertTrue(ok)
        self.assertFalse(pipeline._init_cleanup_done)
        self.assertIsNotNone(pipeline._provider_registry)
        self.assertEqual(provider.close_calls, 0)
        pipeline.stt.cleanup.assert_not_called()
        pipeline.tts.cleanup.assert_not_called()


class AudioMainVoiceModeStartupTest(unittest.TestCase):
    """run_voice_mode の初期化失敗時セーフティネット (cleanup → exit) の検証。"""

    def _voice_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            stt_model="small",
            tts_voice="jf_alpha",
            vad="auto",
            no_streaming_tts=False,
            no_rag=False,
            microphone=False,
            camera=False,
            no_vision=False,
            camera_id=0,
            screen=False,
            monitor=False,
            no_monitor=False,
            no_persona=False,
            wakeword=False,
            wakeword_model="hey_jarvis",
            wakeword_threshold=0.5,
        )

    def test_run_voice_mode_cleans_up_before_exit_on_failure(self) -> None:
        config = ChatConfig()
        pipeline = mock.Mock()
        pipeline.initialize.return_value = False

        with self.assertRaises(SystemExit) as raised:
            with (
                mock.patch("src.chat.config.ChatConfig.load", return_value=config),
                mock.patch(
                    "src.audio.pipeline.VoicePipeline", return_value=pipeline
                ),
                redirect_stdout(io.StringIO()),
            ):
                audio_main.run_voice_mode(self._voice_args())

        self.assertEqual(raised.exception.code, 1)
        # セーフティネット: initialize 失敗時は sys.exit 前に cleanup が呼ばれる。
        pipeline.cleanup.assert_called_once()

    def test_run_voice_mode_no_cleanup_on_success(self) -> None:
        config = ChatConfig()
        pipeline = mock.Mock()
        pipeline.initialize.return_value = True
        pipeline.run_interactive.return_value = None

        with (
            mock.patch("src.chat.config.ChatConfig.load", return_value=config),
            mock.patch(
                "src.audio.pipeline.VoicePipeline", return_value=pipeline
            ),
            redirect_stdout(io.StringIO()),
        ):
            audio_main.run_voice_mode(self._voice_args())

        pipeline.initialize.assert_called_once()
        pipeline.cleanup.assert_not_called()
        pipeline.run_interactive.assert_called_once()


class VoiceRunInteractiveCleanupTest(unittest.TestCase):
    """run_interactive 終了時クリーンアップ (P0-2.5) のオフライン検証。

    通常終了・KeyboardInterrupt・予期しない例外のいずれでも full cleanup が
    一度だけ走り、KeyboardInterrupt 時は従来どおりセッションを保存する。
    """

    def _build(self) -> tuple[VoicePipeline, StartupProvider]:
        config = ChatConfig()
        provider = StartupProvider(available=True, has_model=True)
        registry = ProviderRegistry()
        registry.register(config.resolved_local_provider_id(), provider, local=True)
        pipeline = build_pipeline(config, registry)
        # 停止対象のリソースを確実にモックで用意する
        pipeline.idle_manager = mock.Mock()
        pipeline.wakeword_detector = mock.Mock()
        pipeline.proactive = mock.Mock()
        pipeline.monitor_context = mock.Mock()
        pipeline.vision_context = mock.Mock()
        pipeline.screen_context = mock.Mock()
        pipeline.task_store = mock.Mock()
        pipeline.session.turn_count = 2
        pipeline.session.save.return_value = "data/history/voice.jsonl"
        return pipeline, provider

    def assert_all_resources_stopped_once(self, pipeline: VoicePipeline) -> None:
        pipeline.idle_manager.stop.assert_called_once()
        pipeline.wakeword_detector.cleanup.assert_called_once()
        pipeline.proactive.stop.assert_called_once()
        pipeline.monitor_context.stop.assert_called_once()
        pipeline.vision_context.stop.assert_called_once()
        pipeline.screen_context.stop.assert_called_once()
        pipeline.task_store.close.assert_called_once()

    def test_normal_exit_runs_full_cleanup_once(self) -> None:
        pipeline, provider = self._build()
        # 1ターン処理後にループを自然終了させる
        pipeline.process_voice_turn = mock.Mock(
            side_effect=lambda: setattr(pipeline, "_running", False)
        )

        with redirect_stdout(io.StringIO()):
            pipeline.run_interactive()

        self.assertFalse(pipeline._running)
        self.assertTrue(pipeline._cleanup_done)
        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)

    def test_keyboard_interrupt_saves_session_and_runs_cleanup(self) -> None:
        pipeline, provider = self._build()
        pipeline.process_voice_turn = mock.Mock(side_effect=KeyboardInterrupt)

        with redirect_stdout(io.StringIO()):
            pipeline.run_interactive()

        self.assertFalse(pipeline._running)
        self.assertTrue(pipeline._cleanup_done)
        pipeline.session.save.assert_called_once()
        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)

    def test_unexpected_error_propagates_and_runs_cleanup(self) -> None:
        pipeline, provider = self._build()
        pipeline.process_voice_turn = mock.Mock(side_effect=RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            with redirect_stdout(io.StringIO()):
                pipeline.run_interactive()

        self.assertTrue(pipeline._cleanup_done)
        # 例外経路ではセッション保存しない (KeyboardInterrupt のみ)。
        pipeline.session.save.assert_not_called()
        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)

    def test_keyboard_interrupt_without_turns_skips_session_save(self) -> None:
        pipeline, provider = self._build()
        pipeline.session.turn_count = 0
        pipeline.process_voice_turn = mock.Mock(side_effect=KeyboardInterrupt)

        with redirect_stdout(io.StringIO()):
            pipeline.run_interactive()

        pipeline.session.save.assert_not_called()
        self.assertTrue(pipeline._cleanup_done)
        self.assertEqual(provider.close_calls, 1)

    def test_cleanup_is_idempotent(self) -> None:
        pipeline, provider = self._build()

        pipeline.cleanup()
        pipeline.cleanup()

        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)

    def test_cleanup_after_run_interactive_is_at_most_once(self) -> None:
        pipeline, provider = self._build()
        pipeline.process_voice_turn = mock.Mock(
            side_effect=lambda: setattr(pipeline, "_running", False)
        )

        with redirect_stdout(io.StringIO()):
            pipeline.run_interactive()
        pipeline.cleanup()

        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)

    def test_cleanup_stop_failure_never_leaks_sensor_details(self) -> None:
        """センサー停止失敗は握りつぶされ、経路・デバイス等の内容を漏らさない。"""
        pipeline, provider = self._build()

        class CanaryError(RuntimeError):
            def __str__(self) -> str:
                return "/secret/monitor model-xyz http://secret hidden-content"

        canary = CanaryError()
        pipeline.monitor_context.stop.side_effect = canary
        pipeline.vision_context.stop.side_effect = canary
        pipeline.screen_context.stop.side_effect = canary

        out = io.StringIO()
        with redirect_stdout(out):
            pipeline.cleanup()

        self.assert_all_resources_stopped_once(pipeline)
        self.assertEqual(provider.close_calls, 1)
        # 停止失敗は無言で続行し、例外の中身は一切出ない。
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()