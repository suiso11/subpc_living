"""
音声対話パイプライン
Phase 3: VAD → STT → LLM (ローカルbackend) → TTS → 再生
改善: ストリーミングTTS (文単位で合成・再生)、Silero VAD対応
Phase 10: ウェイクワード検知モード追加
"""
import os
import sys
import re
import time
import threading
import queue
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from src.perception import ActivityRuntime

from src.audio.stt import WhisperSTT
from src.audio.tts_factory import backend_name, create_tts_backend
from src.audio.vad import EnergyVAD, create_vad
from src.audio.audio_io import AudioRecorder, AudioPlayer
from src.audio.wakeword import WakeWordDetector
from src.assistant.factory import build_local_service
from src.assistant.requests import create_request
from src.chat.session import ChatSession
from src.chat.config import ChatConfig, validate_local_provider_kind
from src.chat.emotion import (
    EmotionTagStreamFilter,
    emotion_to_sbv2_style,
    parse_emotion_tag,
)
from src.chat.web_search import WebSearchContext, create_web_search_context
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.screen.context import ScreenContext
from src.screen import create_screen_context
from src.monitor.context import MonitorContext
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.persona.proactive import ProactiveEngine
from src.service.idle import IdleManager, create_idle_manager
from src.growth.tracker import GrowthTracker


class MicrophoneInputError(RuntimeError):
    """マイク入力の開始・収集ループ失敗を表す例外。

    sounddevice 等の生例外をユーザーやログへ漏らさず、固定メッセージ
    (ASCII のみ) で伝える。
    """

    _FIXED_MESSAGE = "microphone input error"

    def __init__(self) -> None:
        super().__init__(self._FIXED_MESSAGE)


class VoicePipeline:
    """音声対話パイプライン"""

    # 状態定義
    STATE_IDLE = "idle"
    STATE_WAITING = "waiting"  # ウェイクワード待機中
    STATE_LISTENING = "listening"
    STATE_PROCESSING = "processing"
    STATE_SPEAKING = "speaking"

    def __init__(
        self,
        chat_config: Optional[ChatConfig] = None,
        stt_model: str = "small",
        tts_models_dir: str = "models/tts/kokoro",
        tts_voice: str = "jf_alpha",
        vad_type: str = "auto",
        streaming_tts: bool = True,
        enable_rag: bool = True,
        enable_vision: bool = False,
        camera_id: int = 0,
        enable_screen: bool = False,
        enable_monitor: bool = False,
        enable_persona: bool = True,
        enable_wakeword: bool = False,
        wakeword_models: Optional[list[str]] = None,
        wakeword_threshold: float = 0.5,
        activity_runtime: Optional["ActivityRuntime"] = None,
        voice_calendar_write_enabled: Optional[bool] = None,
    ):
        # チャット設定
        self.config = chat_config or ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")
        # ローカル推論 backend kind (Ollama または OpenAI互換)。Ollama 専用機能のゲートに使う。
        # 正規化した kind を保持し、空白のみの設定は一貫して Ollama として扱う。
        self.local_provider_kind = validate_local_provider_kind(self.config)

        # Companion 活動収集ランタイム (オプトイン、main 側で起動・停止を管理)
        self.activity_runtime = activity_runtime

        # STT (faster-whisper) — Phase 9: device="auto" でGPU自動検出
        self.stt = WhisperSTT(
            model_size=stt_model,
            language="ja",
        )

        # TTS
        self.tts = create_tts_backend(
            models_dir=PROJECT_ROOT / tts_models_dir,
            voice=tts_voice,
        )

        # VAD (auto: Silero優先、フォールバックでEnergy)
        self.vad_type = vad_type
        self.vad = create_vad(vad_type=vad_type, sample_rate=16000)

        # オーディオ I/O
        self.recorder = AudioRecorder(sample_rate=16000)
        self.player = AudioPlayer(sample_rate=24000)

        # LLM (選択されたローカルbackend: Ollama または OpenAI互換)
        self._assistant_service, self._provider_registry = build_local_service(self.config)
        self.llm = self._provider_registry.get(
            self.config.resolved_local_provider_id()
        ).provider
        self.web_search: Optional[WebSearchContext] = create_web_search_context(self.config)

        # RAG (Phase 4: 長期記憶)
        self.enable_rag = enable_rag
        self.rag = None
        if enable_rag:
            try:
                self.vector_store = VectorStore(
                    persist_dir=str(PROJECT_ROOT / "data" / "vectordb"),
                )
                self.rag = RAGRetriever(vector_store=self.vector_store)
            except Exception as e:
                print(f"⚠️  RAG初期化スキップ: {e}")
                self.rag = None

        # Vision (Phase 5: 映像入力)
        self.enable_vision = enable_vision
        self.vision_context: Optional[VisionContext] = None
        if enable_vision:
            try:
                emotion_model = str(PROJECT_ROOT / "models" / "vision" / "emotion-ferplus-8.onnx")
                self.vision_context = VisionContext(
                    camera_id=camera_id,
                    analysis_interval=2.0,
                    emotion_model_path=emotion_model,
                )
            except Exception as e:
                print(f"! Vision init skipped: {type(e).__name__}")
                self.vision_context = None

        # Screen (画面認識: スクリーンショット → VLM描写)
        # ScreenDescriber は Ollama /api/chat 前提のため Ollama backend 時のみ作成する。
        # OpenAI互換 backend では作成せず、Ollama URL への送信は試みない。
        self.enable_screen = enable_screen
        self.screen_context: Optional[ScreenContext] = None
        if enable_screen and self.local_provider_kind == "ollama":
            try:
                # SCREEN_CONTEXT_MODE (local|remote) でローカル/リモートを切替
                self.screen_context = create_screen_context(
                    analysis_interval=90.0,
                    base_url=self.config.ollama_base_url,
                    model=self.config.model,
                )
            except Exception as e:
                print(f"! Screen init skipped: {type(e).__name__}")
                self.screen_context = None
        elif enable_screen:
            print("! Screen (screen capture) is only available with Ollama backend, skipping")
            self.screen_context = None

        # Monitor (Phase 6: PCログ収集)
        # MonitorContext は backend 種別に依存しない汎用コンポーネントのため、
        # enable_monitor が真なら Ollama / OpenAI互換 どちらの backend でも作成する。
        # SensorPolicy による opt-in ゲートは enable_monitor が pipeline に到達する前に行われる。
        self.enable_monitor = enable_monitor
        self.monitor_context: Optional[MonitorContext] = None
        if enable_monitor:
            try:
                self.monitor_context = MonitorContext(
                    db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
                    collect_interval=30.0,
                )
            except Exception as e:
                print(f"! Monitor init skipped: {type(e).__name__}")
                self.monitor_context = None

        # Persona (Phase 7: パーソナライズ)
        self.enable_persona = enable_persona
        self.profile: Optional[UserProfile] = None
        self.summarizer: Optional[ConversationSummarizer] = None
        self.preloader: Optional[SessionPreloader] = None
        self.proactive: Optional[ProactiveEngine] = None
        if enable_persona:
            try:
                self.profile = UserProfile(
                    profile_path=str(PROJECT_ROOT / "data" / "profile" / "user_profile.json"),
                )
                self.profile.load()
                self.summarizer = ConversationSummarizer(
                    summaries_dir=str(PROJECT_ROOT / "data" / "profile" / "summaries"),
                )
                self.preloader = SessionPreloader(
                    profile=self.profile,
                    summarizer=self.summarizer,
                )
                self.proactive = ProactiveEngine(
                    profile=self.profile,
                    check_interval=60.0,
                    monitor_context=self.monitor_context,
                    companion_getter=self._companion_state,
                )
            except Exception as e:
                print(f"⚠️  Persona初期化スキップ: {e}")
                self.preloader = None
                self.proactive = None

        # Tasks (読み取り専用): 未完了タスクをLLMコンテキストに注入する
        self.task_store = None
        try:
            from src.tasks.store import TaskStore
            self.task_store = TaskStore(
                db_path=str(PROJECT_ROOT / "data" / "tasks" / "tasks.db"),
            ).initialize()
        except Exception as e:
            print(f"⚠️  Tasks初期化スキップ: {e}")
            self.task_store = None

        # Calendar (読み取り専用): Google Calendar の予定を upcoming.json から注入。
        # ワーカーは起動しない (Discord 側だけが取得・書き込みを行う)。
        self.calendar_context = None
        try:
            from src.tasks.calendar_sync import CalendarContext
            self.calendar_context = CalendarContext(
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
            )
        except Exception as e:
            print(f"⚠️  Calendar context 初期化スキップ: {type(e).__name__}")
            self.calendar_context = None

        # Calendar 書き込み (音声発話からの予定登録) はマイク同意とは別の独立 opt-in。
        # マイク同意 (--microphone / SENSOR_MICROPHONE_ENABLED=true) だけではカレンダーへ
        # 書き込まない。VOICE_CALENDAR_WRITE_ENABLED の明示 `true` のみ有効 (fail closed:
        # 未設定 / false / 不正値は無効)。解決はコンストラクタで一度だけ行い、テストでは
        # bool を直接注入できる。無効時は外部カレンダー経路を一切構築・呼び出さない。
        if voice_calendar_write_enabled is None:
            voice_calendar_write_enabled = (
                os.environ.get("VOICE_CALENDAR_WRITE_ENABLED", "").strip().lower() == "true"
            )
        self.voice_calendar_write_enabled = bool(voice_calendar_write_enabled)

        # Calendar 書き込み: 「予定入れて」発話の登録用。定期ワーカーは持たず、
        # 発話時に on-demand で MCP を呼ぶだけ (pull は従来通り Discord 側のみ)。
        # 外部カレンダークライアントの構築も voice opt-in が有効なときだけ行う。
        self.calendar_client = None
        self.tasks_calendar_id = (
            os.environ.get("TASKS_CALENDAR_ID", "").strip()
            or os.environ.get("DIARY_CALENDAR_ID", "").strip()
            or "primary"
        )
        self.tasks_timezone = os.environ.get("DIARY_TIMEZONE", "Asia/Tokyo").strip() or "Asia/Tokyo"
        if self.voice_calendar_write_enabled and os.environ.get(
            "TASKS_CALENDAR_SYNC_ENABLED", ""
        ).strip().lower() == "true":
            try:
                from src.integrations.google_calendar import GoogleCalendarMCPClient
                self.calendar_client = GoogleCalendarMCPClient.from_env()
            except Exception as e:
                print(f"⚠️  Calendar 書き込みクライアント初期化スキップ: {type(e).__name__}")
                self.calendar_client = None

        # 成長台帳（Web/Discord/CLIと同じSQLiteを共有）
        try:
            self.growth_tracker = GrowthTracker(
                PROJECT_ROOT / "data" / "growth" / "growth.db",
                timezone_name=os.environ.get("DIARY_TIMEZONE", "Asia/Tokyo").strip()
                or "Asia/Tokyo",
            )
        except Exception as e:
            print(f"⚠️  Growth tracker 初期化スキップ: {e}")
            self.growth_tracker = None

        # セッション
        self.session = ChatSession(
            system_prompt=self.config.effective_system_prompt(),
            max_history_turns=self.config.max_history_turns,
            history_dir=str(PROJECT_ROOT / self.config.history_dir),
            rag=self.rag,
            vision_context=self.vision_context,
            screen_context=self.screen_context,
            monitor_context=self.monitor_context,
            preloader=self.preloader,
            web_search=self.web_search,
            task_store=self.task_store,
            calendar_context=self.calendar_context,
            growth_tracker=self.growth_tracker,
            conversation_source="voice",
            emotion_tags=self.config.emotion_tag_enabled,
        )

        # ストリーミングTTS設定
        self.streaming_tts = streaming_tts
        self._tts_queue: queue.Queue = queue.Queue()

        # ウェイクワード (Phase 10)
        self.enable_wakeword = enable_wakeword
        self.wakeword_detector: Optional[WakeWordDetector] = None
        if enable_wakeword:
            self.wakeword_detector = WakeWordDetector(
                model_names=wakeword_models,
                threshold=wakeword_threshold,
            )

        # アイドル管理は複数プロセス間でGPU制限が競合するため、既定は無効。
        self.idle_manager: Optional[IdleManager] = create_idle_manager()

        # 状態
        self._state = self.STATE_IDLE
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()

        # 初期化失敗時クリーンアップ用の状態 (冪等)
        self._init_cleanup_done = False
        self._init_loaded: dict[str, bool] = {"stt": False, "tts": False}

        # 通常終了・例外時クリーンアップ用の状態 (冪等)
        self._cleanup_done = False

    # --- companion state getter (ProactiveEngine gate) ---
    def _companion_state(self):
        runtime = self.activity_runtime
        if runtime is None:
            return None
        try:
            return runtime.state
        except Exception:
            return None

    # --- 文分割ユーティリティ ---
    # 日本語の文末パターン: 。！？!? + 改行
    _SENTENCE_SPLIT_RE = re.compile(r'(?<=[。！？!?\n])')

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """テキストを文単位に分割する"""
        parts = VoicePipeline._SENTENCE_SPLIT_RE.split(text)
        return [p for p in parts if p.strip()]

    @property
    def state(self) -> str:
        return self._state

    def _best_effort_stop(self, component: Optional[object]) -> None:
        """start が False を返す・例外を投げたセンサーコンテキストをベストエフォートで停止する。

        部分起動したコンポーネントの stop を一度だけ呼び、失敗は握りつぶす。
        参照を None に置き換える前に呼び、解放漏れを防ぐ。
        """
        if component is None:
            return
        stop = getattr(component, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                pass

    def initialize(self) -> bool:
        """
        全コンポーネントを初期化。起動時に1回呼ぶ。

        Returns:
            成功したら True
        """
        total_steps = 10 if self.enable_wakeword else 9
        print("=" * 50)
        print(" 音声対話パイプライン 初期化")
        print("=" * 50)

        # ローカル推論サーバー接続チェック (backend別)
        print(f"\n[1/{total_steps}] ローカル推論サーバー 接続確認...")
        if not self._check_backend_availability():
            # 起動チェック失敗時は作成済みリソースを解放してから失敗させる。
            self._cleanup_init_failure()
            return False

        # STT モデルロード
        print(f"\n[2/{total_steps}] STT model load...")
        try:
            self.stt.load()
            self._init_loaded["stt"] = True
            print("STT OK")
        except Exception as e:
            print(f"! STT load failed: {type(e).__name__}")
            self._cleanup_init_failure()
            return False

        # TTS チェック
        print(f"\n[3/{total_steps}] TTS 確認...")
        try:
            self.tts.load()
            self._init_loaded["tts"] = True
            print(f"✅ TTS OK ({backend_name(self.tts)})")
        except Exception as e:
            # 例外本文にはモデルパスやprovider情報が含まれ得るため出さない。
            print(f"TTS ロード失敗: {type(e).__name__}")
            self._cleanup_init_failure()
            return False

        # VAD キャリブレーション (Energy VADの場合のみ環境ノイズ計測)
        # 診断は ASCII のみ (Windows cp932 コンソール移植性)。失敗時は固定メッセージと
        # 例外型名だけを出し、raw パス・デバイス・モデル・音声・エラー内容は漏らさない。
        print(f"\n[4/{total_steps}] VAD calibration...")
        try:
            if isinstance(self.vad, EnergyVAD):
                print("  measuring background noise (be quiet for 2 seconds)...")
                noise_sample = self.recorder.record(2.0)
                self.vad.calibrate(noise_sample)
            else:
                self.vad.calibrate(np.zeros(16000, dtype=np.float32))
            print("VAD OK")
        except Exception as e:
            print(f"! VAD calibration failed (using default threshold): {type(e).__name__}")

        # RAG (Phase 4)
        if self.enable_rag and self.rag is not None:
            print(f"\n[5/{total_steps}] RAG (長期記憶) 初期化...")
            try:
                self.vector_store.initialize()
                stats = self.rag.get_stats()
                print(f"✅ RAG OK (会話: {stats['conversations']}件, 知識: {stats['knowledge']}件)")
            except Exception as e:
                print(f"⚠️  RAG 初期化失敗 (RAGなしで続行): {e}")
                self.session.rag = None
        else:
            print(f"\n[5/{total_steps}] RAG (長期記憶) スキップ")

        # Vision (Phase 5)
        if self.enable_vision and self.vision_context is not None:
            print(f"\n[6/{total_steps}] Vision (video input) init...")
            try:
                if self.vision_context.start():
                    import time
                    time.sleep(1.0)  # カメラ安定待ち
                    status = self.vision_context.get_status()
                    emotion_str = "on" if status["emotion_detection"] else "face-only"
                    print(f"OK Vision (camera started, emotion: {emotion_str})")
                else:
                    print("! camera could not be opened (running without vision)")
                    self._best_effort_stop(self.vision_context)
                    self.session.vision_context = None
                    self.vision_context = None
            except Exception as e:
                print(f"! Vision init failed (running without vision): {type(e).__name__}")
                self._best_effort_stop(self.vision_context)
                self.session.vision_context = None
                self.vision_context = None
        else:
            print(f"\n[6/{total_steps}] Vision (video input) skipped")

        # Screen (画面認識: スクリーンショット → VLM描写)
        if self.enable_screen and self.screen_context is not None:
            print("\n[6+] Screen (screen capture) init...")
            try:
                if self.screen_context.start():
                    status = self.screen_context.get_status()
                    print(f"OK Screen (VLM: {status['model']}, analysis interval: {status['analysis_interval']:.0f}s)")
                else:
                    print("! screen could not be captured (DISPLAY not set? running without screen)")
                    self._best_effort_stop(self.screen_context)
                    self.session.screen_context = None
                    self.screen_context = None
            except Exception as e:
                print(f"! Screen init failed (running without screen): {type(e).__name__}")
                self._best_effort_stop(self.screen_context)
                self.session.screen_context = None
                self.screen_context = None
        elif self.enable_screen:
            print("\n[6+] Screen (screen capture) skipped")

        # Monitor (Phase 6)
        if self.enable_monitor and self.monitor_context is not None:
            print(f"\n[7/{total_steps}] Monitor (PC log collection) init...")
            try:
                if self.monitor_context.start():
                    print("OK Monitor (metric collection started)")
                else:
                    print("! Monitor start failed (running without monitor)")
                    self._best_effort_stop(self.monitor_context)
                    self.session.monitor_context = None
                    self.monitor_context = None
            except Exception as e:
                print(f"! Monitor init failed (running without monitor): {type(e).__name__}")
                self._best_effort_stop(self.monitor_context)
                self.session.monitor_context = None
                self.monitor_context = None
        else:
            print(f"\n[7/{total_steps}] Monitor (PC log collection) skipped")

        # Persona (Phase 7)
        if self.enable_persona and self.preloader is not None:
            print(f"\n[8/{total_steps}] Persona (パーソナライズ) 初期化...")
            try:
                profile_name = self.profile.name or "(未設定)"
                facts_count = len(self.profile.extracted_facts)
                today_count = len(self.profile.get_today_schedule())
                print(f"✅ Persona OK (名前: {profile_name}, 抽出済み事実: {facts_count}件, 今日の予定: {today_count}件)")
            except Exception as e:
                print(f"⚠️  Persona 初期化失敗 (Personaなしで続行): {e}")
                self.session.preloader = None
                self.preloader = None
        else:
            print(f"\n[8/{total_steps}] Persona (パーソナライズ) スキップ")

        # Proactive (Phase 7)
        if self.enable_persona and self.proactive is not None:
            print(f"\n[9/{total_steps}] Proactive (プロアクティブ発話) 初期化...")
            try:
                self.proactive.start(callback=self._on_proactive_trigger)
                print("✅ Proactive OK (バックグラウンド監視開始)")
            except Exception as e:
                print(f"⚠️  Proactive 初期化失敗 (Proactiveなしで続行): {type(e).__name__}")
                self.proactive = None
        else:
            print(f"\n[9/{total_steps}] Proactive (プロアクティブ発話) スキップ")

        # ウェイクワード (Phase 10)
        if self.enable_wakeword and self.wakeword_detector is not None:
            print(f"\n[10/{total_steps}] WakeWord (ウェイクワード検知) 初期化...")
            try:
                if self.wakeword_detector.load():
                    # モデル名・パスは設定情報なので起動診断へ出さない。
                    print("WakeWord OK")
                else:
                    print("WakeWord load failed; continuing without wakeword")
                    self.wakeword_detector = None
                    self.enable_wakeword = False
            except Exception as e:
                print(f"WakeWord init failed: {type(e).__name__}")
                self.wakeword_detector = None
                self.enable_wakeword = False
        elif self.enable_wakeword:
            print(f"\n[10/{total_steps}] WakeWord (ウェイクワード検知) スキップ")

        # IdleManager 起動 (明示的に opt-in した場合のみ)
        if self.idle_manager is not None:
            print(f"\n[{total_steps + 1}/{total_steps + 1}] IdleManager (アイドル電力管理) 初期化...")
            try:
                self.idle_manager.start(
                    monitor_context=self.monitor_context,
                    vision_context=self.vision_context,
                )
                if self.idle_manager.gpu_power_control_enabled:
                    print("✅ IdleManager OK (GPU電力の動的切替有効)")
                else:
                    print(f"✅ IdleManager OK (GPU電力制御は無効: {self.idle_manager.gpu_power_control_reason})")
            except Exception as e:
                print(f"⚠️  IdleManager 初期化失敗 (続行): {e}")
        else:
            print("\nℹ️  IdleManager 無効 (IDLE_MANAGER_ENABLED=true で明示的に有効化)")

        print("\n" + "=" * 50)
        print(" ✅ 初期化完了！")
        print("=" * 50)
        return True

    def _check_backend_availability(self) -> bool:
        """ローカルbackendの起動チェック (kind別の厳格さ)。

        Ollama: ``is_available()`` と ``has_model()`` で厳格に判定し、両方成功して
        初めて「接続OK」を表示する。openai_compatible は ``is_available()`` が
        ライフサイクルのみで接続成功を意味しないため、それだけで「接続OK」と
        表示・成功判定しない。``list_models()`` を1回だけ呼び、非空で設定モデルが
        含まれないときだけ失敗し、空 (未実装) なら生成時に検証する警告で続行する。
        """
        if self.local_provider_kind == "ollama":
            if not self.llm.is_available():
                print("❌ ローカル推論サーバーに接続できません")
                return False
            print("✅ 接続OK")
            if not self.llm.has_model():
                print(f"❌ モデル '{self.config.model}' が見つかりません")
                return False
            print("✅ モデル確認OK")
            return True

        discovered = self.llm.list_models()
        if discovered and self.config.model not in discovered:
            print(f"❌ モデル '{self.config.model}' が見つかりません")
            print(f"  利用可能なモデル: {', '.join(discovered)}")
            return False
        if not discovered:
            print("⚠️  モデル情報の取得に失敗しました。生成時に確認します。")
        else:
            print("✅ モデル確認OK")
        return True

    def _try_register_event(self, text: str) -> Optional[str]:
        """予定登録の意図があれば Google Calendar に登録して結果文を返す (無ければ None)。"""
        try:
            from src.tasks.event_intent import try_register_event

            return try_register_event(
                text,
                client=self.calendar_client,
                calendar_id=self.tasks_calendar_id,
                timezone_name=self.tasks_timezone,
                upcoming_path=str(PROJECT_ROOT / "data" / "calendar" / "upcoming.json"),
            )
        except Exception as e:
            print(f"⚠️  予定登録エラー: {type(e).__name__}")
            return None

    def _on_proactive_trigger(self, trigger_type: str, message: str) -> None:
        """Proactiveエンジンからのコールバック: AI発話をTTSで再生"""
        if not self._running:
            return
        if self._state != self.STATE_IDLE:
            return  # 会話中は割り込まない
        try:
            wav_data = self.tts.synthesize(message)
            self.player.play_wav(wav_data, blocking=True)
        except Exception as e:
            print(f"\n⚠️  Proactive 発話失敗: {type(e).__name__}")

    def _summarize_session(self) -> None:
        """セッション終了時に会話を要約・知識抽出 (Phase 7)。"""
        if self.summarizer is None or self.session.turn_count < 2:
            return
        try:
            result = self.summarizer.process_session_end(
                llm=self.llm,
                messages=self.session.messages,
                session_id=self.session.session_id,
                profile=self.profile,
            )
            # 要約本文・抽出事実は診断へ出さず、存在と件数だけを出す。
            summary_present = bool(result.get("summary"))
            facts = result.get("extracted_facts") or []
            print(
                f"session summary: present={summary_present}, "
                f"extracted_facts={len(facts)}"
            )
        except Exception as e:
            print(f"session summary failed: {type(e).__name__}")

    def process_voice_turn(self) -> Optional[str]:
        """
        1ターンの音声対話を処理する:
        録音 → STT → LLM → TTS → 再生

        streaming_tts=True の場合、LLMのストリーミング応答を文単位で
        逐次TTS合成・再生する（全文完成を待たない）。

        Returns:
            AIの応答テキスト。エラー時は None。
        """
        # --- リスニング ---
        self._state = self.STATE_LISTENING
        print("\n🎤 聞いています... (話し終わったら自動検出します)")

        speech_audio = self._listen_for_speech()
        if speech_audio is None or len(speech_audio) < self.vad.sample_rate * 0.3:
            return None

        # --- STT ---
        self._state = self.STATE_PROCESSING
        print("\nSTT transcribing...")
        if self.idle_manager is not None:
            self.idle_manager.notify_inference_start(wait_for_gpu=True)
        try:
            user_text = self.stt.transcribe(speech_audio)
        except Exception as e:
            print(f"! STT transcription failed: {type(e).__name__}")
            if self.idle_manager is not None:
                self.idle_manager.notify_inference_end()
            self._state = self.STATE_IDLE
            return None

        if not user_text:
            print("  (音声を認識できませんでした)")
            if self.idle_manager is not None:
                self.idle_manager.notify_inference_end()
            return None

        # 認識テキストはセッション/LLM経路のみに渡し、既定のパイプライン診断には出さない。

        # --- 予定登録 (「予定入れて」等) は LLM を介さず Google Calendar へ ---
        # 独立 opt-in (VOICE_CALENDAR_WRITE_ENABLED=true) が無い限り呼ばない
        # (fail closed)。マイク同意だけではカレンダー書き込みを許可しない。
        # 無効 / 未設定 / 不正値のときは外部カレンダー経路へ到達せず、そのまま
        # ローカル LLM / セッションの通常経路へフォールスルーする。
        if getattr(self, "voice_calendar_write_enabled", False):
            event_reply = self._try_register_event(user_text)
            if event_reply is not None:
                self.session.add_user_message(user_text)
                # LLMを介さないプライバシー安全な音声ターンと同様、RAG長期記憶と
                # 成長台帳の記録をスキップしてコミットする。
                self.session.add_assistant_message(
                    event_reply, store_memory=False, record_growth=False
                )
                try:
                    wav_data = self.tts.synthesize(event_reply)
                    self.player.play_wav(wav_data, blocking=True)
                except Exception as e:
                    print(f"⚠️  応答再生失敗: {type(e).__name__}")
                if self.idle_manager is not None:
                    self.idle_manager.notify_inference_end()
                return event_reply

        # --- LLM → TTS (ストリーミング) ---
        print("\n🤖 考え中...")
        self.session.add_user_message(user_text)
        blocks = self.session.build_blocks()

        try:
            if self.streaming_tts:
                response_text = self._stream_llm_with_tts(blocks, user_text)
            else:
                response_text = self._sequential_llm_then_tts(blocks, user_text)

            if not response_text:
                return None

            self.session.add_assistant_message(response_text)

        except Exception as e:
            print(f"\n❌ LLM エラー: {type(e).__name__}")
            self.session.rollback_last_user_message()
            return None
        finally:
            # 応答が空だった経路と例外経路でも待機状態へ戻す。ここを抜けたまま
            # STATE_PROCESSING が残ると、次のターンを受け付けなくなる。
            self._state = self.STATE_IDLE
            if self.idle_manager is not None:
                self.idle_manager.notify_inference_end()

        return response_text

    def _stream_llm_with_tts(self, blocks, user_text: str) -> str:
        """
        LLMストリーミング応答を文単位でTTS合成・再生する

        LLMがトークンを生成する間、文の区切りを検出して
        完成した文から順にTTSキューに投入・再生する。
        """
        response_text = ""
        sentence_buffer = ""
        tts_thread = None
        played_sentences: list[str] = []
        abort_response = False
        fallback_text = "出力が乱れたので止めます。"

        # 感情タグフィルタ (有効時のみ)。冒頭タグを除去しつつ感情を確定する。
        emo_filter = (
            EmotionTagStreamFilter() if self.config.emotion_tag_enabled else None
        )

        def current_style() -> str | None:
            if emo_filter is None:
                return None
            return emotion_to_sbv2_style(emo_filter.emotion)

        # TTS再生ワーカースレッド
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()

        self._state = self.STATE_PROCESSING
        request = create_request(
            text=user_text,
            conversation_id=getattr(self.session, "session_id", "voice") or "voice",
            channel="voice",
            profile="voice_fast",
            privacy="local_only",
        )
        stream = None
        try:
            stream = self._assistant_service.respond_stream(request, blocks, base_system=self.session.system_prompt)
            for token in stream:
                piece = emo_filter.feed(token) if emo_filter is not None else token
                if not piece:
                    continue
                response_text += piece
                sentence_buffer += piece

                # 文の区切りをチェック
                sentences = self._split_sentences(sentence_buffer)
                if len(sentences) > 1:
                    # 最後の要素はまだ不完全な可能性があるので保持
                    for sent in sentences[:-1]:
                        sent = sent.strip()
                        if sent:
                            if self._is_repeated_tts_sentence(sent, played_sentences):
                                response_text = fallback_text
                                sentence_buffer = ""
                                self._tts_queue.put((fallback_text, current_style()))
                                abort_response = True
                                break
                            self._tts_queue.put((sent, current_style()))
                            played_sentences.append(sent)
                    if abort_response:
                        break
                    sentence_buffer = sentences[-1]

            # 残りのバッファも送信
            if not abort_response and sentence_buffer.strip():
                self._tts_queue.put((sentence_buffer.strip(), current_style()))
                played_sentences.append(sentence_buffer.strip())

        finally:
            # LLMストリームを明示的に閉じる。正常終了・繰り返しアボート・生成例外・
            # respond_stream失敗のすべての経路で必ず解放する。closeの失敗は主経路の
            # 例外 (生成例外など) を上書きしないよう握りつぶして続行する。
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            # TTS完了を待機
            self._tts_queue.put(None)  # 終了シグナル
            if tts_thread:
                tts_thread.join(timeout=60)

        return response_text

    @staticmethod
    def _is_repeated_tts_sentence(sentence: str, previous_sentences: list[str]) -> bool:
        normalized = re.sub(r"\s+", "", sentence)
        if not normalized:
            return False

        punct_chars = {"。", "…", ".", "、"}
        is_punct_only = len(normalized) <= 8 and set(normalized) <= punct_chars
        recent = [re.sub(r"\s+", "", sent) for sent in previous_sentences[-5:]]

        # 「……。」「...」だけの沈黙文が続く時は、2回読んだ時点で止める。
        if is_punct_only:
            recent_punct = [sent for sent in recent if sent and set(sent) <= punct_chars]
            if len(recent_punct) >= 2:
                return True

        if len(previous_sentences) < 5:
            return False

        if all(sent == normalized for sent in recent):
            return True

        return is_punct_only and all(set(sent) <= punct_chars for sent in recent)

    def _tts_worker(self) -> None:
        """TTS合成・再生のワーカースレッド"""
        while True:
            item = self._tts_queue.get()
            if item is None:
                break
            text, style = item

            self._state = self.STATE_SPEAKING
            try:
                wav_data = self.tts.synthesize(text, style=style)
                self.player.play_wav(wav_data, blocking=True)
            except Exception as e:
                print(f"\nTTS playback failed: {type(e).__name__}")

    def _sequential_llm_then_tts(self, blocks, user_text: str) -> str:
        """従来のシーケンシャル方式: LLM全文完了後にTTS"""
        response_text = ""
        request = create_request(
            text=user_text,
            conversation_id=getattr(self.session, "session_id", "voice") or "voice",
            channel="voice",
            profile="voice_fast",
            privacy="local_only",
        )
        stream = None
        try:
            stream = self._assistant_service.respond_stream(
                request, blocks, base_system=self.session.system_prompt
            )
            for token in stream:
                response_text += token
        finally:
            # LLMストリームを明示的に閉じる。正常終了・空応答・生成例外のすべての
            # 経路で必ず1回だけ解放する (StreamResult.close は冪等)。
            # closeの失敗は主経路の例外 (生成例外など) を上書きしないよう握りつぶす。
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        if not response_text:
            return ""

        # 感情タグを分離 (有効時のみ)。履歴・返り値には clean text を使う。
        if self.config.emotion_tag_enabled:
            emotion, response_text = parse_emotion_tag(response_text)
            style = emotion_to_sbv2_style(emotion)
        else:
            style = None

        if not response_text:
            return ""

        # --- TTS & 再生 ---
        self._state = self.STATE_SPEAKING
        print("\n🔊 読み上げ中...")
        try:
            wav_data = self.tts.synthesize(response_text, style=style)
            self.player.play_wav(wav_data, blocking=True)
        except Exception as e:
            print(f"TTS/playback failed: {type(e).__name__}")

        return response_text

    def _listen_for_speech(self, max_duration: float = 30.0) -> Optional[np.ndarray]:
        """
        VADを使って発話区間を検出し、音声データを返す

        Args:
            max_duration: 最大録音時間 (秒)

        Returns:
            検出された発話の音声データ。タイムアウト時は None。
        """
        self.vad.reset()
        result_audio = None
        start_time = time.time()

        def audio_callback(indata, frames, time_info, status):
            nonlocal result_audio
            if status:
                pass  # オーバーフロー等は無視
            frame = indata[:, 0].copy()  # モノラルに
            speech = self.vad.process_frame(frame)
            if speech is not None:
                self._audio_queue.put(speech)

        try:
            stream = self.recorder.open_stream(
                callback=audio_callback,
                frame_size=self.vad.frame_size,
            )

            with stream:
                while True:
                    try:
                        # 発話検出待ち
                        speech = self._audio_queue.get(timeout=1.0)
                        return speech
                    except queue.Empty:
                        elapsed = time.time() - start_time
                        if elapsed > max_duration:
                            print("  (タイムアウト)")
                            return None
                        if self.vad.is_speaking:
                            # 発話中のインジケータ
                            print(".", end="", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception:
            raise MicrophoneInputError from None

    def _wait_for_wakeword(self) -> Optional[str]:
        """
        ウェイクワードが検知されるまでマイクを監視する

        Returns:
            検知されたウェイクワード名。中断時は None。
        """
        if self.wakeword_detector is None:
            return None

        self._state = self.STATE_WAITING
        self.wakeword_detector.reset()
        detected_word = None

        # ウェイクワードのフレームサイズ (80ms @ 16kHz = 1280 samples)
        ww_frame_size = self.wakeword_detector.frame_size

        def audio_callback(indata, frames, time_info, status):
            nonlocal detected_word
            if status:
                pass
            if detected_word is not None:
                return  # 既に検知済み
            frame = indata[:, 0].copy()
            result = self.wakeword_detector.process_frame(frame)
            if result is not None:
                detected_word = result

        try:
            stream = self.recorder.open_stream(
                callback=audio_callback,
                frame_size=ww_frame_size,
            )

            with stream:
                while self._running and detected_word is None:
                    time.sleep(0.05)  # 50ms ポーリング
        except KeyboardInterrupt:
            raise
        except Exception:
            raise MicrophoneInputError from None

        return detected_word

    def run_interactive(self) -> None:
        """インタラクティブ音声対話ループ"""
        self._running = True
        print("\n" + "=" * 50)

        if self.enable_wakeword and self.wakeword_detector is not None:
            print(" wakeword mode enabled")
        else:
            print(" 🎙️  音声対話モード")
        print("  Ctrl+C で終了")
        print("=" * 50)

        try:
            while self._running:
                if self.enable_wakeword and self.wakeword_detector is not None:
                    # ウェイクワードモード: 検知まで待機
                    print("\n👂 ウェイクワード待機中...")
                    detected = self._wait_for_wakeword()
                    if detected is None:
                        continue
                    print("\nwakeword detected")

                # 対話ターンを処理
                self.process_voice_turn()

                # Proactive: ユーザーアクティビティ通知
                if self.proactive is not None:
                    self.proactive.notify_user_activity()
        except KeyboardInterrupt:
            print("\n\n終了します...")
            self._running = False
            # Phase 7: セッション要約
            self._summarize_session()
            if self.session.turn_count > 0:
                try:
                    saved = self.session.save()
                    print(f"会話を保存しました: {saved}")
                except Exception as e:
                    print(f"会話保存失敗: {type(e).__name__}")
        except MicrophoneInputError:
            print(f"\n\n{MicrophoneInputError._FIXED_MESSAGE}")
            self._running = False
        finally:
            # 通常終了・KeyboardInterrupt・予期しない例外のいずれでも、
            # 全リソースを確実に解放する (冪等なため二重実行されない)。
            self.cleanup()

    def _close_provider_registry(self) -> None:
        """Provider Registryを一度だけ終了する。"""
        registry = self._provider_registry
        self._provider_registry = None
        if registry is not None:
            registry.close()

    def _cleanup_init_failure(self) -> None:
        """初期化失敗時に作成済みリソースを一度だけ解放する (冪等)。

        initialize() の致命的な失敗経路 (backend/STT/TTS ロード失敗、および
        将来の fatal return) から呼ばれる。何度呼んでも Registry の close と
        ロード済みコンポーネントの release は一度だけ実行する。
        成功時の初期化経路では呼ばれない。
        """
        if self._init_cleanup_done:
            return
        self._init_cleanup_done = True
        self._close_provider_registry()
        for name in ("stt", "tts"):
            if not self._init_loaded.get(name):
                continue
            component = getattr(self, name, None)
            if component is None:
                continue
            release = getattr(component, "cleanup", None)
            if release is None:
                release = getattr(component, "close", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

    def cleanup(self) -> None:
        """リソースの解放 (冪等)。複数回呼んでも一度だけ実行する。"""
        if self._cleanup_done:
            return
        self._cleanup_done = True
        self._running = False
        # 各コンポーネントの解放は独立に行う。停止中の例外が主経路の例外
        # (KeyboardInterrupt 等) を上書きしないよう握りつぶして続行する。
        for attr, method in (
            ("idle_manager", "stop"),
            ("wakeword_detector", "cleanup"),
            ("proactive", "stop"),
            ("monitor_context", "stop"),
            ("vision_context", "stop"),
            ("screen_context", "stop"),
            ("task_store", "close"),
        ):
            component = getattr(self, attr, None)
            if component is None:
                continue
            release = getattr(component, method, None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass
        try:
            self._close_provider_registry()
        except Exception:
            pass
