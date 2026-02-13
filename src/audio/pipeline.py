"""
音声対話パイプライン
Phase 3: VAD → STT → LLM (Ollama) → TTS → 再生
改善: ストリーミングTTS (文単位で合成・再生)、Silero VAD対応
Phase 10: ウェイクワード検知モード追加
"""
import sys
import re
import time
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audio.stt import WhisperSTT
from src.audio.tts import KokoroTTS
from src.audio.vad import EnergyVAD, create_vad
from src.audio.audio_io import AudioRecorder, AudioPlayer
from src.audio.wakeword import WakeWordDetector
from src.chat.client import OllamaClient
from src.chat.session import ChatSession
from src.chat.config import ChatConfig
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext
from src.monitor.context import MonitorContext
from src.persona.profile import UserProfile
from src.persona.summarizer import ConversationSummarizer
from src.persona.preloader import SessionPreloader
from src.persona.proactive import ProactiveEngine


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
        enable_vision: bool = True,
        camera_id: int = 0,
        enable_monitor: bool = True,
        enable_persona: bool = True,
        enable_wakeword: bool = False,
        wakeword_models: Optional[list[str]] = None,
        wakeword_threshold: float = 0.5,
    ):
        # チャット設定
        self.config = chat_config or ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")

        # STT (faster-whisper) — Phase 9: device="auto" でGPU自動検出
        self.stt = WhisperSTT(
            model_size=stt_model if stt_model != "small" else "auto",
            language="ja",
        )

        # TTS (kokoro-onnx)
        self.tts = KokoroTTS(
            models_dir=PROJECT_ROOT / tts_models_dir,
            voice=tts_voice,
        )

        # VAD (auto: Silero優先、フォールバックでEnergy)
        self.vad_type = vad_type
        self.vad = create_vad(vad_type=vad_type, sample_rate=16000)

        # オーディオ I/O
        self.recorder = AudioRecorder(sample_rate=16000)
        self.player = AudioPlayer(sample_rate=24000)

        # LLM
        self.llm = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.model,
        )

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
                print(f"⚠️  Vision初期化スキップ: {e}")
                self.vision_context = None

        # Monitor (Phase 6: PCログ収集)
        self.enable_monitor = enable_monitor
        self.monitor_context: Optional[MonitorContext] = None
        if enable_monitor:
            try:
                self.monitor_context = MonitorContext(
                    db_path=str(PROJECT_ROOT / "data" / "metrics" / "system_metrics.db"),
                    collect_interval=30.0,
                )
            except Exception as e:
                print(f"⚠️  Monitor初期化スキップ: {e}")
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
                )
            except Exception as e:
                print(f"⚠️  Persona初期化スキップ: {e}")
                self.preloader = None
                self.proactive = None

        # セッション
        self.session = ChatSession(
            system_prompt=self.config.system_prompt,
            max_history_turns=self.config.max_history_turns,
            history_dir=str(PROJECT_ROOT / self.config.history_dir),
            rag=self.rag,
            vision_context=self.vision_context,
            monitor_context=self.monitor_context,
            preloader=self.preloader,
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

        # 状態
        self._state = self.STATE_IDLE
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()

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

        # Ollama 接続チェック
        print(f"\n[1/{total_steps}] Ollama 接続確認...")
        if not self.llm.is_available():
            print("❌ Ollamaに接続できません")
            return False
        print("✅ Ollama OK")

        # STT モデルロード
        print(f"\n[2/{total_steps}] STT モデルロード...")
        try:
            self.stt.load()
            print("✅ STT OK")
        except Exception as e:
            print(f"❌ STT ロード失敗: {e}")
            return False

        # TTS チェック
        print(f"\n[3/{total_steps}] TTS 確認...")
        try:
            self.tts.load()
            print("✅ TTS OK")
        except Exception as e:
            print(f"❌ TTS ロード失敗: {e}")
            return False

        # VAD キャリブレーション (Energy VADの場合のみ環境ノイズ計測)
        print(f"\n[4/{total_steps}] VAD キャリブレーション...")
        vad_name = type(self.vad).__name__
        print(f"  VAD方式: {vad_name}")
        try:
            if isinstance(self.vad, EnergyVAD):
                print("  環境ノイズを計測中 (2秒間、静かにしてください)...")
                noise_sample = self.recorder.record(2.0)
                self.vad.calibrate(noise_sample)
            else:
                self.vad.calibrate(np.zeros(16000, dtype=np.float32))
            print("✅ VAD OK")
        except Exception as e:
            print(f"⚠️  VAD キャリブレーション失敗 (デフォルト閾値を使用): {e}")

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
            print(f"\n[6/{total_steps}] Vision (映像入力) 初期化...")
            try:
                if self.vision_context.start():
                    import time
                    time.sleep(1.0)  # カメラ安定待ち
                    status = self.vision_context.get_status()
                    emotion_str = "有効" if status["emotion_detection"] else "顔検出のみ"
                    print(f"✅ Vision OK (カメラ起動, 感情推定: {emotion_str})")
                else:
                    print("⚠️  カメラを開けません (Visionなしで続行)")
                    self.session.vision_context = None
                    self.vision_context = None
            except Exception as e:
                print(f"⚠️  Vision 初期化失敗 (Visionなしで続行): {e}")
                self.session.vision_context = None
                self.vision_context = None
        else:
            print(f"\n[6/{total_steps}] Vision (映像入力) スキップ")

        # Monitor (Phase 6)
        if self.enable_monitor and self.monitor_context is not None:
            print(f"\n[7/{total_steps}] Monitor (PCログ収集) 初期化...")
            try:
                if self.monitor_context.start():
                    print("✅ Monitor OK (メトリクス収集開始)")
                else:
                    print("⚠️  Monitor 起動失敗 (Monitorなしで続行)")
                    self.session.monitor_context = None
                    self.monitor_context = None
            except Exception as e:
                print(f"⚠️  Monitor 初期化失敗 (Monitorなしで続行): {e}")
                self.session.monitor_context = None
                self.monitor_context = None
        else:
            print(f"\n[7/{total_steps}] Monitor (PCログ収集) スキップ")

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
                print(f"⚠️  Proactive 初期化失敗 (Proactiveなしで続行): {e}")
                self.proactive = None
        else:
            print(f"\n[9/{total_steps}] Proactive (プロアクティブ発話) スキップ")

        # ウェイクワード (Phase 10)
        if self.enable_wakeword and self.wakeword_detector is not None:
            print(f"\n[10/{total_steps}] WakeWord (ウェイクワード検知) 初期化...")
            try:
                if self.wakeword_detector.load():
                    models = self.wakeword_detector.loaded_models
                    print(f"✅ WakeWord OK (モデル: {', '.join(models)}, 閾値: {self.wakeword_detector.threshold})")
                else:
                    print("⚠️  WakeWord ロード失敗 (ウェイクワードなしで続行)")
                    self.wakeword_detector = None
                    self.enable_wakeword = False
            except Exception as e:
                print(f"⚠️  WakeWord 初期化失敗 (ウェイクワードなしで続行): {e}")
                self.wakeword_detector = None
                self.enable_wakeword = False
        elif self.enable_wakeword:
            print(f"\n[10/{total_steps}] WakeWord (ウェイクワード検知) スキップ")

        print("\n" + "=" * 50)
        print(" ✅ 初期化完了！")
        print("=" * 50)
        return True

    def _on_proactive_trigger(self, trigger_type: str, message: str) -> None:
        """Proactiveエンジンからのコールバック: AI発話をTTSで再生"""
        if not self._running:
            return
        if self._state != self.STATE_IDLE:
            return  # 会話中は割り込まない
        try:
            print(f"\n\U0001f4ac [Proactive/{trigger_type}] {message}")
            wav_data = self.tts.synthesize(message)
            self.player.play_wav(wav_data, blocking=True)
        except Exception as e:
            print(f"\n⚠️  Proactive 発話失敗: {e}")

    def _summarize_session(self) -> None:
        """セッション終了時に会話を要約・知識抽出 (Phase 7)"""
        if self.summarizer is None or self.session.turn_count < 2:
            return
        try:
            print("ℹ️  会話を要約中...")
            result = self.summarizer.process_session_end(
                llm=self.llm,
                messages=self.session.messages,
                session_id=self.session.session_id,
                profile=self.profile,
            )
            if result["summary"]:
                print(f"  要約: {result['summary'][:80]}...")
            if result["extracted_facts"]:
                print(f"  抽出した事実: {len(result['extracted_facts'])}件")
        except Exception as e:
            print(f"⚠️  会話要約失敗: {e}")

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
        print("\n🔄 音声認識中...")
        user_text = self.stt.transcribe(speech_audio)

        if not user_text:
            print("  (音声を認識できませんでした)")
            return None

        print(f"\n👤 あなた: {user_text}")

        # --- LLM → TTS (ストリーミング) ---
        print("\n🤖 考え中...")
        self.session.add_user_message(user_text)
        messages = self.session.build_messages()

        try:
            if self.streaming_tts:
                response_text = self._stream_llm_with_tts(messages)
            else:
                response_text = self._sequential_llm_then_tts(messages)

            if not response_text:
                return None

            self.session.add_assistant_message(response_text)

        except Exception as e:
            print(f"\n❌ LLM エラー: {e}")
            if self.session._messages and self.session._messages[-1]["role"] == "user":
                self.session._messages.pop()
            return None

        self._state = self.STATE_IDLE
        return response_text

    def _stream_llm_with_tts(self, messages: list[dict]) -> str:
        """
        LLMストリーミング応答を文単位でTTS合成・再生する

        LLMがトークンを生成する間、文の区切りを検出して
        完成した文から順にTTSキューに投入・再生する。
        """
        response_text = ""
        sentence_buffer = ""
        tts_thread = None
        played_sentences: list[str] = []

        # TTS再生ワーカースレッド
        self._tts_stop = False
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()

        self._state = self.STATE_PROCESSING
        try:
            for token in self.llm.generate_stream(
                messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                num_ctx=self.config.num_ctx,
                repeat_penalty=self.config.repeat_penalty,
            ):
                response_text += token
                sentence_buffer += token
                print(token, end="", flush=True)

                # 文の区切りをチェック
                sentences = self._split_sentences(sentence_buffer)
                if len(sentences) > 1:
                    # 最後の要素はまだ不完全な可能性があるので保持
                    for sent in sentences[:-1]:
                        sent = sent.strip()
                        if sent:
                            self._tts_queue.put(sent)
                            played_sentences.append(sent)
                    sentence_buffer = sentences[-1]

            # 残りのバッファも送信
            if sentence_buffer.strip():
                self._tts_queue.put(sentence_buffer.strip())
                played_sentences.append(sentence_buffer.strip())

            print()  # 改行

        finally:
            # TTS完了を待機
            self._tts_queue.put(None)  # 終了シグナル
            if tts_thread:
                tts_thread.join(timeout=60)

        return response_text

    def _tts_worker(self) -> None:
        """TTS合成・再生のワーカースレッド"""
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            if self._tts_stop:
                break

            self._state = self.STATE_SPEAKING
            try:
                wav_data = self.tts.synthesize(text)
                self.player.play_wav(wav_data, blocking=True)
            except Exception as e:
                print(f"\n⚠️  TTS再生エラー: {e}")

    def _sequential_llm_then_tts(self, messages: list[dict]) -> str:
        """従来のシーケンシャル方式: LLM全文完了後にTTS"""
        response_text = ""
        for token in self.llm.generate_stream(
            messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            num_ctx=self.config.num_ctx,
            repeat_penalty=self.config.repeat_penalty,
        ):
            response_text += token
            print(token, end="", flush=True)
        print()

        if not response_text:
            return ""

        # --- TTS & 再生 ---
        self._state = self.STATE_SPEAKING
        print("\n🔊 読み上げ中...")
        try:
            wav_data = self.tts.synthesize(response_text)
            self.player.play_wav(wav_data, blocking=True)
        except Exception as e:
            print(f"⚠️  TTS/再生エラー: {e}")

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

        stream = self.recorder.open_stream(
            callback=audio_callback,
            frame_size=ww_frame_size,
        )

        with stream:
            while self._running and detected_word is None:
                try:
                    import time
                    time.sleep(0.05)  # 50ms ポーリング
                except KeyboardInterrupt:
                    raise

        return detected_word

    def run_interactive(self) -> None:
        """インタラクティブ音声対話ループ"""
        self._running = True
        print("\n" + "=" * 50)

        if self.enable_wakeword and self.wakeword_detector is not None:
            models = self.wakeword_detector.loaded_models
            print(f" 🎙️  ウェイクワードモード")
            print(f"  ウェイクワード: {', '.join(models)}")
            print(f"  閾値: {self.wakeword_detector.threshold}")
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
                    print(f"\n✨ ウェイクワード検知: {detected}")

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
                saved = self.session.save()
                print(f"会話を保存しました: {saved}")
            self.llm.close()

    def cleanup(self) -> None:
        """リソースの解放"""
        self._running = False
        if self.wakeword_detector is not None:
            self.wakeword_detector.cleanup()
        if self.proactive is not None:
            self.proactive.stop()
        if self.monitor_context is not None:
            self.monitor_context.stop()
        if self.vision_context is not None:
            self.vision_context.stop()
        self.llm.close()
