"""
音声対話パイプライン
Phase 3: VAD → STT → LLM (Ollama) → TTS → 再生
改善: ストリーミングTTS (文単位で合成・再生)、Silero VAD対応
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
from src.chat.client import OllamaClient
from src.chat.session import ChatSession
from src.chat.config import ChatConfig
from src.memory.vectorstore import VectorStore
from src.memory.rag import RAGRetriever
from src.vision.context import VisionContext


class VoicePipeline:
    """音声対話パイプライン"""

    # 状態定義
    STATE_IDLE = "idle"
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
    ):
        # チャット設定
        self.config = chat_config or ChatConfig.load(PROJECT_ROOT / "config" / "chat_config.json")

        # STT (faster-whisper)
        self.stt = WhisperSTT(
            model_size=stt_model,
            language="ja",
            device="cpu",
            compute_type="int8",
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

        # セッション
        self.session = ChatSession(
            system_prompt=self.config.system_prompt,
            max_history_turns=self.config.max_history_turns,
            history_dir=str(PROJECT_ROOT / self.config.history_dir),
            rag=self.rag,
            vision_context=self.vision_context,
        )

        # ストリーミングTTS設定
        self.streaming_tts = streaming_tts
        self._tts_queue: queue.Queue = queue.Queue()

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
        print("=" * 50)
        print(" 音声対話パイプライン 初期化")
        print("=" * 50)

        # Ollama 接続チェック
        print("\n[1/4] Ollama 接続確認...")
        if not self.llm.is_available():
            print("❌ Ollamaに接続できません")
            return False
        print("✅ Ollama OK")

        # STT モデルロード
        print("\n[2/4] STT モデルロード...")
        try:
            self.stt.load()
            print("✅ STT OK")
        except Exception as e:
            print(f"❌ STT ロード失敗: {e}")
            return False

        # TTS チェック
        print("\n[3/4] TTS 確認...")
        try:
            self.tts.load()
            print("✅ TTS OK")
        except Exception as e:
            print(f"❌ TTS ロード失敗: {e}")
            return False

        # VAD キャリブレーション (Energy VADの場合のみ環境ノイズ計測)
        print("\n[4/4] VAD キャリブレーション...")
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
            print("\n[5/6] RAG (長期記憶) 初期化...")
            try:
                self.vector_store.initialize()
                stats = self.rag.get_stats()
                print(f"✅ RAG OK (会話: {stats['conversations']}件, 知識: {stats['knowledge']}件)")
            except Exception as e:
                print(f"⚠️  RAG 初期化失敗 (RAGなしで続行): {e}")
                self.session.rag = None
        else:
            print("\n[5/6] RAG (長期記憶) スキップ")

        # Vision (Phase 5)
        if self.enable_vision and self.vision_context is not None:
            print("\n[6/6] Vision (映像入力) 初期化...")
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
            print("\n[6/6] Vision (映像入力) スキップ")

        print("\n" + "=" * 50)
        print(" ✅ 初期化完了！")
        print("=" * 50)
        return True

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

    def run_interactive(self) -> None:
        """インタラクティブ音声対話ループ"""
        self._running = True
        print("\n" + "=" * 50)
        print(" 🎙️  音声対話モード")
        print("  Ctrl+C で終了")
        print("=" * 50)

        try:
            while self._running:
                self.process_voice_turn()
        except KeyboardInterrupt:
            print("\n\n終了します...")
            self._running = False
            if self.session.turn_count > 0:
                saved = self.session.save()
                print(f"会話を保存しました: {saved}")
            self.llm.close()

    def cleanup(self) -> None:
        """リソースの解放"""
        self._running = False
        if self.vision_context is not None:
            self.vision_context.stop()
        self.llm.close()
