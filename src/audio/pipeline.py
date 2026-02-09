"""
音声対話パイプライン
Phase 3: VAD → STT → LLM (Ollama) → TTS → 再生
Phase 2 のチャットモジュールと統合
"""
import sys
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
from src.audio.vad import EnergyVAD
from src.audio.audio_io import AudioRecorder, AudioPlayer
from src.chat.client import OllamaClient
from src.chat.session import ChatSession
from src.chat.config import ChatConfig


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

        # VAD
        self.vad = EnergyVAD(
            sample_rate=16000,
            energy_threshold=0.01,
            silence_duration_ms=800,
            min_speech_duration_ms=300,
        )

        # オーディオ I/O
        self.recorder = AudioRecorder(sample_rate=16000)
        self.player = AudioPlayer(sample_rate=24000)

        # LLM
        self.llm = OllamaClient(
            base_url=self.config.ollama_base_url,
            model=self.config.model,
        )

        # セッション
        self.session = ChatSession(
            system_prompt=self.config.system_prompt,
            max_history_turns=self.config.max_history_turns,
            history_dir=str(PROJECT_ROOT / self.config.history_dir),
        )

        # 状態
        self._state = self.STATE_IDLE
        self._running = False
        self._audio_queue: queue.Queue = queue.Queue()

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

        # VAD キャリブレーション
        print("\n[4/4] VAD キャリブレーション...")
        try:
            print("  環境ノイズを計測中 (2秒間、静かにしてください)...")
            noise_sample = self.recorder.record(2.0)
            self.vad.calibrate(noise_sample)
            print("✅ VAD OK")
        except Exception as e:
            print(f"⚠️  VAD キャリブレーション失敗 (デフォルト閾値を使用): {e}")

        print("\n" + "=" * 50)
        print(" ✅ 初期化完了！")
        print("=" * 50)
        return True

    def process_voice_turn(self) -> Optional[str]:
        """
        1ターンの音声対話を処理する:
        録音 → STT → LLM → TTS → 再生

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

        # --- LLM ---
        print("\n🤖 考え中...")
        self.session.add_user_message(user_text)
        messages = self.session.build_messages()

        try:
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
                return None

            self.session.add_assistant_message(response_text)

        except Exception as e:
            print(f"\n❌ LLM エラー: {e}")
            if self.session._messages and self.session._messages[-1]["role"] == "user":
                self.session._messages.pop()
            return None

        # --- TTS & 再生 ---
        self._state = self.STATE_SPEAKING
        print("\n🔊 読み上げ中...")
        try:
            wav_data = self.tts.synthesize(response_text)
            self.player.play_wav(wav_data, blocking=True)
        except Exception as e:
            print(f"⚠️  TTS/再生エラー: {e}")

        self._state = self.STATE_IDLE
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
        self.llm.close()
