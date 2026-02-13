"""
ウェイクワード検知モジュール
Phase 10: OpenWakeWord を使用したウェイクワード検知

常時マイク入力を監視し、特定のウェイクワード（"hey jarvis" 等）を
検知したらアクティブモードに切り替える。
"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    OpenWakeWord ベースのウェイクワード検知

    16kHz/16bit PCM 音声の 1280 サンプル (80ms) フレーム単位で
    process_frame() を呼び出し、ウェイクワード検知時にワード名を返す。

    Args:
        model_names: 検知するウェイクワードモデル名のリスト
                     (例: ["hey_jarvis", "alexa"])
                     None の場合は全プリトレインモデルをロード
        threshold: 検知の閾値 (0.0〜1.0、デフォルト: 0.5)
        vad_threshold: VAD 閾値 (0.0 で無効、0.0〜1.0 で有効)
        enable_speex: Speex ノイズ抑制を有効化 (Linux のみ)
    """

    # 80ms @ 16kHz = 1280 サンプル
    FRAME_SIZE = 1280
    SAMPLE_RATE = 16000

    # 利用可能なプリトレインモデル名
    AVAILABLE_MODELS = ["alexa", "hey_mycroft", "hey_jarvis", "timer", "weather"]

    def __init__(
        self,
        model_names: Optional[list[str]] = None,
        threshold: float = 0.5,
        vad_threshold: float = 0.0,
        enable_speex: bool = False,
    ):
        self.model_names = model_names
        self.threshold = threshold
        self.vad_threshold = vad_threshold
        self.enable_speex = enable_speex
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """
        モデルをロードする

        Returns:
            成功したら True
        """
        try:
            import openwakeword
            from openwakeword.model import Model

            # モデル名 → ファイルパスに変換
            model_paths: list[str] = []
            if self.model_names:
                for name in self.model_names:
                    if name in openwakeword.models:
                        model_paths.append(openwakeword.models[name]["model_path"])
                    else:
                        logger.warning(
                            f"ウェイクワードモデル '{name}' が見つかりません。"
                            f" 利用可能: {list(openwakeword.models.keys())}"
                        )
                if not model_paths:
                    logger.error("有効なウェイクワードモデルがありません")
                    return False
            # model_paths が空リストなら全プリトレインモデルをロード

            # モデルをインスタンス化
            kwargs = {}
            if model_paths:
                kwargs["wakeword_model_paths"] = model_paths
            if self.vad_threshold > 0:
                kwargs["vad_threshold"] = self.vad_threshold
            if self.enable_speex:
                kwargs["enable_speex_noise_suppression"] = True

            self._model = Model(**kwargs)
            self._loaded = True

            # ロードされたモデル名を取得
            loaded_models = list(self._model.models.keys())
            logger.info(f"ウェイクワードモデルロード完了: {loaded_models}")
            return True

        except ImportError:
            logger.error("openwakeword がインストールされていません: pip install openwakeword")
            return False
        except Exception as e:
            logger.error(f"ウェイクワードモデルのロードに失敗: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def loaded_models(self) -> list[str]:
        """ロードされたモデル名のリスト"""
        if self._model is None:
            return []
        return list(self._model.models.keys())

    @property
    def frame_size(self) -> int:
        return self.FRAME_SIZE

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        1フレーム分の音声を処理してウェイクワード検知を行う

        Args:
            frame: 音声データ (float32, モノラル, 16kHz)
                   1280 サンプル (80ms) の倍数が推奨

        Returns:
            検知されたウェイクワード名。検知なしの場合は None。
        """
        if not self._loaded or self._model is None:
            return None

        # float32 [-1.0, 1.0] → int16 [-32768, 32767] に変換
        # OpenWakeWord は int16 PCM を期待する
        audio_int16 = (frame * 32767).astype(np.int16)

        # 予測を実行
        prediction = self._model.predict(audio_int16)

        # 閾値を超えたモデルを検索
        for model_name, score in prediction.items():
            if score >= self.threshold:
                logger.info(f"ウェイクワード検知: {model_name} (score={score:.3f})")
                # 検知後にモデルをリセット（連続誤検知を防ぐ）
                self._model.reset()
                return model_name

        return None

    def reset(self) -> None:
        """検知状態をリセット"""
        if self._model is not None:
            self._model.reset()

    def cleanup(self) -> None:
        """リソースを解放"""
        self._model = None
        self._loaded = False

    def get_info(self) -> dict:
        """検知器の情報を返す"""
        return {
            "loaded": self._loaded,
            "models": self.loaded_models,
            "threshold": self.threshold,
            "vad_threshold": self.vad_threshold,
            "frame_size": self.FRAME_SIZE,
            "sample_rate": self.SAMPLE_RATE,
        }
