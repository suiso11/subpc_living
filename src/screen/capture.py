"""
スクリーンキャプチャ
mss でプライマリモニタをキャプチャし、Pillow で縮小して JPEG バイトを返す。

X11 セッション前提 (DISPLAY が必要)。DISPLAY が無い・キャプチャ失敗時は
None を返し、例外を外に漏らさない。
"""
import io
import os
from typing import Optional

try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ScreenCapture:
    """プライマリモニタのスクリーンショットを JPEG バイトで取得する。

    VLM 推論に渡す前提のため、長辺を max_edge (デフォルト 1344px) 程度に縮小し、
    JPEG (quality 指定) にエンコードして返す。

    mss はスレッドセーフではないため、キャプチャのたびに mss インスタンスを
    生成・破棄する (呼び出し間隔が長いので問題にならない)。
    """

    def __init__(self, max_edge: int = 1344, jpeg_quality: int = 85):
        """
        Args:
            max_edge: 縮小後の長辺ピクセル数の目安
            jpeg_quality: JPEG 品質 (0〜100)
        """
        self.max_edge = max_edge
        self.jpeg_quality = jpeg_quality

    def is_available(self) -> bool:
        """キャプチャ可能な環境か (依存ライブラリ + DISPLAY の有無で判定)。"""
        if not HAS_MSS or not HAS_PIL:
            return False
        # X11: DISPLAY が無ければ GUI セッションに接続できない
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            return False
        return True

    def capture(self) -> Optional[bytes]:
        """プライマリモニタをキャプチャして JPEG バイトを返す。

        失敗時 (DISPLAY 無し・X11 接続不可・エンコード失敗など) は None。
        例外は外に漏らさない。
        """
        if not self.is_available():
            return None

        try:
            with mss.mss() as sct:
                # monitors[0] は全モニタ結合、monitors[1] がプライマリ。
                # 単一モニタでも monitors[1] は存在する。
                monitors = sct.monitors
                monitor = monitors[1] if len(monitors) > 1 else monitors[0]
                shot = sct.grab(monitor)

            # mss の BGRA raw → Pillow Image (RGB)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

            # 長辺を max_edge に縮小 (拡大はしない)
            w, h = img.size
            long_edge = max(w, h)
            if long_edge > self.max_edge:
                scale = self.max_edge / float(long_edge)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                img = img.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
        except Exception:
            # スクリーンキャプチャ失敗は静かに None を返す (ループ継続用)
            return None
