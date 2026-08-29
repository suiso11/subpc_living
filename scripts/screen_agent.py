#!/usr/bin/env python3
"""
画面キャプチャエージェント (メインPC 側で常駐実行する自己完結スクリプト)

Windows / Linux / macOS で動く。プロジェクト内モジュールには依存しない。
依存は mss / Pillow / httpx のみ:

    pip install mss pillow httpx

プライマリモニタを定期キャプチャ → 長辺 1344px に縮小 → JPEG q85 →
    POST {url}/api/screen/ingest
に Content-Type: image/jpeg の生バイトで送信する。
ヘッダ X-Screen-Token に共有トークンを付ける (サーバー側 SCREEN_INGEST_TOKEN と一致)。

前回送信と画像ハッシュ (縮小後 JPEG の sha256) が同じならスキップして帯域と
VLM 負荷を節約する。ただし min-resend 秒 (デフォルト 600=10分) 以上送っていなければ
同一でも送る (鮮度維持)。

送信失敗はログを出してリトライ継続 (指数バックオフ上限 5 分、プロセスは死なない)。
Ctrl+C で綺麗に終了する (シグナルハンドラは使わず KeyboardInterrupt で抜ける)。

使い方の例 (画面キャプチャは明示的な opt-in が必要):
    python screen_agent.py --enable-screen-capture --url http://host:8000 --token TOKEN
    python screen_agent.py --enable-screen-capture --once  # 1 回だけ送って終了
環境変数でも指定可:
    SENSOR_SCREEN_CAPTURE_ENABLED=true
    SCREEN_AGENT_URL / SCREEN_AGENT_TOKEN / SCREEN_AGENT_INTERVAL
"""
import argparse
import hashlib
import io
import os
import sys
import time
from datetime import datetime

# --- 遅延/ガード付き import (mss 等が無くても純粋ロジックは import できる) ---
try:
    import mss  # type: ignore
    HAS_MSS = True
except Exception:
    HAS_MSS = False

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    import httpx  # type: ignore
    HAS_HTTPX = True
except Exception:
    HAS_HTTPX = False


DEFAULT_INTERVAL = 90.0
DEFAULT_MAX_EDGE = 1344
DEFAULT_JPEG_QUALITY = 85
DEFAULT_MIN_RESEND = 600.0   # 同一画像でもこの秒数を超えたら送る (鮮度維持)
BACKOFF_START = 5.0
BACKOFF_MAX = 300.0          # 指数バックオフ上限 5 分
SEND_TIMEOUT = 30.0


# ------------------------- 純粋ロジック (テスト対象) -------------------------

def is_exact_true(value: object) -> bool:
    """明示的な true だけを opt-in として受け入れる。"""
    return isinstance(value, str) and value.strip().lower() == "true"


def image_hash(jpeg_bytes: bytes) -> str:
    """縮小後 JPEG バイトの sha256 hex を返す。"""
    return hashlib.sha256(jpeg_bytes).hexdigest()


def should_send(
    new_hash: str,
    last_hash,
    last_sent_at,
    now: float,
    min_resend_interval: float = DEFAULT_MIN_RESEND,
) -> bool:
    """今回のキャプチャを送信すべきか判定する。

    - まだ一度も送っていない (last_sent_at is None) → 送る
    - ハッシュが前回と異なる → 送る
    - ハッシュ同一でも min_resend_interval を超えていれば送る (鮮度維持)
    - それ以外 (同一かつ間隔内) → 送らない
    """
    if last_sent_at is None or last_hash is None:
        return True
    if new_hash != last_hash:
        return True
    return (now - last_sent_at) >= min_resend_interval


def next_backoff(current: float) -> float:
    """指数バックオフの次の待機秒 (上限 BACKOFF_MAX)。"""
    return min(current * 2.0, BACKOFF_MAX)


# ------------------------- キャプチャ / 送信 -------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def capture_jpeg(max_edge: int, quality: int) -> bytes:
    """プライマリモニタをキャプチャし、縮小 JPEG バイトで返す。

    依存が無い / キャプチャ失敗時は RuntimeError を送出 (呼び出し側でリトライ)。
    """
    if not HAS_MSS or not HAS_PIL:
        raise RuntimeError("mss / Pillow が必要です: pip install mss pillow httpx")

    with mss.mss() as sct:
        monitors = sct.monitors
        # monitors[0] は全モニタ結合、monitors[1] がプライマリ
        monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        shot = sct.grab(monitor)

    img = Image.frombytes("RGB", shot.size, shot.rgb)

    w, h = img.size
    long_edge = max(w, h)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def send_jpeg(url: str, token: str, jpeg_bytes: bytes) -> None:
    """ingest エンドポイントへ生 JPEG を POST する。失敗時は例外を送出。"""
    if not HAS_HTTPX:
        raise RuntimeError("httpx が必要です: pip install httpx")

    endpoint = url.rstrip("/") + "/api/screen/ingest"
    headers = {
        "Content-Type": "image/jpeg",
        "X-Screen-Token": token,
    }
    resp = httpx.post(endpoint, content=jpeg_bytes, headers=headers, timeout=SEND_TIMEOUT)
    resp.raise_for_status()


# ------------------------- メインループ -------------------------

def run(args) -> int:
    if not (
        getattr(args, "enable_screen_capture", False)
        or is_exact_true(os.environ.get("SENSOR_SCREEN_CAPTURE_ENABLED"))
    ):
        _log("ERROR: screen capture is disabled")
        return 4

    url = args.url
    token = args.token
    if not url:
        _log("ERROR: URL is missing")
        return 2
    if not token:
        _log("ERROR: token is missing")
        return 2

    if not (HAS_MSS and HAS_PIL and HAS_HTTPX):
        _log("ERROR: required dependencies are unavailable")
        return 3

    _log("screen_agent started")

    last_hash = None
    last_sent_at = None
    backoff = BACKOFF_START

    try:
        while True:
            # --- キャプチャ ---
            try:
                jpeg = capture_jpeg(args.max_edge, DEFAULT_JPEG_QUALITY)
            except Exception:
                _log("ERROR: capture failed; retrying")
                if args.once:
                    return 1
                time.sleep(args.interval)
                continue

            new_hash = image_hash(jpeg)
            now = time.time()

            if not args.once and not should_send(
                new_hash, last_hash, last_sent_at, now, args.min_resend
            ):
                # 変化なし & 鮮度維持間隔内 → スキップ
                time.sleep(args.interval)
                continue

            # --- 送信 (失敗時は指数バックオフでリトライ、プロセスは死なない) ---
            sent = False
            while not sent:
                try:
                    send_jpeg(url, token, jpeg)
                    sent = True
                    last_hash = new_hash
                    last_sent_at = time.time()
                    backoff = BACKOFF_START
                    _log("screen_agent sent image")
                except Exception:
                    _log("ERROR: send failed; retrying")
                    if args.once:
                        return 1
                    time.sleep(backoff)
                    backoff = next_backoff(backoff)

            if args.once:
                _log("screen_agent completed once")
                return 0

            time.sleep(args.interval)

    except KeyboardInterrupt:
        _log("screen_agent stopped")
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="画面キャプチャエージェント (メインPC 常駐)")
    p.add_argument(
        "--enable-screen-capture",
        action="store_true",
        help="enable screen capture (or set SENSOR_SCREEN_CAPTURE_ENABLED=true)",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("SCREEN_AGENT_URL", ""),
        help="サブPC の Web サーバー URL (例: http://100.x.x.x:8000)。env SCREEN_AGENT_URL",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("SCREEN_AGENT_TOKEN", ""),
        help="共有トークン (サーバーの SCREEN_INGEST_TOKEN と一致)。env SCREEN_AGENT_TOKEN",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("SCREEN_AGENT_INTERVAL", DEFAULT_INTERVAL)),
        help=f"キャプチャ間隔秒 (default {DEFAULT_INTERVAL})。env SCREEN_AGENT_INTERVAL",
    )
    p.add_argument(
        "--max-edge",
        type=int,
        default=DEFAULT_MAX_EDGE,
        help=f"縮小後の長辺ピクセル (default {DEFAULT_MAX_EDGE})",
    )
    p.add_argument(
        "--min-resend",
        type=float,
        default=DEFAULT_MIN_RESEND,
        help=f"同一画像でも送る鮮度間隔秒 (default {DEFAULT_MIN_RESEND})",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="1回だけ送って終了 (動作確認用)",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
