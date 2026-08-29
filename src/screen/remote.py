"""
リモート画面コンテキスト (remote モード)

別PC (Windows メインPC など) 上のキャプチャエージェント
(scripts/screen_agent.py) がスクリーンショットを Web サーバーの
POST /api/screen/ingest に push し、サーバー側が VLM で 1 回だけ描写して
data/screen/latest.json に書き出す。

RemoteScreenContext はその latest.json を定期的に「読むだけ」で、
自分ではキャプチャも VLM 呼び出しも行わない (ローカルの ScreenContext と
同じ公開インターフェースを提供する)。

ファイルを読むだけなので負荷は軽く、ポーリング間隔はデフォルト 15 秒と短め。
captured_at が stale_after (デフォルト 600秒=10分) より古い / ファイルが無い
場合は get_context_text() は空文字を返す (ScreenContext と同じ鮮度ポリシー)。

共有 SensorPolicy.screen_ingest (env SENSOR_SCREEN_INGEST_ENABLED の明示 true) が
有効なときだけ最新情報を読取・公開する。無効 (false / 未設定 / 不正値) のときは
ファイルを一切読まず、start() は False を返し、既存の latest.json の描写は
get_state / get_status / get_context_text のいずれからも公開しない (fail closed)。
消費側 (Web / Discord / 音声 pipeline) の screen_capture ゲートに加えて、remote
モードはこの ingest 政策も併せて必須とする (二重ゲート)。
"""
import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.perception.policy import resolve_sensor_policy

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LATEST_JSON = PROJECT_ROOT / "data" / "screen" / "latest.json"


@dataclass
class RemoteScreenState:
    """latest.json から読み取った最新のリモート画面状態"""
    description: str = ""            # VLM 描写テキスト
    captured_at: float = 0.0         # サーバーが画像を受信した時刻 (epoch秒)
    described_at: float = 0.0        # 描写が完了した時刻 (epoch秒)
    source: str = "remote"           # 供給元 (常に "remote")
    read_count: int = 0              # latest.json の読み取り成功回数


class RemoteScreenContext:
    """
    共有ファイル (data/screen/latest.json) を定期読取するだけの
    軽量スクリーンコンテキスト。

    ScreenContext と同じ公開インターフェース:
      start / stop / is_running / pause / resume /
      get_state / get_context_text / get_status
    """

    def __init__(
        self,
        poll_interval: float = 15.0,
        stale_after: float = 600.0,
        latest_path: Optional[Union[str, Path]] = None,
        screen_ingest: Optional[bool] = None,
    ):
        """
        Args:
            poll_interval: latest.json の読取間隔（秒）。ファイルを読むだけなので短め
            stale_after: この秒数より古い描写はコンテキストに含めない (デフォルト 600秒=10分)
            latest_path: 読み取る JSON ファイルのパス。None なら data/screen/latest.json
            screen_ingest: 共有 SensorPolicy.screen_ingest の明示 true のときだけ
                latest.json の読取・公開を許可する (安全側デフォルト)。None なら
                env SENSOR_SCREEN_INGEST_ENABLED から解決する (明示 true のみ有効)。
                明示 bool を渡すと env に依存せず確定する (テスト注入用)。
        """
        self.poll_interval = poll_interval
        self.stale_after = stale_after
        self.latest_path = Path(latest_path) if latest_path else DEFAULT_LATEST_JSON
        if screen_ingest is None:
            screen_ingest = resolve_sensor_policy().screen_ingest
        self.screen_ingest = bool(screen_ingest)

        self._state = RemoteScreenState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._paused = False
        self._stop = threading.Event()
        self._stop_pending = False

    def start(self) -> bool:
        """バックグラウンドの読取ループを開始。

        共有 SensorPolicy.screen_ingest が有効のときだけファイルを読む。無効
        (false / 未設定 / 不正値) の場合は一切読まず・スレッドも立てず False を返す。
        既存の読取スレッドが生存中は再起動せず False を返す (fail safe)。stop 要求
        (stop_pending) 中にスレッドが生存し続けていても上書きしない。再 start は
        スレッドの死が確認された後でのみ許可する。

        Partial start の硬化:
        - スレッド生成 (factory) と start の例外を握り、生成済みスレッドの ownership
          を self._thread に start より先に確定する
        - start 後に実際に生存しているスレッドは ownership を保持したまま True を返し、
          例外があっても破棄しない (部分起動でも上書きしない)
        - 死んだ / 未 start のスレッドは回収して再 start を許可し、即死した場合は
          False を返す
        例外メッセージは外に漏らさない。
        """
        if not self.screen_ingest:
            return False
        if self.thread_alive:
            # 生存中の読取スレッドがある限り再起動しない。stop 要求 (stop_pending) 中に
            # スレッドが生存し続けている場合も is_running ではなく生の生存判定で
            # 拒否し、既存スレッドを上書きしない (is_running は stop 要求後 False のため、
            # それを見ると誤って再 start を許可してしまう)。
            return False
        self._stop.clear()
        self._paused = False
        self._stop_pending = False
        # 起動直後に一度読んでおく (即座にコンテキストを利用可能にする)
        self._read_once()
        try:
            thread = threading.Thread(target=self._poll_loop, daemon=True)
        except Exception:
            self._thread = None
            return False
        # start 前に ownership を確定する (start 途中の例外でも生成済みスレッドを
        # 失わず、生存中のスレッドを上書きしない)
        self._thread = thread
        try:
            thread.start()
        except Exception:
            # 例外があっても、実際に生存しているスレッドは破棄せず ownership を保持する
            if thread.is_alive():
                return True
            # 未 start のままのスレッドは回収して再 start を許可
            if self._thread is thread:
                self._thread = None
            return False
        if not thread.is_alive():
            # 即死 (読取ループが即終了): 状態をクリーンアップして再 start を許可
            if self._thread is thread:
                self._thread = None
            return False
        return True

    def stop(self):
        """停止。

        読取スレッドの join がタイムアウトして生存し続ける場合は ownership を保持し
        (self._thread を None にしない)、stop_pending を立てる。以後の stop で再試行
        できる。確認済みに死んだ場合のみ self._thread を解放し、stop_pending を
        解除して再 start を許可する。多重呼び出しは安全 (idempotent)。

        未 start のまま保持されたスレッドは join できないため、例外を漏らさずに
        回収して解放する。
        """
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        try:
            thread.join(timeout=5)
        except RuntimeError:
            # 未 start のスレッドは join 不可 → そのまま回収して再 start を許可する
            if self._thread is thread:
                self._thread = None
                self._stop_pending = False
            return
        if thread.is_alive():
            # join タイムアウト: ownership を保持し、以後の stop で再試行可能に保つ。
            # stop を要求した以上、スレッドが生存していても「稼働中」とは報告しない。
            self._stop_pending = True
            return
        self._thread = None
        self._stop_pending = False

    @property
    def is_running(self) -> bool:
        """実際に稼働中 (読取スレッドが生存 かつ 停止要求なし) のときだけ True。

        stop を要求した後 (stop_pending) はスレッドが生存し続けていても False を
        真実として返す。予期せぬスレッド死亡も「稼働中」として報告しない。
        """
        return not self._stop_pending and self.thread_alive

    @property
    def thread_alive(self) -> bool:
        """所有する読取スレッドが実際に生存しているか (生の生存判定)。

        stop 要求 (stop_pending) の有無に関係なく報告する。start はこの値で
        上書きを拒否する。
        """
        thread = self._thread
        if thread is None:
            return False
        try:
            return thread.is_alive()
        except Exception:
            return False

    @property
    def stop_pending(self) -> bool:
        """停止要求済みだが読取スレッドの死亡が未確認か。"""
        return self._stop_pending

    def pause(self) -> None:
        """読取を一時停止"""
        self._paused = True

    def resume(self) -> None:
        """読取を再開"""
        self._paused = False

    def get_state(self) -> RemoteScreenState:
        """現在の状態を取得 (thread-safe)。

        screen_ingest が無効のときは常に空の状態を返し、既存の latest.json の
        描写を決して公開しない。
        """
        if not self.screen_ingest:
            return RemoteScreenState()
        with self._state_lock:
            return copy.copy(self._state)

    def get_context_text(self) -> str:
        """
        LLM のシステムプロンプトに注入する画面コンテキストテキストを生成。

        ScreenContext と同じ整形だが、「メインPCの画面」であることが分かる。
        screen_ingest が無効のときは空文字。未取得 / ファイル無し / captured_at
        が stale_after より古い場合も空文字。
        """
        if not self.screen_ingest or not self.is_running:
            return ""

        state = self.get_state()
        if not state.description or state.captured_at <= 0:
            return ""

        age = time.time() - state.captured_at
        if age > self.stale_after:
            return ""

        minutes = max(0, int(age / 60))
        lines = [
            "\n--- 画面情報 (メインPCの画面) ---",
            f"- ユーザーの画面(メインPC): {state.description} ({minutes}分前時点)",
        ]
        return "\n".join(lines)

    def get_status(self) -> dict:
        """API レスポンス用の状態辞書"""
        state = self.get_state()
        age = (time.time() - state.captured_at) if state.captured_at > 0 else None
        return {
            "mode": "remote",
            "running": self.is_running,
            "thread_alive": self.thread_alive,
            "stop_pending": self.stop_pending,
            "paused": self._paused,
            "screen_ingest": self.screen_ingest,
            "description": state.description,
            "captured_at": state.captured_at,
            "described_at": state.described_at,
            "source": "remote",
            "age_seconds": age,
            "poll_interval": self.poll_interval,
            "stale_after": self.stale_after,
            "latest_path": str(self.latest_path),
            "read_count": state.read_count,
        }

    # --- 内部: バックグラウンド読取 ---

    def _poll_loop(self):
        if not self.screen_ingest:
            return
        while not self._stop.is_set():
            if not self._paused:
                self._read_once()
            self._stop.wait(self.poll_interval)

    def _read_once(self) -> bool:
        """latest.json を 1 回読んで状態を更新。有効な描写を読めたら True。

        screen_ingest が無効のときはファイルを読まずに False を返す (fail closed)。
        - ファイル無し: 状態を空にする (captured_at=0)
        - 壊れた JSON / 読取失敗: 状態は維持 (更新しない)
        例外は外に漏らさない。
        """
        if not self.screen_ingest:
            return False
        try:
            if not self.latest_path.exists():
                with self._state_lock:
                    self._state.description = ""
                    self._state.captured_at = 0.0
                return False
            raw = self.latest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            description = (data.get("description") or "").strip()
            captured_at = float(data.get("captured_at") or 0.0)
        except Exception:
            # 壊れた JSON・部分書き込み・読取失敗は無視して前回状態を維持
            return False

        with self._state_lock:
            self._state.description = description
            self._state.captured_at = captured_at
            self._state.described_at = float(data.get("described_at") or 0.0)
            self._state.source = "remote"
            self._state.read_count += 1

        return bool(description and captured_at > 0)
