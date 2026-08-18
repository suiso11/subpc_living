"""同期streamをバックグラウンドでQueueへ橋渡しするAdapter。"""

from collections.abc import Iterable
import queue
import threading


_PUT_TIMEOUT = 0.05


class QueueStream:
    """バックグラウンドThreadでstreamを消費し、Queueへtokenを流すAdapter。"""

    def __init__(self, stream: Iterable[str], *, maxsize: int = 256) -> None:
        self.queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._source = stream
        self._cancelled = threading.Event()
        self._cancel_lock = threading.Lock()
        self._source_closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def source(self) -> object:
        """元のstreamを返す。"""
        return self._source

    @property
    def is_running(self) -> bool:
        """Workerが実行中ならTrueを返す。"""
        return self._thread.is_alive()

    def cancel(self) -> None:
        """停止を要求し、Queueのブロックを解除して可能ならsourceを閉じる。

        sourceが読み取りでブロック中の場合、Workerは即座には終了しない。終了時間は
        Providerのタイムアウト（現状httpxで300秒）で有界となる。真の即時中断には
        Provider側のリクエスト中断が必要であり、これは別フェーズの課題である。
        """
        self._cancelled.set()
        with self._cancel_lock:
            if self._source_closed:
                return
            self._source_closed = True
            close = getattr(self._source, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def join(self, timeout: float | None = None) -> bool:
        """Workerの終了を待ち、終了済みならTrueを返す。"""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _start(self) -> None:
        self._thread.start()

    def _put(self, item: object) -> bool:
        while not self._cancelled.is_set():
            try:
                self.queue.put(item, timeout=_PUT_TIMEOUT)
                return True
            except queue.Full:
                continue
        return False

    def _put_sentinel(self) -> None:
        while True:
            try:
                self.queue.put(None, timeout=_PUT_TIMEOUT)
                return
            except queue.Full:
                if not self._cancelled.is_set():
                    continue
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def _run(self) -> None:
        try:
            if not self._cancelled.is_set():
                for token in self._source:
                    if not self._put(token):
                        break
        except Exception as exc:
            if not self._cancelled.is_set():
                self._put(exc)
        finally:
            self._put_sentinel()


def stream_to_queue(stream: Iterable[str], *, maxsize: int = 256) -> QueueStream:
    """streamを消費するWorkerを起動し、QueueStreamを返す。"""
    queue_stream = QueueStream(stream, maxsize=maxsize)
    queue_stream._start()
    return queue_stream
