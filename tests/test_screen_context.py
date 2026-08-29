"""
ScreenContext の状態遷移・staleness・コンテキストテキスト整形を検証する。

実キャプチャ・実 Ollama は使わず、capture / describer をフェイクに差し替える。
"""
from __future__ import annotations

import time
import unittest

from src.screen.context import ScreenContext


class FakeCapture:
    """常にダミー JPEG を返すキャプチャ。available フラグを操作可能。"""

    def __init__(self, available: bool = True, jpeg: bytes = b"jpeg-bytes"):
        self.available = available
        self.jpeg = jpeg
        self.calls = 0

    def is_available(self) -> bool:
        return self.available

    def capture(self):
        self.calls += 1
        return self.jpeg


class FakeDescriber:
    """設定した描写を返す / None を返す / 例外を投げる、を切り替えられる。"""

    def __init__(self, description="VSCodeでPythonを編集しています。", model="fake-vlm"):
        self.description = description
        self.model = model
        self.calls = 0

    def describe(self, jpeg_bytes):
        self.calls += 1
        if isinstance(self.description, Exception):
            raise self.description
        return self.description


class FakeThread:
    """is_alive() が常に True を返すフェイクスレッド。

    実スレッドを起動せずに、ScreenContext の「所有するスレッドが生存している
    ときだけ is_running が True」という契約を満たすために使う。
    """

    def __init__(self, *args, **kwargs):
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def start(self):
        self._alive = True


def make_ctx(capture=None, describer=None, **kwargs):
    return ScreenContext(
        capture=capture or FakeCapture(),
        describer=describer or FakeDescriber(),
        thread_factory=FakeThread,
        **kwargs,
    )


class ScreenContextTest(unittest.TestCase):
    # --- 状態遷移 ---

    def test_start_returns_false_when_capture_unavailable(self):
        ctx = make_ctx(capture=FakeCapture(available=False))
        self.assertFalse(ctx.start())
        self.assertFalse(ctx.is_running)

    def test_successful_run_records_description(self):
        ctx = make_ctx(describer=FakeDescriber("Chromeでニュースを見ています。"))
        self.assertTrue(ctx.start())  # スレッドを起動せず直接検証
        self.assertTrue(ctx._run_once())

        state = ctx.get_state()
        self.assertEqual(state.description, "Chromeでニュースを見ています。")
        self.assertEqual(state.analysis_count, 1)
        self.assertEqual(state.consecutive_failures, 0)
        self.assertGreater(state.captured_at, 0)

    def test_context_text_empty_when_not_running(self):
        ctx = make_ctx()
        # 描写はあるが running でない
        ctx._run_once()
        self.assertEqual(ctx.get_context_text(), "")

    def test_context_text_empty_before_any_capture(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        self.assertEqual(ctx.get_context_text(), "")

    # --- コンテキストテキスト整形 ---

    def test_context_text_format_fresh(self):
        ctx = make_ctx(describer=FakeDescriber("Blenderで3Dモデルを制作しています。"))
        self.assertTrue(ctx.start())
        ctx._run_once()
        text = ctx.get_context_text()
        self.assertIn("--- 画面情報 ---", text)
        self.assertIn(
            "- ユーザーの画面: Blenderで3Dモデルを制作しています。 (0分前時点)",
            text,
        )
        # 先頭に改行が入り、既存プロンプトへ自然に連結できる
        self.assertTrue(text.startswith("\n"))

    def test_context_text_reports_minutes_elapsed(self):
        ctx = make_ctx()
        self.assertTrue(ctx.start())
        ctx._run_once()
        # 3分前に取得したことにする
        with ctx._state_lock:
            ctx._state.captured_at = time.time() - 3 * 60
        text = ctx.get_context_text()
        self.assertIn("(3分前時点)", text)

    # --- staleness ---

    def test_context_text_empty_when_stale(self):
        ctx = make_ctx(stale_after=600.0)
        self.assertTrue(ctx.start())
        ctx._run_once()
        # 11分前 → stale_after(10分)超過で空
        with ctx._state_lock:
            ctx._state.captured_at = time.time() - 11 * 60
        self.assertEqual(ctx.get_context_text(), "")

    def test_context_text_present_just_within_stale_window(self):
        ctx = make_ctx(stale_after=600.0)
        self.assertTrue(ctx.start())
        ctx._run_once()
        with ctx._state_lock:
            ctx._state.captured_at = time.time() - 9 * 60
        self.assertIn("画面情報", ctx.get_context_text())

    # --- 失敗ハンドリング ---

    def test_capture_failure_increments_consecutive_failures(self):
        ctx = make_ctx(capture=FakeCapture(jpeg=None))
        self.assertTrue(ctx.start())
        self.assertFalse(ctx._run_once())
        self.assertEqual(ctx.get_state().consecutive_failures, 1)

    def test_describer_none_counts_as_failure(self):
        ctx = make_ctx(describer=FakeDescriber(description=None))
        self.assertTrue(ctx.start())
        self.assertFalse(ctx._run_once())
        self.assertEqual(ctx.get_state().consecutive_failures, 1)

    def test_describer_exception_is_swallowed_and_counted(self):
        ctx = make_ctx(describer=FakeDescriber(description=RuntimeError("boom")))
        self.assertTrue(ctx.start())
        # 例外は外に漏れず失敗としてカウントされる
        self.assertFalse(ctx._run_once())
        self.assertEqual(ctx.get_state().consecutive_failures, 1)

    def test_auto_pause_after_max_failures(self):
        ctx = make_ctx(capture=FakeCapture(jpeg=None), max_failures=5)
        self.assertTrue(ctx.start())
        for _ in range(5):
            ctx._run_once()
        self.assertTrue(ctx._paused)
        self.assertGreaterEqual(ctx.get_state().consecutive_failures, 5)

    def test_auto_pause_safe_under_cp932_stdout(self):
        # CP932 (Windows-31J) では絵文字 ⚠️ をエンコードできず UnicodeEncodeError になる。
        # auto-pause 時の通知は絵文字を含まない logging 経由なので、stdout/stderr を
        # CP932 相当にしても例外を出さず、pause セマンティクスを維持する。
        import io
        import logging

        class CP932StreamIO(io.TextIOBase):
            def __init__(self):
                self.encoder = io.TextIOWrapper(io.BytesIO(), encoding="cp932")

            def write(self, s):
                return self.encoder.write(s)

        ctx = make_ctx(capture=FakeCapture(jpeg=None), max_failures=2)
        self.assertTrue(ctx.start())

        handler = logging.StreamHandler(CP932StreamIO())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("src.screen.context")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            for _ in range(2):
                ctx._run_once()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        self.assertTrue(ctx._paused)
        self.assertGreaterEqual(ctx.get_state().consecutive_failures, 2)

    def test_success_resets_failure_counter(self):
        describer = FakeDescriber(description=None)
        ctx = make_ctx(describer=describer)
        self.assertTrue(ctx.start())
        ctx._run_once()  # 失敗
        ctx._run_once()  # 失敗
        self.assertEqual(ctx.get_state().consecutive_failures, 2)
        describer.description = "ターミナルで作業しています。"
        ctx._run_once()  # 成功
        self.assertEqual(ctx.get_state().consecutive_failures, 0)

    def test_resume_resets_failures_and_unpauses(self):
        ctx = make_ctx(capture=FakeCapture(jpeg=None), max_failures=3)
        self.assertTrue(ctx.start())
        for _ in range(3):
            ctx._run_once()
        self.assertTrue(ctx._paused)
        ctx.resume()
        self.assertFalse(ctx._paused)
        self.assertEqual(ctx.get_state().consecutive_failures, 0)

    # --- get_status ---

    def test_get_status_fields(self):
        ctx = make_ctx(describer=FakeDescriber("Slackを見ています。", model="fake-vlm"))
        self.assertTrue(ctx.start())
        ctx._run_once()
        status = ctx.get_status()
        self.assertTrue(status["running"])
        self.assertEqual(status["description"], "Slackを見ています。")
        self.assertEqual(status["model"], "fake-vlm")
        self.assertEqual(status["analysis_count"], 1)
        self.assertIsNotNone(status["age_seconds"])
        self.assertEqual(status["analysis_interval"], 90.0)


if __name__ == "__main__":
    unittest.main()
