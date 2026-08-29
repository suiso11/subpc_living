from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.discord_bot import bot as bot_module
from src.discord_bot.bot import (
    DiscordConsoleState,
    companion_status_line,
    start_companion_activity_runtime,
    stop_companion_activity_runtime,
)


class _FakeRuntimeStatus:
    def __init__(self) -> None:
        self.running = True
        self.last_update_at = 456.0
        self.failure_count = 2
        self.consecutive_failures = 1
        self.last_error_type = "SomeError"
        self.last_state = SimpleNamespace(
            activity_mode="focused",
            present=True,
            focused_since=123.0,
            interruptible=False,
            display_state="on",
            updated_at=456.0,
            # 公開されてはならない情報
            window_title="secret-window",
            process_name="secret-process",
            error_text="secret-error-body",
        )


class _FakeRuntime:
    def __init__(self) -> None:
        self.status = _FakeRuntimeStatus()
        self.is_running = True
        self.stop_called = False

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_called = True


def _status_state() -> DiscordConsoleState:
    state = DiscordConsoleState()
    state.config = SimpleNamespace(
        ollama_base_url="http://localhost:11434", model="model-a"
    )
    return state


class CompanionWiringTest(unittest.TestCase):
    def tearDown(self) -> None:
        bot_module.activity_runtime = None

    def test_start_when_disabled_calls_factory_and_keeps_none(self) -> None:
        with patch.object(
            bot_module, "create_activity_runtime_from_env", return_value=None
        ) as create:
            start_companion_activity_runtime()
            create.assert_called_once_with(
                bot_module.os.environ, logger=bot_module.logger
            )
            self.assertIsNone(bot_module.activity_runtime)

    def test_start_when_enabled_holds_runtime_and_is_idempotent(self) -> None:
        fake = _FakeRuntime()
        with patch.object(
            bot_module, "create_activity_runtime_from_env", return_value=fake
        ):
            start_companion_activity_runtime()
            self.assertIs(bot_module.activity_runtime, fake)
        # 起動済みなら再び factory を呼ばない
        with patch.object(
            bot_module, "create_activity_runtime_from_env"
        ) as create:
            start_companion_activity_runtime()
            create.assert_not_called()
        stop_companion_activity_runtime()

    def test_stop_calls_runtime_stop_and_clears(self) -> None:
        fake = _FakeRuntime()
        bot_module.activity_runtime = fake
        stop_companion_activity_runtime()
        self.assertTrue(fake.stop_called)
        self.assertIsNone(bot_module.activity_runtime)

    def test_stop_when_none_is_noop(self) -> None:
        bot_module.activity_runtime = None
        stop_companion_activity_runtime()
        self.assertIsNone(bot_module.activity_runtime)

    def test_create_activity_runtime_from_env_true_and_false(self) -> None:
        from src.perception import create_activity_runtime_from_env

        # 未設定 / false は None
        self.assertIsNone(create_activity_runtime_from_env({}, logger=bot_module.logger))
        self.assertIsNone(
            create_activity_runtime_from_env(
                {"COMPANION_ACTIVITY_ENABLED": "false"}, logger=bot_module.logger
            )
        )
        # true は source を mock して実スレッドを開始した runtime を返す
        with patch(
            "src.perception.bootstrap.create_activity_source",
            return_value=MagicMock(),
        ):
            runtime = create_activity_runtime_from_env(
                {"COMPANION_ACTIVITY_ENABLED": "true"}, logger=bot_module.logger
            )
        self.assertIsNotNone(runtime)
        self.assertTrue(runtime.is_running)
        runtime.stop()

    def test_companion_status_line_is_privacy_safe(self) -> None:
        text = companion_status_line(_FakeRuntime())
        self.assertIn("'enabled': True", text)
        self.assertNotIn("secret-window", text)
        self.assertNotIn("secret-process", text)
        self.assertNotIn("secret-error-body", text)

    def test_companion_status_line_none_disabled(self) -> None:
        self.assertEqual(companion_status_line(None), "companion: {'enabled': False}\n")

    def test_status_text_includes_privacy_safe_companion_payload(self) -> None:
        state = _status_state()
        bot_module.activity_runtime = _FakeRuntime()
        with patch.object(bot_module, "HealthChecker") as checker_cls:
            checker_cls.return_value.check_all.return_value = {
                "status": "ok",
                "checks": "{}",
            }
            text = state.status_text()
        self.assertIn("companion: {'enabled': True", text)
        self.assertNotIn("secret-window", text)
        self.assertNotIn("secret-process", text)
        self.assertNotIn("secret-error-body", text)


if __name__ == "__main__":
    unittest.main()
