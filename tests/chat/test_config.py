"""
ChatConfig の基本テスト
"""
import json
import unittest
from pathlib import Path

from src.chat.config import ChatConfig

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "chat_config.json"

GUARD_MARKERS = ("タスク状態 (権威)", "会話履歴", "RAG")
NO_PROMPT_MARKERS = ("候補がない", "自発的な催促")


def test_default_config():
    """デフォルト値が正しく設定されること"""
    config = ChatConfig()
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.model == "qwen2.5:7b-instruct-q4_K_M"
    assert config.stream is True


def test_load_config(tmp_path):
    """JSON から設定をロードできること"""
    config_path = tmp_path / "test_chat_config.json"
    data = {
        "model": "test-model",
        "temperature": 0.5,
        "unknown_key": "ignored",
    }
    config_path.write_text("""{
  "model": "test-model",
  "temperature": 0.5,
  "unknown_key": "ignored"
}""", encoding="utf-8")

    config = ChatConfig.load(config_path)
    assert config.model == "test-model"
    assert config.temperature == 0.5
    # 未定義のキーは無視される
    assert not hasattr(config, "unknown_key")


def test_save_config(tmp_path):
    """設定を JSON に保存できること"""
    config = ChatConfig(model="saved-model")
    config_path = tmp_path / "saved_config.json"
    config.save(config_path)

    loaded = ChatConfig.load(config_path)
    assert loaded.model == "saved-model"


def _load_raw_config() -> dict:
    """追跡対象の実 JSON を読み込む (環境変数や非追跡 env は読まない)"""
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _assert_guard(prompt: str, label: str) -> None:
    assert prompt, f"{label}: プロンプトが空です"
    for marker in GUARD_MARKERS:
        assert marker in prompt, f"{label}: 守護文言に必須語「{marker}」が欠けています"
    # 自発催促禁止の否定表記が含まれること
    assert "推測" in prompt and "催促" in prompt, (
        f"{label}: 履歴/RAGからの推測禁止または自発催促禁止の記述がありません"
    )


class TaskStateGuardTests(unittest.TestCase):
    """未完了タスク判断の権威をタスク状態 (権威) ブロックに限定する guard の回帰テスト"""

    def test_base_system_prompt_has_task_state_guard(self):
        """base system_prompt が task-state tuning guard を含むこと"""
        data = _load_raw_config()
        _assert_guard(data["system_prompt"], "base system_prompt")

    def test_every_model_prompt_override_has_task_state_guard(self):
        """全 model_prompt_overrides が guard を持つこと"""
        data = _load_raw_config()
        overrides = data.get("model_prompt_overrides", {})
        self.assertTrue(overrides, "model_prompt_overrides が空です")
        for model, prompt in overrides.items():
            _assert_guard(prompt, f"override[{model}]")

    def test_active_effective_override_has_guard(self):
        """現在の model に対応する有効な override が guard を持つこと"""
        data = _load_raw_config()
        active_model = data["model"]
        overrides = data.get("model_prompt_overrides", {})
        self.assertIn(
            active_model, overrides,
            f"active model {active_model} の override が存在しません",
        )
        _assert_guard(overrides[active_model], f"active override[{active_model}]")

    def test_no_prompt_phrasing_present(self):
        """no-candidate / no-reminder で自発催促しない文言が各プロンプトにあること

候補がない (no-candidate) と自発的な催促 (no-reminder) の両方が
各プロンプトに明示されていることを検証する。"""
        data = _load_raw_config()
        prompts = [data["system_prompt"], *data.get("model_prompt_overrides", {}).values()]
        for idx, prompt in enumerate(prompts):
            for marker in NO_PROMPT_MARKERS:
                self.assertIn(marker, prompt, f"prompt[{idx}]: no-candidate / no-reminder 文言がありません")
