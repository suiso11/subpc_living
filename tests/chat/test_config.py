"""
ChatConfig の基本テスト
"""
from pathlib import Path

from src.chat.config import ChatConfig


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
