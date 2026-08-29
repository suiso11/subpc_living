"""
ChatConfig の基本テスト
"""
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from src.chat.config import (
    LOCAL_OPENAI_DEFAULT_BASE_URL,
    ChatConfig,
    resolve_local_api_key,
    resolve_local_base_url,
    resolve_local_provider_id,
    validate_local_base_url,
    validate_local_provider_kind,
)
from src.llm.local_endpoint import validate_loopback_openai_base_url

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


class LocalProviderConfigTests(unittest.TestCase):
    """P0-2: ローカル推論 backend 選択設定の既定値・解決・検証"""

    def test_local_provider_defaults(self):
        config = ChatConfig()
        self.assertEqual(config.local_provider_kind, "ollama")
        self.assertEqual(config.local_base_url, "")
        self.assertEqual(config.local_provider_id, "")
        self.assertEqual(config.local_api_key_env, "")

    def test_resolved_provider_id_defaults_by_kind(self):
        self.assertEqual(ChatConfig().resolved_local_provider_id(), "ollama")
        self.assertEqual(
            ChatConfig(local_provider_kind="openai_compatible").resolved_local_provider_id(),
            "local-openai",
        )
        config = ChatConfig(local_provider_id="llama-server")
        self.assertEqual(config.resolved_local_provider_id(), "llama-server")

    def test_empty_or_whitespace_kind_normalizes_to_ollama(self):
        for raw in ("", "   ", None):
            config = ChatConfig(local_provider_kind=raw)
            self.assertEqual(validate_local_provider_kind(config), "ollama")
            self.assertEqual(config.resolved_local_provider_id(), "ollama")
            self.assertEqual(config.resolved_local_base_url(), "http://localhost:11434")
            self.assertIsNone(config.resolve_local_api_key())
            config.validate_local_provider()

    def test_kind_normalization_consistency_across_helpers(self):
        config = ChatConfig(local_provider_kind="  openai_compatible  ")
        self.assertEqual(validate_local_provider_kind(config), "openai_compatible")
        self.assertEqual(config.resolved_local_provider_id(), "local-openai")
        self.assertEqual(
            config.resolved_local_base_url(), LOCAL_OPENAI_DEFAULT_BASE_URL
        )

    def test_resolve_local_base_url_ollama_honors_legacy_with_blank_kind(self):
        config = ChatConfig(
            local_provider_kind="   ",
            ollama_base_url="http://legacy:11434",
            local_base_url="http://other/v1",
        )
        self.assertEqual(config.resolved_local_base_url(), "http://legacy:11434")
        config.validate_local_provider()

    def test_resolved_base_url_ollama_honors_legacy(self):
        config = ChatConfig(
            ollama_base_url="http://legacy:11434",
            local_base_url="http://other/v1",
        )
        self.assertEqual(config.resolved_local_base_url(), "http://legacy:11434")

    def test_resolved_base_url_openai_empty_uses_conventional_default(self):
        config = ChatConfig(local_provider_kind="openai_compatible")
        self.assertEqual(
            config.resolved_local_base_url(), LOCAL_OPENAI_DEFAULT_BASE_URL
        )
        self.assertEqual(LOCAL_OPENAI_DEFAULT_BASE_URL, "http://localhost:8080/v1")

    def test_resolved_base_url_openai_explicit(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:1234/v1",
        )
        self.assertEqual(config.resolved_local_base_url(), "http://localhost:1234/v1")

    def test_validate_rejects_unknown_kind(self):
        config = ChatConfig(local_provider_kind="banana")
        with self.assertRaises(ValueError):
            config.validate_local_provider()
        with self.assertRaises(ValueError):
            validate_local_provider_kind(config)
        with self.assertRaises(ValueError):
            resolve_local_provider_id(config)
        with self.assertRaises(ValueError):
            resolve_local_base_url(config)
        with self.assertRaises(ValueError):
            resolve_local_api_key(config)

    def test_validate_rejects_non_string_non_none_kind(self):
        """int / list / bool などは型エラー (AttributeError 等) でなく ValueError で拒否する。"""
        for raw in (123, [1, 2], ["ollama"], True, False, {"ollama": 1}, 0):
            with self.subTest(raw=raw):
                config = ChatConfig(local_provider_kind=raw)
                with self.assertRaises(ValueError):
                    validate_local_provider_kind(config)
                with self.assertRaises(ValueError):
                    resolve_local_provider_id(config)
                with self.assertRaises(ValueError):
                    resolve_local_base_url(config)
                with self.assertRaises(ValueError):
                    resolve_local_api_key(config)
                with self.assertRaises(ValueError):
                    config.validate_local_provider()

    def test_resolve_api_key_runtime_env_only(self):
        with patch.dict(os.environ, {"LOCAL_DUMMY_KEY": "dummy-secret-value"}):
            config = ChatConfig(
                local_provider_kind="openai_compatible",
                local_api_key_env="LOCAL_DUMMY_KEY",
            )
            self.assertEqual(config.resolve_local_api_key(), "dummy-secret-value")
            # キーは設定オブジェクトへ保持されない (env 名のみ)
            self.assertEqual(config.local_api_key_env, "LOCAL_DUMMY_KEY")
            self.assertNotIn("dummy-secret-value", asdict(config).values())
            # Ollama 時は env があっても解決しない
            self.assertIsNone(
                ChatConfig(local_api_key_env="LOCAL_DUMMY_KEY").resolve_local_api_key()
            )
            # env 名未指定
            self.assertIsNone(
                ChatConfig(local_provider_kind="openai_compatible").resolve_local_api_key()
            )
            # env が未設定
            self.assertIsNone(
                ChatConfig(
                    local_provider_kind="openai_compatible",
                    local_api_key_env="LOCAL_UNSET_KEY",
                ).resolve_local_api_key()
            )

    def test_save_never_persists_resolved_key(self):
        with patch.dict(os.environ, {"LOCAL_DUMMY_KEY": "dummy-secret-value"}):
            config = ChatConfig(
                local_provider_kind="openai_compatible",
                local_api_key_env="LOCAL_DUMMY_KEY",
            )
            self.assertEqual(config.resolve_local_api_key(), "dummy-secret-value")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "chat_config.json"
                config.save(path)
                raw = path.read_text(encoding="utf-8")
        self.assertNotIn("dummy-secret-value", raw)
        self.assertIn("LOCAL_DUMMY_KEY", raw)

    def test_load_and_save_round_trip_local_provider_fields(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://localhost:1234/v1",
            local_provider_id="llama-server",
            local_api_key_env="LOCAL_ENV_NAME",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_config.json"
            config.save(path)
            loaded = ChatConfig.load(path)
        self.assertEqual(loaded.local_provider_kind, "openai_compatible")
        self.assertEqual(loaded.local_base_url, "http://localhost:1234/v1")
        self.assertEqual(loaded.local_provider_id, "llama-server")
        self.assertEqual(loaded.local_api_key_env, "LOCAL_ENV_NAME")

    def test_load_defaults_local_fields_when_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chat_config.json"
            path.write_text(json.dumps({"model": "m"}), encoding="utf-8")
            loaded = ChatConfig.load(path)
        self.assertEqual(loaded.local_provider_kind, "ollama")
        self.assertEqual(loaded.local_base_url, "")
        self.assertEqual(loaded.local_provider_id, "")
        self.assertEqual(loaded.local_api_key_env, "")


class LocalBaseUrlValidationTests(unittest.TestCase):
    """P0-2: openai_compatible の local_base_url は loopback 限定で厳格検証される"""

    def test_rejects_malformed_scheme(self):
        for url in (
            "ftp://localhost:8080/v1",
            "localhost:8080/v1",
            "localhost",
            "//localhost:8080/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_accepts_https_scheme(self):
        self.assertEqual(
            validate_local_base_url("https://localhost:8443/v1"),
            "https://localhost:8443/v1",
        )

    def test_rejects_userinfo(self):
        for url in (
            "http://user:pass@localhost:8080/v1",
            "http://user@127.0.0.1:8080/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_rejects_missing_host(self):
        for url in ("http://:8080/v1", "http:///v1"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_rejects_public_ip(self):
        for url in ("http://8.8.8.8/v1", "http://1.1.1.1:8080/v1"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_rejects_private_lan_ip(self):
        for url in (
            "http://192.168.1.5:8080/v1",
            "http://10.0.0.2/v1",
            "http://172.16.1.1/v1",
            "http://[fe80::1]:8080/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_rejects_ambiguous_hostname(self):
        for url in (
            "http://my-laptop:8080/v1",
            "http://server.local/v1",
            "http://nas:8080/v1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_local_base_url(url)

    def test_accepts_localhost_and_loopback_ips(self):
        for url in (
            "http://localhost:8080/v1",
            "http://LOCALHOST:8080/v1",
            "http://127.0.0.1:8080/v1",
            "http://127.5.6.7:8080/v1",
            "http://[::1]:8080/v1",
        ):
            with self.subTest(url=url):
                self.assertEqual(validate_local_base_url(url), url)

    def test_resolved_base_url_rejects_non_loopback(self):
        config = ChatConfig(
            local_provider_kind="openai_compatible",
            local_base_url="http://192.168.1.5:8080/v1",
        )
        with self.assertRaises(ValueError):
            resolve_local_base_url(config)
        with self.assertRaises(ValueError):
            config.resolved_local_base_url()
        with self.assertRaises(ValueError):
            config.validate_local_provider()

    def test_default_openai_base_url_is_loopback_valid(self):
        config = ChatConfig(local_provider_kind="openai_compatible")
        self.assertEqual(validate_local_base_url(LOCAL_OPENAI_DEFAULT_BASE_URL), LOCAL_OPENAI_DEFAULT_BASE_URL)
        self.assertEqual(config.resolved_local_base_url(), LOCAL_OPENAI_DEFAULT_BASE_URL)
        config.validate_local_provider()

    def test_ollama_base_url_not_restricted_by_loopback_rule(self):
        # Ollama は従来の ollama_base_url を後方互換のまま尊重し、本マイルストーンでは制限しない
        config = ChatConfig(
            local_provider_kind="ollama",
            ollama_base_url="http://192.168.1.10:11434",
            local_base_url="ftp://example.invalid/v1",
        )
        self.assertEqual(config.resolved_local_base_url(), "http://192.168.1.10:11434")
        config.validate_local_provider()

    def test_config_validator_delegates_to_shared_loopback_validator(self):
        valid = (
            "http://localhost:8080/v1",
            "http://LOCALHOST:8080/v1",
            "http://127.0.0.1:8080/v1",
            "https://[::1]:8443/v1",
        )
        invalid = (
            "http://192.168.1.5:8080/v1",
            "http://8.8.8.8/v1",
            "ftp://localhost:8080/v1",
            "http://user:pass@localhost:8080/v1",
            "http://my-laptop:8080/v1",
            "localhost",
        )
        for url in valid:
            self.assertEqual(
                validate_local_base_url(url),
                validate_loopback_openai_base_url(url),
            )
        for url in invalid:
            with self.assertRaises(ValueError):
                validate_local_base_url(url)
            with self.assertRaises(ValueError):
                validate_loopback_openai_base_url(url)
