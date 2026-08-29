import os
import unittest

from src.llm.cloud_config import CloudConfig, CloudConfigError
from src.llm.errors import ProviderRequestError
from src.llm.providers.cloud import FakeCloudProvider
from src.llm.registry import ProviderRegistry


class CloudConfigTest(unittest.TestCase):
    def test_disabled_by_default(self):
        cfg = CloudConfig()
        self.assertFalse(cfg.enabled)
        self.assertIsNone(cfg.resolve_api_key())

    def test_no_key_when_disabled(self):
        cfg = CloudConfig(enabled=False, api_key_env="MY_KEY")
        self.assertIsNone(cfg.resolve_api_key())

    def test_resolve_key_only_when_enabled(self):
        os.environ["MY_CLOUD_KEY"] = "secret"
        try:
            cfg = CloudConfig(enabled=True, api_key_env="MY_CLOUD_KEY", model="m")
            self.assertEqual(cfg.resolve_api_key(), "secret")
            disabled = CloudConfig(enabled=False, api_key_env="MY_CLOUD_KEY")
            self.assertIsNone(disabled.resolve_api_key())
        finally:
            del os.environ["MY_CLOUD_KEY"]

    def test_validate_ok_when_disabled(self):
        CloudConfig().validate()  # no raise

    def test_validate_requires_model(self):
        with self.assertRaises(CloudConfigError):
            CloudConfig(enabled=True).validate()

    def test_validate_requires_key(self):
        with self.assertRaises(CloudConfigError):
            CloudConfig(enabled=True, model="m", api_key_env="MISSING_KEY").validate()

    def test_default_provider_kind_is_fake(self):
        cfg = CloudConfig()
        self.assertEqual(cfg.provider_kind, "fake")

    def test_validate_openai_compatible_requires_key(self):
        with self.assertRaises(CloudConfigError):
            CloudConfig(
                enabled=True,
                model="gpt-4",
                provider_kind="openai_compatible",
            ).validate()

    def test_validate_openai_compatible_passes_with_key(self):
        os.environ["TEST_OPENAI_KEY"] = "test-key-value"
        try:
            cfg = CloudConfig(
                enabled=True,
                model="gpt-4",
                provider_kind="openai_compatible",
                api_key_env="TEST_OPENAI_KEY",
            )
            cfg.validate()  # should not raise
        finally:
            del os.environ["TEST_OPENAI_KEY"]


class FakeCloudProviderTest(unittest.TestCase):
    def test_generate_records_payload(self):
        p = FakeCloudProvider(model="cloud-x")
        out = p.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(out, "cloud response")
        self.assertEqual(len(p.sent_payloads), 1)
        self.assertEqual(p.sent_payloads[0][0]["content"], "hi")

    def test_generate_stream_yields(self):
        p = FakeCloudProvider(model="cloud-x")
        chunks = list(p.generate_stream([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["cloud ", "response"])

    def test_fail_raises(self):
        p = FakeCloudProvider(fail=True)
        with self.assertRaises(ProviderRequestError):
            p.generate([{"role": "user", "content": "hi"}])

    def test_registry_marks_non_local(self):
        reg = ProviderRegistry()
        reg.register("cloud", FakeCloudProvider(), local=False)
        self.assertFalse(reg.get("cloud").local)

    def test_available_and_close(self):
        p = FakeCloudProvider()
        self.assertTrue(p.is_available())
        p.close()
        self.assertFalse(p.is_available())


if __name__ == "__main__":
    unittest.main()
