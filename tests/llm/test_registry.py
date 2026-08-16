from __future__ import annotations

import unittest

from src.llm.providers.fake import FakeProvider
from src.llm.registry import (
    DuplicateProviderError,
    ProviderRegistry,
    UnknownProviderError,
)


class ProviderRegistryTest(unittest.TestCase):
    def test_register_and_get_preserve_metadata(self) -> None:
        registry = ProviderRegistry()
        provider = FakeProvider(model="fast")

        entry = registry.register(
            "local-fast", provider, local=True, profiles=("voice_fast",)
        )

        self.assertIs(registry.get("local-fast"), entry)
        self.assertTrue(entry.local)
        self.assertEqual(entry.profiles, ("voice_fast",))
        self.assertEqual(len(registry), 1)
        self.assertIn("local-fast", registry)

    def test_empty_duplicate_and_unknown_ids_are_rejected(self) -> None:
        registry = ProviderRegistry()
        provider = FakeProvider()

        with self.assertRaises(ValueError):
            registry.register("  ", provider, local=True)
        registry.register("local", provider, local=True)
        with self.assertRaises(DuplicateProviderError):
            registry.register("local", FakeProvider(), local=True)
        with self.assertRaises(UnknownProviderError):
            registry.get("missing")

    def test_close_closes_shared_provider_once(self) -> None:
        class CountingProvider(FakeProvider):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            def close(self) -> None:
                self.close_count += 1
                super().close()

        registry = ProviderRegistry()
        provider = CountingProvider()
        registry.register("first", provider, local=True)
        registry.register("alias", provider, local=True)

        registry.close()

        self.assertEqual(provider.close_count, 1)

    def test_close_attempts_all_providers_before_raising_first_error(self) -> None:
        class BrokenProvider(FakeProvider):
            def close(self) -> None:
                raise RuntimeError("close failed")

        registry = ProviderRegistry()
        healthy = FakeProvider()
        registry.register("broken", BrokenProvider(), local=True)
        registry.register("healthy", healthy, local=True)

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            registry.close()

        self.assertTrue(healthy.closed)


if __name__ == "__main__":
    unittest.main()
