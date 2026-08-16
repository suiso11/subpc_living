from __future__ import annotations

from dataclasses import dataclass
import unittest

from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry, UnknownProviderError
from src.llm.routing.contracts import NoRouteError, PrivacyMode
from src.llm.routing.static import StaticRouter


@dataclass(frozen=True)
class Request:
    profile: str = "chat_auto"
    privacy: PrivacyMode = "local_preferred"
    requested_provider: str | None = None
    allow_cloud: bool = False


class StaticRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ProviderRegistry()
        self.fast = FakeProvider(model="fast-model")
        self.strong = FakeProvider(model="strong-model")
        self.cloud = FakeProvider(model="cloud-model")
        self.registry.register("local-fast", self.fast, local=True)
        self.registry.register("local-strong", self.strong, local=True)
        self.registry.register("cloud", self.cloud, local=False)

    def make_router(self) -> StaticRouter:
        return StaticRouter(
            self.registry,
            default_provider_id="local-strong",
            profile_routes={"voice_fast": "local-fast", "deep": "cloud"},
            fallback_provider_ids=("local-strong", "local-fast"),
        )

    def test_default_and_profile_routes_are_deterministic(self) -> None:
        router = self.make_router()

        default = router.route(Request())
        voice = router.route(Request(profile="voice_fast"))

        self.assertEqual(default.provider_id, "local-strong")
        self.assertEqual(default.reason, "default route")
        self.assertEqual(voice.provider_id, "local-fast")
        self.assertIn("profile route", voice.reason)

    def test_explicit_provider_is_used_when_allowed(self) -> None:
        decision = self.make_router().route(
            Request(requested_provider="local-fast", privacy="local_only")
        )

        self.assertEqual(decision.provider_id, "local-fast")
        self.assertTrue(decision.local)
        self.assertEqual(decision.model, "fast-model")
        self.assertEqual(decision.reason, "explicit provider request")

    def test_cloud_requires_both_privacy_and_explicit_permission(self) -> None:
        router = self.make_router()

        denied_by_privacy = router.route(
            Request(profile="deep", privacy="local_preferred", allow_cloud=True)
        )
        denied_by_flag = router.route(
            Request(profile="deep", privacy="cloud_allowed", allow_cloud=False)
        )
        allowed = router.route(
            Request(profile="deep", privacy="cloud_allowed", allow_cloud=True)
        )

        self.assertEqual(denied_by_privacy.provider_id, "local-strong")
        self.assertIn("cloud-disallowed", denied_by_privacy.reason)
        self.assertEqual(denied_by_flag.provider_id, "local-strong")
        self.assertEqual(allowed.provider_id, "cloud")
        self.assertFalse(allowed.local)

    def test_unavailable_primary_uses_configured_fallback(self) -> None:
        self.fast.available = False

        decision = self.make_router().route(Request(profile="voice_fast"))

        self.assertEqual(decision.provider_id, "local-strong")
        self.assertIn("local-fast=unavailable", decision.reason)

    def test_no_route_and_unknown_explicit_provider_fail_closed(self) -> None:
        self.fast.available = False
        self.strong.available = False
        router = self.make_router()

        with self.assertRaises(NoRouteError):
            router.route(Request(privacy="local_only"))
        with self.assertRaises(UnknownProviderError):
            router.route(Request(requested_provider="typo"))

    def test_router_configuration_is_validated(self) -> None:
        with self.assertRaises(UnknownProviderError):
            StaticRouter(self.registry, default_provider_id="missing")
        with self.assertRaises(ValueError):
            StaticRouter(
                self.registry,
                default_provider_id="local-strong",
                fallback_provider_ids=("local-fast", "local-fast"),
            )


if __name__ == "__main__":
    unittest.main()
