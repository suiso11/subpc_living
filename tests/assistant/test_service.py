from collections.abc import Generator
import unittest

from src.assistant import (
    AssistantError,
    AssistantGenerationError,
    AssistantRequest,
    AssistantService,
)
from src.llm.errors import ProviderRequestError
from src.llm.providers.fake import FakeProvider
from src.llm.registry import ProviderRegistry
from src.llm.routing.contracts import NoRouteError, RouteDecision


class FixedRouter:
    def __init__(self, decision: RouteDecision | None = None) -> None:
        self.decision = decision

    def route(self, request: AssistantRequest) -> RouteDecision:
        if self.decision is None:
            raise NoRouteError("no route")
        return self.decision


class FailingProvider(FakeProvider):
    def generate(self, messages, **options):
        self.calls.append(
            {"kind": "generate", "messages": list(messages), "options": options}
        )
        raise ProviderRequestError("failing", "generate", "planned failure")


class FailingStreamProvider(FakeProvider):
    def __init__(self, *, fail_after_token: bool) -> None:
        super().__init__(model="failing-stream")
        self.fail_after_token = fail_after_token

    def generate_stream(self, messages, **options) -> Generator[str, None, None]:
        self.calls.append(
            {"kind": "generate_stream", "messages": list(messages), "options": options}
        )
        if self.fail_after_token:
            yield "partial"
        raise ProviderRequestError("failing", "generate_stream", "planned failure")


class ClosingFailingStreamIterator:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> str:
        raise ProviderRequestError("failing", "generate_stream", "planned failure")

    def close(self) -> None:
        self.closed = True


class ClosingFailingStreamProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(model="closing-failing-stream")
        self.iterator = ClosingFailingStreamIterator()

    def generate_stream(self, messages, **options):
        self.calls.append(
            {"kind": "generate_stream", "messages": list(messages), "options": options}
        )
        return self.iterator


class AssistantServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = ProviderRegistry()
        self.primary = FakeProvider(
            "hello", model="local-model", stats={"eval_count": 7}
        )
        self.registry.register("primary", self.primary, local=True)
        self.request = AssistantRequest(
            text="hello", conversation_id="conversation", channel="web"
        )
        self.messages = [{"role": "user", "content": "hello"}]

    @staticmethod
    def decision(*fallbacks: str) -> RouteDecision:
        return RouteDecision(
            provider_id="primary",
            model="router-model",
            local=True,
            reason="test route",
            fallback_provider_ids=tuple(fallbacks),
        )

    def service(self, decision: RouteDecision | None = None, **kwargs):
        return AssistantService(
            self.registry, FixedRouter(decision or self.decision()), **kwargs
        )

    def test_normal_response_uses_actual_provider_metadata_and_stats(self) -> None:
        response = self.service().generate(self.request, self.messages)

        self.assertEqual(response.text, "hello")
        self.assertEqual(response.route.provider_id, "primary")
        self.assertEqual(response.route.model, "local-model")
        self.assertTrue(response.route.reason)
        self.assertGreaterEqual(response.latency_ms, 0)
        self.assertEqual(response.stats, {"eval_count": 7})

    def test_default_options_and_messages_are_passed_to_provider(self) -> None:
        self.service().generate(self.request, self.messages)

        call = self.primary.calls[0]
        self.assertEqual(call["messages"], self.messages)
        self.assertEqual(
            call["options"],
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 8192,
                "num_predict": None,
            },
        )

    def test_caller_messages_are_not_modified(self) -> None:
        original = [dict(message) for message in self.messages]

        self.service().generate(self.request, self.messages)

        self.assertEqual(self.messages, original)
        self.assertIsNot(self.primary.calls[0]["messages"], self.messages)

    def test_stream_exposes_tokens_and_response_only_after_completion(self) -> None:
        provider = FakeProvider(
            stream_chunks=("hel", "lo"), model="stream-model", stats={"eval_count": 2}
        )
        registry = ProviderRegistry()
        registry.register("primary", provider, local=True)
        result = AssistantService(registry, FixedRouter(self.decision())).generate_stream(
            self.request, self.messages
        )

        with self.assertRaisesRegex(AssistantError, "stream not finished"):
            _ = result.response
        self.assertEqual(list(result), ["hel", "lo"])
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.response.route.model, "stream-model")
        self.assertEqual(result.response.stats, {"eval_count": 2})

    def test_generation_error_uses_only_defined_fallback(self) -> None:
        failing = FailingProvider(model="failed")
        fallback = FakeProvider("fallback", model="fallback-model")
        unrelated = FakeProvider("unrelated")
        registry = ProviderRegistry()
        registry.register("primary", failing, local=True)
        registry.register("fallback", fallback, local=True)
        registry.register("unrelated", unrelated, local=True)
        service = AssistantService(
            registry, FixedRouter(self.decision("fallback", "fallback"))
        )

        response = service.generate(self.request, self.messages)

        self.assertEqual(response.text, "fallback")
        self.assertEqual(response.route.provider_id, "fallback")
        self.assertIn("fallback after", response.route.reason)
        self.assertIn("primary=ProviderRequestError", response.route.reason)
        self.assertEqual(len(failing.calls), 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertEqual(unrelated.calls, [])

    def test_all_candidates_fail_with_attempts_and_last_cause(self) -> None:
        primary = FailingProvider(model="failed-primary")
        fallback = FailingProvider(model="failed-fallback")
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        service = AssistantService(registry, FixedRouter(self.decision("fallback")))

        with self.assertRaises(AssistantGenerationError) as raised:
            service.generate(self.request, self.messages)

        self.assertEqual(
            [provider_id for provider_id, _ in raised.exception.attempts],
            ["primary", "fallback"],
        )
        self.assertIsInstance(raised.exception.__cause__, ProviderRequestError)
        self.assertIn("failing.generate", str(raised.exception.__cause__))

        message = str(raised.exception)
        self.assertIn("all provider candidates failed", message)
        self.assertIn("primary=ProviderRequestError", message)
        self.assertIn("fallback=ProviderRequestError", message)
        self.assertIn("failing.generate: planned failure", message)

    def test_non_local_provider_is_never_called(self) -> None:
        cloud = FakeProvider("cloud")
        fallback = FakeProvider("local")
        registry = ProviderRegistry()
        registry.register("primary", cloud, local=False)
        registry.register("fallback", fallback, local=True)
        service = AssistantService(registry, FixedRouter(self.decision("fallback")))

        response = service.generate(self.request, self.messages)

        self.assertEqual(response.text, "local")
        self.assertEqual(cloud.calls, [])
        self.assertIn("non-local provider rejected", response.route.reason)

    def test_only_non_local_candidate_raises_generation_error(self) -> None:
        cloud = FakeProvider("cloud")
        registry = ProviderRegistry()
        registry.register("primary", cloud, local=False)
        service = AssistantService(registry, FixedRouter(self.decision()))

        with self.assertRaises(AssistantGenerationError) as raised:
            service.generate(self.request, self.messages)

        self.assertEqual(cloud.calls, [])
        self.assertEqual(raised.exception.attempts[0][0], "primary")

    def test_stream_non_local_provider_is_never_called(self) -> None:
        cloud = FakeProvider(stream_chunks=("cloud",))
        fallback = FakeProvider(stream_chunks=("local",))
        registry = ProviderRegistry()
        registry.register("primary", cloud, local=False)
        registry.register("fallback", fallback, local=True)
        result = AssistantService(
            registry, FixedRouter(self.decision("fallback"))
        ).generate_stream(self.request, self.messages)

        self.assertEqual(list(result), ["local"])
        self.assertEqual(cloud.calls, [])
        self.assertEqual(result.response.route.provider_id, "fallback")

    def test_cloud_allowed_request_still_rejects_non_local_provider(self) -> None:
        request = AssistantRequest(
            text="hello",
            conversation_id="conversation",
            channel="web",
            privacy="cloud_allowed",
            allow_cloud=True,
        )
        cloud = FakeProvider("cloud")
        local = FakeProvider("local")
        registry = ProviderRegistry()
        registry.register("primary", cloud, local=False)
        registry.register("fallback", local, local=True)
        service = AssistantService(
            registry, FixedRouter(self.decision("fallback"))
        )

        response = service.generate(request, self.messages)

        self.assertEqual(response.text, "local")
        self.assertEqual(cloud.calls, [])

        cloud_only_registry = ProviderRegistry()
        cloud_only = FakeProvider("cloud")
        cloud_only_registry.register("primary", cloud_only, local=False)
        cloud_only_service = AssistantService(
            cloud_only_registry, FixedRouter(self.decision())
        )
        with self.assertRaises(AssistantGenerationError):
            cloud_only_service.generate(request, self.messages)
        self.assertEqual(cloud_only.calls, [])

    def test_unavailable_provider_is_skipped(self) -> None:
        primary = FakeProvider(available=False)
        fallback = FakeProvider("available")
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        service = AssistantService(registry, FixedRouter(self.decision("fallback")))

        response = service.generate(self.request, self.messages)

        self.assertEqual(response.text, "available")
        self.assertEqual(primary.calls, [])
        self.assertIn("primary=unavailable", response.route.reason)

    def test_stream_failure_before_first_token_uses_fallback(self) -> None:
        primary = FailingStreamProvider(fail_after_token=False)
        fallback = FakeProvider(stream_chunks=("safe",), model="fallback")
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        service = AssistantService(registry, FixedRouter(self.decision("fallback")))

        result = service.generate_stream(self.request, self.messages)

        self.assertEqual(list(result), ["safe"])
        self.assertEqual(result.response.route.provider_id, "fallback")

    def test_failed_stream_iterator_is_closed_before_fallback(self) -> None:
        primary = ClosingFailingStreamProvider()
        fallback = FakeProvider(stream_chunks=("safe",), model="fallback")
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        result = AssistantService(
            registry, FixedRouter(self.decision("fallback"))
        ).generate_stream(self.request, self.messages)

        self.assertEqual(list(result), ["safe"])
        self.assertTrue(primary.iterator.closed)

    def test_stream_does_not_modify_caller_messages(self) -> None:
        original = [dict(message) for message in self.messages]

        result = self.service().generate_stream(self.request, self.messages)
        list(result)

        self.assertEqual(len(self.messages), len(original))
        self.assertEqual(self.messages, original)

    def test_duplicate_candidates_are_called_once_for_sync_and_stream(self) -> None:
        primary = FakeProvider("winner", stream_chunks=("winner",))
        fallback = FakeProvider("unused", stream_chunks=("unused",))
        registry = ProviderRegistry()
        registry.register("p", primary, local=True)
        registry.register("f", fallback, local=True)
        decision = RouteDecision(
            provider_id="p",
            model="router-model",
            local=True,
            reason="duplicate route",
            fallback_provider_ids=("p", "f"),
        )
        service = AssistantService(registry, FixedRouter(decision))

        self.assertEqual(service.generate(self.request, self.messages).text, "winner")
        self.assertEqual(
            list(service.generate_stream(self.request, self.messages)), ["winner"]
        )
        self.assertEqual(
            [call["kind"] for call in primary.calls],
            ["generate", "generate_stream"],
        )
        self.assertEqual(fallback.calls, [])

    def test_requested_model_is_ignored(self) -> None:
        request = AssistantRequest(
            text="hello",
            conversation_id="conversation",
            channel="web",
            requested_model="other-model",
        )

        response = self.service().generate(request, self.messages)

        self.assertEqual(response.route.model, "local-model")

    def test_stream_failure_after_token_does_not_fallback(self) -> None:
        primary = FailingStreamProvider(fail_after_token=True)
        fallback = FakeProvider(stream_chunks=("duplicate",))
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        result = AssistantService(
            registry, FixedRouter(self.decision("fallback"))
        ).generate_stream(self.request, self.messages)

        iterator = iter(result)
        self.assertEqual(next(iterator), "partial")
        with self.assertRaises(ProviderRequestError):
            next(iterator)
        self.assertEqual(fallback.calls, [])
        with self.assertRaisesRegex(AssistantError, "stream not finished"):
            _ = result.response

    def test_no_route_error_is_not_wrapped(self) -> None:
        service = AssistantService(self.registry, FixedRouter())

        with self.assertRaises(NoRouteError):
            service.generate(self.request, self.messages)
        with self.assertRaises(NoRouteError):
            service.generate_stream(self.request, self.messages)

    def test_injected_clock_makes_latency_deterministic(self) -> None:
        ticks = iter((10.0, 10.125))

        response = self.service(clock=lambda: next(ticks)).generate(
            self.request, self.messages
        )

        self.assertEqual(response.latency_ms, 125)

    def test_stream_latency_covers_all_provider_attempts(self) -> None:
        primary = FailingStreamProvider(fail_after_token=False)
        fallback = FakeProvider(stream_chunks=("done",))
        registry = ProviderRegistry()
        registry.register("primary", primary, local=True)
        registry.register("fallback", fallback, local=True)
        ticks = iter((3.0, 3.25))
        result = AssistantService(
            registry,
            FixedRouter(self.decision("fallback")),
            clock=lambda: next(ticks),
        ).generate_stream(self.request, self.messages)

        self.assertEqual(list(result), ["done"])
        self.assertEqual(result.response.latency_ms, 250)


if __name__ == "__main__":
    unittest.main()
