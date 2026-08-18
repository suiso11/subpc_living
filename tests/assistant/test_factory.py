from dataclasses import dataclass
import unittest

from src.assistant.contracts import AssistantRequest
from src.assistant.factory import build_local_service
from src.llm.providers.fake import FakeProvider


@dataclass
class FactoryConfig:
    ollama_base_url: str = "http://unused.invalid"
    model: str = "unused-model"
    temperature: float = 0.23
    top_p: float = 0.81
    top_k: int = 17
    repeat_penalty: float = 1.37
    num_ctx: int = 3456
    num_predict: int | None = 123


class BuildLocalServiceTest(unittest.TestCase):
    @staticmethod
    def request() -> AssistantRequest:
        return AssistantRequest(
            text="質問",
            conversation_id="factory-test",
            channel="internal",
        )

    def test_injected_provider_receives_configured_options(self) -> None:
        config = FactoryConfig()
        provider = FakeProvider(response="応答")

        service, registry = build_local_service(config, provider=provider)
        response = service.generate(
            self.request(), [{"role": "user", "content": "質問"}]
        )

        self.assertEqual(response.text, "応答")
        self.assertEqual(
            provider.calls[0]["options"],
            {
                "temperature": 0.23,
                "top_p": 0.81,
                "top_k": 17,
                "repeat_penalty": 1.37,
                "num_ctx": 3456,
                "num_predict": 123,
            },
        )
        entry = registry.get("ollama")
        self.assertIs(entry.provider, provider)
        self.assertTrue(entry.local)

    def test_custom_provider_id_is_registered_and_used_as_default(self) -> None:
        provider = FakeProvider(response="custom response")

        service, registry = build_local_service(
            FactoryConfig(), provider=provider, provider_id="local-custom"
        )
        response = service.generate(
            self.request(), [{"role": "user", "content": "質問"}]
        )

        self.assertIs(registry.get("local-custom").provider, provider)
        self.assertEqual(response.text, "custom response")
        self.assertEqual(response.route.provider_id, "local-custom")
        self.assertEqual(response.route.reason, "default route")


if __name__ == "__main__":
    unittest.main()
