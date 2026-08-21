import unittest
from dataclasses import replace

from src.assistant.factory import build_assistant_service, build_local_service
from src.assistant.contracts import AssistantRequest
from src.context.contracts import ContextBlock
from src.llm.approval import ApprovalGate
from src.llm.cloud_config import CloudConfig


class _FakeConfig:
    ollama_base_url = "http://localhost:11434"
    model = "local-model"
    temperature = 0.7
    top_p = 0.9
    top_k = 40
    repeat_penalty = 1.1
    num_ctx = 8192
    num_predict = None
    system_prompt = "You are a helpful assistant."
    model_prompt_overrides: dict = {}

    def effective_system_prompt(self, model=None) -> str:
        target = model if model is not None else self.model
        if target and target in self.model_prompt_overrides:
            return self.model_prompt_overrides[target]
        return self.system_prompt


def _request(request_id="r1"):
    return AssistantRequest(
        text="hi", conversation_id="c1", channel="web",
        privacy="cloud_allowed", allow_cloud=True, request_id=request_id,
    )


def _blocks():
    return [
        ContextBlock(source="calendar", content="calendar secret", sensitivity="personal", local_only=True),
        ContextBlock(source="public_note", content="public info", sensitivity="public", local_only=False),
    ]


class FactoryCloudWiringTest(unittest.TestCase):
    def test_local_service_has_no_cloud(self):
        service, reg = build_local_service(_FakeConfig())
        self.assertNotIn("cloud", reg)

    def test_default_no_cloud(self):
        service, reg, bridge = build_assistant_service(_FakeConfig())
        self.assertIsNone(bridge)
        self.assertNotIn("cloud", reg)

    def test_opt_in_enables_cloud(self):
        cfg = CloudConfig(enabled=True, model="cloud-m", provider_id="cloud")
        service, reg, bridge = build_assistant_service(_FakeConfig(), cloud_config=cfg)
        self.assertIsNotNone(bridge)
        self.assertIn("cloud", reg)
        self.assertFalse(reg.get("cloud").local)

    def test_cloud_send_via_factory_bridge(self):
        gate = ApprovalGate()
        gate.approve("r1")
        cfg = CloudConfig(enabled=True, model="cloud-m", provider_id="cloud")
        service, reg, bridge = build_assistant_service(
            _FakeConfig(), cloud_config=cfg, approval=gate
        )
        resp = bridge.send(_request(), _blocks())
        self.assertFalse(resp.route.local)
        self.assertEqual(resp.text, "cloud response")
        sent = reg.get("cloud").provider.sent_payloads[0]
        self.assertNotIn("calendar", str(sent))
        self.assertIn("public info", str(sent))

    def test_bridge_wired_to_service(self):
        """Factory should wire bridge into service via set_cloud_bridge."""
        gate = ApprovalGate()
        cfg = CloudConfig(enabled=True, model="cloud-m", provider_id="cloud")
        service, reg, bridge = build_assistant_service(
            _FakeConfig(), cloud_config=cfg, approval=gate
        )
        self.assertIsNotNone(bridge)
        self.assertIs(service._cloud_bridge, bridge)

    def test_enabled_without_key_is_invalid(self):
        cfg = CloudConfig(enabled=True, model="m", api_key_env="NOPE")
        with self.assertRaises(Exception):
            build_assistant_service(_FakeConfig(), cloud_config=cfg)

    def test_bridge_base_system_from_config(self):
        """Factory should wire config.effective_system_prompt() into bridge._base_system."""
        gate = ApprovalGate()
        cfg = CloudConfig(enabled=True, model="cloud-m", provider_id="cloud")
        service, reg, bridge = build_assistant_service(
            _FakeConfig(), cloud_config=cfg, approval=gate
        )
        self.assertIsNotNone(bridge)
        self.assertEqual(bridge._base_system, "You are a helpful assistant.")


if __name__ == "__main__":
    unittest.main()
