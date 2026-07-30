from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.chat.client import OllamaClient


class OllamaClientTimeoutTest(unittest.TestCase):
    def test_generate_passes_request_timeout_to_httpx(self) -> None:
        client = OllamaClient()
        response = Mock()
        response.json.return_value = {"message": {"content": "ok"}}
        fake_http = Mock()
        fake_http.post.return_value = response
        original_http = client._client
        client._client = fake_http
        try:
            result = client.generate(
                [{"role": "user", "content": "sample"}], timeout=4.5
            )
        finally:
            original_http.close()
        self.assertEqual(result, "ok")
        _, kwargs = fake_http.post.call_args
        self.assertEqual(kwargs["timeout"], 4.5)
