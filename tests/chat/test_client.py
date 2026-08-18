from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, Mock

from src.chat.client import OllamaClient, OllamaResponseError


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


class OllamaClientGenerateResponseErrorTest(unittest.TestCase):
    def _client_with_response(self, response: Mock) -> OllamaClient:
        client = OllamaClient()
        fake_http = Mock()
        fake_http.post.return_value = response
        original_http = client._client
        client._client = fake_http
        self.addCleanup(original_http.close)
        return client

    def test_invalid_json_is_wrapped_as_response_error(self) -> None:
        response = Mock()
        response.json.side_effect = json.JSONDecodeError("Expecting value", "x", 0)
        client = self._client_with_response(response)

        with self.assertRaises(OllamaResponseError) as raised:
            client.generate([{"role": "user", "content": "hi"}])

        self.assertIsInstance(raised.exception.__cause__, json.JSONDecodeError)

    def test_missing_message_key_is_wrapped_as_response_error(self) -> None:
        response = Mock()
        response.json.return_value = {"unexpected": True}
        client = self._client_with_response(response)

        with self.assertRaises(OllamaResponseError) as raised:
            client.generate([{"role": "user", "content": "hi"}])

        self.assertIsInstance(raised.exception.__cause__, KeyError)

    def test_type_mismatch_is_wrapped_as_response_error(self) -> None:
        response = Mock()
        response.json.return_value = {"message": None}
        client = self._client_with_response(response)

        with self.assertRaises(OllamaResponseError) as raised:
            client.generate([{"role": "user", "content": "hi"}])

        self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_plain_value_error_is_not_wrapped(self) -> None:
        response = Mock()
        response.json.side_effect = ValueError("boom")
        client = self._client_with_response(response)

        with self.assertRaisesRegex(ValueError, "boom"):
            client.generate([{"role": "user", "content": "hi"}])


class OllamaClientStreamResponseErrorTest(unittest.TestCase):
    def _client_with_lines(self, lines: list[str]) -> OllamaClient:
        client = OllamaClient()
        response = Mock()
        response.iter_lines.return_value = iter(lines)
        fake_http = MagicMock()
        fake_http.stream.return_value.__enter__.return_value = response
        original_http = client._client
        client._client = fake_http
        self.addCleanup(original_http.close)
        return client

    def test_valid_chunks_are_yielded_with_stats(self) -> None:
        client = self._client_with_lines(
            [
                '{"done": false, "message": {"content": "he"}}',
                '{"done": false, "message": {"content": "llo"}}',
                '{"done": true, "message": {"content": ""}, "eval_count": 2}',
            ]
        )

        tokens = list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertEqual(tokens, ["he", "llo"])
        self.assertEqual(client.last_stats["eval_count"], 2)

    def test_invalid_json_chunk_is_wrapped_as_response_error(self) -> None:
        client = self._client_with_lines(["not json"])

        with self.assertRaises(OllamaResponseError) as raised:
            list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertIsInstance(raised.exception.__cause__, json.JSONDecodeError)

    def test_missing_message_key_chunk_is_wrapped_as_response_error(self) -> None:
        client = self._client_with_lines(['{"done": false}'])

        with self.assertRaises(OllamaResponseError) as raised:
            list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertIsInstance(raised.exception.__cause__, KeyError)

    def test_non_dict_chunk_is_wrapped_as_response_error(self) -> None:
        client = self._client_with_lines(["[1, 2]"])

        with self.assertRaises(OllamaResponseError) as raised:
            list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_non_str_message_content_chunk_is_wrapped_as_response_error(self) -> None:
        client = self._client_with_lines(
            ['{"done": false, "message": {"content": 123}}']
        )

        with self.assertRaises(OllamaResponseError) as raised:
            list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertIsInstance(raised.exception.__cause__, TypeError)

    def test_non_bool_done_chunk_is_wrapped_as_response_error(self) -> None:
        client = self._client_with_lines(
            ['{"done": "false", "message": {"content": "he"}}']
        )

        with self.assertRaises(OllamaResponseError) as raised:
            list(client.generate_stream([{"role": "user", "content": "hi"}]))

        self.assertIsInstance(raised.exception.__cause__, TypeError)


if __name__ == "__main__":
    unittest.main()
