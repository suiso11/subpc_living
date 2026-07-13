from __future__ import annotations

import queue
import unittest

from src.integrations.mcp_stdio import MCPStdioClient, MCPStdioError


class _FakeProcess:
    returncode = None

    @staticmethod
    def poll():
        return None


class MCPStdioClientTest(unittest.TestCase):
    def test_tool_error_result_raises_with_server_message(self) -> None:
        messages = queue.Queue()
        messages.put(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": "Authentication token is invalid or expired.",
                        }
                    ],
                },
            }
        )
        client = MCPStdioClient(["unused"])

        with self.assertRaisesRegex(MCPStdioError, "invalid or expired"):
            client._wait_for_response(_FakeProcess(), messages, 2, [])

    def test_tool_error_without_text_uses_fallback(self) -> None:
        messages = queue.Queue()
        messages.put(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"isError": True, "content": []},
            }
        )
        client = MCPStdioClient(["unused"])

        with self.assertRaisesRegex(MCPStdioError, "returned an error"):
            client._wait_for_response(_FakeProcess(), messages, 2, [])


if __name__ == "__main__":
    unittest.main()
