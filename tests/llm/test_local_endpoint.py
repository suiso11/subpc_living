"""Table-driven tests for the shared loopback-only base URL validator."""

from __future__ import annotations

import unittest

from src.llm.local_endpoint import validate_loopback_openai_base_url


class ValidateLoopbackOpenaiBaseUrlTest(unittest.TestCase):
    def test_accepts_valid_loopback_urls(self) -> None:
        valid = (
            "http://localhost:8080/v1",
            "https://localhost:8443/v1",
            "http://127.0.0.1:8080/v1",
            "https://[::1]:8443/v1",
            "http://LOCALHOST:8080/v1",
            "http://localhost/v1",
            "http://localhost:8080",
            "http://127.0.0.1:8080/v1/",
            "http://localhost:8080/v1///",
        )
        for url in valid:
            with self.subTest(url=url):
                self.assertEqual(validate_loopback_openai_base_url(url), url)

    def test_rejects_non_loopback_and_malformed_urls(self) -> None:
        invalid = (
            "http://192.168.1.5:8080/v1",
            "http://8.8.8.8/v1",
            "http://10.0.0.5:8080/v1",
            "ftp://localhost:8080/v1",
            "http://my-laptop:8080/v1",
            "localhost:8080/v1",
            "http://user:pass@localhost:8080/v1",
            "http://user@localhost:8080/v1",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_query_string(self) -> None:
        invalid = (
            "http://localhost:8080/v1?key=value",
            "http://localhost:8080/v1?apikey=sk-secret-123",
            "http://127.0.0.1:8080/v1/?x=1",
            "http://[::1]:8080/v1?token=abc",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_fragment(self) -> None:
        invalid = (
            "http://localhost:8080/v1#frag",
            "http://localhost:8080/v1#token=sk-secret-123",
            "http://127.0.0.1:8080/v1/#x",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_query_and_fragment_together(self) -> None:
        with self.assertRaises(ValueError):
            validate_loopback_openai_base_url(
                "http://localhost:8080/v1?k=v#frag"
            )

    def test_query_error_does_not_echo_query_value(self) -> None:
        secret = "sk-super-secret-query-value"
        url = f"http://localhost:8080/v1?key={secret}"
        with self.assertRaises(ValueError) as ctx:
            validate_loopback_openai_base_url(url)
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn(url, str(ctx.exception))
        self.assertNotIn("sk-", str(ctx.exception))

    def test_fragment_error_does_not_echo_fragment_value(self) -> None:
        secret = "frag-secret-value"
        url = f"http://localhost:8080/v1#{secret}"
        with self.assertRaises(ValueError) as ctx:
            validate_loopback_openai_base_url(url)
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn(url, str(ctx.exception))

    def test_query_rejected_even_for_loopback_ip(self) -> None:
        for url in (
            "http://127.0.0.1:8080/v1?x=1",
            "http://[::1]:8080/v1?x=1",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_bare_query_delimiter_with_empty_query(self) -> None:
        invalid = (
            "http://localhost:8080/v1?",
            "http://localhost:8080/v1/?",
            "http://127.0.0.1:8080/v1?",
            "http://[::1]:8080/v1?",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_bare_fragment_delimiter_with_empty_fragment(self) -> None:
        invalid = (
            "http://localhost:8080/v1#",
            "http://localhost:8080/v1/#",
            "http://127.0.0.1:8080/v1#",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_rejects_bare_delimiters_combined_and_repeated(self) -> None:
        invalid = (
            "http://localhost:8080/v1?#",
            "http://localhost:8080/v1?x#",
            "http://localhost:8080/v1??",
            "http://localhost:8080/v1##",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_openai_base_url(url)

    def test_accepts_encoded_delimiters_in_path(self) -> None:
        valid = (
            "http://localhost:8080/v1%3F",
            "http://localhost:8080/v1%23",
            "http://localhost:8080/v1%3Fpath%23",
            "http://127.0.0.1:8080/v1%3F",
            "http://[::1]:8080/v1%3F",
        )
        for url in valid:
            with self.subTest(url=url):
                self.assertEqual(validate_loopback_openai_base_url(url), url)

    def test_error_messages_are_fixed_and_category_only(self) -> None:
        cases = (
            ("http://[::1", "local endpoint URL is malformed"),
            ("ftp://localhost:8080/v1",
             "local endpoint URL scheme must be http or https"),
            ("localhost:8080/v1",
             "local endpoint URL scheme must be http or https"),
            ("http://user:pass@localhost:8080/v1",
             "local endpoint URL must not include userinfo (user:password@)"),
            ("http://localhost:8080/v1?",
             "local endpoint URL must not include a query string "
             "(query values may contain credentials)"),
            ("http://localhost:8080/v1#",
             "local endpoint URL must not include a fragment"),
            ("http://my-laptop:8080/v1",
             "local endpoint URL host must be 'localhost' or a loopback IP"),
            ("http://192.168.1.5:8080/v1",
             "local endpoint URL host must be loopback"),
        )
        for url, expected in cases:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_loopback_openai_base_url(url)
                self.assertEqual(str(ctx.exception), expected)

    def test_validation_errors_never_echo_canary_url_parts(self) -> None:
        canary = "canary-secret-42"
        invalid = (
            f"http://user:{canary}@localhost:8080/v1",
            f"http://{canary}:8080/v1",
            f"http://localhost:8080/v1?key={canary}",
            f"http://localhost:8080/v1?",
            f"http://localhost:8080/v1#{canary}",
            f"http://10.1.{canary}.5:8080/v1",
            "http://[::1",
        )
        for url in invalid:
            with self.subTest(url=url):
                with self.assertRaises(ValueError) as ctx:
                    validate_loopback_openai_base_url(url)
                message = str(ctx.exception)
                self.assertNotIn(canary, message)
                self.assertNotIn(url, message)
                self.assertNotIn("http://", message)
                self.assertNotIn("IPv6", message)


if __name__ == "__main__":
    unittest.main()