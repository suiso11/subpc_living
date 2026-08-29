"""Low-level loopback-only base URL validation for OpenAI-compatible endpoints.

Shared by the chat configuration boundary (``src.chat.config``) and the
``LocalOpenAICompatibleProvider`` constructor so that non-loopback destinations
are rejected even when the configuration layer is bypassed (defense in depth).
No dependency on the chat module, keeping the provider free of an
``llm -> chat`` edge.

All rejection messages are fixed, category-only strings: they never echo the
input URL, userinfo, host string, query, fragment, or the underlying parse
exception, so credentials carried in a rejected URL cannot leak into logs.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def validate_loopback_openai_base_url(url: str) -> str:
    """Validate an OpenAI-compatible ``/v1`` base URL and return it unchanged.

    The destination is restricted to this machine's loopback so that a
    ``local=True`` provider (which skips cloud approval/redaction semantics)
    cannot be pointed at arbitrary LAN / public / ambiguous hosts. Rules:

    - scheme is ``http`` or ``https``
    - host is required (no empty / bare ``//host`` forms)
    - no userinfo (``user:pass@``)
    - no query string and no fragment; a raw ``?`` or ``#`` delimiter is
      rejected even when the query or fragment content is empty (ambiguous,
      and may carry credentials that would otherwise be leaked into probe/join
      URLs). Percent-encoded ``%3F`` / ``%23`` in the path are preserved and
      remain valid
    - host is ``localhost`` (case-insensitive) or an IP for which
      ``ipaddress.is_loopback`` is true; other hostnames are ambiguous and
      rejected

    Every ``ValueError`` message is a fixed, category-only string that never
    echoes the offending URL or any of its parts.

    Raises ``ValueError`` on any violation. IPv6 literals must be bracketed
    (``http://[::1]:8080/v1``) so that ``urlsplit`` extracts the host cleanly.
    """
    if "?" in url:
        raise ValueError(
            "local endpoint URL must not include a query string "
            "(query values may contain credentials)"
        )
    if "#" in url:
        raise ValueError("local endpoint URL must not include a fragment")
    try:
        parts = urlsplit(url)
    except ValueError:
        raise ValueError("local endpoint URL is malformed") from None
    if parts.scheme not in ("http", "https"):
        raise ValueError("local endpoint URL scheme must be http or https")
    host = parts.hostname
    if not host:
        raise ValueError("local endpoint URL must include a host")
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            "local endpoint URL must not include userinfo (user:password@)"
        )
    if host.lower() == "localhost":
        return url
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise ValueError(
            "local endpoint URL host must be 'localhost' or a loopback IP"
        ) from None
    if not ip.is_loopback:
        raise ValueError("local endpoint URL host must be loopback")
    return url