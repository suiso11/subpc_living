"""High-confidence credential detection shared by extraction and persistence."""
from __future__ import annotations

import re
from typing import Any

_PATTERNS = (
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{24,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"hooks\.slack\.com/services/T[0-9A-Z]+/B[0-9A-Z]+/[A-Za-z0-9]+"),
    re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(r"\b[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{8,}", re.IGNORECASE),
    # Explicit assignment is sensitive regardless of value length.
    re.compile(
        r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|"
        r"client[_-]?secret|refresh[_-]?token|private[_-]?key|token)"
        r"\s*[:=]\s*['\"]?[^\s'\"]+",
        re.IGNORECASE,
    ),
    # Japanese label + an ASCII-like value (avoids matching ordinary prose such as パスワードは忘れた).
    re.compile(
        r"(?:パスワード|暗証番号|APIキー|アクセストークン|認証トークン|秘密鍵)"
        r"\s*(?:は|[:：=])\s*[`'\"]?[A-Za-z0-9._/+\-=]{4,}"
    ),
)


def is_sensitive_text(text: Any) -> bool:
    """Return True when text contains a credential or an explicit secret assignment."""
    return isinstance(text, str) and any(pattern.search(text) for pattern in _PATTERNS)
