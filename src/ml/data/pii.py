"""PII redaction for training data exports.

Phase 0 uses regex-based redaction. Phase 1+ will integrate with
Google Cloud DLP for production-grade PII handling.
"""

from __future__ import annotations

import re

# Patterns for common PII types
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("[EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")),
    ("[PHONE]", re.compile(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("[CC]", re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")),
    ("[IP]", re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]


def redact_pii(text: str) -> str:
    """Replace known PII patterns with placeholder tokens."""
    for placeholder, pattern in _PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def redact_record(record: dict, text_fields: list[str] | None = None) -> dict:
    """Redact PII from specified text fields in a record dict."""
    fields = text_fields or ["text", "narrative"]
    result = dict(record)
    for field in fields:
        if field in result and isinstance(result[field], str):
            result[field] = redact_pii(result[field])
    return result
