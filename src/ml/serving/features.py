"""Feature computation for serving (inline Phase 0, pre-computed Phase 1+).

Phase 0: features are computed inline at prediction time from raw text.
Phase 1+: features are pre-computed in BigQuery and looked up at predict.
"""

from __future__ import annotations

import re
from typing import Any


def compute_inline_features(text: str) -> dict[str, Any]:
    """Compute features inline from raw text for Phase 0 serving."""
    words = text.split()
    unique_words = set(w.lower() for w in words)

    return {
        "text_length": len(text),
        "word_count": len(words),
        "lexical_diversity": len(unique_words) / len(words) if words else 0.0,
        "has_crypto_wallet": bool(
            re.search(r"\b(0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-z0-9]{39,59})\b", text)
        ),
        "has_email": bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)),
        "has_phone": bool(re.search(r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b", text)),
        "has_bank_account": bool(re.search(r"\b\d{8,17}\b", text)),
    }
