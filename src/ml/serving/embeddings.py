"""Embedding model for document similarity (Sprint 5).

Uses sentence-transformers for text embedding generation.
Phase 3 scope: in-process FAISS for similarity search (< 10K cases).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL: Any = None
_EMBEDDING_DIM: int = 384


def _load_embedding_model() -> Any:
    """Load the sentence-transformer embedding model."""
    global _EMBEDDING_MODEL, _EMBEDDING_DIM
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    model_name = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        _EMBEDDING_DIM = _EMBEDDING_MODEL.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model: %s (dim=%d)", model_name, _EMBEDDING_DIM)
    except ImportError:
        logger.warning("sentence-transformers not installed — embedding features unavailable")
        raise
    return _EMBEDDING_MODEL


def compute_embedding(text: str) -> list[float]:
    """Compute a dense text embedding vector.

    Args:
        text: Input text to embed.

    Returns:
        Embedding vector as a list of floats.
    """
    model = _load_embedding_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def get_embedding_dim() -> int:
    """Return the embedding dimension for the loaded model."""
    return _EMBEDDING_DIM
