"""FAISS-based case similarity search (Sprint 5).

In-process FAISS index for finding similar cases based on
text embeddings. Designed for < 10K cases; uses flat L2 index.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimilarCase:
    """A case returned by similarity search."""

    case_id: str
    distance: float
    score: float  # 1 / (1 + distance) — higher is more similar


@dataclass
class SimilarityIndex:
    """FAISS flat-L2 similarity index over case embeddings."""

    _index: object = field(default=None, repr=False)
    _case_ids: list[str] = field(default_factory=list)
    _dim: int = 384

    def build(self, case_ids: list[str], embeddings: np.ndarray) -> None:
        """Build the FAISS index from scratch.

        Args:
            case_ids: Ordered list of case IDs matching embedding rows.
            embeddings: 2-D float32 array of shape (n_cases, dim).
        """
        import faiss

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {embeddings.shape}")
        n, dim = embeddings.shape
        if n != len(case_ids):
            raise ValueError(f"case_ids length {len(case_ids)} != embeddings rows {n}")

        self._dim = dim
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings.astype(np.float32))
        self._case_ids = list(case_ids)
        logger.info("Built similarity index: %d cases, dim=%d", n, dim)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[SimilarCase]:
        """Find the k nearest cases.

        Args:
            query_embedding: 1-D or 2-D float32 array.
            k: Number of neighbours to return.

        Returns:
            Sorted list of SimilarCase (most similar first).
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(query_embedding.astype(np.float32), k)

        results: list[SimilarCase] = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append(
                SimilarCase(
                    case_id=self._case_ids[idx],
                    distance=float(dist),
                    score=1.0 / (1.0 + float(dist)),
                )
            )
        return results

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0

    def save(self, path: str) -> None:
        """Persist the index to disk."""
        import faiss

        if self._index is None:
            raise RuntimeError("Cannot save empty index")
        faiss.write_index(self._index, path)
        logger.info("Saved similarity index to %s (%d vectors)", path, self.size)

    def load(self, path: str, case_ids: list[str]) -> None:
        """Load a previously saved index."""
        import faiss

        self._index = faiss.read_index(path)
        self._case_ids = list(case_ids)
        self._dim = self._index.d
        logger.info("Loaded similarity index from %s (%d vectors)", path, self.size)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_SIMILARITY_INDEX = SimilarityIndex()


def get_similarity_index() -> SimilarityIndex:
    """Return the module-level similarity index singleton."""
    return _SIMILARITY_INDEX


def rebuild_index_from_bq(project_id: str | None = None) -> SimilarityIndex:
    """Rebuild the similarity index from the BigQuery embeddings table.

    Reads case_embeddings table, builds FAISS index, optionally caches
    to local disk for fast restarts.

    Args:
        project_id: GCP project ID (defaults to I4G_ML_PROJECT).
    """
    from google.cloud import bigquery

    project_id = project_id or os.environ.get("I4G_ML_PROJECT", "i4g-ml")
    dataset = os.environ.get("I4G_ML_BQ_DATASET", "i4g_ml")

    client = bigquery.Client(project=project_id)
    query = f"""
        SELECT case_id, embedding
        FROM `{project_id}.{dataset}.case_embeddings`
        ORDER BY case_id
    """
    rows = list(client.query(query).result())
    if not rows:
        logger.warning("No embeddings found in case_embeddings table")
        return _SIMILARITY_INDEX

    case_ids = [row["case_id"] for row in rows]
    embeddings = np.array([row["embedding"] for row in rows], dtype=np.float32)

    _SIMILARITY_INDEX.build(case_ids, embeddings)

    # Cache to local file for fast restarts
    cache_dir = os.environ.get("SIMILARITY_CACHE_DIR")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        _SIMILARITY_INDEX.save(os.path.join(cache_dir, "similarity.index"))

    return _SIMILARITY_INDEX
