"""Unit tests for similarity search (Sprint 5)."""

from __future__ import annotations

import numpy as np
import pytest

from ml.serving.similarity import SimilarityIndex


class TestSimilarityIndex:
    """FAISS flat-L2 similarity index."""

    def test_build_and_search(self):
        pytest.importorskip("faiss")

        idx = SimilarityIndex()
        case_ids = ["case-1", "case-2", "case-3"]
        embeddings = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        idx.build(case_ids, embeddings)
        assert idx.size == 3

        # Search for the first vector — should find itself as nearest
        results = idx.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
        assert len(results) == 2
        assert results[0].case_id == "case-1"
        assert results[0].distance == pytest.approx(0.0, abs=1e-6)

    def test_empty_index_returns_empty(self):
        idx = SimilarityIndex()
        results = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=5)
        assert results == []

    def test_k_larger_than_index(self):
        pytest.importorskip("faiss")

        idx = SimilarityIndex()
        case_ids = ["case-1"]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        idx.build(case_ids, embeddings)

        results = idx.search(np.array([1.0, 0.0], dtype=np.float32), k=10)
        assert len(results) == 1

    def test_build_mismatched_dimensions(self):
        idx = SimilarityIndex()
        with pytest.raises(ValueError, match="case_ids length"):
            idx.build(["a", "b"], np.array([[1.0]], dtype=np.float32))

    def test_score_is_inverse_distance(self):
        pytest.importorskip("faiss")

        idx = SimilarityIndex()
        case_ids = ["a", "b"]
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        idx.build(case_ids, embeddings)

        results = idx.search(np.array([0.0, 0.0], dtype=np.float32), k=2)
        assert results[0].score == pytest.approx(1.0, abs=1e-6)  # distance 0 → score 1
        assert results[1].score < results[0].score
