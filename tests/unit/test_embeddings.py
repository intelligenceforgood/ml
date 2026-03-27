"""Unit tests for embedding model (Sprint 5.1)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestComputeEmbedding:
    """Embedding generation."""

    @patch("ml.serving.embeddings._load_embedding_model")
    def test_returns_list_of_floats(self, mock_load):
        mock_model = MagicMock()
        import numpy as np

        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_load.return_value = mock_model

        from ml.serving.embeddings import compute_embedding

        result = compute_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    @patch("ml.serving.embeddings._load_embedding_model")
    def test_deterministic_same_input(self, mock_load):
        """Same input should produce same output."""
        import numpy as np

        embedding = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = embedding
        mock_load.return_value = mock_model

        from ml.serving.embeddings import compute_embedding

        result1 = compute_embedding("identical text")
        result2 = compute_embedding("identical text")

        assert result1 == result2

    @patch("ml.serving.embeddings._load_embedding_model")
    def test_called_without_progress_bar(self, mock_load):
        """Embedding model should be called with show_progress_bar=False."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        mock_load.return_value = mock_model

        from ml.serving.embeddings import compute_embedding

        compute_embedding("test")

        mock_model.encode.assert_called_once_with("test", show_progress_bar=False)


class TestGetEmbeddingDim:
    def test_default_dimension(self):
        from ml.serving.embeddings import get_embedding_dim

        # Default is 384 (MiniLM)
        dim = get_embedding_dim()
        assert dim == 384


class TestLoadEmbeddingModel:
    @patch.dict(os.environ, {"EMBEDDING_MODEL_NAME": "test-model"})
    def test_uses_env_var(self):
        """Model name should come from EMBEDDING_MODEL_NAME env var."""
        import sys
        import types

        fake_st = types.ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = MagicMock()
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 768
        fake_st.SentenceTransformer.return_value = mock_instance

        with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
            import ml.serving.embeddings as emb_mod

            # Reset module state
            emb_mod._EMBEDDING_MODEL = None

            emb_mod._load_embedding_model()

            fake_st.SentenceTransformer.assert_called_once_with("test-model")
            assert emb_mod._EMBEDDING_DIM == 768
