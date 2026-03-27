"""Unit tests for Feature Store integration (Sprint 3)."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

from ml.data.feature_store import _feature_cache, fetch_online_features


class TestFeatureCache:
    """LRU cache for Feature Store online serving."""

    def setup_method(self):
        _feature_cache.clear()

    def teardown_method(self):
        _feature_cache.clear()

    def test_returns_none_when_feature_store_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            result = fetch_online_features("case-1")
        assert result is None

    @patch("ml.data.feature_store.aiplatform")
    def test_caches_result(self, mock_aiplatform):
        """Second call with same entity_id should hit cache."""
        mock_fs = MagicMock()
        mock_entity_type = MagicMock()
        mock_entity_type.read.return_value.to_dict.return_value = {"case-1": {"feature_a": 1.0}}
        mock_fs.get_entity_type.return_value = mock_entity_type
        mock_aiplatform.Featurestore.return_value = mock_fs

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}, clear=True):
            result1 = fetch_online_features("case-1")
            result2 = fetch_online_features("case-1")

        # Should only call Feature Store once — second is from cache
        assert mock_entity_type.read.call_count == 1
        assert result1 == result2

    @patch("ml.data.feature_store.aiplatform")
    def test_cache_expires(self, mock_aiplatform):
        """Cache entries older than TTL should be evicted."""
        mock_fs = MagicMock()
        mock_entity_type = MagicMock()
        mock_entity_type.read.return_value.to_dict.return_value = {"case-x": {"feature_a": 2.0}}
        mock_fs.get_entity_type.return_value = mock_entity_type
        mock_aiplatform.Featurestore.return_value = mock_fs

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}, clear=True):
            # Insert an expired entry
            _feature_cache["case-x"] = (time.time() - 120, {"feature_a": 1.0})
            fetch_online_features("case-x")

        # Should have called Feature Store (cache expired)
        assert mock_entity_type.read.call_count == 1
