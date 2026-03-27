"""Unit tests for Feature Store → inline feature fallback in predict.py (Sprint 3.3)."""

from __future__ import annotations

import os
from unittest.mock import patch

from ml.serving.predict import _fetch_features_with_fallback


class TestFetchFeaturesWithFallback:
    """Tests for _fetch_features_with_fallback() in predict.py."""

    @patch("ml.serving.features.compute_inline_features")
    def test_uses_inline_when_feature_store_not_configured(self, mock_inline):
        mock_inline.return_value = {"text_length": 200, "word_count": 50}

        with patch.dict(os.environ, {}, clear=True):
            result = _fetch_features_with_fallback("some text", "case-1")

        mock_inline.assert_called_once_with("some text")
        assert result == {"text_length": 200, "word_count": 50}

    @patch("ml.data.feature_store.fetch_online_features")
    @patch("ml.serving.features.compute_inline_features")
    def test_uses_feature_store_when_configured(self, mock_inline, mock_fs):
        mock_fs.return_value = {"text_length": 200, "entity_count": 5}

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}):
            result = _fetch_features_with_fallback("some text", "case-1")

        mock_fs.assert_called_once_with("case-1")
        mock_inline.assert_not_called()
        assert result == {"text_length": 200, "entity_count": 5}

    @patch("ml.data.feature_store.fetch_online_features")
    @patch("ml.serving.features.compute_inline_features")
    def test_falls_back_to_inline_when_feature_store_returns_none(self, mock_inline, mock_fs):
        mock_fs.return_value = None
        mock_inline.return_value = {"text_length": 100}

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}):
            result = _fetch_features_with_fallback("some text", "case-1")

        mock_fs.assert_called_once_with("case-1")
        mock_inline.assert_called_once_with("some text")
        assert result == {"text_length": 100}
