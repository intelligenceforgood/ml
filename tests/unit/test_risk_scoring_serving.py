"""Unit tests for risk scoring serving (Sprint 4.4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.serving.predict import _RISK_MODEL_STATE, is_risk_ready, predict_risk_score


class TestIsRiskReady:
    def setup_method(self):
        _RISK_MODEL_STATE.clear()

    def test_not_ready_when_empty(self):
        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False
        assert is_risk_ready() is False

    def test_not_ready_when_load_failed(self):
        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = True
        _RISK_MODEL_STATE["framework"] = "xgboost"
        assert is_risk_ready() is False

    def test_ready_when_loaded(self):
        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False
        _RISK_MODEL_STATE["framework"] = "xgboost"
        assert is_risk_ready() is True


class TestPredictRiskScore:
    def setup_method(self):
        _RISK_MODEL_STATE.clear()

    def test_raises_when_not_loaded(self):
        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False
        with pytest.raises(RuntimeError, match="Risk scoring model not loaded"):
            predict_risk_score("some text")

    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features")
    def test_returns_clamped_score(self, mock_features, mock_dmat):
        """Score should be clamped to [0.0, 1.0]."""
        import numpy as np

        mock_features.return_value = {"text_length": 100, "word_count": 20}

        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False

        mock_booster = MagicMock()
        # Return a value > 1 to test clamping
        mock_booster.predict.return_value = np.array([1.5])
        _RISK_MODEL_STATE["framework"] = "xgboost"
        _RISK_MODEL_STATE["booster"] = mock_booster
        _RISK_MODEL_STATE["feature_cols"] = ["text_length", "word_count"]

        score = predict_risk_score("some text")

        assert score == 1.0  # clamped

    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features")
    def test_returns_valid_score(self, mock_features, mock_dmat):
        import numpy as np

        mock_features.return_value = {"text_length": 100, "word_count": 20}

        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.75])
        _RISK_MODEL_STATE["framework"] = "xgboost"
        _RISK_MODEL_STATE["booster"] = mock_booster
        _RISK_MODEL_STATE["feature_cols"] = ["text_length", "word_count"]

        score = predict_risk_score("some text")

        assert score == 0.75

    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features")
    def test_clamps_negative_score(self, mock_features, mock_dmat):
        """Negative predictions should be clamped to 0.0."""
        import numpy as np

        mock_features.return_value = {"text_length": 50, "word_count": 10}

        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([-0.3])
        _RISK_MODEL_STATE["framework"] = "xgboost"
        _RISK_MODEL_STATE["booster"] = mock_booster
        _RISK_MODEL_STATE["feature_cols"] = ["text_length", "word_count"]

        score = predict_risk_score("some text")

        assert score == 0.0

    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features")
    def test_uses_provided_features(self, mock_inline, mock_dmat):
        """When features dict is provided, should use it instead of computing inline."""
        import numpy as np

        import ml.serving.predict as pred_mod

        pred_mod._RISK_LOAD_FAILED = False

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.6])
        _RISK_MODEL_STATE["framework"] = "xgboost"
        _RISK_MODEL_STATE["booster"] = mock_booster
        _RISK_MODEL_STATE["feature_cols"] = ["text_length"]

        provided_features = {"text_length": 200}
        score = predict_risk_score("some text", features=provided_features)

        # compute_inline_features should NOT be called
        mock_inline.assert_not_called()
        assert score == 0.6
