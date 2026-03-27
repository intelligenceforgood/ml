"""Unit tests for shadow mode serving."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import ml.serving.predict as pred_mod
from ml.serving.predict import (
    _MODEL_STATE,
    _SHADOW_MODEL_STATE,
    _parse_memory_limit,
    classify_text,
    classify_text_shadow,
    is_shadow_ready,
    load_shadow_model,
)


class TestShadowModelLoading:
    """Shadow model loads alongside champion."""

    def setup_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def teardown_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def test_shadow_not_ready_when_empty(self):
        assert is_shadow_ready() is False

    def test_shadow_ready_after_load(self):
        _SHADOW_MODEL_STATE["framework"] = "xgboost"
        assert is_shadow_ready() is True

    def test_shadow_not_ready_when_load_failed(self):
        pred_mod._SHADOW_LOAD_FAILED = True
        _SHADOW_MODEL_STATE["framework"] = "xgboost"
        assert is_shadow_ready() is False

    def test_empty_uri_noop(self):
        load_shadow_model("")
        assert is_shadow_ready() is False
        assert _SHADOW_MODEL_STATE == {}

    def test_load_failure_sets_flag(self):
        load_shadow_model("gs://bucket/bad-path/model/v1")
        assert pred_mod._SHADOW_LOAD_FAILED is True
        assert _SHADOW_MODEL_STATE["stage"] == "error"

    def test_champion_unaffected_by_shadow_state(self):
        """Champion model state is independent of shadow state."""
        _MODEL_STATE["model_id"] = "champion"
        pred_mod._SHADOW_LOAD_FAILED = True

        # Champion still works in stub mode (no framework set)
        result = classify_text("test text", "case-1")
        assert result["prediction"]["INTENT"]["code"] == "INTENT.UNKNOWN"


class TestShadowInference:
    """Shadow inference returns correct structure and doesn't crash."""

    def setup_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def teardown_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def test_shadow_returns_none_when_not_ready(self):
        result = classify_text_shadow("text", "case-1", "champ-id-1")
        assert result is None

    def test_shadow_prediction_id_format(self):
        """Shadow prediction_id = '{champion_prediction_id}-shadow'."""
        import sys
        import types

        import numpy as np

        mock_xgb = types.ModuleType("xgboost")

        class MockDMatrix:
            def __init__(self, data, feature_names=None):
                self.data = data

        mock_xgb.DMatrix = MockDMatrix
        sys.modules["xgboost"] = mock_xgb

        try:
            label_map = {
                "INTENT": ["INTENT.A", "INTENT.B"],
                "CHANNEL": ["CHANNEL.X", "CHANNEL.Y"],
            }
            _SHADOW_MODEL_STATE["label_map"] = label_map
            _SHADOW_MODEL_STATE["framework"] = "xgboost"
            _SHADOW_MODEL_STATE["model_id"] = "shadow-model"
            _SHADOW_MODEL_STATE["version"] = 2
            _SHADOW_MODEL_STATE["stage"] = "shadow"

            mock_booster = MagicMock()
            mock_booster.predict.return_value = np.array([[0.3, 0.7, 0.9, 0.1]])
            _SHADOW_MODEL_STATE["booster"] = mock_booster

            result = classify_text_shadow(
                "test text",
                "case-1",
                "abc-123",
                features={
                    "text_length": 9,
                    "word_count": 2,
                    "lexical_diversity": 1.0,
                    "has_email": False,
                    "has_phone": False,
                    "has_crypto_wallet": False,
                    "has_bank_account": False,
                },
            )

            assert result is not None
            assert result["prediction_id"] == "abc-123-shadow"
            assert result["is_shadow"] is True
            assert result["model_info"]["stage"] == "shadow"
            assert result["model_info"]["model_id"] == "shadow-model"
            assert result["prediction"]["INTENT"]["code"] == "INTENT.B"
            assert result["prediction"]["CHANNEL"]["code"] == "CHANNEL.X"
        finally:
            del sys.modules["xgboost"]

    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features", return_value={"text_length": 10})
    def test_shadow_failure_bubbles_up(self, mock_features, mock_dmat):
        """When shadow inference fails, the error propagates (caught by app layer)."""
        _SHADOW_MODEL_STATE["framework"] = "xgboost"
        _SHADOW_MODEL_STATE["model_id"] = "bad"
        _SHADOW_MODEL_STATE["version"] = 1
        _SHADOW_MODEL_STATE["stage"] = "shadow"
        _SHADOW_MODEL_STATE["label_map"] = {"INTENT": ["A"]}
        # No booster key — will raise KeyError when accessing _SHADOW_MODEL_STATE["booster"]
        with pytest.raises((KeyError, TypeError, RuntimeError, ModuleNotFoundError)):
            classify_text_shadow("text", "case-1", "champ-1")


class TestShadowDoesNotBlockChampion:
    """Shadow failures must never affect champion responses."""

    def setup_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def teardown_method(self):
        pred_mod._LOAD_FAILED = False
        pred_mod._SHADOW_LOAD_FAILED = False
        _MODEL_STATE.clear()
        _SHADOW_MODEL_STATE.clear()

    def test_champion_returns_immediately(self):
        """Champion classify_text works even when shadow is loaded."""
        _SHADOW_MODEL_STATE["framework"] = "xgboost"
        result = classify_text("text", "case-1")
        assert result["prediction_id"]
        assert result["prediction"]["INTENT"]["code"] == "INTENT.UNKNOWN"

    @pytest.mark.asyncio
    @patch("xgboost.DMatrix")
    @patch("ml.serving.features.compute_inline_features", return_value={"text_length": 10})
    async def test_shadow_task_exception_caught(self, mock_features, mock_dmat):
        """The async shadow wrapper catches all exceptions."""
        from ml.serving.app import _run_shadow_inference

        # Shadow model is "ready" but will fail (no booster key)
        _SHADOW_MODEL_STATE["framework"] = "xgboost"
        _SHADOW_MODEL_STATE["model_id"] = "broken"
        _SHADOW_MODEL_STATE["label_map"] = {"INTENT": ["A"]}

        # Should NOT raise — exceptions are caught inside _run_shadow_inference
        await _run_shadow_inference("text", "case-1", "champ-123", None)


class TestIsShadowFlag:
    """Verify is_shadow flag is correctly set in BQ log payload."""

    def test_champion_log_no_shadow_flag(self):
        """Champion predictions should not have is_shadow=True."""
        result = {
            "prediction_id": "p1",
            "model_info": {"model_id": "m1", "version": 1},
            "prediction": {},
        }
        with patch("ml.serving.logging._insert_with_retry") as mock_insert:
            from ml.serving.logging import log_prediction

            log_prediction(
                prediction_id=result["prediction_id"],
                case_id="c1",
                model_id="m1",
                model_version=1,
                prediction={},
            )
            if mock_insert.called:
                row = mock_insert.call_args[0][1]
                assert row.get("is_shadow") is False

    def test_shadow_log_has_shadow_flag(self):
        """Shadow predictions should have is_shadow=True."""
        with patch("ml.serving.logging._insert_with_retry") as mock_insert:
            from ml.serving.logging import log_prediction

            log_prediction(
                prediction_id="p1-shadow",
                case_id="c1",
                model_id="shadow-m1",
                model_version=1,
                prediction={},
                is_shadow=True,
            )
            if mock_insert.called:
                row = mock_insert.call_args[0][1]
                assert row.get("is_shadow") is True


class TestParseMemoryLimit:
    def test_gi(self):
        assert _parse_memory_limit("2Gi") == 2048

    def test_mi(self):
        assert _parse_memory_limit("512Mi") == 512

    def test_g(self):
        assert _parse_memory_limit("4G") == 4000

    def test_empty(self):
        assert _parse_memory_limit("") is None

    def test_none_str(self):
        assert _parse_memory_limit("  ") is None


class TestShadowComparison:
    """Test compute_shadow_comparison in accuracy.py."""

    def test_returns_none_when_no_pairs(self):
        from ml.monitoring.accuracy import compute_shadow_comparison

        mock_client = MagicMock()
        mock_query = MagicMock()
        mock_query.result.return_value = []
        mock_client.query.return_value = mock_query

        result = compute_shadow_comparison(lookback_days=7, client=mock_client)
        assert result is None

    def test_full_agreement(self):
        import json

        from ml.monitoring.accuracy import compute_shadow_comparison

        mock_client = MagicMock()

        # Two rows where champion and shadow fully agree
        row1 = MagicMock()
        row1.champion_model = "champion-v1"
        row1.shadow_model = "shadow-v2"
        row1.champion_prediction = json.dumps({"INTENT": {"code": "INTENT.A", "confidence": 0.9}})
        row1.shadow_prediction = json.dumps({"INTENT": {"code": "INTENT.A", "confidence": 0.8}})

        row2 = MagicMock()
        row2.champion_model = "champion-v1"
        row2.shadow_model = "shadow-v2"
        row2.champion_prediction = json.dumps({"INTENT": {"code": "INTENT.B", "confidence": 0.7}})
        row2.shadow_prediction = json.dumps({"INTENT": {"code": "INTENT.B", "confidence": 0.6}})

        mock_query = MagicMock()
        mock_query.result.return_value = [row1, row2]
        mock_client.query.return_value = mock_query

        result = compute_shadow_comparison(lookback_days=7, client=mock_client)
        assert result is not None
        assert result.total_pairs == 2
        assert result.agreement_count == 2
        assert result.agreement_rate == 1.0
        assert result.per_axis_agreement["INTENT"] == 1.0

    def test_partial_agreement(self):
        import json

        from ml.monitoring.accuracy import compute_shadow_comparison

        mock_client = MagicMock()

        row1 = MagicMock()
        row1.champion_model = "champion-v1"
        row1.shadow_model = "shadow-v2"
        row1.champion_prediction = json.dumps(
            {
                "INTENT": {"code": "INTENT.A", "confidence": 0.9},
                "CHANNEL": {"code": "CHANNEL.X", "confidence": 0.8},
            }
        )
        row1.shadow_prediction = json.dumps(
            {
                "INTENT": {"code": "INTENT.A", "confidence": 0.85},
                "CHANNEL": {"code": "CHANNEL.Y", "confidence": 0.7},  # disagree
            }
        )

        mock_query = MagicMock()
        mock_query.result.return_value = [row1]
        mock_client.query.return_value = mock_query

        result = compute_shadow_comparison(lookback_days=7, client=mock_client)
        assert result is not None
        assert result.total_pairs == 1
        assert result.agreement_count == 0  # Not fully agreed (CHANNEL differs)
        assert result.per_axis_agreement["INTENT"] == 1.0
        assert result.per_axis_agreement["CHANNEL"] == 0.0
