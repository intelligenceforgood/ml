"""Unit tests for variant comparison and accuracy materialization (Sprint 1.3)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.monitoring.accuracy import (
    AxisAccuracy,
    VariantComparison,
    VariantMetrics,
    compute_variant_comparison,
    materialize_variant_comparison,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings():
    return SimpleNamespace(
        platform=SimpleNamespace(project_id="test-project"),
        bigquery=SimpleNamespace(
            dataset_id="test_dataset",
            prediction_log_table="predictions_prediction_log",
            outcome_log_table="predictions_outcome_log",
        ),
    )


def _make_variant_row(
    prediction_id: str,
    model_id: str,
    model_version: int,
    prediction: dict,
    correction: dict,
    variant: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        prediction_id=prediction_id,
        model_id=model_id,
        model_version=model_version,
        prediction=json.dumps(prediction),
        correction=json.dumps(correction),
        variant=variant,
    )


# ---------------------------------------------------------------------------
# Tests — compute_variant_comparison
# ---------------------------------------------------------------------------


class TestComputeVariantComparison:
    @patch("ml.monitoring.accuracy.get_settings")
    def test_no_variants(self, mock_settings):
        """No rows → both champion and challenger are None."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        result = compute_variant_comparison(lookback_days=7, client=mock_client)

        assert isinstance(result, VariantComparison)
        assert result.champion is None
        assert result.challenger is None

    @patch("ml.monitoring.accuracy.get_settings")
    def test_champion_only(self, mock_settings):
        """Only champion predictions → challenger is None."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        rows = [
            _make_variant_row(
                "p1",
                "model-champ",
                1,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
                {"INTENT": "INTENT.ROMANCE"},
                "champion",
            ),
            _make_variant_row(
                "p2",
                "model-champ",
                1,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.8}},
                {"INTENT": "INTENT.CRYPTO"},
                "champion",
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        result = compute_variant_comparison(lookback_days=7, client=mock_client)

        assert result.champion is not None
        assert result.champion.accuracy == 1.0
        assert result.challenger is None

    @patch("ml.monitoring.accuracy.get_settings")
    def test_both_variants(self, mock_settings):
        """Champion and challenger with different accuracies."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        rows = [
            # Champion: 2/2 correct
            _make_variant_row(
                "p1",
                "model-champ",
                1,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
                {"INTENT": "INTENT.ROMANCE"},
                "champion",
            ),
            _make_variant_row(
                "p2",
                "model-champ",
                1,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.8}},
                {"INTENT": "INTENT.CRYPTO"},
                "champion",
            ),
            # Challenger: 1/2 correct
            _make_variant_row(
                "p3",
                "model-chal",
                2,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.7}},
                {"INTENT": "INTENT.ROMANCE"},
                "challenger",
            ),
            _make_variant_row(
                "p4",
                "model-chal",
                2,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.6}},
                {"INTENT": "INTENT.ROMANCE"},
                "challenger",
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        result = compute_variant_comparison(lookback_days=7, client=mock_client)

        assert result.champion is not None
        assert result.challenger is not None
        assert result.champion.accuracy == 1.0
        assert result.challenger.accuracy == 0.5
        assert result.champion.model_id == "model-champ"
        assert result.challenger.model_id == "model-chal"
        assert "INTENT" in result.champion.per_axis
        assert "INTENT" in result.challenger.per_axis

    @patch("ml.monitoring.accuracy.get_settings")
    def test_multi_axis_variant(self, mock_settings):
        """Variant comparison across multiple axes."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        rows = [
            _make_variant_row(
                "p1",
                "model-a",
                1,
                {
                    "INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9},
                    "CHANNEL": {"code": "CHANNEL.SOCIAL", "confidence": 0.8},
                },
                {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
                "champion",
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        result = compute_variant_comparison(lookback_days=7, client=mock_client)

        assert result.champion is not None
        assert result.champion.per_axis["INTENT"].correct == 1
        assert result.champion.per_axis["CHANNEL"].overridden == 1


# ---------------------------------------------------------------------------
# Tests — materialize_variant_comparison
# ---------------------------------------------------------------------------


class TestMaterializeVariantComparison:
    @patch("ml.monitoring.accuracy.compute_variant_comparison")
    @patch("ml.monitoring.accuracy.get_settings")
    def test_no_data_skipped(self, mock_settings, mock_compute):
        """No variant data → nothing written."""
        mock_settings.return_value = _fake_settings()
        mock_compute.return_value = VariantComparison(
            champion=None,
            challenger=None,
            lookback_days=7,
            computed_at="2026-03-27",
        )
        mock_client = MagicMock()

        rows = materialize_variant_comparison(lookback_days=7, client=mock_client)

        assert rows == 0
        mock_client.insert_rows_json.assert_not_called()

    @patch("ml.monitoring.accuracy.compute_variant_comparison")
    @patch("ml.monitoring.accuracy.get_settings")
    def test_writes_variant_rows(self, mock_settings, mock_compute):
        """Verify rows are written with variant-specific metrics."""
        mock_settings.return_value = _fake_settings()
        champion = VariantMetrics(
            variant="champion",
            model_id="model-a",
            total_outcomes=50,
            correct=40,
            accuracy=0.8,
            override_rate=0.2,
            f1=0.8,
            per_axis={
                "INTENT": AxisAccuracy(
                    axis="INTENT",
                    total=50,
                    correct=40,
                    overridden=10,
                    accuracy=0.8,
                    override_rate=0.2,
                    precision=0.8,
                    recall=0.8,
                    f1=0.8,
                ),
            },
        )
        challenger = VariantMetrics(
            variant="challenger",
            model_id="model-b",
            total_outcomes=30,
            correct=21,
            accuracy=0.7,
            override_rate=0.3,
            f1=0.7,
            per_axis={
                "INTENT": AxisAccuracy(
                    axis="INTENT",
                    total=30,
                    correct=21,
                    overridden=9,
                    accuracy=0.7,
                    override_rate=0.3,
                    precision=0.7,
                    recall=0.7,
                    f1=0.7,
                ),
            },
        )
        mock_compute.return_value = VariantComparison(
            champion=champion,
            challenger=challenger,
            lookback_days=7,
            computed_at="2026-03-27",
        )
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []

        rows = materialize_variant_comparison(lookback_days=7, client=mock_client)

        assert rows == 2
        mock_client.insert_rows_json.assert_called_once()
        written = mock_client.insert_rows_json.call_args[0][1]
        variants = {r["variant"] for r in written}
        assert variants == {"champion", "challenger"}
        champ_row = next(r for r in written if r["variant"] == "champion")
        assert champ_row["accuracy"] == 0.8
        assert champ_row["model_id"] == "model-a"
