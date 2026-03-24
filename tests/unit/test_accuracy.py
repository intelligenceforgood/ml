"""Unit tests for accuracy monitoring with mock BigQuery results."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.monitoring.accuracy import (
    AccuracyReport,
    AxisAccuracy,
    ModelAccuracy,
    _compute_axis_metrics,
    compute_accuracy_metrics,
    materialize_performance,
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


def _make_row(
    prediction_id: str,
    model_id: str,
    model_version: int,
    prediction: dict,
    correction: dict,
) -> SimpleNamespace:
    """Create a mock BigQuery row."""
    return SimpleNamespace(
        prediction_id=prediction_id,
        model_id=model_id,
        model_version=model_version,
        prediction=json.dumps(prediction),
        correction=json.dumps(correction),
    )


# ---------------------------------------------------------------------------
# Tests — axis metrics computation
# ---------------------------------------------------------------------------


class TestComputeAxisMetrics:
    def test_all_correct(self):
        predicted = ["INTENT.ROMANCE", "INTENT.ROMANCE", "INTENT.CRYPTO"]
        corrected = ["INTENT.ROMANCE", "INTENT.ROMANCE", "INTENT.CRYPTO"]
        correct, overridden, prec, rec, f1 = _compute_axis_metrics(predicted, corrected)
        assert correct == 3
        assert overridden == 0
        assert prec == 1.0
        assert f1 == 1.0

    def test_all_wrong(self):
        predicted = ["INTENT.ROMANCE", "INTENT.CRYPTO"]
        corrected = ["INTENT.CRYPTO", "INTENT.ROMANCE"]
        correct, overridden, prec, rec, f1 = _compute_axis_metrics(predicted, corrected)
        assert correct == 0
        assert overridden == 2
        assert prec == 0.0
        assert f1 == 0.0

    def test_partial(self):
        predicted = ["INTENT.ROMANCE", "INTENT.CRYPTO", "INTENT.ROMANCE"]
        corrected = ["INTENT.ROMANCE", "INTENT.ROMANCE", "INTENT.ROMANCE"]
        correct, overridden, prec, rec, f1 = _compute_axis_metrics(predicted, corrected)
        assert correct == 2
        assert overridden == 1

    def test_empty(self):
        correct, overridden, prec, rec, f1 = _compute_axis_metrics([], [])
        assert correct == 0
        assert overridden == 0
        assert f1 == 0.0


# ---------------------------------------------------------------------------
# Tests — compute_accuracy_metrics
# ---------------------------------------------------------------------------


class TestComputeAccuracyMetrics:
    @patch("ml.monitoring.accuracy.get_settings")
    def test_no_outcomes(self, mock_settings):
        """When there are no outcomes, report should have empty models list."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        report = compute_accuracy_metrics(lookback_days=7, client=mock_client)

        assert isinstance(report, AccuracyReport)
        assert report.models == []
        assert report.lookback_days == 7

    @patch("ml.monitoring.accuracy.get_settings")
    def test_single_model_perfect_accuracy(self, mock_settings):
        """All predictions correct → 100% accuracy, 0% override."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        rows = [
            _make_row(
                f"pred-{i}",
                "model-a",
                1,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
                {"INTENT": "INTENT.ROMANCE"},
            )
            for i in range(5)
        ]
        mock_client.query.return_value.result.return_value = rows

        report = compute_accuracy_metrics(lookback_days=7, client=mock_client)

        assert len(report.models) == 1
        m = report.models[0]
        assert m.model_id == "model-a"
        assert m.model_version == 1
        assert m.accuracy == 1.0
        assert m.override_rate == 0.0
        assert "INTENT" in m.per_axis
        assert m.per_axis["INTENT"].correct == 5
        assert m.per_axis["INTENT"].overridden == 0

    @patch("ml.monitoring.accuracy.get_settings")
    def test_partial_overrides(self, mock_settings):
        """Some predictions overridden — verify accuracy and override rate."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        rows = [
            # 3 correct
            _make_row(
                "p1",
                "model-b",
                2,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
                {"INTENT": "INTENT.ROMANCE"},
            ),
            _make_row(
                "p2",
                "model-b",
                2,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.8}},
                {"INTENT": "INTENT.CRYPTO"},
            ),
            _make_row(
                "p3",
                "model-b",
                2,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.7}},
                {"INTENT": "INTENT.ROMANCE"},
            ),
            # 2 overridden
            _make_row(
                "p4",
                "model-b",
                2,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.6}},
                {"INTENT": "INTENT.CRYPTO"},
            ),
            _make_row(
                "p5",
                "model-b",
                2,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.5}},
                {"INTENT": "INTENT.ROMANCE"},
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        report = compute_accuracy_metrics(lookback_days=30, client=mock_client)

        m = report.models[0]
        assert m.accuracy == 0.6  # 3/5
        assert m.override_rate == 0.4  # 2/5
        assert m.per_axis["INTENT"].correct == 3
        assert m.per_axis["INTENT"].overridden == 2

    @patch("ml.monitoring.accuracy.get_settings")
    def test_multi_axis(self, mock_settings):
        """Corrections on multiple axes should produce per-axis metrics."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        rows = [
            _make_row(
                "p1",
                "model-c",
                1,
                {
                    "INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9},
                    "CHANNEL": {"code": "CHANNEL.SOCIAL_MEDIA", "confidence": 0.8},
                },
                {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        report = compute_accuracy_metrics(lookback_days=7, client=mock_client)

        m = report.models[0]
        assert "INTENT" in m.per_axis
        assert "CHANNEL" in m.per_axis
        assert m.per_axis["INTENT"].correct == 1  # INTENT matched
        assert m.per_axis["CHANNEL"].overridden == 1  # CHANNEL was corrected

    @patch("ml.monitoring.accuracy.get_settings")
    def test_multiple_models(self, mock_settings):
        """Two different models produce two entries in the report."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        rows = [
            _make_row(
                "p1",
                "model-x",
                1,
                {"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
                {"INTENT": "INTENT.ROMANCE"},
            ),
            _make_row(
                "p2",
                "model-y",
                1,
                {"INTENT": {"code": "INTENT.CRYPTO", "confidence": 0.8}},
                {"INTENT": "INTENT.ROMANCE"},
            ),
        ]
        mock_client.query.return_value.result.return_value = rows

        report = compute_accuracy_metrics(lookback_days=7, client=mock_client)

        assert len(report.models) == 2
        model_ids = {m.model_id for m in report.models}
        assert model_ids == {"model-x", "model-y"}


# ---------------------------------------------------------------------------
# Tests — materialize_performance
# ---------------------------------------------------------------------------


class TestMaterializePerformance:
    @patch("ml.monitoring.accuracy.compute_accuracy_metrics")
    @patch("ml.monitoring.accuracy.get_settings")
    def test_no_models_skipped(self, mock_settings, mock_compute):
        """When no models have outcomes, no rows should be written."""
        mock_settings.return_value = _fake_settings()
        mock_compute.return_value = AccuracyReport(lookback_days=7, computed_at="2026-03-23", models=[])
        mock_client = MagicMock()

        rows = materialize_performance(lookback_days=7, client=mock_client)

        assert rows == 0
        mock_client.insert_rows_json.assert_not_called()

    @patch("ml.monitoring.accuracy.compute_accuracy_metrics")
    @patch("ml.monitoring.accuracy.get_settings")
    def test_writes_correct_rows(self, mock_settings, mock_compute):
        """Verify rows are written to BQ with expected schema."""
        mock_settings.return_value = _fake_settings()
        mock_compute.return_value = AccuracyReport(
            lookback_days=7,
            computed_at="2026-03-23",
            models=[
                ModelAccuracy(
                    model_id="model-a",
                    model_version=1,
                    total_predictions=100,
                    outcomes_received=50,
                    correct_predictions=40,
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
            ],
        )
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []  # no errors

        rows = materialize_performance(lookback_days=7, client=mock_client)

        assert rows == 1
        mock_client.insert_rows_json.assert_called_once()
        written_rows = mock_client.insert_rows_json.call_args[0][1]
        assert len(written_rows) == 1
        assert written_rows[0]["model_id"] == "model-a"
        assert written_rows[0]["accuracy"] == 0.8
        assert written_rows[0]["correction_rate"] == 0.2
        assert "per_axis_metrics" in written_rows[0]
        per_axis = json.loads(written_rows[0]["per_axis_metrics"])
        assert "INTENT" in per_axis
