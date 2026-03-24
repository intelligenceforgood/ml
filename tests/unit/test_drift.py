"""Unit tests for drift monitoring with mock BigQuery results."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.monitoring.drift import (
    DriftReport,
    FeatureDrift,
    PredictionDrift,
    _compute_numeric_psi,
    compute_drift_report,
    compute_feature_drift,
    compute_prediction_drift,
    compute_psi,
    materialize_drift_metrics,
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


# ---------------------------------------------------------------------------
# Tests — PSI computation
# ---------------------------------------------------------------------------


class TestComputePsi:
    def test_identical_distributions(self):
        probs = [0.3, 0.3, 0.4]
        psi = compute_psi(probs, probs)
        assert psi < 0.001

    def test_slightly_shifted(self):
        baseline = [0.3, 0.3, 0.4]
        current = [0.35, 0.25, 0.4]
        psi = compute_psi(baseline, current)
        assert 0.0 < psi < 0.2  # minor shift, not drifted

    def test_large_shift(self):
        baseline = [0.5, 0.3, 0.2]
        current = [0.1, 0.1, 0.8]
        psi = compute_psi(baseline, current)
        assert psi > 0.2  # significant drift

    def test_mismatched_lengths_raises(self):
        import pytest

        with pytest.raises(ValueError, match="same number of bins"):
            compute_psi([0.5, 0.5], [0.3, 0.3, 0.4])

    def test_zero_probabilities_handled(self):
        """Zero probabilities should be clamped to epsilon, not cause log(0)."""
        baseline = [0.5, 0.5, 0.0]
        current = [0.0, 0.5, 0.5]
        psi = compute_psi(baseline, current)
        assert psi > 0.0


class TestNumericPsi:
    def test_identical_values(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        psi = _compute_numeric_psi(vals, vals)
        assert psi < 0.001

    def test_shifted_distribution(self):
        baseline = [float(i) for i in range(100)]
        current = [float(i + 50) for i in range(100)]
        psi = _compute_numeric_psi(baseline, current)
        assert psi > 0.0

    def test_empty_baseline(self):
        psi = _compute_numeric_psi([], [1.0, 2.0])
        assert psi == 0.0

    def test_constant_values(self):
        psi = _compute_numeric_psi([5.0] * 10, [5.0] * 10)
        assert psi == 0.0


# ---------------------------------------------------------------------------
# Tests — prediction drift
# ---------------------------------------------------------------------------


class TestComputePredictionDrift:
    @patch("ml.monitoring.drift.get_settings")
    def test_no_data_returns_empty(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        result = compute_prediction_drift("model-v1", window_days=7, client=mock_client)
        assert result == []

    @patch("ml.monitoring.drift.get_settings")
    def test_stable_distribution(self, mock_settings):
        """Same distributions should produce low PSI."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        baseline_rows = [
            SimpleNamespace(label="ROMANCE", cnt=50),
            SimpleNamespace(label="CRYPTO", cnt=50),
        ]
        current_rows = [
            SimpleNamespace(label="ROMANCE", cnt=48),
            SimpleNamespace(label="CRYPTO", cnt=52),
        ]
        mock_client.query.return_value.result.side_effect = [baseline_rows, current_rows]

        result = compute_prediction_drift("model-v1", window_days=7, client=mock_client)
        assert len(result) == 2
        for drift in result:
            assert isinstance(drift, PredictionDrift)
            assert not drift.is_drifted

    @patch("ml.monitoring.drift.get_settings")
    def test_drifted_distribution(self, mock_settings):
        """Large distribution shift should produce high PSI."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        baseline_rows = [
            SimpleNamespace(label="ROMANCE", cnt=90),
            SimpleNamespace(label="CRYPTO", cnt=10),
        ]
        current_rows = [
            SimpleNamespace(label="ROMANCE", cnt=10),
            SimpleNamespace(label="CRYPTO", cnt=90),
        ]
        mock_client.query.return_value.result.side_effect = [baseline_rows, current_rows]

        result = compute_prediction_drift("model-v1", window_days=7, client=mock_client)
        assert len(result) == 2
        assert any(d.is_drifted for d in result)


# ---------------------------------------------------------------------------
# Tests — feature drift
# ---------------------------------------------------------------------------


class TestComputeFeatureDrift:
    @patch("ml.monitoring.drift.get_settings")
    def test_stable_features(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        # Return similar distributions for baseline and current
        vals = [SimpleNamespace(val=str(float(i))) for i in range(100)]
        mock_client.query.return_value.result.side_effect = [vals, vals] * 4  # 4 features × 2 windows

        result = compute_feature_drift("model-v1", window_days=7, client=mock_client)
        assert len(result) == 4
        for fd in result:
            assert isinstance(fd, FeatureDrift)
            assert not fd.is_drifted

    @patch("ml.monitoring.drift.get_settings")
    def test_empty_data(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = []

        result = compute_feature_drift("model-v1", features=["text_length"], window_days=7, client=mock_client)
        assert len(result) == 1
        assert result[0].psi == 0.0


# ---------------------------------------------------------------------------
# Tests — drift report
# ---------------------------------------------------------------------------


class TestComputeDriftReport:
    @patch("ml.monitoring.drift.compute_feature_drift")
    @patch("ml.monitoring.drift.compute_prediction_drift")
    def test_report_structure(self, mock_pred, mock_feat):
        mock_pred.return_value = [
            PredictionDrift(label="A", baseline_rate=0.5, current_rate=0.5, psi=0.0, is_drifted=False),
        ]
        mock_feat.return_value = [
            FeatureDrift(feature_name="text_length", psi=0.05, is_drifted=False),
        ]

        report = compute_drift_report("model-v1", window_days=7)
        assert isinstance(report, DriftReport)
        assert report.model_id == "model-v1"
        assert len(report.prediction_drift) == 1
        assert len(report.feature_drift) == 1
        assert report.report_id


# ---------------------------------------------------------------------------
# Tests — materialization
# ---------------------------------------------------------------------------


class TestMaterializeDriftMetrics:
    @patch("ml.monitoring.drift.get_settings")
    def test_materialize_rows(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []

        report = DriftReport(
            report_id="test-id",
            model_id="model-v1",
            window_start="2026-03-17T00:00:00",
            window_end="2026-03-24T00:00:00",
            baseline_start="2026-03-10T00:00:00",
            baseline_end="2026-03-17T00:00:00",
            prediction_drift=[
                PredictionDrift(label="A", baseline_rate=0.5, current_rate=0.6, psi=0.05, is_drifted=False),
            ],
            feature_drift=[
                FeatureDrift(feature_name="text_length", psi=0.03, is_drifted=False),
            ],
            computed_at="2026-03-24T00:00:00",
        )

        count = materialize_drift_metrics(report, client=mock_client)
        assert count == 2
        mock_client.insert_rows_json.assert_called_once()
        rows = mock_client.insert_rows_json.call_args[0][1]
        assert rows[0]["report_type"] == "prediction"
        assert rows[1]["report_type"] == "feature"

    @patch("ml.monitoring.drift.get_settings")
    def test_materialize_empty_report(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        report = DriftReport(
            report_id="test-id",
            model_id="model-v1",
            window_start="2026-03-17T00:00:00",
            window_end="2026-03-24T00:00:00",
            baseline_start="2026-03-10T00:00:00",
            baseline_end="2026-03-17T00:00:00",
            computed_at="2026-03-24T00:00:00",
        )

        count = materialize_drift_metrics(report, client=mock_client)
        assert count == 0
        mock_client.insert_rows_json.assert_not_called()

    @patch("ml.monitoring.drift.get_settings")
    def test_materialize_bq_error_raises(self, mock_settings):
        import pytest

        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = [{"errors": ["something"]}]

        report = DriftReport(
            report_id="test-id",
            model_id="model-v1",
            window_start="2026-03-17T00:00:00",
            window_end="2026-03-24T00:00:00",
            baseline_start="2026-03-10T00:00:00",
            baseline_end="2026-03-17T00:00:00",
            prediction_drift=[
                PredictionDrift(label="A", baseline_rate=0.5, current_rate=0.6, psi=0.05, is_drifted=False),
            ],
            computed_at="2026-03-24T00:00:00",
        )

        with pytest.raises(RuntimeError, match="Failed to materialize"):
            materialize_drift_metrics(report, client=mock_client)
