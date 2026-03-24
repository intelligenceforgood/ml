"""Unit tests for retraining trigger logic with mock BigQuery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ml.monitoring.triggers import RetrainingTrigger, evaluate_retraining_conditions, record_trigger_event

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
# Tests — evaluate_retraining_conditions
# ---------------------------------------------------------------------------


class TestEvaluateRetrainingConditions:
    @patch("ml.monitoring.triggers.get_settings")
    def test_force_always_retrains(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        result = evaluate_retraining_conditions("classification", force=True, client=MagicMock())
        assert result.should_retrain is True
        assert "forced" in result.reasons

    @patch("ml.monitoring.triggers.get_settings")
    def test_no_previous_training(self, mock_settings):
        """No training record should trigger retraining."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        # last training query → None
        last_train_result = MagicMock()
        last_train_result.result.return_value = [SimpleNamespace(last_training=None)]

        # label count query → 250 (above threshold)
        label_result = MagicMock()
        label_result.result.return_value = [SimpleNamespace(cnt=250)]

        # drift query → no drift
        drift_result = MagicMock()
        drift_result.result.return_value = []

        mock_client.query.side_effect = [last_train_result, label_result, drift_result]

        result = evaluate_retraining_conditions("classification", client=mock_client)
        assert result.should_retrain is True
        assert any("no_previous_training" in r for r in result.reasons)

    @patch("ml.monitoring.triggers.get_settings")
    def test_data_volume_trigger(self, mock_settings):
        """200+ new analyst labels should trigger retraining."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        last_training = datetime.now(UTC) - timedelta(days=10)

        last_train_result = MagicMock()
        last_train_result.result.return_value = [SimpleNamespace(last_training=last_training)]

        label_result = MagicMock()
        label_result.result.return_value = [SimpleNamespace(cnt=250)]

        drift_result = MagicMock()
        drift_result.result.return_value = []

        mock_client.query.side_effect = [last_train_result, label_result, drift_result]

        result = evaluate_retraining_conditions("classification", client=mock_client)
        assert result.should_retrain is True
        assert result.new_analyst_label_count == 250
        assert any("data_volume" in r for r in result.reasons)

    @patch("ml.monitoring.triggers.get_settings")
    def test_drift_trigger(self, mock_settings):
        """Drifted axis should trigger retraining."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        last_training = datetime.now(UTC) - timedelta(days=10)

        last_train_result = MagicMock()
        last_train_result.result.return_value = [SimpleNamespace(last_training=last_training)]

        label_result = MagicMock()
        label_result.result.return_value = [SimpleNamespace(cnt=50)]  # below threshold

        drift_result = MagicMock()
        drift_result.result.return_value = [
            SimpleNamespace(axis_or_feature="intent", psi=0.35, is_drifted=True),
        ]

        mock_client.query.side_effect = [last_train_result, label_result, drift_result]

        result = evaluate_retraining_conditions("classification", client=mock_client)
        assert result.should_retrain is True
        assert result.max_drift_psi == 0.35
        assert any("drift" in r for r in result.reasons)

    @patch("ml.monitoring.triggers.get_settings")
    def test_time_elapsed_trigger(self, mock_settings):
        """Training older than 30 days should trigger retraining."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        last_training = datetime.now(UTC) - timedelta(days=45)

        last_train_result = MagicMock()
        last_train_result.result.return_value = [SimpleNamespace(last_training=last_training)]

        label_result = MagicMock()
        label_result.result.return_value = [SimpleNamespace(cnt=50)]

        drift_result = MagicMock()
        drift_result.result.return_value = []

        mock_client.query.side_effect = [last_train_result, label_result, drift_result]

        result = evaluate_retraining_conditions("classification", client=mock_client)
        assert result.should_retrain is True
        assert any("time_elapsed" in r for r in result.reasons)

    @patch("ml.monitoring.triggers.get_settings")
    def test_no_conditions_met_skips(self, mock_settings):
        """Recent training, few labels, no drift → should not retrain."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        last_training = datetime.now(UTC) - timedelta(days=5)

        last_train_result = MagicMock()
        last_train_result.result.return_value = [SimpleNamespace(last_training=last_training)]

        label_result = MagicMock()
        label_result.result.return_value = [SimpleNamespace(cnt=50)]

        drift_result = MagicMock()
        drift_result.result.return_value = [
            SimpleNamespace(axis_or_feature="intent", psi=0.05, is_drifted=False),
        ]

        mock_client.query.side_effect = [last_train_result, label_result, drift_result]

        result = evaluate_retraining_conditions("classification", client=mock_client)
        assert result.should_retrain is False
        assert len(result.reasons) == 0


# ---------------------------------------------------------------------------
# Tests — record_trigger_event
# ---------------------------------------------------------------------------


class TestRecordTriggerEvent:
    @patch("ml.monitoring.triggers.get_settings")
    def test_logs_event(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = []

        trigger = RetrainingTrigger(
            should_retrain=True,
            reasons=["data_volume: 300 new labels"],
            new_analyst_label_count=300,
            max_drift_psi=0.0,
        )

        record_trigger_event(trigger, capability="classification", pipeline_job_name="job-123", client=mock_client)

        mock_client.insert_rows_json.assert_called_once()
        row = mock_client.insert_rows_json.call_args[0][1][0]
        assert row["capability"] == "classification"
        assert row["should_retrain"] is True
        assert row["pipeline_job_name"] == "job-123"

    @patch("ml.monitoring.triggers.get_settings")
    def test_bq_error_raises(self, mock_settings):
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.insert_rows_json.return_value = [{"errors": ["insert failed"]}]

        trigger = RetrainingTrigger(should_retrain=False, reasons=[])

        with pytest.raises(RuntimeError, match="Failed to record trigger event"):
            record_trigger_event(trigger, client=mock_client)
