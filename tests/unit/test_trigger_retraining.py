"""Unit tests for the trigger_retraining Cloud Run Job entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


class TestTriggerRetraining:
    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        yield
        sys.path.remove(str(SCRIPTS_DIR))

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    @patch("submit_pipeline.aiplatform")
    def test_retrain_submitted_when_conditions_met(self, mock_aip, mock_eval, mock_record):
        from trigger_retraining import run

        from ml.monitoring.triggers import RetrainingTrigger

        trigger = RetrainingTrigger(
            should_retrain=True,
            reasons=["data_volume: 300 new analyst labels (threshold: 200)"],
            new_analyst_label_count=300,
        )
        mock_eval.return_value = trigger

        mock_job = MagicMock()
        mock_job.resource_name = "projects/i4g-ml/pipelineJobs/retrain-123"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification")

        mock_eval.assert_called_once_with("classification", force=False)
        mock_record.assert_called_once_with(
            trigger,
            capability="classification",
            pipeline_job_name="projects/i4g-ml/pipelineJobs/retrain-123",
        )

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_retrain_skipped_when_no_conditions(self, mock_eval, mock_record):
        from trigger_retraining import run

        from ml.monitoring.triggers import RetrainingTrigger

        trigger = RetrainingTrigger(should_retrain=False, reasons=[])
        mock_eval.return_value = trigger

        run(capability="classification")

        mock_eval.assert_called_once()
        # Should still record the event (skipped)
        mock_record.assert_called_once_with(trigger, capability="classification")

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    @patch("submit_pipeline.aiplatform")
    def test_force_flag_passed_through(self, mock_aip, mock_eval, mock_record):
        from trigger_retraining import run

        from ml.monitoring.triggers import RetrainingTrigger

        trigger = RetrainingTrigger(should_retrain=True, reasons=["forced"])
        mock_eval.return_value = trigger

        mock_job = MagicMock()
        mock_job.resource_name = "job/forced-1"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification", force=True)

        mock_eval.assert_called_once_with("classification", force=True)

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    @patch("submit_pipeline.aiplatform")
    def test_trigger_reason_extracted_from_first_reason(self, mock_aip, mock_eval, mock_record):
        from trigger_retraining import run

        from ml.monitoring.triggers import RetrainingTrigger

        trigger = RetrainingTrigger(
            should_retrain=True,
            reasons=["drift: some_axis PSI=0.3000", "time_elapsed: 45 days"],
        )
        mock_eval.return_value = trigger

        mock_job = MagicMock()
        mock_job.resource_name = "job/drift-1"
        mock_aip.PipelineJob.return_value = mock_job

        run()

        # The trigger_reason passed to submit_pipeline should be "drift"
        call_kwargs = mock_aip.PipelineJob.call_args[1]
        assert call_kwargs["labels"]["trigger_reason"] == "drift"

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    @patch("submit_pipeline.aiplatform")
    def test_missing_config_proceeds_without(self, mock_aip, mock_eval, mock_record, tmp_path, monkeypatch):
        from trigger_retraining import _DEFAULT_CONFIGS, run

        from ml.monitoring.triggers import RetrainingTrigger

        # Point config to a nonexistent path
        monkeypatch.setitem(_DEFAULT_CONFIGS, "classification", str(tmp_path / "nope.yaml"))

        trigger = RetrainingTrigger(should_retrain=True, reasons=["forced"])
        mock_eval.return_value = trigger

        mock_job = MagicMock()
        mock_job.resource_name = "job/noconfig"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification")

        # Pipeline still submitted (config_path=None fallback)
        mock_aip.PipelineJob.assert_called_once()
