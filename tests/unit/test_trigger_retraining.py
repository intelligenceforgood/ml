"""Unit tests for the retraining trigger CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.cli.retrain import _DEFAULT_CONFIGS, run


class TestTriggerRetraining:
    @patch("ml.training.submission.aiplatform")
    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_retrain_submitted_when_conditions_met(self, mock_eval, mock_record, mock_aip):
        from ml.monitoring.triggers import RetrainingTrigger

        trigger_result = RetrainingTrigger(
            should_retrain=True,
            reasons=["data_volume: 300 new analyst labels (threshold: 200)"],
            new_analyst_label_count=300,
        )
        mock_eval.return_value = trigger_result

        mock_job = MagicMock()
        mock_job.resource_name = "projects/i4g-ml/pipelineJobs/retrain-123"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification")

        mock_eval.assert_called_once_with("classification", force=False)
        mock_record.assert_called_once_with(
            trigger_result,
            capability="classification",
            pipeline_job_name="projects/i4g-ml/pipelineJobs/retrain-123",
        )

    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_retrain_skipped_when_no_conditions(self, mock_eval, mock_record):
        from ml.monitoring.triggers import RetrainingTrigger

        trigger_result = RetrainingTrigger(should_retrain=False, reasons=[])
        mock_eval.return_value = trigger_result

        run(capability="classification")

        mock_eval.assert_called_once()
        mock_record.assert_called_once_with(trigger_result, capability="classification")

    @patch("ml.training.submission.aiplatform")
    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_force_flag_passed_through(self, mock_eval, mock_record, mock_aip):
        from ml.monitoring.triggers import RetrainingTrigger

        trigger_result = RetrainingTrigger(should_retrain=True, reasons=["forced"])
        mock_eval.return_value = trigger_result

        mock_job = MagicMock()
        mock_job.resource_name = "job/forced-1"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification", force=True)

        mock_eval.assert_called_once_with("classification", force=True)

    @patch("ml.training.submission.aiplatform")
    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_trigger_reason_extracted_from_first_reason(self, mock_eval, mock_record, mock_aip):
        from ml.monitoring.triggers import RetrainingTrigger

        trigger_result = RetrainingTrigger(
            should_retrain=True,
            reasons=["drift: some_axis PSI=0.3000", "time_elapsed: 45 days"],
        )
        mock_eval.return_value = trigger_result

        mock_job = MagicMock()
        mock_job.resource_name = "job/drift-1"
        mock_aip.PipelineJob.return_value = mock_job

        run()

        call_kwargs = mock_aip.PipelineJob.call_args[1]
        assert call_kwargs["labels"]["trigger_reason"] == "drift"

    @patch("ml.training.submission.aiplatform")
    @patch("ml.monitoring.triggers.record_trigger_event")
    @patch("ml.monitoring.triggers.evaluate_retraining_conditions")
    def test_missing_config_proceeds_without(self, mock_eval, mock_record, mock_aip, tmp_path, monkeypatch):
        from ml.monitoring.triggers import RetrainingTrigger

        monkeypatch.setitem(_DEFAULT_CONFIGS, "classification", str(tmp_path / "nope.yaml"))

        trigger_result = RetrainingTrigger(should_retrain=True, reasons=["forced"])
        mock_eval.return_value = trigger_result

        mock_job = MagicMock()
        mock_job.resource_name = "job/noconfig"
        mock_aip.PipelineJob.return_value = mock_job

        run(capability="classification")

        mock_aip.PipelineJob.assert_called_once()
