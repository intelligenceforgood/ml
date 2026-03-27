"""Unit tests for pipeline submission library."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestSubmitPipeline:
    @patch("ml.training.submission.aiplatform")
    def test_submit_with_config(self, mock_aip, tmp_path):
        from ml.training.submission import submit_pipeline

        # Create a minimal config YAML
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "model_id: test-model\n"
            "capability: classification\n"
            "framework: xgboost\n"
            "eval_gate:\n"
            "  min_overall_f1: 0.6\n"
            "  max_per_axis_regression: 0.05\n"
            "resources:\n"
            "  machine_type: n1-standard-8\n"
        )

        mock_job = MagicMock()
        mock_job.resource_name = "projects/i4g-ml/locations/us-central1/pipelineJobs/test-123"
        mock_aip.PipelineJob.return_value = mock_job

        result = submit_pipeline(
            config_path=str(config_file),
            experiment_name="test-experiment",
            trigger_reason="data_volume",
        )

        assert result == "projects/i4g-ml/locations/us-central1/pipelineJobs/test-123"
        mock_job.submit.assert_called_once()

        # Verify parameter values
        call_kwargs = mock_aip.PipelineJob.call_args[1]
        params = call_kwargs["parameter_values"]
        assert params["capability"] == "classification"
        assert params["min_overall_f1"] == 0.6
        assert params["machine_type"] == "n1-standard-8"

        # Verify labels include trigger reason
        labels = call_kwargs["labels"]
        assert labels["trigger_reason"] == "data_volume"
        assert labels["framework"] == "xgboost"

    @patch("ml.training.submission.aiplatform")
    def test_submit_without_config(self, mock_aip):
        from ml.training.submission import submit_pipeline

        mock_job = MagicMock()
        mock_job.resource_name = "pipelineJobs/manual-123"
        mock_aip.PipelineJob.return_value = mock_job

        result = submit_pipeline(trigger_reason="manual")

        assert result == "pipelineJobs/manual-123"
        mock_job.submit.assert_called_once()

    @patch("ml.training.submission.aiplatform")
    def test_framework_maps_to_container(self, mock_aip, tmp_path):
        from ml.training.submission import submit_pipeline

        config_file = tmp_path / "xgb.yaml"
        config_file.write_text("framework: xgboost\ncapability: classification\n")

        mock_job = MagicMock()
        mock_job.resource_name = "job/1"
        mock_aip.PipelineJob.return_value = mock_job

        submit_pipeline(config_path=str(config_file))

        params = mock_aip.PipelineJob.call_args[1]["parameter_values"]
        assert "train-xgboost" in params["container_uri"]

    @patch("ml.training.submission.aiplatform")
    def test_pytorch_default_container(self, mock_aip, tmp_path):
        from ml.training.submission import submit_pipeline

        config_file = tmp_path / "pt.yaml"
        config_file.write_text("framework: pytorch\ncapability: classification\n")

        mock_job = MagicMock()
        mock_job.resource_name = "job/2"
        mock_aip.PipelineJob.return_value = mock_job

        submit_pipeline(config_path=str(config_file))

        params = mock_aip.PipelineJob.call_args[1]["parameter_values"]
        assert "train-pytorch" in params["container_uri"]

    @patch("ml.training.submission.aiplatform")
    def test_custom_image_tag(self, mock_aip, tmp_path):
        from ml.training.submission import submit_pipeline

        config_file = tmp_path / "c.yaml"
        config_file.write_text("framework: xgboost\ncapability: classification\n")

        mock_job = MagicMock()
        mock_job.resource_name = "job/3"
        mock_aip.PipelineJob.return_value = mock_job

        submit_pipeline(config_path=str(config_file), image_tag="prod")

        params = mock_aip.PipelineJob.call_args[1]["parameter_values"]
        assert ":prod" in params["container_uri"]

    @patch("ml.training.submission.aiplatform")
    def test_experiment_name_auto_generated(self, mock_aip, tmp_path):
        from ml.training.submission import submit_pipeline

        config_file = tmp_path / "c.yaml"
        config_file.write_text("model_id: my-model\ncapability: classification\n")

        mock_job = MagicMock()
        mock_job.resource_name = "job/4"
        mock_aip.PipelineJob.return_value = mock_job

        submit_pipeline(config_path=str(config_file))

        call_kwargs = mock_aip.PipelineJob.call_args[1]
        assert call_kwargs["display_name"].startswith("my-model-")

    def test_auto_compile_skips_when_no_source(self, tmp_path, monkeypatch):
        from ml.training.submission import _auto_compile_if_stale

        monkeypatch.chdir(tmp_path)
        # No source file, no compilation attempt, no error
        _auto_compile_if_stale("nonexistent_pipeline.yaml")
