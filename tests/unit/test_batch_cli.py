"""Unit tests for batch prediction CLI command (Sprint 2.2)."""

from __future__ import annotations

from unittest.mock import patch

from ml.cli.serve import run_batch


class TestRunBatch:
    """Batch prediction CLI wiring."""

    @patch("ml.serving.batch.run_batch_prediction")
    def test_defaults(self, mock_run):
        run_batch()

        mock_run.assert_called_once_with(
            capability="classification",
            model_artifact_uri="",
            source_query=None,
            dest_table=None,
            batch_size=100,
        )

    @patch("ml.serving.batch.run_batch_prediction")
    def test_all_args(self, mock_run):
        run_batch(
            capability="embedding",
            model_artifact_uri="gs://bucket/model/v1",
            source_query="SELECT * FROM t",
            dest_table="project.dataset.table",
            batch_size=50,
        )

        mock_run.assert_called_once_with(
            capability="embedding",
            model_artifact_uri="gs://bucket/model/v1",
            source_query="SELECT * FROM t",
            dest_table="project.dataset.table",
            batch_size=50,
        )

    @patch("ml.serving.batch.run_batch_prediction")
    def test_ner_capability(self, mock_run):
        run_batch(capability="ner")

        mock_run.assert_called_once()
        assert mock_run.call_args[1]["capability"] == "ner"

    @patch("ml.serving.batch.run_batch_prediction")
    def test_risk_scoring_capability(self, mock_run):
        run_batch(capability="risk_scoring")

        mock_run.assert_called_once()
        assert mock_run.call_args[1]["capability"] == "risk_scoring"

    @patch("ml.serving.batch.run_batch_prediction")
    def test_custom_batch_size(self, mock_run):
        run_batch(capability="embedding", batch_size=25)

        mock_run.assert_called_once_with(
            capability="embedding",
            model_artifact_uri="",
            source_query=None,
            dest_table=None,
            batch_size=25,
        )
