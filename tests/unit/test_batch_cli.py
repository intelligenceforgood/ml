"""Unit tests for batch prediction CLI entry point (Sprint 2.2)."""

from __future__ import annotations

from unittest.mock import patch

from scripts.run_batch_prediction import main, parse_args


class TestParseArgs:
    """CLI argument parsing."""

    def test_defaults(self):
        args = parse_args([])
        assert args.capability == "classification"
        assert args.model_artifact_uri == ""
        assert args.source_query is None
        assert args.dest_table is None
        assert args.batch_size == 100

    def test_all_args(self):
        args = parse_args(
            [
                "--capability",
                "embedding",
                "--model-artifact-uri",
                "gs://bucket/model/v1",
                "--source-query",
                "SELECT * FROM t",
                "--dest-table",
                "project.dataset.table",
                "--batch-size",
                "50",
            ]
        )
        assert args.capability == "embedding"
        assert args.model_artifact_uri == "gs://bucket/model/v1"
        assert args.source_query == "SELECT * FROM t"
        assert args.dest_table == "project.dataset.table"
        assert args.batch_size == 50

    def test_ner_capability(self):
        args = parse_args(["--capability", "ner"])
        assert args.capability == "ner"

    def test_risk_scoring_capability(self):
        args = parse_args(["--capability", "risk_scoring"])
        assert args.capability == "risk_scoring"


class TestMain:
    """Entry point wiring."""

    @patch("ml.serving.batch.run_batch_prediction")
    def test_calls_batch_with_parsed_args(self, mock_run):
        main(["--capability", "embedding", "--batch-size", "25"])

        mock_run.assert_called_once_with(
            capability="embedding",
            model_artifact_uri="",
            source_query=None,
            dest_table=None,
            batch_size=25,
        )
