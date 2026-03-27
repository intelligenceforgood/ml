"""Unit tests for batch prediction module (Sprint 2)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.serving.batch import _extract_confidence, _write_batch_results, run_batch_prediction


class TestExtractConfidence:
    """Confidence extraction from prediction dicts."""

    def test_multi_axis_classification(self):
        pred = {
            "INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9},
            "CHANNEL": {"code": "CHANNEL.SOCIAL", "confidence": 0.7},
        }
        assert _extract_confidence(pred) == (0.9 + 0.7) / 2

    def test_single_float_value(self):
        pred = {"risk_score": 0.65}
        assert _extract_confidence(pred) == 0.65

    def test_empty_prediction(self):
        assert _extract_confidence({}) == 0.0

    def test_mixed_types(self):
        pred = {
            "INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.8},
            "extra": "not-a-dict",
        }
        assert _extract_confidence(pred) == 0.8


class TestWriteBatchResults:
    """Writing results to BigQuery."""

    def test_creates_table_and_writes(self):
        mock_client = MagicMock()
        mock_client.create_table.return_value = None
        mock_client.insert_rows_json.return_value = []

        results = [
            {
                "case_id": "c1",
                "prediction_id": "p1",
                "capability": "classification",
                "prediction": '{"INTENT": {"code": "INTENT.ROMANCE"}}',
                "confidence": 0.9,
                "model_artifact_uri": "gs://bucket/model/v1",
                "predicted_at": "2026-03-27T00:00:00",
            },
        ]

        _write_batch_results(mock_client, "project.dataset.batch_table", results)

        mock_client.create_table.assert_called_once()
        mock_client.insert_rows_json.assert_called_once_with("project.dataset.batch_table", results)

    def test_skips_table_creation_if_exists(self):
        mock_client = MagicMock()
        mock_client.create_table.side_effect = Exception("Already exists")
        mock_client.insert_rows_json.return_value = []

        results = [
            {
                "case_id": "c1",
                "prediction_id": "p1",
                "capability": "classification",
                "prediction": "{}",
                "confidence": 0.5,
                "model_artifact_uri": "",
                "predicted_at": "2026-03-27",
            }
        ]

        # Should not raise — table creation failure is tolerated
        _write_batch_results(mock_client, "project.dataset.batch_table", results)
        mock_client.insert_rows_json.assert_called_once()


class TestRunBatchPrediction:
    """End-to-end batch prediction with mocked BQ and model."""

    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_classification_stub_mode(self, mock_get_settings, mock_bq_client_cls):
        """Without a real model, should produce stub predictions."""
        mock_settings = SimpleNamespace(
            platform=SimpleNamespace(project_id="test-project"),
            bigquery=SimpleNamespace(dataset_id="test_dataset"),
        )
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_bq_client_cls.return_value = mock_client

        # Fake source rows
        mock_rows = [
            SimpleNamespace(
                case_id="case-1",
                text="test text",
                text_length=9,
                word_count=2,
                entity_count=0,
                has_crypto_wallet=False,
                has_bank_account=False,
                has_phone=False,
                has_email=False,
                classification_axis_count=0,
                current_classification_conf=0.0,
            ),
            SimpleNamespace(
                case_id="case-2",
                text="another case",
                text_length=12,
                word_count=2,
                entity_count=0,
                has_crypto_wallet=False,
                has_bank_account=False,
                has_phone=False,
                has_email=False,
                classification_axis_count=0,
                current_classification_conf=0.0,
            ),
        ]
        mock_client.query.return_value.result.return_value = mock_rows
        mock_client.insert_rows_json.return_value = []
        mock_client.create_table.return_value = None

        run_batch_prediction(
            capability="classification",
            model_artifact_uri="",
            dest_table="test-project.test_dataset.batch_test",
            batch_size=10,
        )

        # Should have written 2 result rows
        mock_client.insert_rows_json.assert_called()
        written = mock_client.insert_rows_json.call_args[0][1]
        assert len(written) == 2
        assert all(r["capability"] == "classification" for r in written)

    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_batch_chunking(self, mock_get_settings, mock_bq_client_cls):
        """Progress logging happens at batch boundaries."""
        mock_settings = SimpleNamespace(
            platform=SimpleNamespace(project_id="tp"),
            bigquery=SimpleNamespace(dataset_id="td"),
        )
        mock_get_settings.return_value = mock_settings

        mock_client = MagicMock()
        mock_bq_client_cls.return_value = mock_client

        # 5 rows, batch_size=2 → 3 batches
        mock_rows = [
            SimpleNamespace(
                case_id=f"c{i}",
                text=f"text {i}",
                text_length=6,
                word_count=2,
                entity_count=0,
                has_crypto_wallet=False,
                has_bank_account=False,
                has_phone=False,
                has_email=False,
                classification_axis_count=0,
                current_classification_conf=0.0,
            )
            for i in range(5)
        ]
        mock_client.query.return_value.result.return_value = mock_rows
        mock_client.insert_rows_json.return_value = []
        mock_client.create_table.return_value = None

        run_batch_prediction(
            capability="classification",
            dest_table="tp.td.batch_chunk",
            batch_size=2,
        )

        written = mock_client.insert_rows_json.call_args[0][1]
        assert len(written) == 5
