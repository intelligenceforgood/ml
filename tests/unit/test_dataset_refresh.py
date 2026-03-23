"""Unit tests for automated dataset refresh — Sprint 2 task 2.3."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestDatasetLabelSourcePriority:
    """2.3.1 / 2.3.2 / 2.3.6 — Verify analyst labels take priority over bootstrap."""

    def test_default_query_contains_label_source(self):
        """The default dataset query should include label_source column."""
        from ml.data.datasets import create_dataset_version

        # We need to inspect the query that gets built, so we patch BQ
        with (
            patch("ml.data.datasets.bigquery.Client") as mock_bq,
            patch("ml.data.datasets.storage.Client"),
        ):
            bq_client = mock_bq.return_value

            # Make the query call capture the SQL
            captured_queries = []
            _original_query = bq_client.query

            def capture_query(sql, **kwargs):
                captured_queries.append(sql)
                mock_result = MagicMock()
                # Return a small DataFrame for the main query
                mock_result.to_dataframe.return_value = pd.DataFrame(
                    {
                        "case_id": [f"c{i}" for i in range(100)],
                        "text": [f"text {i}" for i in range(100)],
                        "label_code": ["INTENT.ROMANCE"] * 100,
                        "label_source": ["analyst"] * 50 + ["llm_bootstrap"] * 50,
                        "label_timestamp": ["2026-01-01"] * 100,
                    }
                )
                # For version query
                mock_row = MagicMock()
                mock_row.next_v = 1
                mock_result.result.return_value = [mock_row]
                return mock_result

            bq_client.query.side_effect = capture_query
            bq_client.insert_rows_json.return_value = []

            # Mock GCS blob
            mock_blob = MagicMock()

            with patch("ml.data.datasets.storage.Client") as mock_storage:
                mock_storage.return_value.bucket.return_value.blob.return_value = mock_blob

                _metadata = create_dataset_version(
                    capability="classification",
                    version=1,
                    min_samples_per_class=1,
                )

            # Verify the default query uses the label source priority logic
            main_query = captured_queries[0]
            assert "analyst_corrections" in main_query
            assert "bootstrap_labels" in main_query
            assert "label_source" in main_query
            assert "COALESCE" in main_query
            assert "outcome_log" in main_query

    def test_query_prefers_analyst_over_bootstrap(self):
        """The COALESCE logic should prefer analyst corrections."""
        from ml.data.datasets import create_dataset_version

        with (
            patch("ml.data.datasets.bigquery.Client") as mock_bq,
            patch("ml.data.datasets.storage.Client") as mock_storage,
        ):
            bq_client = mock_bq.return_value

            captured_queries = []

            def capture_query(sql, **kwargs):
                captured_queries.append(sql)
                mock_result = MagicMock()
                mock_result.to_dataframe.return_value = pd.DataFrame(
                    {
                        "case_id": [f"c{i}" for i in range(100)],
                        "text": [f"text {i}" for i in range(100)],
                        "label_code": ["INTENT.ROMANCE"] * 100,
                        "label_source": ["analyst"] * 100,
                        "label_timestamp": ["2026-01-01"] * 100,
                    }
                )
                mock_row = MagicMock()
                mock_row.next_v = 5
                mock_result.result.return_value = [mock_row]
                return mock_result

            bq_client.query.side_effect = capture_query
            bq_client.insert_rows_json.return_value = []
            mock_storage.return_value.bucket.return_value.blob.return_value = MagicMock()

            create_dataset_version(
                capability="classification",
                version=5,
                min_samples_per_class=1,
            )

            # Verify COALESCE puts analyst first
            main_query = captured_queries[0]
            coalesce_pos = main_query.index("COALESCE(ac.intent_label")
            assert coalesce_pos > 0  # analyst (ac) is first arg


class TestDatasetVersionAutoIncrement:
    """2.3.4 — Verify auto-increment queries registry for latest version."""

    @patch("ml.data.datasets.bigquery.Client")
    @patch("ml.data.datasets.storage.Client")
    def test_auto_increment_from_registry(self, mock_storage, mock_bq):
        from ml.data.datasets import create_dataset_version

        bq_client = mock_bq.return_value

        call_count = 0

        def query_side_effect(sql, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()

            if call_count == 1:
                # Main data query
                mock_result.to_dataframe.return_value = pd.DataFrame(
                    {
                        "case_id": [f"c{i}" for i in range(100)],
                        "text": [f"text {i}" for i in range(100)],
                        "label_code": ["INTENT.ROMANCE"] * 100,
                        "label_source": ["analyst"] * 100,
                        "label_timestamp": ["2026-01-01"] * 100,
                    }
                )
                return mock_result
            elif call_count == 2:
                # Version query: simulate existing latest = 3 → next = 4
                mock_row = MagicMock()
                mock_row.next_v = 4
                mock_result.result.return_value = [mock_row]
                return mock_result
            return mock_result

        bq_client.query.side_effect = query_side_effect
        bq_client.insert_rows_json.return_value = []
        mock_storage.return_value.bucket.return_value.blob.return_value = MagicMock()

        metadata = create_dataset_version(
            capability="classification",
            min_samples_per_class=1,
        )

        assert metadata["version"] == 4
        assert metadata["dataset_id"] == "classification-v4"


class TestDatasetValidationGate:
    """2.3.5 — Verify validation gate checks min_samples and class_balance."""

    @patch("ml.data.datasets.bigquery.Client")
    @patch("ml.data.datasets.storage.Client")
    def test_empty_dataset_fails_validation(self, mock_storage, mock_bq):
        from ml.data.datasets import create_dataset_version

        bq_client = mock_bq.return_value
        bq_client.query.return_value.to_dataframe.return_value = pd.DataFrame()

        with pytest.raises(ValueError, match="Dataset is empty"):
            create_dataset_version(capability="classification", version=1)

    @patch("ml.data.datasets.bigquery.Client")
    @patch("ml.data.datasets.storage.Client")
    def test_duplicates_fail_validation(self, mock_storage, mock_bq):
        from ml.data.datasets import create_dataset_version

        bq_client = mock_bq.return_value
        bq_client.query.return_value.to_dataframe.return_value = pd.DataFrame(
            {
                "case_id": ["c1", "c1", "c2"],
                "text": ["a", "b", "c"],
                "label_code": ["INTENT.ROMANCE"] * 3,
            }
        )

        with pytest.raises(ValueError, match="duplicate"):
            create_dataset_version(
                capability="classification",
                version=1,
                label_column="label_code",
            )


class TestRefreshPipeline:
    """2.3.3 — Verify data refresh pipeline orchestration."""

    @patch("ml.data.refresh.create_dataset_version")
    @patch("ml.data.refresh.run_incremental_ingest")
    def test_refresh_calls_etl_then_dataset(self, mock_ingest, mock_dataset):
        from ml.data.refresh import refresh_dataset

        mock_ingest.return_value = {"raw_cases": 10, "raw_entities": 5, "raw_analyst_labels": 3}
        mock_dataset.return_value = {
            "dataset_id": "classification-v2",
            "version": 2,
            "gcs_path": "gs://bucket/datasets/classification/v2",
            "train_count": 70,
            "eval_count": 15,
            "test_count": 15,
        }

        result = refresh_dataset()

        mock_ingest.assert_called_once()
        mock_dataset.assert_called_once_with(
            capability="classification",
            min_samples_per_class=50,
            redact=True,
        )
        assert result["version"] == 2

    @patch("ml.data.refresh.create_dataset_version")
    @patch("ml.data.refresh.run_incremental_ingest")
    def test_refresh_continues_on_etl_failure(self, mock_ingest, mock_dataset):
        """ETL failure for one table should not prevent dataset export."""
        from ml.data.refresh import refresh_dataset

        mock_ingest.return_value = {"raw_cases": 10, "raw_entities": -1, "raw_analyst_labels": 3}
        mock_dataset.return_value = {
            "dataset_id": "classification-v3",
            "version": 3,
            "gcs_path": "gs://bucket/datasets/classification/v3",
            "train_count": 70,
            "eval_count": 15,
            "test_count": 15,
        }

        result = refresh_dataset()
        assert result["version"] == 3
        mock_dataset.assert_called_once()
