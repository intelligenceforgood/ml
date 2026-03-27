"""Unit tests for risk dataset creation (Sprint 4.2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestCreateRiskDatasetVersion:
    """Tests for create_risk_dataset_version()."""

    def _make_settings(self):
        s = MagicMock()
        s.platform.project_id = "proj"
        s.bigquery.dataset_id = "i4g_ml"
        s.storage.datasets_prefix = "datasets"
        s.storage.data_bucket = "bucket"
        return s

    def _make_risk_df(self, n: int = 200):
        """Build a DataFrame that passes validation (n rows, non-degenerate)."""
        import numpy as np

        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "case_id": [f"c{i}" for i in range(n)],
                "text": ["narrative"] * n,
                "risk_label": rng.uniform(0.0, 1.0, size=n),
                "label_source": ["analyst_severity"] * (n // 2) + ["loss_proxy"] * (n - n // 2),
                "text_length": [100] * n,
                "word_count": [20] * n,
                "entity_count": [3] * n,
                "has_crypto_wallet": [False] * n,
                "has_bank_account": [True] * n,
                "has_phone": [False] * n,
                "has_email": [True] * n,
                "classification_axis_count": [2] * n,
                "current_classification_conf": [0.8] * n,
                "shared_entity_count": [1] * n,
                "entity_reuse_frequency": [0.5] * n,
                "cluster_size": [3] * n,
            }
        )

    @patch("ml.data.datasets._export_jsonl")
    @patch("ml.data.datasets.bigquery")
    @patch("ml.data.datasets.get_settings")
    def test_creates_dataset_and_registers(self, mock_settings, mock_bq, mock_export):
        from ml.data.datasets import create_risk_dataset_version

        mock_settings.return_value = self._make_settings()
        mock_client = MagicMock()
        mock_bq.Client.return_value = mock_client
        mock_bq.QueryJobConfig = MagicMock()
        mock_bq.ScalarQueryParameter = MagicMock()

        df = self._make_risk_df(200)
        mock_client.query.return_value.to_dataframe.return_value = df
        # Version auto-increment query
        mock_version_row = MagicMock()
        mock_version_row.next_v = 3
        mock_client.query.return_value.result.return_value = [mock_version_row]

        result = create_risk_dataset_version()

        assert result["capability"] == "risk_scoring"
        assert result["version"] == 3
        assert result["train_count"] > 0
        # 3 exports: train, eval, test
        assert mock_export.call_count == 3
        # Should register in BQ
        mock_client.insert_rows_json.assert_called_once()

    @patch("ml.data.datasets.bigquery")
    @patch("ml.data.datasets.get_settings")
    def test_raises_on_insufficient_samples(self, mock_settings, mock_bq):
        from ml.data.datasets import create_risk_dataset_version

        mock_settings.return_value = self._make_settings()
        mock_client = MagicMock()
        mock_bq.Client.return_value = mock_client

        df = self._make_risk_df(50)  # below default 100 min
        mock_client.query.return_value.to_dataframe.return_value = df

        with pytest.raises(ValueError, match="minimum is 100"):
            create_risk_dataset_version()

    @patch("ml.data.datasets.bigquery")
    @patch("ml.data.datasets.get_settings")
    def test_raises_on_degenerate_distribution(self, mock_settings, mock_bq):
        from ml.data.datasets import create_risk_dataset_version

        mock_settings.return_value = self._make_settings()
        mock_client = MagicMock()
        mock_bq.Client.return_value = mock_client

        df = self._make_risk_df(200)
        df["risk_label"] = 0.5  # constant → std = 0
        mock_client.query.return_value.to_dataframe.return_value = df

        with pytest.raises(ValueError, match="degenerate"):
            create_risk_dataset_version()
