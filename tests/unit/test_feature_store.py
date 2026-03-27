"""Unit tests for Feature Store integration (Sprint 3)."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

from ml.data.feature_store import _CACHE_MAX_SIZE, _cache_get, _cache_put, _feature_cache, fetch_online_features

# ---------------------------------------------------------------------------
# Online feature serving — cache
# ---------------------------------------------------------------------------


class TestFeatureCache:
    """LRU cache for Feature Store online serving."""

    def setup_method(self):
        _feature_cache.clear()

    def teardown_method(self):
        _feature_cache.clear()

    def test_returns_none_when_feature_store_not_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            result = fetch_online_features("case-1")
        assert result is None

    @patch("ml.config.get_settings")
    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    def test_caches_result(self, mock_aip_init, mock_et_cls, mock_settings):
        """Second call with same entity_id should hit cache."""
        settings = MagicMock()
        settings.platform.project_id = "proj"
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        mock_entity_type = MagicMock()
        mock_result_df = MagicMock()
        mock_result_df.empty = False
        row_dict = {"feature_a": 1.0}
        mock_result_df.iloc.__getitem__.return_value.to_dict.return_value = row_dict
        mock_entity_type.read.return_value = mock_result_df
        mock_et_cls.return_value = mock_entity_type

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}, clear=True):
            result1 = fetch_online_features("case-cache-a")
            result2 = fetch_online_features("case-cache-a")

        # Should only call EntityType once — second is from cache
        assert mock_et_cls.call_count == 1
        assert result1 == result2

    @patch("ml.config.get_settings")
    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    def test_cache_expires(self, mock_aip_init, mock_et_cls, mock_settings):
        """Cache entries older than TTL should be evicted."""
        settings = MagicMock()
        settings.platform.project_id = "proj"
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        mock_entity_type = MagicMock()
        mock_result_df = MagicMock()
        mock_result_df.empty = False
        mock_result_df.iloc.__getitem__.return_value.to_dict.return_value = {"feature_a": 2.0}
        mock_entity_type.read.return_value = mock_result_df
        mock_et_cls.return_value = mock_entity_type

        with patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}, clear=True):
            _feature_cache["case-expire-a"] = (time.time() - 120, {"feature_a": 1.0})
            fetch_online_features("case-expire-a")

        # Should have called Feature Store (cache expired)
        assert mock_et_cls.call_count == 1

    def test_cache_put_evicts_oldest(self):
        """When cache is full, oldest entry should be evicted."""
        for i in range(_CACHE_MAX_SIZE):
            _cache_put(f"case-{i}", {"v": i})

        assert len(_feature_cache) == _CACHE_MAX_SIZE

        _cache_put("case-new", {"v": "new"})
        assert len(_feature_cache) == _CACHE_MAX_SIZE
        assert _cache_get("case-new") == {"v": "new"}

    def test_cache_get_returns_none_for_missing(self):
        assert _cache_get("nonexistent") is None

    def test_fetch_returns_none_on_exception(self):
        """Feature Store errors should return None, not raise."""
        with (
            patch("google.cloud.aiplatform.EntityType", side_effect=RuntimeError("connection failed")),
            patch("google.cloud.aiplatform.init"),
            patch("ml.config.get_settings") as mock_settings,
            patch.dict(os.environ, {"FEATURE_STORE_ID": "test-store"}),
        ):
            settings = MagicMock()
            settings.platform.project_id = "proj"
            settings.platform.region = "us-central1"
            mock_settings.return_value = settings
            result = fetch_online_features("case-err-3")

        assert result is None


# ---------------------------------------------------------------------------
# Feature Store sync
# ---------------------------------------------------------------------------


class TestSyncFeaturesToStore:
    """Tests for sync_features_to_store()."""

    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_sync_returns_zero_when_no_rows(self, mock_settings, mock_bq_cls, mock_aip_init, mock_et):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        settings.bigquery.dataset_id = "i4g_ml"
        mock_settings.return_value = settings

        mock_bq_client = MagicMock()
        mock_bq_cls.return_value = mock_bq_client
        mock_bq_client.query.return_value.result.return_value = []

        from ml.data.feature_store import sync_features_to_store

        count = sync_features_to_store("proj", "store-1")
        assert count == 0

    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_sync_ingests_and_returns_count(self, mock_settings, mock_bq_cls, mock_aip_init, mock_et):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        settings.bigquery.dataset_id = "i4g_ml"
        mock_settings.return_value = settings

        mock_bq_client = MagicMock()
        mock_bq_cls.return_value = mock_bq_client
        fake_rows = [{"entity_id": "c1"}, {"entity_id": "c2"}, {"entity_id": "c3"}]
        mock_bq_client.query.return_value.result.return_value = fake_rows

        mock_entity_type = MagicMock()
        mock_et.return_value = mock_entity_type

        from ml.data.feature_store import sync_features_to_store

        count = sync_features_to_store("proj", "store-1", incremental=False)
        assert count == 3
        mock_entity_type.ingest_from_bq.assert_called_once()

    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_incremental_query_includes_watermark(self, mock_settings, mock_bq_cls, mock_aip_init, mock_et):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        settings.bigquery.dataset_id = "i4g_ml"
        mock_settings.return_value = settings

        mock_bq_client = MagicMock()
        mock_bq_cls.return_value = mock_bq_client
        mock_bq_client.query.return_value.result.return_value = []

        from ml.data.feature_store import sync_features_to_store

        sync_features_to_store("proj", "store-1", incremental=True)

        query_arg = mock_bq_client.query.call_args[0][0]
        assert "feature_store_sync_log" in query_arg
        assert "COALESCE(MAX(ingested_at)" in query_arg

    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_full_sync_omits_watermark(self, mock_settings, mock_bq_cls, mock_aip_init, mock_et):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        settings.bigquery.dataset_id = "i4g_ml"
        mock_settings.return_value = settings

        mock_bq_client = MagicMock()
        mock_bq_cls.return_value = mock_bq_client
        mock_bq_client.query.return_value.result.return_value = []

        from ml.data.feature_store import sync_features_to_store

        sync_features_to_store("proj", "store-1", incremental=False)

        query_arg = mock_bq_client.query.call_args[0][0]
        assert "feature_store_sync_log" not in query_arg

    @patch("google.cloud.aiplatform.EntityType")
    @patch("google.cloud.aiplatform.init")
    @patch("google.cloud.bigquery.Client")
    @patch("ml.config.get_settings")
    def test_sync_logs_event_to_bq(self, mock_settings, mock_bq_cls, mock_aip_init, mock_et):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        settings.bigquery.dataset_id = "i4g_ml"
        mock_settings.return_value = settings

        mock_bq_client = MagicMock()
        mock_bq_cls.return_value = mock_bq_client
        fake_rows = [{"entity_id": "c1"}]
        mock_bq_client.query.return_value.result.return_value = fake_rows
        mock_et.return_value = MagicMock()

        from ml.data.feature_store import sync_features_to_store

        sync_features_to_store("proj", "store-1")

        mock_bq_client.insert_rows_json.assert_called_once()
        log_row = mock_bq_client.insert_rows_json.call_args[0][1][0]
        assert log_row["entity_type"] == "case"
        assert log_row["entities_synced"] == 1
