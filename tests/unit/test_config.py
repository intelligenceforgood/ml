"""Unit tests for ML platform settings/config."""

from __future__ import annotations

import os
from unittest.mock import patch

from ml.config import EtlSettings, Settings, get_settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.platform.project_id == "i4g-ml"
        assert s.bigquery.dataset_id == "i4g_ml"
        assert s.etl.batch_size == 1000
        assert s.etl.source_instance == ""
        assert s.etl.source_db_connection == ""

    def test_etl_cloud_sql_fields(self):
        etl = EtlSettings(
            source_instance="proj:region:inst",
            source_db_name="mydb",
            source_db_user="sa@proj.iam",
            source_enable_iam_auth=True,
        )
        assert etl.source_instance == "proj:region:inst"
        assert etl.source_db_name == "mydb"
        assert etl.source_db_user == "sa@proj.iam"
        assert etl.source_enable_iam_auth is True


class TestEnvVarOverrides:
    def test_env_var_overrides_section_key(self):
        get_settings.cache_clear()
        env = {
            "I4G_ML_ETL__SOURCE_INSTANCE": "my-proj:us-central1:my-db",
            "I4G_ML_ETL__SOURCE_DB_NAME": "testdb",
            "I4G_ML_ETL__BATCH_SIZE": "500",
        }
        with patch.dict(os.environ, env, clear=False):
            s = get_settings()
            assert s.etl.source_instance == "my-proj:us-central1:my-db"
            assert s.etl.source_db_name == "testdb"
            assert s.etl.batch_size == 500
        get_settings.cache_clear()

    def test_env_var_overrides_platform(self):
        get_settings.cache_clear()
        with patch.dict(os.environ, {"I4G_ML_PLATFORM__PROJECT_ID": "other-proj"}, clear=False):
            s = get_settings()
            assert s.platform.project_id == "other-proj"
        get_settings.cache_clear()

    def test_non_prefixed_env_ignored(self):
        get_settings.cache_clear()
        with patch.dict(os.environ, {"SOURCE_DB_CONNECTION": "should-be-ignored"}, clear=False):
            s = get_settings()
            assert s.etl.source_db_connection != "should-be-ignored"
        get_settings.cache_clear()
