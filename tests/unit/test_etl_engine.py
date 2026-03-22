"""Unit tests for ETL engine factory and Cloud SQL connector logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.config import EtlSettings
from ml.data.etl import _build_engine


class TestBuildEngine:
    def test_raises_when_nothing_configured(self):
        etl = EtlSettings()
        with pytest.raises(ValueError, match="ETL source is not configured"):
            _build_engine(etl)

    def test_uses_connection_string_when_set(self):
        etl = EtlSettings(source_db_connection="sqlite:///:memory:")
        engine = _build_engine(etl)
        assert str(engine.url) == "sqlite:///:memory:"
        engine.dispose()

    def test_prefers_instance_over_connection_string(self):
        """When source_instance is set, Cloud SQL Connector is used."""
        mock_connector = MagicMock()
        mock_module = MagicMock()
        mock_module.Connector.return_value = mock_connector
        mock_connector.connect.return_value = MagicMock()

        etl = EtlSettings(
            source_instance="proj:region:inst",
            source_db_name="testdb",
            source_db_user="sa@proj.iam",
            source_db_connection="sqlite:///:memory:",
        )
        with patch.dict("sys.modules", {"google.cloud.sql.connector": mock_module}):
            engine = _build_engine(etl)
            assert "pg8000" in str(engine.url)
            engine.dispose()
