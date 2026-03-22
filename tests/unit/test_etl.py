"""Unit tests for ETL pipeline."""

from __future__ import annotations

import pytest

from ml.data.etl import TABLE_CONFIGS, IngestConfig


class TestIngestConfig:
    def test_table_configs_not_empty(self):
        assert len(TABLE_CONFIGS) > 0

    def test_all_configs_have_required_fields(self):
        for config in TABLE_CONFIGS:
            assert config.source_table
            assert config.target_table
            assert config.primary_key
            assert config.watermark_column
            assert len(config.columns) > 0

    def test_primary_key_in_columns(self):
        for config in TABLE_CONFIGS:
            assert (
                config.primary_key in config.columns
            ), f"{config.source_table}: primary key '{config.primary_key}' not in columns"

    def test_watermark_column_in_columns(self):
        for config in TABLE_CONFIGS:
            assert (
                config.watermark_column in config.columns
            ), f"{config.source_table}: watermark '{config.watermark_column}' not in columns"

    def test_source_tables_unique(self):
        sources = [c.source_table for c in TABLE_CONFIGS]
        assert len(sources) == len(set(sources))

    def test_target_tables_unique(self):
        targets = [c.target_table for c in TABLE_CONFIGS]
        assert len(targets) == len(set(targets))

    def test_frozen(self):
        config = TABLE_CONFIGS[0]
        with pytest.raises(AttributeError):
            config.source_table = "other"

    def test_custom_config(self):
        config = IngestConfig(
            source_table="my_table",
            target_table="raw_my_table",
            primary_key="id",
            watermark_column="updated_at",
            columns=["id", "name", "updated_at"],
        )
        assert config.source_table == "my_table"
        assert config.primary_key == "id"


class TestTableMappings:
    """Verify the specific table mappings match the TDD."""

    def test_cases_mapping(self):
        cases = next(c for c in TABLE_CONFIGS if c.source_table == "cases")
        assert cases.target_table == "raw_cases"
        assert cases.watermark_column == "updated_at"
        assert "case_id" in cases.columns
        assert "classification_result" in cases.columns
        assert "risk_score" in cases.columns

    def test_no_classification_results_table(self):
        """classification_result is a JSON column on cases, not a standalone table."""
        tables = [c.source_table for c in TABLE_CONFIGS]
        assert "classification_results" not in tables

    def test_entities_mapping(self):
        ent = next(c for c in TABLE_CONFIGS if c.source_table == "entities")
        assert ent.target_table == "raw_entities"
        assert "entity_type" in ent.columns
        assert "canonical_value" in ent.columns

    def test_analyst_labels_mapping(self):
        al = next(c for c in TABLE_CONFIGS if c.source_table == "analyst_labels")
        assert al.target_table == "raw_analyst_labels"
        assert al.primary_key == "id"
        assert "axis" in al.columns
        assert "label_code" in al.columns
        assert "analyst_id" in al.columns
