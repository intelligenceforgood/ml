"""Unit tests for data validation."""

from __future__ import annotations

import pandas as pd

from ml.data.validation import ValidationResult, validate_dataset


class TestValidateDataset:
    def test_empty_dataframe_fails(self):
        df = pd.DataFrame()
        result = validate_dataset(df)
        assert not result.passed
        assert "empty" in result.errors[0].lower()

    def test_valid_dataset_passes(self):
        df = pd.DataFrame(
            {
                "case_id": [f"case-{i}" for i in range(100)],
                "text": ["some narrative text"] * 100,
                "labels": ["INTENT.ROMANCE"] * 50 + ["INTENT.INVESTMENT"] * 50,
            }
        )
        result = validate_dataset(df, min_samples_per_class=10)
        assert result.passed
        assert len(result.errors) == 0

    def test_duplicate_case_ids_flagged(self):
        df = pd.DataFrame(
            {
                "case_id": ["case-1", "case-1", "case-2"],
                "text": ["text"] * 3,
                "labels": ["A", "A", "B"],
            }
        )
        result = validate_dataset(df, min_samples_per_class=1)
        assert not result.passed
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_high_null_rate_flagged(self):
        df = pd.DataFrame(
            {
                "case_id": [f"case-{i}" for i in range(10)],
                "text": [None] * 5 + ["text"] * 5,
                "labels": ["A"] * 10,
            }
        )
        result = validate_dataset(df, max_null_rate=0.2, min_samples_per_class=1)
        assert not result.passed
        assert any("null rate" in e.lower() for e in result.errors)

    def test_low_class_count_warns(self):
        df = pd.DataFrame(
            {
                "case_id": [f"case-{i}" for i in range(5)],
                "text": ["text"] * 5,
                "labels": ["A"] * 5,
            }
        )
        result = validate_dataset(df, min_samples_per_class=10)
        assert result.passed  # Warning, not error
        assert len(result.warnings) > 0

    def test_stats_populated(self):
        df = pd.DataFrame(
            {
                "case_id": ["c1", "c2"],
                "text": ["a", "b"],
                "labels": ["X", "Y"],
            }
        )
        result = validate_dataset(df, min_samples_per_class=1)
        assert result.stats["total_rows"] == 2
        assert "null_rates" in result.stats


class TestValidationResult:
    def test_construction(self):
        r = ValidationResult(passed=True)
        assert r.passed
        assert r.errors == []
        assert r.warnings == []

    def test_with_errors(self):
        r = ValidationResult(passed=False, errors=["bad data"])
        assert not r.passed
        assert len(r.errors) == 1
