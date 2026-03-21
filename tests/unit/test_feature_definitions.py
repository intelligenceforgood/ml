"""Tests for FeatureDefinition catalog and helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml.data.features import (
    FEATURE_CATALOG,
    ComputeMethod,
    FeatureDefinition,
    FeatureType,
    get_feature_names,
)


class TestFeatureDefinition:
    def test_catalog_is_nonempty(self):
        assert len(FEATURE_CATALOG) > 0

    def test_names_are_unique(self):
        names = [f.name for f in FEATURE_CATALOG]
        assert len(names) == len(set(names))

    def test_all_versions_positive(self):
        for feat in FEATURE_CATALOG:
            assert feat.version >= 1

    def test_frozen_model(self):
        feat = FEATURE_CATALOG[0]
        with pytest.raises(ValidationError):
            feat.name = "hacked"

    def test_custom_feature(self):
        feat = FeatureDefinition(
            name="test_feat",
            feature_type=FeatureType.CATEGORICAL,
            description="A test feature",
            compute_method=ComputeMethod.PYTHON,
            version=2,
        )
        assert feat.name == "test_feat"
        assert feat.feature_type == FeatureType.CATEGORICAL
        assert feat.version == 2


class TestGetFeatureNames:
    def test_all_names(self):
        names = get_feature_names()
        assert len(names) == len(FEATURE_CATALOG)

    def test_filter_by_boolean(self):
        names = get_feature_names(FeatureType.BOOLEAN)
        assert all("has_" in n for n in names)

    def test_filter_by_numeric(self):
        names = get_feature_names(FeatureType.NUMERIC)
        assert len(names) > 0
        assert "text_length" in names
