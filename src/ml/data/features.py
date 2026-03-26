"""Feature definitions — declarative catalog of engineered features.

Each ``FeatureDefinition`` describes a single feature: its name, type,
how it is computed, and its current schema version.  The authoritative
list lives in ``FEATURE_CATALOG`` and is consumed by the ETL pipeline,
dataset export, and serving feature-computation code.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FeatureType(StrEnum):
    """Data type of a computed feature value."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    EMBEDDING = "embedding"


class ComputeMethod(StrEnum):
    """Backend used to materialise the feature."""

    BIGQUERY_SQL = "bigquery_sql"
    DATAFLOW = "dataflow"
    PYTHON = "python"


class FeatureDefinition(BaseModel):
    """Immutable descriptor for a single engineered feature."""

    model_config = {"frozen": True}

    name: str
    feature_type: FeatureType
    description: str
    compute_method: ComputeMethod
    version: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------------
# Authoritative catalog — single source of truth for feature names/types
# ---------------------------------------------------------------------------

FEATURE_CATALOG: list[FeatureDefinition] = [
    FeatureDefinition(
        name="text_length",
        feature_type=FeatureType.NUMERIC,
        description="Character count of case narrative",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="word_count",
        feature_type=FeatureType.NUMERIC,
        description="Word count of case narrative",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="lexical_diversity",
        feature_type=FeatureType.NUMERIC,
        description="Unique words / total words ratio",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="entity_count",
        feature_type=FeatureType.NUMERIC,
        description="Total entities extracted from case",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="unique_entity_types",
        feature_type=FeatureType.NUMERIC,
        description="Distinct entity types in case",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="has_crypto_wallet",
        feature_type=FeatureType.BOOLEAN,
        description="Case contains a crypto wallet entity",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="has_bank_account",
        feature_type=FeatureType.BOOLEAN,
        description="Case contains a bank account entity",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="has_phone",
        feature_type=FeatureType.BOOLEAN,
        description="Case contains a phone entity",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="has_email",
        feature_type=FeatureType.BOOLEAN,
        description="Case contains an email entity",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="classification_axis_count",
        feature_type=FeatureType.NUMERIC,
        description="Number of taxonomy axes with classifications",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    FeatureDefinition(
        name="current_classification_conf",
        feature_type=FeatureType.NUMERIC,
        description="Confidence of latest classification",
        compute_method=ComputeMethod.BIGQUERY_SQL,
    ),
    # ── Graph features (computed by Dataflow/Beam pipeline) ────────────────
    FeatureDefinition(
        name="shared_entity_count",
        feature_type=FeatureType.NUMERIC,
        description="Distinct entities this case shares with other cases",
        compute_method=ComputeMethod.DATAFLOW,
    ),
    FeatureDefinition(
        name="entity_reuse_frequency",
        feature_type=FeatureType.NUMERIC,
        description="Average number of cases each of this case's entities appears in",
        compute_method=ComputeMethod.DATAFLOW,
    ),
    FeatureDefinition(
        name="cluster_size",
        feature_type=FeatureType.NUMERIC,
        description="Size of the connected component this case belongs to in entity co-occurrence graph",
        compute_method=ComputeMethod.DATAFLOW,
    ),
]


def get_feature_names(feature_type: FeatureType | None = None) -> list[str]:
    """Return feature names, optionally filtered by type."""
    if feature_type is None:
        return [f.name for f in FEATURE_CATALOG]
    return [f.name for f in FEATURE_CATALOG if f.feature_type == feature_type]
