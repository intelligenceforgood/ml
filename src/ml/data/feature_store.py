"""Vertex AI Feature Store integration for online feature serving.

Provides high-speed feature retrieval (sub-100ms) for serving,
with LRU cache and fallback to inline computation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def sync_features_to_store(
    project_id: str,
    feature_store_id: str,
    entity_type_id: str = "case",
    *,
    incremental: bool = True,
) -> int:
    """Sync features from BigQuery to Vertex AI Feature Store.

    Reads from features_case_features + features_graph_features tables,
    writes to the Feature Store entity type.

    Args:
        project_id: GCP project ID.
        feature_store_id: Feature Store name (e.g. 'i4g_ml_features').
        entity_type_id: Entity type ID (e.g. 'case').
        incremental: If True, only sync rows updated since last watermark.

    Returns:
        Number of entities synced.
    """
    from google.cloud import aiplatform, bigquery

    from ml.config import get_settings

    settings = get_settings()
    region = settings.platform.region
    dataset = settings.bigquery.dataset_id

    aiplatform.init(project=project_id, location=region)
    bq_client = bigquery.Client(project=project_id)

    # Get watermark for incremental sync
    watermark_query = ""
    if incremental:
        watermark_query = f"""
            AND f._computed_at > (
                SELECT COALESCE(MAX(ingested_at), TIMESTAMP('1970-01-01'))
                FROM `{project_id}.{dataset}.feature_store_sync_log`
                WHERE entity_type = '{entity_type_id}'
            )
        """

    # Query features to ingest
    query = f"""
        SELECT
            f.case_id AS entity_id,
            f.text_length, f.word_count, f.entity_count,
            f.unique_entity_types,
            f.has_crypto_wallet, f.has_bank_account, f.has_phone, f.has_email,
            f.classification_axis_count, f.current_classification_conf,
            COALESCE(gf.shared_entity_count, 0) AS shared_entity_count,
            COALESCE(gf.entity_reuse_frequency, 0.0) AS entity_reuse_frequency,
            COALESCE(gf.cluster_size, 0) AS cluster_size
        FROM `{project_id}.{dataset}.features_case_features` f
        LEFT JOIN `{project_id}.{dataset}.features_graph_features` gf
            ON f.case_id = gf.case_id
        WHERE 1=1
        {watermark_query}
    """

    rows = list(bq_client.query(query).result())
    if not rows:
        logger.info("No new features to sync")
        return 0

    logger.info("Syncing %d entities to Feature Store", len(rows))

    # Get Feature Store entity type
    entity_type = aiplatform.EntityType(
        entity_type_name=f"projects/{project_id}/locations/{region}/featurestores/{feature_store_id}/entityTypes/{entity_type_id}"
    )

    # Ingest from BigQuery source table
    # Use BQ as source for bulk ingest
    source_table = f"bq://{project_id}.{dataset}.features_case_features"
    entity_type.ingest_from_bq(
        feature_ids=[
            "text_length",
            "word_count",
            "entity_count",
            "unique_entity_types",
            "has_crypto_wallet",
            "has_bank_account",
            "has_phone",
            "has_email",
            "classification_axis_count",
            "current_classification_conf",
        ],
        feature_time="_computed_at",
        bq_source_uri=source_table,
        entity_id_field="case_id",
    )

    # Log sync event
    sync_row = {
        "entity_type": entity_type_id,
        "entities_synced": len(rows),
        "ingested_at": time.time(),
    }
    try:
        sync_table = f"{project_id}.{dataset}.feature_store_sync_log"
        bq_client.insert_rows_json(sync_table, [sync_row])
    except Exception:
        logger.debug("Could not log sync event (table may not exist yet)")

    logger.info("Feature Store sync complete: %d entities", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Online feature retrieval
# ---------------------------------------------------------------------------

# LRU cache: 128 entries, caller must handle TTL externally if needed
_feature_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SECONDS = 60.0
_CACHE_MAX_SIZE = 128


def _cache_get(case_id: str) -> dict[str, Any] | None:
    """Get cached features if not expired."""
    entry = _feature_cache.get(case_id)
    if entry is None:
        return None
    cached_at, features = entry
    if time.time() - cached_at > _CACHE_TTL_SECONDS:
        del _feature_cache[case_id]
        return None
    return features


def _cache_put(case_id: str, features: dict[str, Any]) -> None:
    """Cache features with TTL. Evicts oldest if at capacity."""
    if len(_feature_cache) >= _CACHE_MAX_SIZE:
        # Evict oldest entry
        oldest_key = min(_feature_cache, key=lambda k: _feature_cache[k][0])
        del _feature_cache[oldest_key]
    _feature_cache[case_id] = (time.time(), features)


def fetch_online_features(case_id: str) -> dict[str, Any] | None:
    """Fetch pre-computed features from Vertex AI Feature Store.

    Uses an LRU cache (128 entries, 60s TTL) to avoid repeated lookups.
    Returns None if Feature Store is not configured or unavailable.

    Args:
        case_id: Case identifier to look up.

    Returns:
        Feature dict or None if unavailable.
    """
    feature_store_id = os.environ.get("FEATURE_STORE_ID", "")
    if not feature_store_id:
        return None

    # Check cache first
    cached = _cache_get(case_id)
    if cached is not None:
        return cached

    try:
        from google.cloud import aiplatform

        from ml.config import get_settings

        settings = get_settings()
        aiplatform.init(project=settings.platform.project_id, location=settings.platform.region)

        entity_type = aiplatform.EntityType(
            entity_type_name=(
                f"projects/{settings.platform.project_id}/locations/{settings.platform.region}"
                f"/featurestores/{feature_store_id}/entityTypes/case"
            )
        )

        result = entity_type.read(entity_ids=[case_id])
        if result.empty:
            return None

        features = result.iloc[0].to_dict()
        # Remove entity_id from features dict
        features.pop("entity_id", None)

        _cache_put(case_id, features)
        return features

    except Exception:
        logger.debug("Feature Store read failed for %s", case_id, exc_info=True)
        return None
