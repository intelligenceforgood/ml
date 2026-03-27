"""Batch prediction for historical re-classification and embedding generation.

Reads cases from BigQuery, runs inference in batches, writes results to
a destination BigQuery table.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def run_batch_prediction(
    *,
    capability: str = "classification",
    model_artifact_uri: str = "",
    source_query: str | None = None,
    dest_table: str | None = None,
    batch_size: int = 100,
) -> None:
    """Run batch prediction over BigQuery data.

    Args:
        capability: ML capability (classification, ner, risk_scoring, embedding).
        model_artifact_uri: GCS URI to model artifacts. Falls back to env var.
        source_query: Custom BigQuery source query. Default: all cases.
        dest_table: Destination BQ table. Auto-generated if omitted.
        batch_size: Number of rows to process per batch.
    """
    from google.cloud import bigquery

    from ml.config import get_settings

    settings = get_settings()
    project = settings.platform.project_id
    dataset = settings.bigquery.dataset_id
    bq_client = bigquery.Client(project=project)

    # Resolve model URI from env if not provided
    if not model_artifact_uri:
        env_key = {
            "classification": "MODEL_ARTIFACT_URI",
            "ner": "NER_MODEL_ARTIFACT_URI",
            "risk_scoring": "RISK_MODEL_ARTIFACT_URI",
            "embedding": "EMBEDDING_MODEL_NAME",
        }.get(capability, "MODEL_ARTIFACT_URI")
        model_artifact_uri = os.environ.get(env_key, "")

    # Default source query
    if source_query is None:
        source_query = f"""
            SELECT c.case_id, c.narrative AS text,
                   f.text_length, f.word_count, f.entity_count,
                   f.has_crypto_wallet, f.has_bank_account, f.has_phone, f.has_email,
                   f.classification_axis_count, f.current_classification_conf
            FROM `{project}.{dataset}.raw_cases` c
            JOIN `{project}.{dataset}.features_case_features` f ON c.case_id = f.case_id
        """

    # Default dest table
    if not dest_table:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        dest_table = f"{project}.{dataset}.batch_predictions_{capability}_{timestamp}"

    logger.info("Batch prediction: capability=%s, dest=%s", capability, dest_table)

    # Load model for non-embedding capabilities
    if capability == "embedding":
        _run_embedding_batch(bq_client, source_query, dest_table, batch_size, project, dataset)
    else:
        _run_model_batch(
            bq_client, source_query, dest_table, batch_size, capability, model_artifact_uri, project, dataset
        )


def _run_model_batch(
    bq_client: Any,
    source_query: str,
    dest_table: str,
    batch_size: int,
    capability: str,
    model_artifact_uri: str,
    project: str,
    dataset: str,
) -> None:
    """Run batch prediction using a classification/NER/risk model."""
    import tempfile
    from pathlib import Path

    from ml.serving.predict import _detect_model_type, _download_artifacts, _load_label_map

    # Load model artifacts
    if model_artifact_uri:
        dest_dir = Path(tempfile.mkdtemp(prefix="ml_batch_"))
        _download_artifacts(model_artifact_uri, dest_dir)
        label_map = _load_label_map(dest_dir)
        model_type = _detect_model_type(dest_dir)
    else:
        model_type = "stub"
        label_map = {}

    # Query source data
    logger.info("Querying source data...")
    rows = list(bq_client.query(source_query).result())
    total = len(rows)
    logger.info("Total rows to process: %d", total)

    results = []
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        for row in batch:
            prediction_id = str(uuid.uuid4())
            case_id = row.case_id
            text = getattr(row, "text", "") or ""

            if capability == "risk_scoring":
                prediction = {"risk_score": 0.5}  # placeholder until model loaded
            elif model_type == "stub":
                prediction = {
                    "INTENT": {"code": "INTENT.UNKNOWN", "confidence": 0.5},
                    "CHANNEL": {"code": "CHANNEL.UNKNOWN", "confidence": 0.5},
                }
            else:
                # Use loaded model for actual inference
                prediction = _batch_infer(text, model_type, label_map, dest_dir, row)

            results.append(
                {
                    "case_id": case_id,
                    "prediction_id": prediction_id,
                    "capability": capability,
                    "prediction": json.dumps(prediction),
                    "confidence": _extract_confidence(prediction),
                    "model_artifact_uri": model_artifact_uri,
                    "predicted_at": datetime.now(UTC).isoformat(),
                }
            )

        elapsed = time.time() - start_time
        processed = min(i + batch_size, total)
        logger.info("Progress: %d/%d rows (%.1fs elapsed)", processed, total, elapsed)

    # Write results to BigQuery
    if results:
        _write_batch_results(bq_client, dest_table, results)

    elapsed = time.time() - start_time
    logger.info("Batch prediction complete: %d rows in %.1fs → %s", total, elapsed, dest_table)


def _batch_infer(
    text: str,
    model_type: str,
    label_map: dict,
    artifact_dir: Any,
    row: Any,
) -> dict:
    """Run single-row inference for batch processing."""
    if model_type == "xgboost":
        import numpy as np
        import xgboost as xgb

        from ml.serving.features import compute_inline_features

        feat = compute_inline_features(text)
        feature_keys = sorted(feat.keys())
        values = [float(feat.get(k) or 0) for k in feature_keys]
        dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)

        # Load booster if not already loaded
        booster = xgb.Booster()
        booster.load_model(str(artifact_dir / "xgboost_model.json"))
        raw_pred = booster.predict(dmat)

        result: dict = {}
        offset = 0
        for axis, labels in label_map.items():
            n = len(labels)
            axis_probs = raw_pred[0][offset : offset + n] if raw_pred.ndim > 1 else raw_pred[offset : offset + n]
            best_idx = int(np.argmax(axis_probs))
            result[axis] = {"code": labels[best_idx], "confidence": round(float(axis_probs[best_idx]), 4)}
            offset += n
        return result
    else:
        # Stub for other types
        return {"INTENT": {"code": "INTENT.UNKNOWN", "confidence": 0.5}}


def _extract_confidence(prediction: dict) -> float:
    """Extract average confidence from a prediction dict."""
    confidences = []
    for val in prediction.values():
        if isinstance(val, dict) and "confidence" in val:
            confidences.append(val["confidence"])
        elif isinstance(val, int | float):
            confidences.append(float(val))
    return sum(confidences) / len(confidences) if confidences else 0.0


def _write_batch_results(bq_client: Any, dest_table: str, results: list[dict]) -> None:
    """Write batch results to BigQuery, creating the table if needed."""
    from google.cloud import bigquery

    # Create table with schema
    project_dataset, table_name = dest_table.rsplit(".", 1)
    schema = [
        bigquery.SchemaField("case_id", "STRING"),
        bigquery.SchemaField("prediction_id", "STRING"),
        bigquery.SchemaField("capability", "STRING"),
        bigquery.SchemaField("prediction", "STRING"),
        bigquery.SchemaField("confidence", "FLOAT64"),
        bigquery.SchemaField("model_artifact_uri", "STRING"),
        bigquery.SchemaField("predicted_at", "TIMESTAMP"),
    ]

    table = bigquery.Table(dest_table, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field="predicted_at")
    table.clustering_fields = ["capability"]
    try:
        bq_client.create_table(table)
        logger.info("Created table %s", dest_table)
    except Exception:
        logger.debug("Table %s already exists or creation skipped", dest_table)

    errors = bq_client.insert_rows_json(dest_table, results)
    if errors:
        logger.error("BQ insert errors: %s", errors)
    else:
        logger.info("Wrote %d rows to %s", len(results), dest_table)


def _run_embedding_batch(
    bq_client: Any,
    source_query: str,
    dest_table: str,
    batch_size: int,
    project: str,
    dataset: str,
) -> None:
    """Run batch embedding generation."""
    from ml.serving.embeddings import compute_embedding

    logger.info("Querying source data for embedding generation...")
    rows = list(bq_client.query(source_query).result())
    total = len(rows)
    logger.info("Total rows: %d", total)

    results = []
    start_time = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        for row in batch:
            text = getattr(row, "text", "") or ""
            embedding = compute_embedding(text)
            results.append(
                {
                    "case_id": row.case_id,
                    "embedding": embedding,
                    "_computed_at": datetime.now(UTC).isoformat(),
                }
            )

        processed = min(i + batch_size, total)
        elapsed = time.time() - start_time
        logger.info("Embedding progress: %d/%d rows (%.1fs elapsed)", processed, total, elapsed)

    # Write embeddings
    if results:
        from google.cloud import bigquery

        embed_table = f"{project}.{dataset}.features_case_embeddings"
        schema = [
            bigquery.SchemaField("case_id", "STRING"),
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("_computed_at", "TIMESTAMP"),
        ]
        table = bigquery.Table(embed_table, schema=schema)
        with contextlib.suppress(Exception):
            bq_client.create_table(table)

        errors = bq_client.insert_rows_json(embed_table, results)
        if errors:
            logger.error("BQ insert errors: %s", errors)
        else:
            logger.info("Wrote %d embeddings to %s", len(results), embed_table)

    logger.info("Embedding batch complete: %d rows in %.1fs", total, time.time() - start_time)
