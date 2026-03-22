"""Prediction and outcome logging to BigQuery.

All logging is fire-and-forget — a failed log write does not block
the prediction response.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ml.config import get_settings

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

_bq_client = None


def _get_bq_client() -> bigquery.Client:
    """Return a lazily-initialized BigQuery client."""
    global _bq_client
    if _bq_client is None:
        from google.cloud import bigquery

        settings = get_settings()
        _bq_client = bigquery.Client(project=settings.platform.project_id)
    return _bq_client


def log_prediction(
    *,
    prediction_id: str,
    case_id: str,
    model_id: str,
    model_version: int,
    prediction: dict,
    features: dict | None = None,
    latency_ms: int = 0,
    endpoint: str = "",
) -> None:
    """Log a prediction to BigQuery ``predictions_prediction_log``."""
    try:
        settings = get_settings()
        table = (
            f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.{settings.bigquery.prediction_log_table}"
        )
        row = {
            "prediction_id": prediction_id,
            "case_id": case_id,
            "model_id": model_id,
            "model_version": model_version,
            "endpoint": endpoint or settings.serving.dev_endpoint_name,
            "request_payload": json.dumps({"case_id": case_id}),
            "prediction": json.dumps(prediction),
            "latency_ms": latency_ms,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _get_bq_client().insert_rows_json(table, [row])
    except Exception:  # noqa: BLE001 — fire-and-forget; must not block prediction response
        logger.exception("Failed to log prediction %s", prediction_id)


def log_outcome(
    *,
    prediction_id: str,
    case_id: str,
    correction: dict,
    analyst_id: str,
) -> str:
    """Log an outcome (analyst correction) to BigQuery ``predictions_outcome_log``."""
    outcome_id = str(uuid.uuid4())
    try:
        settings = get_settings()
        table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.{settings.bigquery.outcome_log_table}"
        row = {
            "outcome_id": outcome_id,
            "prediction_id": prediction_id,
            "case_id": case_id,
            "correction": json.dumps(correction),
            "analyst_id": analyst_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _get_bq_client().insert_rows_json(table, [row])
    except Exception:  # noqa: BLE001 — fire-and-forget; must not block feedback response
        logger.exception("Failed to log outcome for prediction %s", prediction_id)
    return outcome_id
