"""Prediction and outcome logging to BigQuery.

All logging is fire-and-forget — a failed log write does not block
the prediction response.  Failed writes are retried up to ``MAX_RETRIES``
times with exponential back-off and dead-lettered to a local log on
exhaustion.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ml.config import get_settings

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

_bq_client = None

MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.5  # seconds; doubles each retry

# Structured dead-letter logger for failed writes
_dead_letter_logger = logging.getLogger("ml.serving.logging.dead_letter")


def _get_bq_client() -> bigquery.Client:
    """Return a lazily-initialized BigQuery client."""
    global _bq_client
    if _bq_client is None:
        from google.cloud import bigquery

        settings = get_settings()
        _bq_client = bigquery.Client(project=settings.platform.project_id)
    return _bq_client


def _insert_with_retry(table: str, row: dict, *, context: str) -> None:
    """Insert a row into BigQuery with retry and dead-letter on exhaustion."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            errors = _get_bq_client().insert_rows_json(table, [row])
            if not errors:
                return
            # BigQuery streaming insert returned row-level errors
            raise RuntimeError(f"BQ insert errors: {errors}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Retry %d/%d for %s: %s",
                    attempt,
                    MAX_RETRIES,
                    context,
                    exc,
                )
                time.sleep(delay)

    # Exhausted retries — dead-letter the row so it can be replayed later
    _dead_letter_logger.error(
        "Dead-letter %s after %d retries: %s | row=%s",
        context,
        MAX_RETRIES,
        last_exc,
        json.dumps(row, default=str),
    )


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
    is_shadow: bool = False,
    capability: str = "classification",
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
            "is_shadow": is_shadow,
            "capability": capability,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        _insert_with_retry(table, row, context=f"prediction:{prediction_id}")
    except Exception:  # noqa: BLE001 — must not block prediction response
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
        _insert_with_retry(table, row, context=f"outcome:{prediction_id}")
    except Exception:  # noqa: BLE001 — must not block feedback response
        logger.exception("Failed to log outcome for prediction %s", prediction_id)
    return outcome_id
