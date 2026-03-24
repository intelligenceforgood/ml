"""Accuracy monitoring — prediction vs outcome analysis.

Joins ``prediction_log`` and ``outcome_log`` in BigQuery to compute per-model
per-axis accuracy, override rate, and F1.  Results are materialized to the
``analytics_model_performance`` table for dashboarding and alerting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ml.config import get_settings

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

_bq_client: bigquery.Client | None = None


def _get_bq_client() -> bigquery.Client:
    """Return a lazily-initialized BigQuery client."""
    global _bq_client
    if _bq_client is None:
        from google.cloud import bigquery as bq

        settings = get_settings()
        _bq_client = bq.Client(project=settings.platform.project_id)
    return _bq_client


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisAccuracy:
    """Accuracy breakdown for a single taxonomy axis."""

    axis: str
    total: int
    correct: int
    overridden: int
    accuracy: float
    override_rate: float
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ModelAccuracy:
    """Aggregate accuracy for one model version."""

    model_id: str
    model_version: int
    total_predictions: int
    outcomes_received: int
    correct_predictions: int
    accuracy: float
    override_rate: float
    f1: float
    per_axis: dict[str, AxisAccuracy] = field(default_factory=dict)


@dataclass(frozen=True)
class AccuracyReport:
    """Full accuracy report across all models."""

    lookback_days: int
    computed_at: str
    models: list[ModelAccuracy] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

_JOINED_QUERY = """
SELECT
    p.prediction_id,
    p.model_id,
    p.model_version,
    p.prediction,
    o.correction
FROM `{project}.{dataset}.{pred_table}` p
INNER JOIN `{project}.{dataset}.{out_table}` o
    ON p.prediction_id = o.prediction_id
WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
"""


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _compute_axis_metrics(
    predicted: list[str | None],
    corrected: list[str],
) -> tuple[int, int, float, float, float]:
    """Return (correct, overridden, precision, recall, f1) for a list of
    predicted vs corrected labels on a single axis."""
    correct = sum(1 for p, c in zip(predicted, corrected, strict=False) if p == c)
    overridden = len(corrected) - correct

    # For a single-axis classification we treat each (pred, truth) pair:
    #   TP = correct, FP = overridden (predicted wrong), FN = overridden (missed right)
    # This gives precision = recall = accuracy in the per-axis view, but we
    # compute full P/R/F1 via micro-averaged counts for consistency.
    precision = _safe_div(correct, correct + overridden)
    recall = precision  # micro-averaged: symmetric in single-label classification
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return correct, overridden, precision, recall, f1


def compute_accuracy_metrics(
    *,
    lookback_days: int = 7,
    model_id: str | None = None,
    client: bigquery.Client | None = None,
) -> AccuracyReport:
    """Query BigQuery and compute per-model per-axis accuracy metrics.

    Args:
        lookback_days: Number of days of prediction history to include.
        model_id: Restrict to a single model (optional).
        client: Optional pre-configured BigQuery client (for testing).

    Returns:
        An ``AccuracyReport`` with per-model, per-axis breakdowns.
    """
    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()

    query = _JOINED_QUERY.format(
        project=settings.platform.project_id,
        dataset=settings.bigquery.dataset_id,
        pred_table=settings.bigquery.prediction_log_table,
        out_table=settings.bigquery.outcome_log_table,
    )
    if model_id:
        query += "\n    AND p.model_id = @model_id"

    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ScalarQueryParameter("lookback_days", "INT64", lookback_days),
            *([bq.ScalarQueryParameter("model_id", "STRING", model_id)] if model_id else []),
        ],
    )

    rows = list(bq_client.query(query, job_config=job_config).result())

    # Group by (model_id, model_version)
    # Track per-axis predictions and corrections
    model_axis_data: dict[
        tuple[str, int],
        dict[str, tuple[list[str | None], list[str]]],
    ] = {}
    model_total_preds: dict[tuple[str, int], int] = {}

    for row in rows:
        key = (row.model_id, row.model_version)
        prediction_raw = row.prediction
        correction_raw = row.correction

        prediction = json.loads(prediction_raw) if isinstance(prediction_raw, str) else prediction_raw or {}
        correction = json.loads(correction_raw) if isinstance(correction_raw, str) else correction_raw or {}

        if key not in model_axis_data:
            model_axis_data[key] = {}
        model_total_preds[key] = model_total_preds.get(key, 0) + 1

        for axis, corrected_label in correction.items():
            pred_entry = prediction.get(axis, {})
            predicted_label = pred_entry.get("code") if isinstance(pred_entry, dict) else None

            if axis not in model_axis_data[key]:
                model_axis_data[key][axis] = ([], [])
            preds, corrs = model_axis_data[key][axis]
            preds.append(predicted_label)
            corrs.append(corrected_label)

    # Build report
    models: list[ModelAccuracy] = []
    for (mid, mver), axis_data in sorted(model_axis_data.items()):
        per_axis: dict[str, AxisAccuracy] = {}
        total_correct = 0
        total_overridden = 0
        total_outcomes = 0

        for axis, (preds, corrs) in sorted(axis_data.items()):
            correct, overridden, prec, rec, f1 = _compute_axis_metrics(preds, corrs)
            total = len(corrs)
            total_correct += correct
            total_overridden += overridden
            total_outcomes += total
            per_axis[axis] = AxisAccuracy(
                axis=axis,
                total=total,
                correct=correct,
                overridden=overridden,
                accuracy=_safe_div(correct, total),
                override_rate=_safe_div(overridden, total),
                precision=prec,
                recall=rec,
                f1=f1,
            )

        overall_accuracy = _safe_div(total_correct, total_outcomes)
        overall_override = _safe_div(total_overridden, total_outcomes)
        overall_f1 = _safe_div(
            2 * overall_accuracy * overall_accuracy, overall_accuracy + overall_accuracy
        )  # micro-average

        models.append(
            ModelAccuracy(
                model_id=mid,
                model_version=mver,
                total_predictions=model_total_preds.get((mid, mver), 0),
                outcomes_received=total_outcomes,
                correct_predictions=total_correct,
                accuracy=overall_accuracy,
                override_rate=overall_override,
                f1=overall_f1,
                per_axis=per_axis,
            )
        )

    return AccuracyReport(
        lookback_days=lookback_days,
        computed_at=datetime.now(UTC).isoformat(),
        models=models,
    )


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def materialize_performance(
    *,
    lookback_days: int = 7,
    client: bigquery.Client | None = None,
) -> int:
    """Compute accuracy metrics and write to ``analytics_model_performance``.

    Returns:
        Number of rows written.
    """
    report = compute_accuracy_metrics(lookback_days=lookback_days, client=client)
    if not report.models:
        logger.info("No models with outcomes in the last %d days — nothing to materialize", lookback_days)
        return 0

    settings = get_settings()
    bq_client = client or _get_bq_client()
    table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_model_performance"

    # Compute the ISO week start (Monday) for the current period
    now = datetime.now(UTC)
    week_start = (now - __import__("datetime").timedelta(days=now.weekday())).strftime("%Y-%m-%d")

    rows_to_insert = []
    for m in report.models:
        per_axis_json = json.dumps(
            {
                axis: {
                    "accuracy": round(am.accuracy, 4),
                    "override_rate": round(am.override_rate, 4),
                    "f1": round(am.f1, 4),
                    "precision": round(am.precision, 4),
                    "recall": round(am.recall, 4),
                    "total": am.total,
                    "correct": am.correct,
                    "overridden": am.overridden,
                }
                for axis, am in m.per_axis.items()
            }
        )
        rows_to_insert.append(
            {
                "model_id": m.model_id,
                "model_version": m.model_version,
                "capability": "classification",
                "week": week_start,
                "total_predictions": m.total_predictions,
                "outcomes_received": m.outcomes_received,
                "correct_predictions": m.correct_predictions,
                "accuracy": round(m.accuracy, 4),
                "correction_rate": round(m.override_rate, 4),
                "per_axis_metrics": per_axis_json,
                "f1": round(m.f1, 4),
            }
        )

    errors = bq_client.insert_rows_json(table, rows_to_insert)
    if errors:
        logger.error("BigQuery insert errors for analytics_model_performance: %s", errors)
        raise RuntimeError(f"Failed to materialize performance: {errors}")

    logger.info("Materialized %d model performance rows for week %s", len(rows_to_insert), week_start)
    return len(rows_to_insert)


# ---------------------------------------------------------------------------
# CLI entry point (for Cloud Run Job / Cloud Scheduler)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run accuracy materialization — intended as Cloud Run Job entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = materialize_performance()
    logger.info("Accuracy materialization complete — %d rows written", rows)


if __name__ == "__main__":
    main()
