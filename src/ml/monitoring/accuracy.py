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
    p.variant,
    o.correction
FROM `{project}.{dataset}.{pred_table}` p
INNER JOIN `{project}.{dataset}.{out_table}` o
    ON p.prediction_id = o.prediction_id
WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
  AND p.is_shadow IS NOT TRUE
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
    variant: str | None = None,
    client: bigquery.Client | None = None,
) -> AccuracyReport:
    """Query BigQuery and compute per-model per-axis accuracy metrics.

    Args:
        lookback_days: Number of days of prediction history to include.
        model_id: Restrict to a single model (optional).
        variant: Restrict to a single variant, e.g. ``"champion"`` or
            ``"challenger"`` (optional).
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
    if variant:
        query += "\n    AND p.variant = @variant"

    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ScalarQueryParameter("lookback_days", "INT64", lookback_days),
            *([bq.ScalarQueryParameter("model_id", "STRING", model_id)] if model_id else []),
            *([bq.ScalarQueryParameter("variant", "STRING", variant)] if variant else []),
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

    # Fetch cost-per-prediction for cost_per_correct metric (Sprint 6.3)
    cost_per_prediction: float | None = None
    try:
        from ml.monitoring.cost import compute_cost_summary

        cost_summary = compute_cost_summary(period_days=lookback_days, client=bq_client)
        if cost_summary and cost_summary.cost_per_prediction_usd > 0:
            cost_per_prediction = cost_summary.cost_per_prediction_usd
    except Exception:  # noqa: BLE001 — cost data is supplementary
        logger.debug("Could not fetch cost data for cost_per_correct metric", exc_info=True)

    now = datetime.now(UTC)
    computed_date = now.strftime("%Y-%m-%d")

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
                "computed_at": computed_date,
                "total_predictions": m.total_predictions,
                "outcomes_received": m.outcomes_received,
                "correct_predictions": m.correct_predictions,
                "accuracy": round(m.accuracy, 4),
                "correction_rate": round(m.override_rate, 4),
                "per_axis_metrics": per_axis_json,
                "f1": round(m.f1, 4),
                "cost_per_correct_prediction": (
                    round(_safe_div(cost_per_prediction, m.accuracy), 6) if cost_per_prediction is not None else None
                ),
            }
        )

    errors = bq_client.insert_rows_json(table, rows_to_insert)
    if errors:
        logger.error("BigQuery insert errors for analytics_model_performance: %s", errors)
        raise RuntimeError(f"Failed to materialize performance: {errors}")

    logger.info("Materialized %d model performance rows for %s", len(rows_to_insert), computed_date)
    return len(rows_to_insert)


# ---------------------------------------------------------------------------
# Shadow comparison
# ---------------------------------------------------------------------------

_SHADOW_QUERY = """
SELECT
    c.prediction_id AS champion_id,
    s.prediction_id AS shadow_id,
    c.model_id AS champion_model,
    s.model_id AS shadow_model,
    c.prediction AS champion_prediction,
    s.prediction AS shadow_prediction,
    c.case_id,
    c.timestamp
FROM `{project}.{dataset}.{pred_table}` c
INNER JOIN `{project}.{dataset}.{pred_table}` s
    ON s.prediction_id = CONCAT(c.prediction_id, '-shadow')
WHERE c.is_shadow IS NOT TRUE
  AND s.is_shadow = TRUE
  AND c.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
"""


@dataclass(frozen=True)
class ShadowComparison:
    """Comparison between champion and shadow model predictions."""

    champion_model: str
    shadow_model: str
    total_pairs: int
    agreement_count: int
    agreement_rate: float
    per_axis_agreement: dict[str, float]
    lookback_days: int
    computed_at: str


def compute_shadow_comparison(
    *,
    lookback_days: int = 7,
    client: bigquery.Client | None = None,
) -> ShadowComparison | None:
    """Compare champion and shadow predictions from the prediction log.

    Returns None if no shadow prediction pairs are found.
    """
    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()

    query = _SHADOW_QUERY.format(
        project=settings.platform.project_id,
        dataset=settings.bigquery.dataset_id,
        pred_table=settings.bigquery.prediction_log_table,
    )

    job_config = bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("lookback_days", "INT64", lookback_days)],
    )

    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None

    champion_model = rows[0].champion_model
    shadow_model = rows[0].shadow_model

    axis_matches: dict[str, list[bool]] = {}
    fully_agree = 0

    for row in rows:
        champion_raw = row.champion_prediction
        shadow_raw = row.shadow_prediction
        c_pred = json.loads(champion_raw) if isinstance(champion_raw, str) else champion_raw or {}
        s_pred = json.loads(shadow_raw) if isinstance(shadow_raw, str) else shadow_raw or {}

        all_match = True
        for axis in c_pred:
            c_code = c_pred[axis].get("code") if isinstance(c_pred[axis], dict) else None
            s_code = s_pred.get(axis, {}).get("code") if isinstance(s_pred.get(axis), dict) else None
            match = c_code == s_code
            axis_matches.setdefault(axis, []).append(match)
            if not match:
                all_match = False
        if all_match:
            fully_agree += 1

    total = len(rows)
    per_axis = {axis: _safe_div(sum(m), len(m)) for axis, m in sorted(axis_matches.items())}

    return ShadowComparison(
        champion_model=champion_model,
        shadow_model=shadow_model,
        total_pairs=total,
        agreement_count=fully_agree,
        agreement_rate=_safe_div(fully_agree, total),
        per_axis_agreement=per_axis,
        lookback_days=lookback_days,
        computed_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Champion vs Challenger variant comparison (Sprint 1.3)
# ---------------------------------------------------------------------------

_VARIANT_QUERY = """
SELECT
    p.prediction_id,
    p.model_id,
    p.model_version,
    p.prediction,
    p.variant,
    o.correction
FROM `{project}.{dataset}.{pred_table}` p
INNER JOIN `{project}.{dataset}.{out_table}` o
    ON p.prediction_id = o.prediction_id
WHERE p.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
  AND p.variant IN ('champion', 'challenger')
  AND p.is_shadow IS NOT TRUE
"""


@dataclass(frozen=True)
class VariantMetrics:
    """Accuracy metrics for a single variant (champion or challenger)."""

    variant: str
    model_id: str
    total_outcomes: int
    correct: int
    accuracy: float
    override_rate: float
    f1: float
    per_axis: dict[str, AxisAccuracy] = field(default_factory=dict)


@dataclass(frozen=True)
class VariantComparison:
    """Side-by-side comparison of champion vs challenger accuracy."""

    champion: VariantMetrics | None
    challenger: VariantMetrics | None
    lookback_days: int
    computed_at: str


def compute_variant_comparison(
    *,
    lookback_days: int = 7,
    client: bigquery.Client | None = None,
) -> VariantComparison:
    """Compare champion vs challenger accuracy using analyst corrections.

    Queries prediction_log joined with outcome_log, grouped by variant.
    """
    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()

    query = _VARIANT_QUERY.format(
        project=settings.platform.project_id,
        dataset=settings.bigquery.dataset_id,
        pred_table=settings.bigquery.prediction_log_table,
        out_table=settings.bigquery.outcome_log_table,
    )

    job_config = bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("lookback_days", "INT64", lookback_days)],
    )

    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return VariantComparison(
            champion=None,
            challenger=None,
            lookback_days=lookback_days,
            computed_at=datetime.now(UTC).isoformat(),
        )

    # Group by variant
    variant_data: dict[str, dict] = {}
    for row in rows:
        v = row.variant or "champion"
        if v not in variant_data:
            variant_data[v] = {
                "model_id": row.model_id,
                "axis_data": {},
                "total": 0,
            }
        vd = variant_data[v]
        vd["total"] += 1

        prediction = json.loads(row.prediction) if isinstance(row.prediction, str) else row.prediction or {}
        correction = json.loads(row.correction) if isinstance(row.correction, str) else row.correction or {}

        for axis, corrected_label in correction.items():
            pred_entry = prediction.get(axis, {})
            pred_label = pred_entry.get("code") if isinstance(pred_entry, dict) else None

            if axis not in vd["axis_data"]:
                vd["axis_data"][axis] = ([], [])
            preds, corrs = vd["axis_data"][axis]
            preds.append(pred_label)
            corrs.append(corrected_label)

    def _build_variant_metrics(variant_name: str, data: dict) -> VariantMetrics:
        per_axis: dict[str, AxisAccuracy] = {}
        total_correct = 0
        total_overridden = 0
        total_outcomes = 0

        for axis, (preds, corrs) in sorted(data["axis_data"].items()):
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
        overall_f1 = _safe_div(2 * overall_accuracy * overall_accuracy, overall_accuracy + overall_accuracy)

        return VariantMetrics(
            variant=variant_name,
            model_id=data["model_id"],
            total_outcomes=total_outcomes,
            correct=total_correct,
            accuracy=overall_accuracy,
            override_rate=overall_override,
            f1=overall_f1,
            per_axis=per_axis,
        )

    champion_metrics = (
        _build_variant_metrics("champion", variant_data["champion"]) if "champion" in variant_data else None
    )
    challenger_metrics = (
        _build_variant_metrics("challenger", variant_data["challenger"]) if "challenger" in variant_data else None
    )

    return VariantComparison(
        champion=champion_metrics,
        challenger=challenger_metrics,
        lookback_days=lookback_days,
        computed_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Variant comparison materialization (Sprint 1.3)
# ---------------------------------------------------------------------------


def materialize_variant_comparison(
    *,
    lookback_days: int = 7,
    client: bigquery.Client | None = None,
) -> int:
    """Compute variant comparison and write to ``analytics_variant_comparison``.

    Returns:
        Number of rows written.
    """
    comparison = compute_variant_comparison(lookback_days=lookback_days, client=client)
    variants = [v for v in (comparison.champion, comparison.challenger) if v is not None]
    if not variants:
        logger.info("No variant data in the last %d days — nothing to materialize", lookback_days)
        return 0

    settings = get_settings()
    bq_client = client or _get_bq_client()
    table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_variant_comparison"

    rows_to_insert = []
    for vm in variants:
        per_axis_json = json.dumps(
            {
                axis: {
                    "accuracy": round(am.accuracy, 4),
                    "override_rate": round(am.override_rate, 4),
                    "f1": round(am.f1, 4),
                    "total": am.total,
                }
                for axis, am in vm.per_axis.items()
            }
        )
        rows_to_insert.append(
            {
                "variant": vm.variant,
                "model_id": vm.model_id,
                "total_outcomes": vm.total_outcomes,
                "correct": vm.correct,
                "accuracy": round(vm.accuracy, 4),
                "override_rate": round(vm.override_rate, 4),
                "f1": round(vm.f1, 4),
                "per_axis_metrics": per_axis_json,
                "lookback_days": lookback_days,
                "computed_at": comparison.computed_at,
            }
        )

    errors = bq_client.insert_rows_json(table, rows_to_insert)
    if errors:
        logger.error("BigQuery insert errors for analytics_variant_comparison: %s", errors)
        raise RuntimeError(f"Failed to materialize variant comparison: {errors}")

    logger.info("Materialized %d variant comparison rows", len(rows_to_insert))
    return len(rows_to_insert)


# ---------------------------------------------------------------------------
# CLI entry point (for Cloud Run Job / Cloud Scheduler)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run accuracy materialization — intended as Cloud Run Job entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rows = materialize_performance()
    logger.info("Accuracy materialization complete — %d rows written", rows)

    variant_rows = materialize_variant_comparison()
    logger.info("Variant comparison materialization complete — %d rows written", variant_rows)


if __name__ == "__main__":
    main()
