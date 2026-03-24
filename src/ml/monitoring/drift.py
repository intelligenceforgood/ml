"""Input and prediction drift detection via PSI (Population Stability Index).

Computes drift between a baseline window and a current window for both
prediction label distributions and numeric feature distributions.  Results
are materialized to ``analytics_drift_metrics`` in BigQuery.
"""

from __future__ import annotations

import logging
import math
import uuid
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

PSI_DRIFT_THRESHOLD = 0.2


@dataclass(frozen=True)
class PredictionDrift:
    """Drift result for a single prediction axis label."""

    label: str
    baseline_rate: float
    current_rate: float
    psi: float
    is_drifted: bool


@dataclass(frozen=True)
class FeatureDrift:
    """Drift result for a single numeric feature."""

    feature_name: str
    psi: float
    is_drifted: bool


@dataclass(frozen=True)
class DriftReport:
    """Aggregated drift report for a model."""

    report_id: str
    model_id: str
    window_start: str
    window_end: str
    baseline_start: str
    baseline_end: str
    prediction_drift: list[PredictionDrift] = field(default_factory=list)
    feature_drift: list[FeatureDrift] = field(default_factory=list)
    computed_at: str = ""


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

_EPSILON = 1e-6  # Avoid log(0)


def compute_psi(baseline_probs: list[float], current_probs: list[float]) -> float:
    """Compute Population Stability Index between two probability distributions.

    Args:
        baseline_probs: Probability distribution for baseline period.
        current_probs: Probability distribution for current period.

    Returns:
        PSI score (0 = identical, >0.2 = significant drift).
    """
    if len(baseline_probs) != len(current_probs):
        raise ValueError("Distributions must have the same number of bins")
    psi = 0.0
    for b, c in zip(baseline_probs, current_probs, strict=True):
        b = max(b, _EPSILON)
        c = max(c, _EPSILON)
        psi += (c - b) * math.log(c / b)
    return psi


def _distribution_from_counts(counts: dict[str, int]) -> tuple[list[str], list[float]]:
    """Convert label→count dict to (labels, probabilities)."""
    total = sum(counts.values())
    if total == 0:
        labels = sorted(counts.keys())
        return labels, [0.0] * len(labels)
    labels = sorted(counts.keys())
    probs = [counts.get(label, 0) / total for label in labels]
    return labels, probs


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------

_LABEL_DISTRIBUTION_QUERY = """
SELECT
    JSON_EXTRACT_SCALAR(prediction, '$.{axis}.code') AS label,
    COUNT(*) AS cnt
FROM `{project}.{dataset}.{pred_table}`
WHERE
    timestamp >= @window_start
    AND timestamp < @window_end
    AND model_id = @model_id
GROUP BY label
"""


def compute_prediction_drift(
    model_id: str,
    *,
    window_days: int = 7,
    client: bigquery.Client | None = None,
) -> list[PredictionDrift]:
    """Compare prediction label distributions between baseline and current window.

    Baseline = [now - 2*window, now - window], current = [now - window, now].

    Args:
        model_id: Model identifier to filter predictions.
        window_days: Size of each comparison window.
        client: Optional pre-configured BigQuery client.

    Returns:
        List of per-label drift results.
    """
    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()

    # We need to determine which axes exist. We'll query for the overall
    # intent axis as the primary classification axis.
    axes = ["intent"]  # Expand as needed

    all_drift: list[PredictionDrift] = []
    for axis in axes:
        query = _LABEL_DISTRIBUTION_QUERY.format(
            axis=axis,
            project=settings.platform.project_id,
            dataset=settings.bigquery.dataset_id,
            pred_table=settings.bigquery.prediction_log_table,
        )

        # Use a simpler approach: compute dates in Python
        now = datetime.now(UTC)
        from datetime import timedelta

        window_end = now
        window_start = now - timedelta(days=window_days)
        baseline_end = window_start
        baseline_start = now - timedelta(days=2 * window_days)

        # Query baseline
        baseline_params = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("model_id", "STRING", model_id),
                bq.ScalarQueryParameter("window_start", "TIMESTAMP", baseline_start),
                bq.ScalarQueryParameter("window_end", "TIMESTAMP", baseline_end),
            ],
        )
        baseline_rows = list(bq_client.query(query, job_config=baseline_params).result())
        baseline_counts: dict[str, int] = {row.label: row.cnt for row in baseline_rows if row.label}

        # Query current
        current_params = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("model_id", "STRING", model_id),
                bq.ScalarQueryParameter("window_start", "TIMESTAMP", window_start),
                bq.ScalarQueryParameter("window_end", "TIMESTAMP", window_end),
            ],
        )
        current_rows = list(bq_client.query(query, job_config=current_params).result())
        current_counts: dict[str, int] = {row.label: row.cnt for row in current_rows if row.label}

        # Combine label sets
        all_labels = sorted(set(baseline_counts.keys()) | set(current_counts.keys()))
        if not all_labels:
            continue

        baseline_total = sum(baseline_counts.values())
        current_total = sum(current_counts.values())

        if baseline_total == 0 or current_total == 0:
            logger.warning("Insufficient data for drift computation on axis=%s, model=%s", axis, model_id)
            continue

        for label in all_labels:
            b_rate = baseline_counts.get(label, 0) / baseline_total
            c_rate = current_counts.get(label, 0) / current_total
            b_safe = max(b_rate, _EPSILON)
            c_safe = max(c_rate, _EPSILON)
            psi = (c_safe - b_safe) * math.log(c_safe / b_safe)
            all_drift.append(
                PredictionDrift(
                    label=label,
                    baseline_rate=round(b_rate, 6),
                    current_rate=round(c_rate, 6),
                    psi=round(psi, 6),
                    is_drifted=psi > PSI_DRIFT_THRESHOLD,
                )
            )

    return all_drift


# ---------------------------------------------------------------------------
# Feature drift
# ---------------------------------------------------------------------------

_FEATURE_STATS_QUERY = """
SELECT
    JSON_EXTRACT_SCALAR(features_used, '$.{feature}') AS val
FROM `{project}.{dataset}.{pred_table}`
WHERE
    timestamp >= @window_start
    AND timestamp < @window_end
    AND model_id = @model_id
    AND JSON_EXTRACT_SCALAR(features_used, '$.{feature}') IS NOT NULL
"""


def _compute_numeric_psi(baseline_vals: list[float], current_vals: list[float], n_bins: int = 10) -> float:
    """Compute PSI for numeric distributions using equal-width bins."""
    if not baseline_vals or not current_vals:
        return 0.0

    all_vals = baseline_vals + current_vals
    min_val = min(all_vals)
    max_val = max(all_vals)

    if min_val == max_val:
        return 0.0

    bin_width = (max_val - min_val) / n_bins

    def _bin_counts(vals: list[float]) -> list[int]:
        counts = [0] * n_bins
        for v in vals:
            idx = min(int((v - min_val) / bin_width), n_bins - 1)
            counts[idx] += 1
        return counts

    b_counts = _bin_counts(baseline_vals)
    c_counts = _bin_counts(current_vals)

    b_total = len(baseline_vals)
    c_total = len(current_vals)

    b_probs = [c / b_total for c in b_counts]
    c_probs = [c / c_total for c in c_counts]

    return compute_psi(b_probs, c_probs)


def compute_feature_drift(
    model_id: str,
    *,
    window_days: int = 7,
    features: list[str] | None = None,
    client: bigquery.Client | None = None,
) -> list[FeatureDrift]:
    """Compare numeric feature distributions between baseline and current window.

    Args:
        model_id: Model identifier.
        window_days: Size of each comparison window.
        features: List of feature names to check. If None, uses default set.
        client: Optional BigQuery client.

    Returns:
        List of per-feature drift results.
    """
    from datetime import timedelta

    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()

    if features is None:
        features = [
            "text_length",
            "word_count",
            "entity_count",
            "indicator_count",
        ]

    now = datetime.now(UTC)
    window_start = now - timedelta(days=window_days)
    baseline_end = window_start
    baseline_start = now - timedelta(days=2 * window_days)

    results: list[FeatureDrift] = []
    for feature in features:
        query = _FEATURE_STATS_QUERY.format(
            feature=feature,
            project=settings.platform.project_id,
            dataset=settings.bigquery.dataset_id,
            pred_table=settings.bigquery.prediction_log_table,
        )

        # Baseline values
        baseline_params = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("model_id", "STRING", model_id),
                bq.ScalarQueryParameter("window_start", "TIMESTAMP", baseline_start),
                bq.ScalarQueryParameter("window_end", "TIMESTAMP", baseline_end),
            ],
        )
        baseline_rows = list(bq_client.query(query, job_config=baseline_params).result())
        baseline_vals = [float(r.val) for r in baseline_rows if r.val is not None]

        # Current values
        current_params = bq.QueryJobConfig(
            query_parameters=[
                bq.ScalarQueryParameter("model_id", "STRING", model_id),
                bq.ScalarQueryParameter("window_start", "TIMESTAMP", window_start),
                bq.ScalarQueryParameter("window_end", "TIMESTAMP", now),
            ],
        )
        current_rows = list(bq_client.query(query, job_config=current_params).result())
        current_vals = [float(r.val) for r in current_rows if r.val is not None]

        psi = _compute_numeric_psi(baseline_vals, current_vals)
        results.append(
            FeatureDrift(
                feature_name=feature,
                psi=round(psi, 6),
                is_drifted=psi > PSI_DRIFT_THRESHOLD,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Full drift report
# ---------------------------------------------------------------------------


def compute_drift_report(
    model_id: str,
    *,
    window_days: int = 7,
    client: bigquery.Client | None = None,
) -> DriftReport:
    """Compute a full drift report (prediction + feature drift).

    Args:
        model_id: Model identifier.
        window_days: Size of each comparison window.
        client: Optional BigQuery client.

    Returns:
        A ``DriftReport`` with prediction and feature drift results.
    """
    from datetime import timedelta

    now = datetime.now(UTC)
    window_start = now - timedelta(days=window_days)
    baseline_start = now - timedelta(days=2 * window_days)

    pred_drift = compute_prediction_drift(model_id, window_days=window_days, client=client)
    feat_drift = compute_feature_drift(model_id, window_days=window_days, client=client)

    return DriftReport(
        report_id=str(uuid.uuid4()),
        model_id=model_id,
        window_start=window_start.isoformat(),
        window_end=now.isoformat(),
        baseline_start=baseline_start.isoformat(),
        baseline_end=window_start.isoformat(),
        prediction_drift=pred_drift,
        feature_drift=feat_drift,
        computed_at=now.isoformat(),
    )


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def materialize_drift_metrics(
    report: DriftReport,
    *,
    client: bigquery.Client | None = None,
) -> int:
    """Write drift report rows to ``analytics_drift_metrics`` in BigQuery.

    Args:
        report: The drift report to materialize.
        client: Optional BigQuery client.

    Returns:
        Number of rows written.
    """
    bq_client = client or _get_bq_client()
    settings = get_settings()
    table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_drift_metrics"

    rows: list[dict] = []

    for pd in report.prediction_drift:
        rows.append(
            {
                "report_id": report.report_id,
                "model_id": report.model_id,
                "report_type": "prediction",
                "axis_or_feature": pd.label,
                "baseline_rate": pd.baseline_rate,
                "current_rate": pd.current_rate,
                "psi": pd.psi,
                "is_drifted": pd.is_drifted,
                "window_start": report.window_start,
                "window_end": report.window_end,
                "computed_at": report.computed_at,
            }
        )

    for fd in report.feature_drift:
        rows.append(
            {
                "report_id": report.report_id,
                "model_id": report.model_id,
                "report_type": "feature",
                "axis_or_feature": fd.feature_name,
                "baseline_rate": None,
                "current_rate": None,
                "psi": fd.psi,
                "is_drifted": fd.is_drifted,
                "window_start": report.window_start,
                "window_end": report.window_end,
                "computed_at": report.computed_at,
            }
        )

    if not rows:
        logger.info("No drift metrics to materialize for model=%s", report.model_id)
        return 0

    errors = bq_client.insert_rows_json(table, rows)
    if errors:
        logger.error("BigQuery insert errors for analytics_drift_metrics: %s", errors)
        raise RuntimeError(f"Failed to materialize drift metrics: {errors}")

    logger.info("Materialized %d drift metric rows for model=%s", len(rows), report.model_id)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI entry point (for Cloud Run Job / Cloud Scheduler)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run drift computation and materialization — Cloud Run Job entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Compute and materialize drift metrics")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument("--window-days", type=int, default=7, help="Comparison window size in days")
    args = parser.parse_args()

    report = compute_drift_report(args.model_id, window_days=args.window_days)
    rows = materialize_drift_metrics(report)
    logger.info("Drift analysis complete — %d metrics written for model=%s", rows, args.model_id)


if __name__ == "__main__":
    main()
