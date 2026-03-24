"""Cost monitoring and per-capability cost attribution.

Queries GCP billing export and ML-platform BigQuery tables to compute:
- Per-component cost breakdown (Vertex AI, Cloud Run, BigQuery, Storage)
- Per-prediction cost of the ML platform
- Comparison to LLM API cost-per-prediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ml.config import get_settings

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentCost:
    """Cost for a single GCP component."""

    component: str
    cost_usd: float
    description: str = ""


@dataclass(frozen=True)
class CostSummary:
    """Aggregate cost summary for the ML platform."""

    period_days: int
    total_cost_usd: float
    total_predictions: int
    cost_per_prediction_usd: float
    components: list[ComponentCost] = field(default_factory=list)
    computed_at: str = ""


@dataclass(frozen=True)
class CostComparison:
    """ML platform vs LLM API cost comparison."""

    ml_cost_per_prediction: float
    llm_cost_per_prediction: float
    savings_per_prediction: float
    savings_percentage: float
    ml_total_cost: float
    llm_equivalent_cost: float
    total_predictions: int
    period_days: int


# ---------------------------------------------------------------------------
# Billing export query
# ---------------------------------------------------------------------------

_BILLING_QUERY = """
SELECT
    service.description AS service_name,
    SUM(cost) AS total_cost
FROM `{billing_table}`
WHERE
    project.id = @project_id
    AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @period_days DAY)
    AND service.description IN (
        'Vertex AI',
        'Cloud Run',
        'BigQuery',
        'Cloud Storage'
    )
GROUP BY service.description
ORDER BY total_cost DESC
"""

_PREDICTION_COUNT_QUERY = """
SELECT COUNT(*) AS total
FROM `{project}.{dataset}.{pred_table}`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @period_days DAY)
"""


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_cost_summary(
    *,
    period_days: int = 30,
    billing_table: str = "",
    client: bigquery.Client | None = None,
) -> CostSummary:
    """Query GCP billing export and prediction volume to produce cost summary.

    Args:
        period_days: Lookback period.
        billing_table: Fully-qualified billing export table
            (e.g. ``project.dataset.gcp_billing_export_v1_XXXX``).
            If empty, returns a zero-cost summary (billing export not configured).
        client: Optional pre-configured BigQuery client.

    Returns:
        A ``CostSummary`` with per-component costs and per-prediction cost.
    """
    from google.cloud import bigquery as bq

    settings = get_settings()
    bq_client = client or _get_bq_client()

    # Get prediction count
    count_query = _PREDICTION_COUNT_QUERY.format(
        project=settings.platform.project_id,
        dataset=settings.bigquery.dataset_id,
        pred_table=settings.bigquery.prediction_log_table,
    )
    count_config = bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("period_days", "INT64", period_days)],
    )
    count_result = list(bq_client.query(count_query, job_config=count_config).result())
    total_predictions = count_result[0].total if count_result else 0

    if not billing_table:
        logger.warning("No billing_table configured — returning zero-cost summary")
        return CostSummary(
            period_days=period_days,
            total_cost_usd=0.0,
            total_predictions=total_predictions,
            cost_per_prediction_usd=0.0,
            computed_at=datetime.now(UTC).isoformat(),
        )

    # Query billing export
    billing_query = _BILLING_QUERY.format(billing_table=billing_table)
    billing_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ScalarQueryParameter("project_id", "STRING", settings.platform.project_id),
            bq.ScalarQueryParameter("period_days", "INT64", period_days),
        ],
    )
    billing_rows = list(bq_client.query(billing_query, job_config=billing_config).result())

    components = [ComponentCost(component=row.service_name, cost_usd=float(row.total_cost)) for row in billing_rows]
    total_cost = sum(c.cost_usd for c in components)

    return CostSummary(
        period_days=period_days,
        total_cost_usd=round(total_cost, 2),
        total_predictions=total_predictions,
        cost_per_prediction_usd=round(_safe_div(total_cost, total_predictions), 6),
        components=components,
        computed_at=datetime.now(UTC).isoformat(),
    )


def compare_to_llm_cost(
    *,
    period_days: int = 30,
    llm_cost_per_prediction: float = 0.03,
    billing_table: str = "",
    client: bigquery.Client | None = None,
) -> CostComparison:
    """Compare ML platform cost to LLM API cost on a per-prediction basis.

    Args:
        period_days: Lookback period.
        llm_cost_per_prediction: Estimated cost per LLM API call (default $0.03
            based on GPT-4o mini pricing for classification prompt + response).
        billing_table: Billing export table (same as ``compute_cost_summary``).
        client: Optional BigQuery client.

    Returns:
        A ``CostComparison`` with savings metrics.
    """
    summary = compute_cost_summary(
        period_days=period_days,
        billing_table=billing_table,
        client=client,
    )

    llm_equivalent = llm_cost_per_prediction * summary.total_predictions
    savings = llm_equivalent - summary.total_cost_usd
    savings_pct = _safe_div(savings, llm_equivalent) * 100

    return CostComparison(
        ml_cost_per_prediction=summary.cost_per_prediction_usd,
        llm_cost_per_prediction=llm_cost_per_prediction,
        savings_per_prediction=round(llm_cost_per_prediction - summary.cost_per_prediction_usd, 6),
        savings_percentage=round(savings_pct, 1),
        ml_total_cost=summary.total_cost_usd,
        llm_equivalent_cost=round(llm_equivalent, 2),
        total_predictions=summary.total_predictions,
        period_days=period_days,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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
# CLI entry point (for Cloud Run Job / Cloud Scheduler)
# ---------------------------------------------------------------------------


def main() -> None:
    """Run cost materialization — intended as Cloud Run Job entry point."""
    import uuid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    comparison = compare_to_llm_cost()
    settings = get_settings()
    bq_client = _get_bq_client()
    table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_cost_summary"

    from datetime import timedelta

    now = datetime.now(UTC)
    row = {
        "summary_id": str(uuid.uuid4()),
        "model_id": None,
        "capability": "classification",
        "prediction_count": comparison.total_predictions,
        "ml_cost_per_prediction": comparison.ml_cost_per_prediction,
        "llm_cost_per_prediction": comparison.llm_cost_per_prediction,
        "ml_total": comparison.ml_total_cost,
        "llm_total": comparison.llm_equivalent_cost,
        "savings_pct": comparison.savings_percentage,
        "period_start": (now - timedelta(days=comparison.period_days)).isoformat(),
        "period_end": now.isoformat(),
        "computed_at": now.isoformat(),
    }

    errors = bq_client.insert_rows_json(table, [row])
    if errors:
        logger.error("BigQuery insert errors for analytics_cost_summary: %s", errors)
    else:
        logger.info("Cost materialization complete — savings: %.1f%%", comparison.savings_percentage)


if __name__ == "__main__":
    main()
