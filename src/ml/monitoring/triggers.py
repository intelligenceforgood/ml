"""Retraining trigger logic — data volume, drift, time-based.

Evaluates whether a capability should trigger a retraining pipeline run.
Conditions: new analyst label volume, prediction/feature drift, time since
last training.  Results are logged to ``analytics_trigger_log`` in BigQuery.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ml.config import get_settings

if TYPE_CHECKING:
    from google.cloud import bigquery

logger = logging.getLogger(__name__)

_bq_client: bigquery.Client | None = None

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_ANALYST_LABELS = 200
DRIFT_PSI_THRESHOLD = 0.2
MAX_DAYS_SINCE_TRAINING = 30


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


@dataclass
class RetrainingTrigger:
    """Result of retraining condition evaluation."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    new_analyst_label_count: int = 0
    max_drift_psi: float = 0.0
    last_training_date: datetime | None = None


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

_ANALYST_LABEL_COUNT_QUERY = """
SELECT COUNT(*) AS cnt
FROM `{project}.{dataset}.raw_analyst_labels`
WHERE
    label_source = 'analyst'
    AND _ingested_at > @since
"""

_LAST_TRAINING_QUERY = """
SELECT MAX(created_at) AS last_training
FROM `{project}.{dataset}.training_dataset_registry`
WHERE capability = @capability
"""

_LATEST_DRIFT_QUERY = """
SELECT
    axis_or_feature,
    psi,
    is_drifted
FROM `{project}.{dataset}.analytics_drift_metrics`
WHERE
    model_id = (
        SELECT model_id
        FROM `{project}.{dataset}.analytics_drift_metrics`
        ORDER BY computed_at DESC
        LIMIT 1
    )
    AND computed_at = (
        SELECT MAX(computed_at)
        FROM `{project}.{dataset}.analytics_drift_metrics`
    )
"""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def evaluate_retraining_conditions(
    capability: str = "classification",
    *,
    force: bool = False,
    client: bigquery.Client | None = None,
) -> RetrainingTrigger:
    """Evaluate whether a capability should trigger retraining.

    Args:
        capability: ML capability to evaluate (e.g., ``"classification"``).
        force: If True, unconditionally recommend retraining.
        client: Optional BigQuery client.

    Returns:
        A ``RetrainingTrigger`` with evaluation results.
    """
    from google.cloud import bigquery as bq

    bq_client = client or _get_bq_client()
    settings = get_settings()
    project = settings.platform.project_id
    dataset = settings.bigquery.dataset_id

    if force:
        return RetrainingTrigger(
            should_retrain=True,
            reasons=["forced"],
        )

    reasons: list[str] = []
    new_label_count = 0
    max_psi = 0.0
    last_training: datetime | None = None

    # 1. Get last training date
    last_train_query = _LAST_TRAINING_QUERY.format(project=project, dataset=dataset)
    last_train_config = bq.QueryJobConfig(
        query_parameters=[bq.ScalarQueryParameter("capability", "STRING", capability)]
    )
    last_train_rows = list(bq_client.query(last_train_query, job_config=last_train_config).result())
    if last_train_rows and last_train_rows[0].last_training:
        last_training = last_train_rows[0].last_training
        if isinstance(last_training, str):
            last_training = datetime.fromisoformat(last_training)
    else:
        last_training = None

    # 2. Data volume check — new analyst labels since last training
    since = last_training or datetime(2020, 1, 1, tzinfo=UTC)
    label_query = _ANALYST_LABEL_COUNT_QUERY.format(project=project, dataset=dataset)
    label_config = bq.QueryJobConfig(query_parameters=[bq.ScalarQueryParameter("since", "TIMESTAMP", since)])
    label_rows = list(bq_client.query(label_query, job_config=label_config).result())
    if label_rows:
        new_label_count = label_rows[0].cnt or 0

    if new_label_count >= MIN_ANALYST_LABELS:
        reasons.append(f"data_volume: {new_label_count} new analyst labels (threshold: {MIN_ANALYST_LABELS})")

    # 3. Drift check — latest drift metrics
    drift_query = _LATEST_DRIFT_QUERY.format(project=project, dataset=dataset)
    try:
        drift_rows = list(bq_client.query(drift_query).result())
        for row in drift_rows:
            if row.psi and row.psi > max_psi:
                max_psi = row.psi
            if row.is_drifted:
                reasons.append(f"drift: {row.axis_or_feature} PSI={row.psi:.4f}")
    except Exception:  # noqa: BLE001 — table may not exist yet
        logger.warning("Could not query drift metrics — table may not exist yet", exc_info=True)

    # 4. Time-based check
    if last_training:
        days_since = (datetime.now(UTC) - last_training).days
        if days_since > MAX_DAYS_SINCE_TRAINING:
            reasons.append(
                f"time_elapsed: {days_since} days since last training (threshold: {MAX_DAYS_SINCE_TRAINING})"
            )
    else:
        reasons.append("no_previous_training: no training record found")

    return RetrainingTrigger(
        should_retrain=len(reasons) > 0,
        reasons=reasons,
        new_analyst_label_count=new_label_count,
        max_drift_psi=round(max_psi, 4),
        last_training_date=last_training,
    )


# ---------------------------------------------------------------------------
# Trigger event logging
# ---------------------------------------------------------------------------


def record_trigger_event(
    trigger: RetrainingTrigger,
    *,
    capability: str = "classification",
    pipeline_job_name: str | None = None,
    client: bigquery.Client | None = None,
) -> None:
    """Write trigger evaluation result to ``analytics_trigger_log``.

    Args:
        trigger: The evaluated trigger result.
        capability: ML capability.
        pipeline_job_name: Vertex AI pipeline job name (if retrain was submitted).
        client: Optional BigQuery client.
    """
    bq_client = client or _get_bq_client()
    settings = get_settings()
    table = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_trigger_log"

    row = {
        "event_id": str(uuid.uuid4()),
        "capability": capability,
        "should_retrain": trigger.should_retrain,
        "reasons": json.dumps(trigger.reasons),
        "new_label_count": trigger.new_analyst_label_count,
        "max_drift_psi": trigger.max_drift_psi,
        "pipeline_job_name": pipeline_job_name,
        "triggered_at": datetime.now(UTC).isoformat(),
    }

    errors = bq_client.insert_rows_json(table, [row])
    if errors:
        logger.error("BigQuery insert errors for analytics_trigger_log: %s", errors)
        raise RuntimeError(f"Failed to record trigger event: {errors}")

    logger.info(
        "Recorded trigger event: capability=%s, should_retrain=%s, reasons=%s",
        capability,
        trigger.should_retrain,
        trigger.reasons,
    )
