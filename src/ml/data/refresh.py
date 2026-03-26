"""Automated dataset refresh pipeline.

Orchestrates: ETL ingest → feature re-materialization → dataset re-export.
Designed to run as a Cloud Run Job on a weekly schedule.
"""

from __future__ import annotations

import logging
import sys

from ml.data.datasets import create_dataset_version
from ml.data.etl import run_incremental_ingest

logger = logging.getLogger(__name__)


def refresh_dataset(
    *,
    capability: str = "classification",
    min_samples_per_class: int = 50,
    redact: bool = True,
) -> dict:
    """Run the full data refresh pipeline.

    Steps:
      1. Incremental ETL from Cloud SQL → BigQuery raw tables
      2. Create a new dataset version (auto-incremented) with PII redaction
         and label-source priority (analyst > llm_bootstrap)

    Feature re-materialization is handled by BigQuery scheduled query
    (``materialize_features.sql``) and runs independently at 3 AM UTC.
    """
    # Step 1 — ETL ingest
    logger.info("Starting incremental ETL ingest")
    ingest_results = run_incremental_ingest()
    failed = {k: v for k, v in ingest_results.items() if v < 0}
    if failed:
        logger.error("ETL ingest had failures: %s — proceeding with dataset export", list(failed.keys()))

    logger.info("ETL ingest results: %s", ingest_results)

    # Step 2 — Dataset export (auto-version, PII redaction, label priority)
    logger.info("Creating new dataset version (capability=%s, redact=%s)", capability, redact)
    metadata = create_dataset_version(
        capability=capability,
        min_samples_per_class=min_samples_per_class,
        redact=redact,
    )

    logger.info(
        "Dataset refresh complete: %s v%d — train=%d, eval=%d, test=%d",
        metadata["dataset_id"],
        metadata["version"],
        metadata["train_count"],
        metadata["eval_count"],
        metadata["test_count"],
    )
    return metadata


def main() -> None:
    """CLI entry point for the data refresh pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    try:
        metadata = refresh_dataset()
        logger.info("Success: %s", metadata["gcs_path"])
    except Exception:
        logger.exception("Dataset refresh failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
