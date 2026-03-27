#!/usr/bin/env python3
"""Cloud Run Job entry point for batch prediction.

Usage (local):
    python scripts/run_batch_prediction.py --capability classification \\
        --model-artifact-uri gs://i4g-ml-data/models/classifier-xgboost-v1/v1

Usage (Cloud Run Job):
    Configured via env vars (see below) and invoked by Cloud Scheduler
    or ``gcloud run jobs execute batch-prediction``.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch prediction on BigQuery data.",
    )
    parser.add_argument(
        "--capability",
        choices=["classification", "ner", "risk_scoring", "embedding"],
        default="classification",
        help="ML capability to run (default: classification)",
    )
    parser.add_argument(
        "--model-artifact-uri",
        default="",
        help="GCS URI to model artifacts. Falls back to env var per capability.",
    )
    parser.add_argument(
        "--source-query",
        default=None,
        help="Custom BigQuery source query. Default: all cases joined with features.",
    )
    parser.add_argument(
        "--dest-table",
        default=None,
        help="Destination BQ table. Auto-generated with timestamp if omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows per batch (default: 100)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for batch prediction Cloud Run Job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args = parse_args(argv)

    logger.info(
        "Starting batch prediction: capability=%s, batch_size=%d",
        args.capability,
        args.batch_size,
    )

    from ml.serving.batch import run_batch_prediction

    run_batch_prediction(
        capability=args.capability,
        model_artifact_uri=args.model_artifact_uri,
        source_query=args.source_query,
        dest_table=args.dest_table,
        batch_size=args.batch_size,
    )

    logger.info("Batch prediction job complete.")


if __name__ == "__main__":
    main()
