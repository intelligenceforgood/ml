"""Batch prediction and serving utility commands."""

from __future__ import annotations

import logging

import typer

logger = logging.getLogger(__name__)

serve_app = typer.Typer(help="Batch prediction and serving utilities.")


def run_batch(
    capability: str = "classification",
    model_artifact_uri: str = "",
    source_query: str | None = None,
    dest_table: str | None = None,
    batch_size: int = 100,
) -> None:
    """Core batch prediction logic — callable from CLI and tests."""
    from ml.serving.batch import run_batch_prediction

    run_batch_prediction(
        capability=capability,
        model_artifact_uri=model_artifact_uri,
        source_query=source_query,
        dest_table=dest_table,
        batch_size=batch_size,
    )


@serve_app.command("batch")
def batch_predict(
    capability: str = typer.Option(
        "classification", "--capability", "-c", help="Capability: classification, ner, risk_scoring, embedding."
    ),
    model_artifact_uri: str = typer.Option(
        "", "--model-artifact-uri", "-m", help="gs:// URI to model artifacts (uses env default if omitted)."
    ),
    source_query: str | None = typer.Option(
        None, "--source-query", "-q", help="Custom BigQuery source query (default: all cases)."
    ),
    dest_table: str | None = typer.Option(
        None, "--dest-table", "-d", help="Destination BQ table (auto-generated if omitted)."
    ),
    batch_size: int = typer.Option(100, "--batch-size", "-b", help="Batch size for processing."),
) -> None:
    """Run batch prediction for historical re-classification or embedding generation.

    Reads cases from BigQuery, runs inference in batches, and writes results
    to a destination table. Supports classification, NER, risk_scoring, and
    embedding capabilities.
    """
    run_batch(
        capability=capability,
        model_artifact_uri=model_artifact_uri,
        source_query=source_query,
        dest_table=dest_table,
        batch_size=batch_size,
    )
    typer.echo(f"Batch prediction complete for capability={capability}")
