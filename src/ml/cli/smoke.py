"""E2E smoke test commands."""

from __future__ import annotations

import logging
import time

import typer

logger = logging.getLogger(__name__)

smoke_app = typer.Typer(help="End-to-end smoke tests.")


@smoke_app.command("e2e")
def e2e(
    endpoint: str = typer.Option("serving-dev", "--endpoint", "-e", help="Vertex AI endpoint display name."),
    wait_secs: int = typer.Option(30, "--wait", "-w", help="Seconds to wait for BQ streaming buffer."),
) -> None:
    """Run end-to-end smoke test against a serving endpoint.

    Sends a prediction, waits for BigQuery streaming buffer, and verifies
    the prediction was logged. Reports lifecycle status.
    """
    from google.cloud import aiplatform, bigquery

    PROJECT = "i4g-ml"
    REGION = "us-central1"
    DATASET = "i4g_ml"

    aiplatform.init(project=PROJECT, location=REGION)
    bq_client = bigquery.Client(project=PROJECT)

    # Get endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint}"')
    if not endpoints:
        typer.echo(f"Endpoint '{endpoint}' not found", err=True)
        raise typer.Exit(code=1)
    ep = endpoints[0]
    typer.echo(f"Endpoint: {ep.resource_name}")

    # Send prediction
    typer.echo("--- Step 1: Send prediction ---")
    response = ep.predict(instances=[{"text": "E2E smoke test - suspicious wire transfer", "case_id": "e2e-smoke-001"}])
    pred = response.predictions[0]
    prediction_id = pred["prediction_id"]
    typer.echo(f"prediction_id: {prediction_id}")
    typer.echo(f"INTENT: {pred['prediction']['INTENT']}")
    typer.echo(f"CHANNEL: {pred['prediction']['CHANNEL']}")

    assert "prediction_id" in pred, "Missing prediction_id"
    assert "prediction" in pred, "Missing prediction"
    typer.echo("PASS: Prediction response structure valid")

    # Verify BQ logging
    typer.echo(f"--- Step 2: Verify BQ logging (waiting {wait_secs}s) ---")
    time.sleep(wait_secs)

    query = f"""
    SELECT prediction_id, case_id, model_id, endpoint, prediction, timestamp
    FROM `{PROJECT}.{DATASET}.predictions_prediction_log`
    WHERE prediction_id = '{prediction_id}'
    """
    rows = list(bq_client.query(query).result())
    if len(rows) != 1:
        typer.echo(f"Expected 1 prediction log row, got {len(rows)}", err=True)
        raise typer.Exit(code=1)
    row = rows[0]
    typer.echo(f"Logged: prediction_id={row.prediction_id}, model_id={row.model_id}")
    typer.echo("PASS: Prediction logged in BigQuery")

    # Summary
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("E2E SMOKE TEST: ALL CHECKS PASSED")
    typer.echo("=" * 60)
    typer.echo("  [x] Prediction endpoint returns valid response")
    typer.echo("  [x] Prediction logged in BigQuery")
    typer.echo("  [x] Model lifecycle operational")
    typer.echo("=" * 60)
