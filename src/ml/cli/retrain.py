"""Retraining trigger evaluation and submission commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

retrain_app = typer.Typer(help="Retraining trigger evaluation and submission.")

_DEFAULT_CONFIGS: dict[str, str] = {
    "classification": "pipelines/configs/classification_xgboost.yaml",
    "risk_scoring": "pipelines/configs/risk_scoring_xgboost.yaml",
}


@retrain_app.command("trigger")
def trigger(
    capability: str = typer.Option(
        "classification", "--capability", "-c", help="ML capability to evaluate (classification, ner, risk_scoring)."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force retraining regardless of conditions."),
    image_tag: str = typer.Option("dev", "--image-tag", "-t", help="Container image tag (dev or prod)."),
) -> None:
    """Evaluate retraining conditions and submit pipeline if warranted.

    Checks three conditions: data volume (≥200 new labels), drift (PSI > 0.2),
    and time (>30 days since last training). If any condition is met (or --force),
    submits a training pipeline.

    Always exits with code 0 — Cloud Run Jobs treat exit 1 as failure.
    """
    try:
        from ml.monitoring.triggers import evaluate_retraining_conditions, record_trigger_event

        trigger_result = evaluate_retraining_conditions(capability, force=force)

        if not trigger_result.should_retrain:
            typer.echo(
                json.dumps({"action": "retrain_skipped", "capability": capability, "reasons": trigger_result.reasons})
            )
            record_trigger_event(trigger_result, capability=capability)
            return

        config_path = _DEFAULT_CONFIGS.get(capability)
        if config_path and not Path(config_path).exists():
            typer.echo(f"Config {config_path} not found — submitting without config")
            config_path = None

        trigger_reason = trigger_result.reasons[0].split(":")[0] if trigger_result.reasons else "unknown"

        # Submit pipeline
        # Use the underlying submit_pipeline function directly
        from scripts.submit_pipeline import submit_pipeline

        pipeline_job_name = submit_pipeline(
            config_path=config_path,
            trigger_reason=trigger_reason,
            image_tag=image_tag,
        )

        record_trigger_event(trigger_result, capability=capability, pipeline_job_name=pipeline_job_name)

        typer.echo(
            json.dumps(
                {
                    "action": "retrain_submitted",
                    "capability": capability,
                    "reasons": trigger_result.reasons,
                    "pipeline_job": pipeline_job_name,
                }
            )
        )

    except Exception:
        logger.exception("Trigger evaluation failed")
        # Always exit 0 — Cloud Run Jobs convention
