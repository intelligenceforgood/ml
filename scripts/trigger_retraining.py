"""Cloud Run Job entry point — evaluate retraining triggers and submit pipeline.

Usage:
    conda run -n ml python scripts/trigger_retraining.py
    conda run -n ml python scripts/trigger_retraining.py --capability classification --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='{"severity":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# Default config per capability
_DEFAULT_CONFIGS: dict[str, str] = {
    "classification": "pipelines/configs/classification_xgboost.yaml",
}


def run(capability: str = "classification", *, force: bool = False) -> None:
    """Evaluate retraining conditions and submit pipeline if warranted.

    Always exits with code 0 — Cloud Run Jobs treat exit code 1 as failure.
    Use structured logging for alerting, not exit codes.
    """
    from ml.monitoring.triggers import evaluate_retraining_conditions, record_trigger_event

    trigger = evaluate_retraining_conditions(capability, force=force)

    if not trigger.should_retrain:
        logger.info(json.dumps({"action": "retrain_skipped", "capability": capability, "reasons": trigger.reasons}))
        record_trigger_event(trigger, capability=capability)
        return

    # Resolve config path
    config_path = _DEFAULT_CONFIGS.get(capability)
    if config_path and not Path(config_path).exists():
        logger.warning("Config %s not found — submitting without config", config_path)
        config_path = None

    # Submit pipeline
    # Add scripts/ to sys.path so we can import submit_pipeline
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from submit_pipeline import submit_pipeline

    trigger_reason = trigger.reasons[0].split(":")[0] if trigger.reasons else "unknown"

    pipeline_job_name = submit_pipeline(
        config_path=config_path,
        trigger_reason=trigger_reason,
    )

    # Record trigger event with pipeline job name
    record_trigger_event(trigger, capability=capability, pipeline_job_name=pipeline_job_name)

    logger.info(
        json.dumps(
            {
                "action": "retrain_submitted",
                "capability": capability,
                "reasons": trigger.reasons,
                "pipeline_job": pipeline_job_name,
            }
        )
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate retraining triggers and submit pipeline")
    parser.add_argument("--capability", default="classification", help="ML capability to evaluate")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of conditions")
    args = parser.parse_args()

    try:
        run(capability=args.capability, force=args.force)
    except Exception:
        logger.exception("Trigger evaluation failed")
    # Always exit 0 — Cloud Run Jobs treat exit 1 as failure


if __name__ == "__main__":
    main()
