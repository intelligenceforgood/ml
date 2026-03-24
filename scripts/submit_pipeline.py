"""Submit the training pipeline to Vertex AI.

Usage:
    conda run -n ml python scripts/submit_pipeline.py
    conda run -n ml python scripts/submit_pipeline.py --config pipelines/configs/classification_xgboost.yaml
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "i4g-ml"
REGION = "us-central1"
SA = "sa-ml-platform@i4g-ml.iam.gserviceaccount.com"
REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"
PIPELINE_YAML = "pipelines/training_pipeline.yaml"


def _auto_compile_if_stale(pipeline_yaml: str) -> None:
    """Re-compile the KFP pipeline YAML if the source is newer."""
    yaml_path = Path(pipeline_yaml)
    source_path = Path("src/ml/training/pipeline.py")
    if not source_path.exists():
        return
    if not yaml_path.exists() or source_path.stat().st_mtime > yaml_path.stat().st_mtime:
        logger.info("Pipeline source newer than compiled YAML — recompiling")
        from kfp import compiler

        from ml.training.pipeline import training_pipeline

        compiler.Compiler().compile(training_pipeline, str(yaml_path))
        logger.info("Compiled to %s", yaml_path)


def submit_pipeline(
    config_path: str | None = None,
    pipeline_yaml_path: str = PIPELINE_YAML,
    experiment_name: str | None = None,
    trigger_reason: str = "manual",
    *,
    image_tag: str = "dev",
) -> str:
    """Submit a training pipeline to Vertex AI.

    Args:
        config_path: Path to a pipeline config YAML (e.g. classification_xgboost.yaml).
        pipeline_yaml_path: Path to the compiled KFP pipeline YAML.
        experiment_name: Experiment name for tracking.
        trigger_reason: Why this run was triggered (manual, data_volume, drift, etc.).
        image_tag: Container image tag (dev or prod).

    Returns:
        Pipeline job resource name.
    """
    _auto_compile_if_stale(pipeline_yaml_path)

    aiplatform.init(project=PROJECT, location=REGION)

    # Load config if provided
    config: dict[str, Any] = {}
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    capability = config.get("capability", "classification")
    model_id = config.get("model_id", "classification-v1")
    framework = config.get("framework", "pytorch")

    # Determine container image based on framework
    framework_container_map = {
        "pytorch": f"{REGISTRY}/train-pytorch:{image_tag}",
        "xgboost": f"{REGISTRY}/train-xgboost:{image_tag}",
    }
    container_uri = framework_container_map.get(framework, f"{REGISTRY}/train-pytorch:{image_tag}")

    if not experiment_name:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        experiment_name = f"{model_id}-{timestamp}"

    # Build parameter values
    eval_gate = config.get("eval_gate", {})
    resources = config.get("resources", {})
    dataset_version = config.get("dataset_version", 1)

    parameter_values = {
        "project_id": PROJECT,
        "region": REGION,
        "dataset_id": "i4g_ml",
        "capability": capability,
        "dataset_version": dataset_version,
        "container_uri": container_uri,
        "serving_container_uri": f"{REGISTRY}/serve:{image_tag}",
        "experiment_name": experiment_name,
        "config_path": f"gs://i4g-ml-data/configs/{Path(config_path).name}" if config_path else "",
        "golden_set_uri": f"gs://i4g-ml-data/datasets/{capability}/golden/test.jsonl",
        "endpoint_name": "serving-dev",
        "min_overall_f1": eval_gate.get("min_overall_f1", 0.0),
        "max_per_axis_regression": eval_gate.get("max_per_axis_regression", 0.05),
        "machine_type": resources.get("machine_type", "n1-standard-4"),
        "min_replicas": 0,
        "max_replicas": 1,
    }

    # Metadata tags for run tracking
    labels = {
        "capability": capability,
        "trigger_reason": trigger_reason,
        "framework": framework,
    }

    job = aiplatform.PipelineJob(
        display_name=experiment_name,
        template_path=pipeline_yaml_path,
        parameter_values=parameter_values,
        enable_caching=False,
        labels=labels,
    )

    job.submit(service_account=SA)
    logger.info(
        "Pipeline submitted: %s (capability=%s, trigger=%s)",
        job.resource_name,
        capability,
        trigger_reason,
    )
    return job.resource_name


def main() -> None:
    """CLI entry point for pipeline submission."""
    parser = argparse.ArgumentParser(description="Submit training pipeline to Vertex AI")
    parser.add_argument("--config", help="Path to pipeline config YAML")
    parser.add_argument("--experiment", help="Experiment name")
    parser.add_argument("--trigger-reason", default="manual", help="Trigger reason tag")
    parser.add_argument("--image-tag", default="dev", help="Container image tag (dev/prod)")
    args = parser.parse_args()

    submit_pipeline(
        config_path=args.config,
        experiment_name=args.experiment,
        trigger_reason=args.trigger_reason,
        image_tag=args.image_tag,
    )


if __name__ == "__main__":
    main()
