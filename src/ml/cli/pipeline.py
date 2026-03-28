"""Training pipeline submission and management commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

pipeline_app = typer.Typer(help="Training pipeline submission and management.")


@pipeline_app.command("submit")
def submit(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to pipeline config YAML (e.g. pipelines/configs/classification_xgboost.yaml).",
    ),
    experiment: str | None = typer.Option(None, "--experiment", "-e", help="Experiment name for tracking."),
    trigger_reason: str = typer.Option(
        "manual", "--trigger-reason", help="Why this run was triggered (manual, data_volume, drift)."
    ),
    image_tag: str = typer.Option("dev", "--image-tag", "-t", help="Container image tag (dev or prod)."),
) -> None:
    """Submit a training pipeline to Vertex AI Pipelines.

    Auto-compiles KFP YAML if the pipeline source is newer. Uploads config
    to GCS and maps framework to the correct training container image.
    """
    # Guard against direct Python calls where typer.Option defaults are OptionInfo objects
    if not isinstance(trigger_reason, str):
        trigger_reason = "manual"
    if not isinstance(image_tag, str):
        image_tag = "dev"

    from datetime import UTC, datetime

    import yaml
    from google.cloud import aiplatform, storage

    PROJECT = "i4g-ml"
    REGION = "us-central1"
    SA = "sa-ml-platform@i4g-ml.iam.gserviceaccount.com"
    REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"
    PIPELINE_YAML = "pipelines/training_pipeline.yaml"

    # Auto-compile if stale
    yaml_path = Path(PIPELINE_YAML)
    source_path = Path("src/ml/training/pipeline.py")
    if source_path.exists() and (not yaml_path.exists() or source_path.stat().st_mtime > yaml_path.stat().st_mtime):
        typer.echo("Pipeline source newer than compiled YAML — recompiling...")
        from kfp import compiler

        from ml.training.pipeline import training_pipeline

        compiler.Compiler().compile(training_pipeline, str(yaml_path))
        typer.echo(f"Compiled to {yaml_path}")

    aiplatform.init(project=PROJECT, location=REGION)

    pipeline_config: dict = {}
    if config:
        with open(config) as f:
            pipeline_config = yaml.safe_load(f)
        gcs_config_blob = f"configs/{config.name}"
        gcs_client = storage.Client(project=PROJECT)
        bucket = gcs_client.bucket("i4g-ml-data")
        bucket.blob(gcs_config_blob).upload_from_filename(str(config))
        typer.echo(f"Uploaded config to gs://i4g-ml-data/{gcs_config_blob}")

    capability = pipeline_config.get("capability", "classification")
    model_id = pipeline_config.get("model_id", "classification-v1")
    framework = pipeline_config.get("framework", "pytorch")

    framework_container_map = {
        "pytorch": f"{REGISTRY}/train-pytorch:{image_tag}",
        "xgboost": f"{REGISTRY}/train-xgboost:{image_tag}",
    }
    # NER uses a dedicated training container regardless of framework setting
    if capability == "ner":
        container_uri = f"{REGISTRY}/train-ner:{image_tag}"
    else:
        container_uri = framework_container_map.get(framework, f"{REGISTRY}/train-pytorch:{image_tag}")

    experiment_name = experiment
    if not experiment_name:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
        experiment_name = f"{model_id}-{timestamp}"

    eval_gate = pipeline_config.get("eval_gate", {})
    resources = pipeline_config.get("resources", {})
    dataset_version = pipeline_config.get("dataset_version", 1)

    parameter_values = {
        "project_id": PROJECT,
        "region": REGION,
        "dataset_id": "i4g_ml",
        "capability": capability,
        "dataset_version": dataset_version,
        "container_uri": container_uri,
        "serving_container_uri": f"{REGISTRY}/serve:{image_tag}",
        "experiment_name": experiment_name,
        "config_path": f"gs://i4g-ml-data/configs/{config.name}" if config else "",
        "golden_set_uri": f"gs://i4g-ml-data/datasets/{capability}/golden/test.jsonl",
        "endpoint_name": "serving-dev",
        # For risk_scoring, min_overall_f1 slot is repurposed as max_mse threshold
        "min_overall_f1": (
            eval_gate.get("max_mse", eval_gate.get("min_overall_f1", 0.0))
            if capability == "risk_scoring"
            else eval_gate.get("min_overall_f1", 0.0)
        ),
        "max_per_axis_regression": eval_gate.get("max_per_axis_regression", 0.05),
        "machine_type": resources.get("machine_type", "n1-standard-4"),
        "min_replicas": 0,
        "max_replicas": 1,
    }

    labels = {
        "capability": capability,
        "trigger_reason": trigger_reason,
        "framework": framework,
    }

    job = aiplatform.PipelineJob(
        display_name=experiment_name,
        template_path=str(PIPELINE_YAML),
        parameter_values=parameter_values,
        enable_caching=False,
        labels=labels,
    )

    job.submit(service_account=SA)
    typer.echo(f"Pipeline submitted: {job.resource_name}")
    typer.echo(f"  capability={capability}  trigger={trigger_reason}  framework={framework}")


@pipeline_app.command("compile")
def compile_pipeline() -> None:
    """Compile the KFP pipeline YAML from source."""
    from kfp import compiler

    from ml.training.pipeline import training_pipeline

    output = "pipelines/training_pipeline.yaml"
    compiler.Compiler().compile(training_pipeline, output)
    typer.echo(f"Compiled to {output}")
