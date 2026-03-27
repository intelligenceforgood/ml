"""Vertex AI pipeline submission library.

Shared logic for submitting training pipelines. Used by:
- ``i4g-ml pipeline submit`` CLI command
- ``i4g-ml retrain trigger`` CLI command
- Cloud Run Job entry points
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT = os.getenv("I4G_ML_PROJECT", "i4g-ml")
_REGION = os.getenv("I4G_ML_REGION", "us-central1")
_ARTIFACT_REGISTRY = f"{_REGION}-docker.pkg.dev/{_PROJECT}/containers"
_PIPELINE_ROOT = os.getenv(
    "I4G_ML_PIPELINE_ROOT",
    f"gs://{_PROJECT}-data/pipelines",
)

_FRAMEWORK_CONTAINER_MAP: dict[str, str] = {
    "xgboost": "train-xgboost",
    "pytorch": "train-pytorch",
    "sklearn": "train-sklearn",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auto_compile_if_stale(source_path: str) -> None:
    """Re-compile a pipeline YAML from source if stale.

    No-op when the source file does not exist.
    """
    src = Path(source_path)
    if not src.exists():
        return
    # Future: check timestamps and compile if source is newer than compiled YAML


def _load_config(config_path: str | None) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict if *config_path* is ``None``."""
    if config_path is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Core submission function
# ---------------------------------------------------------------------------


def submit_pipeline(
    config_path: str | None = None,
    experiment_name: str | None = None,
    trigger_reason: str | None = None,
    image_tag: str = "dev",
) -> str:
    """Submit a Vertex AI Pipeline job.

    Args:
        config_path: Optional path to a YAML training config.
        experiment_name: Display name / experiment. Auto-generated from model_id if omitted.
        trigger_reason: Why training was triggered (``manual``, ``drift``, ``data_volume``, …).
        image_tag: Docker image tag to use (default ``dev``).

    Returns:
        The ``resource_name`` of the submitted pipeline job.
    """
    cfg = _load_config(config_path)

    model_id: str = cfg.get("model_id", "model")
    capability: str = cfg.get("capability", "classification")
    framework: str = cfg.get("framework", "xgboost")
    eval_gate: dict[str, Any] = cfg.get("eval_gate", {})
    resources: dict[str, Any] = cfg.get("resources", {})

    container_name = _FRAMEWORK_CONTAINER_MAP.get(framework, f"train-{framework}")
    container_uri = f"{_ARTIFACT_REGISTRY}/{container_name}:{image_tag}"

    display_name = experiment_name or f"{model_id}-{capability}"

    parameter_values: dict[str, Any] = {
        "capability": capability,
        "container_uri": container_uri,
        **{k: v for k, v in eval_gate.items()},
        **{k: v for k, v in resources.items()},
    }

    labels: dict[str, str] = {
        "framework": framework,
    }
    if trigger_reason:
        labels["trigger_reason"] = trigger_reason

    aiplatform.init(project=_PROJECT, location=_REGION)

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=cfg.get("template_path", f"{_PIPELINE_ROOT}/{capability}_pipeline.yaml"),
        pipeline_root=_PIPELINE_ROOT,
        parameter_values=parameter_values,
        labels=labels,
    )
    job.submit()

    return job.resource_name
