"""Model registry helpers for Vertex AI Model Registry."""

from __future__ import annotations

import logging

from google.cloud import aiplatform

from ml.config import get_settings

logger = logging.getLogger(__name__)


def register_model(
    *,
    display_name: str,
    artifact_uri: str,
    serving_container_image_uri: str,
    capability: str = "classification",
    framework: str = "pytorch",
    version: int = 1,
) -> aiplatform.Model:
    """Upload a trained model to Vertex AI Model Registry."""
    settings = get_settings()
    aiplatform.init(
        project=settings.platform.project_id,
        location=settings.platform.region,
    )

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        labels={
            "capability": capability,
            "framework": framework,
            "stage": "experimental",
            "version": str(version),
        },
    )
    logger.info("Registered model %s (resource: %s)", display_name, model.resource_name)
    return model


def get_champion_model(capability: str = "classification") -> aiplatform.Model | None:
    """Find the current champion model for a capability, if any."""
    settings = get_settings()
    aiplatform.init(
        project=settings.platform.project_id,
        location=settings.platform.region,
    )

    models = aiplatform.Model.list(
        filter=f'labels.capability="{capability}" AND labels.stage="champion"',
    )
    if not models:
        return None
    # Return the most recently created champion
    return sorted(models, key=lambda m: m.create_time, reverse=True)[0]


def deploy_model_to_endpoint(
    model: aiplatform.Model,
    endpoint_name: str | None = None,
    *,
    machine_type: str | None = None,
    min_replicas: int | None = None,
    max_replicas: int | None = None,
) -> aiplatform.Endpoint:
    """Deploy a model to a Vertex AI Endpoint."""
    settings = get_settings()
    ep_name = endpoint_name or settings.serving.dev_endpoint_name
    mtype = machine_type or settings.serving.machine_type
    min_r = min_replicas if min_replicas is not None else settings.serving.min_replicas
    max_r = max_replicas if max_replicas is not None else settings.serving.max_replicas

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{ep_name}"',
    )
    if not endpoints:
        raise ValueError(f"Endpoint '{ep_name}' not found")
    endpoint = endpoints[0]

    model.deploy(
        endpoint=endpoint,
        machine_type=mtype,
        min_replica_count=min_r,
        max_replica_count=max_r,
        traffic_percentage=100,
    )
    logger.info("Deployed %s to endpoint %s", model.display_name, ep_name)
    return endpoint
