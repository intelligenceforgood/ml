"""Redeploy the serving container after updating the image.

Usage:
    conda run -n ml python scripts/redeploy_serving.py
"""

from __future__ import annotations

import logging

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "i4g-ml"
REGION = "us-central1"
REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"
ENDPOINT_NAME = "serving-dev"


def main() -> None:
    """Upload a new model version and redeploy to the serving-dev endpoint."""
    aiplatform.init(project=PROJECT, location=REGION)

    # Upload new model version with updated container
    logger.info("Uploading new model version...")
    model = aiplatform.Model.upload(
        display_name="classification-stub-v1",
        serving_container_image_uri=f"{REGISTRY}/serve:dev",
        serving_container_ports=[8080],
        serving_container_predict_route="/predict/classify",
        serving_container_health_route="/health",
        labels={"stage": "experimental", "capability": "classification", "phase": "0"},
        is_default_version=True,
    )
    logger.info("New model version: %s", model.resource_name)

    # Get endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
    endpoint = endpoints[0]
    logger.info("Endpoint: %s", endpoint.resource_name)

    # Undeploy all existing models
    for dm in endpoint.gca_resource.deployed_models:
        logger.info("Undeploying: %s", dm.id)
        endpoint.undeploy(deployed_model_id=dm.id)

    # Deploy new version
    logger.info("Deploying new model version...")
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=0,
        max_replica_count=1,
        traffic_percentage=100,
        service_account="sa-ml-platform@i4g-ml.iam.gserviceaccount.com",
    )
    logger.info("DONE - new model deployed")


if __name__ == "__main__":
    main()
