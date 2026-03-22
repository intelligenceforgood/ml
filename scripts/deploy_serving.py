"""Deploy the serving container to Vertex AI serving-dev endpoint.

Phase 0: No real training yet — the serving container uses stub predictions.
This script proves the model lifecycle works end-to-end:
1. Upload model to Vertex AI Model Registry
2. Deploy to serving-dev endpoint

Usage:
    conda run -n ml python scripts/deploy_serving.py
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
    """Upload a stub model and deploy it to the serving-dev endpoint."""
    aiplatform.init(project=PROJECT, location=REGION)

    # Upload model to registry
    logger.info("Uploading model to Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name="classification-stub-v1",
        serving_container_image_uri=f"{REGISTRY}/serve:dev",
        serving_container_ports=[8080],
        serving_container_predict_route="/predict/classify",
        serving_container_health_route="/health",
        labels={"stage": "experimental", "capability": "classification", "phase": "0"},
    )
    logger.info("Model registered: %s", model.resource_name)

    # Find serving-dev endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
    if not endpoints:
        logger.error("Endpoint '%s' not found. Creating it...", ENDPOINT_NAME)
        endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    else:
        endpoint = endpoints[0]
    logger.info("Endpoint: %s", endpoint.resource_name)

    # Deploy model to endpoint
    logger.info("Deploying model to endpoint (this may take several minutes)...")
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=0,
        max_replica_count=1,
        traffic_percentage=100,
        service_account="sa-ml-platform@i4g-ml.iam.gserviceaccount.com",
    )
    logger.info("Model deployed to %s", ENDPOINT_NAME)
    logger.info("Endpoint resource: %s", endpoint.resource_name)


if __name__ == "__main__":
    main()
