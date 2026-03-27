"""Model deployment and redeployment commands."""

from __future__ import annotations

import logging

import typer

logger = logging.getLogger(__name__)

deploy_app = typer.Typer(help="Model deployment and redeployment.")


@deploy_app.command("serving")
def deploy_serving(
    endpoint: str = typer.Option("serving-dev", "--endpoint", "-e", help="Vertex AI endpoint display name."),
    model_name: str = typer.Option("classification-stub-v1", "--model-name", "-m", help="Display name for the model."),
    image_tag: str = typer.Option("dev", "--image-tag", "-t", help="Container image tag (dev or prod)."),
    machine_type: str = typer.Option("n1-standard-2", "--machine-type", help="Machine type for serving."),
    min_replicas: int = typer.Option(0, "--min-replicas", help="Minimum replica count."),
    max_replicas: int = typer.Option(1, "--max-replicas", help="Maximum replica count."),
) -> None:
    """Upload model to Vertex AI Model Registry and deploy to an endpoint.

    Creates the endpoint if it doesn't exist. Deploys with 100% traffic.
    Typical run time: 5-10 minutes for initial deployment.
    """
    from google.cloud import aiplatform

    PROJECT = "i4g-ml"
    REGION = "us-central1"
    REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"

    aiplatform.init(project=PROJECT, location=REGION)

    typer.echo("Uploading model to Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=f"{REGISTRY}/serve:{image_tag}",
        serving_container_ports=[8080],
        serving_container_predict_route="/predict/classify",
        serving_container_health_route="/health",
        labels={"stage": "experimental", "capability": "classification", "phase": "0"},
    )
    typer.echo(f"Model registered: {model.resource_name}")

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint}"')
    if not endpoints:
        typer.echo(f"Endpoint '{endpoint}' not found. Creating it...")
        ep = aiplatform.Endpoint.create(display_name=endpoint)
    else:
        ep = endpoints[0]
    typer.echo(f"Endpoint: {ep.resource_name}")

    typer.echo("Deploying model to endpoint (this may take several minutes)...")
    model.deploy(
        endpoint=ep,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
        service_account="sa-ml-platform@i4g-ml.iam.gserviceaccount.com",
    )
    typer.echo(f"Model deployed to {endpoint}")


@deploy_app.command("redeploy")
def redeploy_serving(
    endpoint: str = typer.Option("serving-dev", "--endpoint", "-e", help="Vertex AI endpoint display name."),
    model_name: str = typer.Option("classification-stub-v1", "--model-name", "-m", help="Display name for the model."),
    image_tag: str = typer.Option("dev", "--image-tag", "-t", help="Container image tag (dev or prod)."),
    machine_type: str = typer.Option("n1-standard-2", "--machine-type", help="Machine type for serving."),
) -> None:
    """Redeploy serving container after updating the image.

    Undeploys all existing models on the endpoint before deploying the new version.
    Typical run time: 5-10 minutes.
    """
    from google.cloud import aiplatform

    PROJECT = "i4g-ml"
    REGION = "us-central1"
    REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"

    aiplatform.init(project=PROJECT, location=REGION)

    typer.echo("Uploading new model version...")
    model = aiplatform.Model.upload(
        display_name=model_name,
        serving_container_image_uri=f"{REGISTRY}/serve:{image_tag}",
        serving_container_ports=[8080],
        serving_container_predict_route="/predict/classify",
        serving_container_health_route="/health",
        labels={"stage": "experimental", "capability": "classification", "phase": "0"},
        is_default_version=True,
    )
    typer.echo(f"New model version: {model.resource_name}")

    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint}"')
    ep = endpoints[0]
    typer.echo(f"Endpoint: {ep.resource_name}")

    for dm in ep.gca_resource.deployed_models:
        typer.echo(f"Undeploying: {dm.id}")
        ep.undeploy(deployed_model_id=dm.id)

    typer.echo("Deploying new model version...")
    model.deploy(
        endpoint=ep,
        machine_type=machine_type,
        min_replica_count=0,
        max_replica_count=1,
        traffic_percentage=100,
        service_account="sa-ml-platform@i4g-ml.iam.gserviceaccount.com",
    )
    typer.echo("Done — new model deployed.")
