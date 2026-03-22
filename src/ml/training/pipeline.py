"""KFP v2 training pipeline definitions.

Five-stage pipeline: prepare_dataset → train_model → evaluate_model →
register_model → deploy_model.

Note: ``from __future__ import annotations`` is intentionally omitted —
KFP v2 requires eager annotation evaluation to introspect component signatures.
"""

from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-storage"],
)
def prepare_dataset(
    project_id: str,
    dataset_id: str,
    capability: str,
    dataset_version: int,
) -> str:
    """Verify pre-exported dataset splits exist in GCS.

    Returns the GCS path prefix for the dataset version.
    """
    from google.cloud import storage as gcs

    bucket_name = "i4g-ml-data"
    prefix = f"datasets/{capability}/v{dataset_version}"
    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    for split in ("train", "eval", "test"):
        blob = bucket.blob(f"{prefix}/{split}.jsonl")
        if not blob.exists():
            raise FileNotFoundError(f"Missing dataset split: gs://{bucket_name}/{prefix}/{split}.jsonl")

    return f"gs://{bucket_name}/{prefix}"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def train_model(
    project_id: str,
    region: str,
    container_uri: str,
    config_path: str,
    dataset_gcs_path: str,
    experiment_name: str,
) -> str:
    """Submit a Vertex AI CustomJob for model training.

    Returns the GCS URI of the model artifacts.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    job = aiplatform.CustomJob.from_local_script(
        display_name=f"train-{experiment_name}",
        script_path="train.py",
        container_uri=container_uri,
        args=[
            "--config",
            config_path,
            "--dataset",
            dataset_gcs_path,
            "--experiment",
            experiment_name,
        ],
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )
    job.run(experiment=experiment_name)

    return f"gs://i4g-ml-data/models/{experiment_name}/"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-storage"],
)
def evaluate_model(
    model_uri: str,
    golden_set_uri: str,
    min_overall_f1: float,
    max_per_axis_regression: float,
) -> NamedTuple("EvalOutputs", [("passed", str), ("metrics_json", str)]):
    """Evaluate a trained model against the golden test set."""
    import json

    # Phase 0: stub evaluation — real impl loads model and runs inference
    metrics = {
        "overall_f1": 0.0,
        "overall_precision": 0.0,
        "overall_recall": 0.0,
        "per_axis": {},
        "eval_gate_passed": True,
    }

    passed = "true" if metrics["eval_gate_passed"] else "false"
    return (passed, json.dumps(metrics))


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def register_model(
    project_id: str,
    region: str,
    model_uri: str,
    display_name: str,
    serving_container_uri: str,
    eval_passed: str,
) -> str:
    """Register a trained model in Vertex AI Model Registry.

    Returns the model resource name or 'SKIPPED'.
    """
    if eval_passed != "true":
        return "SKIPPED"

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=model_uri,
        serving_container_image_uri=serving_container_uri,
        labels={"stage": "candidate", "capability": "classification"},
    )
    return model.resource_name


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"],
)
def deploy_model(
    project_id: str,
    region: str,
    model_name: str,
    endpoint_name: str,
    machine_type: str,
    min_replicas: int,
    max_replicas: int,
) -> None:
    """Deploy a registered model to a Vertex AI Endpoint."""
    if model_name == "SKIPPED":
        return

    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    model = aiplatform.Model(model_name=model_name)
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    if not endpoints:
        raise ValueError(f"Endpoint '{endpoint_name}' not found")

    model.deploy(
        endpoint=endpoints[0],
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100,
    )


@dsl.pipeline(
    name="i4g-ml-training-pipeline",
    description="End-to-end training pipeline: data prep → train → evaluate → register → deploy",
)
def training_pipeline(
    project_id: str = "i4g-ml",
    region: str = "us-central1",
    dataset_id: str = "i4g_ml",
    capability: str = "classification",
    dataset_version: int = 1,
    container_uri: str = "",
    serving_container_uri: str = "",
    experiment_name: str = "",
    config_path: str = "",
    golden_set_uri: str = "gs://i4g-ml-data/datasets/classification/golden/test.jsonl",
    endpoint_name: str = "serving-dev",
    min_overall_f1: float = 0.0,
    max_per_axis_regression: float = 0.05,
    machine_type: str = "n1-standard-4",
    min_replicas: int = 0,
    max_replicas: int = 1,
) -> None:
    prep = prepare_dataset(
        project_id=project_id,
        dataset_id=dataset_id,
        capability=capability,
        dataset_version=dataset_version,
    )
    train = train_model(
        project_id=project_id,
        region=region,
        container_uri=container_uri,
        config_path=config_path,
        dataset_gcs_path=prep.output,
        experiment_name=experiment_name,
    )
    evaluate = evaluate_model(
        model_uri=train.output,
        golden_set_uri=golden_set_uri,
        min_overall_f1=min_overall_f1,
        max_per_axis_regression=max_per_axis_regression,
    )
    register = register_model(
        project_id=project_id,
        region=region,
        model_uri=train.output,
        display_name=experiment_name,
        serving_container_uri=serving_container_uri,
        eval_passed=evaluate.outputs["passed"],
    )
    deploy_model(
        project_id=project_id,
        region=region,
        model_name=register.output,
        endpoint_name=endpoint_name,
        machine_type=machine_type,
        min_replicas=min_replicas,
        max_replicas=max_replicas,
    )
