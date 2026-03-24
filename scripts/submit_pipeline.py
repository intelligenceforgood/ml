"""Submit the training pipeline to Vertex AI.

Usage:
    conda run -n ml python scripts/submit_pipeline.py
"""

from __future__ import annotations

import logging

from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "i4g-ml"
REGION = "us-central1"
SA = "sa-ml-platform@i4g-ml.iam.gserviceaccount.com"
REGISTRY = "us-central1-docker.pkg.dev/i4g-ml/containers"


def main() -> None:
    """Compile and submit the KFP training pipeline to Vertex AI."""
    aiplatform.init(project=PROJECT, location=REGION)

    job = aiplatform.PipelineJob(
        display_name="classification-opt125m-v1",
        template_path="pipeline.yaml",
        parameter_values={
            "project_id": PROJECT,
            "region": REGION,
            "dataset_id": "i4g_ml",
            "capability": "classification",
            "dataset_version": 1,
            "container_uri": f"{REGISTRY}/train-pytorch:dev",
            "serving_container_uri": f"{REGISTRY}/serve:dev",
            "experiment_name": "classification-opt125m-v1",
            "config_path": "gs://i4g-ml-data/configs/classification_opt125m.yaml",
            "golden_set_uri": "gs://i4g-ml-data/datasets/classification/golden/test.jsonl",
            "endpoint_name": "serving-dev",
            "min_overall_f1": 0.0,
            "max_per_axis_regression": 0.05,
            "machine_type": "n1-standard-4",
            "min_replicas": 0,
            "max_replicas": 1,
        },
        enable_caching=False,
    )

    job.submit(service_account=SA)
    logger.info("Pipeline submitted: %s", job.resource_name)


if __name__ == "__main__":
    main()
