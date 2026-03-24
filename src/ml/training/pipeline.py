"""KFP v2 training pipeline definitions.

Five-stage pipeline: prepare_dataset → train_model → evaluate_model →
register_model → deploy_model.

Note: ``from __future__ import annotations`` is intentionally omitted —
KFP v2 requires eager annotation evaluation to introspect component signatures.
"""

from typing import NamedTuple

from kfp import dsl

# All pipeline component dependencies (kfp, google-cloud-aiplatform,
# google-cloud-storage) are pre-installed in this image at compatible pinned
# versions.  Using a shared base image eliminates runtime pip installs and
# the protobuf version conflicts they cause.  Rebuild with:
#   scripts/build_image.sh pipeline-base dev
_PIPELINE_BASE_IMAGE = "us-central1-docker.pkg.dev/i4g-ml/containers/pipeline-base:dev"


@dsl.component(base_image=_PIPELINE_BASE_IMAGE)
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


@dsl.component(base_image=_PIPELINE_BASE_IMAGE)
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

    aiplatform.init(
        project=project_id,
        location=region,
        staging_bucket="gs://i4g-ml-vertex-pipelines-us-central1",
        experiment=experiment_name,
    )

    model_output = f"gs://i4g-ml-data/models/{experiment_name}/"
    job = aiplatform.CustomJob(
        display_name=f"train-{experiment_name}",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-highmem-4",
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": container_uri,
                    "args": [
                        "--config",
                        config_path,
                        "--dataset",
                        dataset_gcs_path,
                        "--experiment",
                        experiment_name,
                        "--output",
                        model_output,
                    ],
                },
            }
        ],
    )
    job.run(experiment=experiment_name)

    return f"gs://i4g-ml-data/models/{experiment_name}/"


@dsl.component(base_image=_PIPELINE_BASE_IMAGE)
def evaluate_model(
    model_uri: str,
    golden_set_uri: str,
    min_overall_f1: float,
    max_per_axis_regression: float,
) -> NamedTuple("EvalOutputs", [("passed", str), ("metrics_json", str)]):
    """Evaluate a trained model against the golden test set.

    Downloads model artifacts + golden test JSONL from GCS, runs inference
    on each sample, computes per-axis P/R/F1, and applies the eval gate.
    """
    import json
    import tempfile
    from pathlib import Path

    from google.cloud import storage as gcs

    # ── Download golden test set ─────────────────────────────────────────
    def _download_gcs_file(uri: str, dest: Path) -> None:
        """Download a single GCS object to a local path."""
        without_scheme = uri[5:]
        bucket_name, _, blob_path = without_scheme.partition("/")
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))

    def _download_gcs_prefix(uri: str, dest: Path) -> None:
        """Download all objects under a GCS prefix to a local directory."""
        without_scheme = uri[5:]
        bucket_name, _, prefix = without_scheme.partition("/")
        prefix = prefix.rstrip("/")
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            rel = blob.name[len(prefix) :].lstrip("/")
            if not rel:
                continue
            local_path = dest / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))

    # ── Load golden test data ────────────────────────────────────────────
    golden_path = Path(tempfile.mkdtemp()) / "test.jsonl"
    _download_gcs_file(golden_set_uri, golden_path)

    golden_data: list[dict] = []
    with open(golden_path) as f:
        for line in f:
            line = line.strip()
            if line:
                golden_data.append(json.loads(line))

    if not golden_data:
        metrics = {
            "overall_f1": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "per_axis": {},
            "eval_gate_passed": False,
            "error": "Empty golden test set",
        }
        return ("false", json.dumps(metrics))

    # ── Download and detect model ────────────────────────────────────────
    model_dir = Path(tempfile.mkdtemp())
    _download_gcs_prefix(model_uri.rstrip("/") + "/", model_dir)

    # Load label map
    label_map_path = model_dir / "label_map.json"
    if not label_map_path.exists():
        metrics = {
            "overall_f1": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "per_axis": {},
            "eval_gate_passed": False,
            "error": "label_map.json not found in model artifacts",
        }
        return ("false", json.dumps(metrics))

    with open(label_map_path) as f:
        label_map = json.load(f)

    # Detect model type and run inference
    is_xgboost = (model_dir / "xgboost_model.json").exists()
    is_pytorch = (model_dir / "model").is_dir() or (model_dir / "config.json").exists()

    predictions: list[dict[str, str]] = []
    ground_truth: list[dict[str, str]] = []

    if is_pytorch:
        # Install PyTorch + transformers at runtime in this lightweight component.
        # For KFP, this runs inside a container that has pip available.
        import subprocess

        subprocess.check_call(
            ["pip", "install", "--quiet", "torch", "transformers"],
            stdout=subprocess.DEVNULL,
        )

        import importlib

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch = importlib.import_module("torch")

        pt_model_dir = model_dir / "model" if (model_dir / "model").is_dir() else model_dir
        tokenizer = AutoTokenizer.from_pretrained(str(pt_model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(pt_model_dir))
        model.eval()

        for sample in golden_data:
            text = sample.get("text", "")
            gt_labels = sample.get("labels", {})
            ground_truth.append(gt_labels)

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            pred: dict[str, str] = {}
            offset = 0
            for axis, labels in label_map.items():
                n = len(labels)
                axis_probs = probs[offset : offset + n]
                best_idx = int(torch.argmax(axis_probs).item())
                pred[axis] = labels[best_idx]
                offset += n
            predictions.append(pred)

    elif is_xgboost:
        import subprocess

        subprocess.check_call(
            ["pip", "install", "--quiet", "xgboost", "numpy"],
            stdout=subprocess.DEVNULL,
        )
        import importlib

        xgb = importlib.import_module("xgboost")
        np = importlib.import_module("numpy")

        booster = xgb.Booster()
        booster.load_model(str(model_dir / "xgboost_model.json"))

        for sample in golden_data:
            gt_labels = sample.get("labels", {})
            ground_truth.append(gt_labels)

            features = sample.get("features", {})
            feature_keys = sorted(features.keys())
            values = [float(features.get(k, 0)) for k in feature_keys]
            dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)
            raw_pred = booster.predict(dmat)

            pred: dict[str, str] = {}
            offset = 0
            for axis, labels in label_map.items():
                n = len(labels)
                axis_probs = raw_pred[0][offset : offset + n] if raw_pred.ndim > 1 else raw_pred[offset : offset + n]
                best_idx = int(np.argmax(axis_probs))
                pred[axis] = labels[best_idx]
                offset += n
            predictions.append(pred)

    else:
        metrics = {
            "overall_f1": 0.0,
            "overall_precision": 0.0,
            "overall_recall": 0.0,
            "per_axis": {},
            "eval_gate_passed": False,
            "error": "Unknown model type in artifacts",
        }
        return ("false", json.dumps(metrics))

    # ── Compute metrics (inline — avoids importing ml package in KFP) ────
    all_axes: set[str] = set()
    for gt in ground_truth:
        all_axes.update(gt.keys())

    axis_tp: dict[str, int] = {a: 0 for a in all_axes}
    axis_fp: dict[str, int] = {a: 0 for a in all_axes}
    axis_fn: dict[str, int] = {a: 0 for a in all_axes}
    axis_support: dict[str, int] = {a: 0 for a in all_axes}

    for pred_item, gt_item in zip(predictions, ground_truth, strict=False):
        for axis in all_axes:
            gt_label = gt_item.get(axis)
            pred_label = pred_item.get(axis)
            if gt_label is not None:
                axis_support[axis] += 1
                if pred_label == gt_label:
                    axis_tp[axis] += 1
                else:
                    axis_fn[axis] += 1
                    if pred_label is not None:
                        axis_fp[axis] += 1
            elif pred_label is not None:
                axis_fp[axis] += 1

    per_axis: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    weights: list[int] = []

    for axis in sorted(all_axes):
        tp, fp, fn = axis_tp[axis], axis_fp[axis], axis_fn[axis]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        support = axis_support[axis]
        per_axis[axis] = {"precision": p, "recall": r, "f1": f1, "support": support}
        f1_scores.append(f1)
        precisions.append(p)
        recalls.append(r)
        weights.append(support)

    total_weight = sum(weights)
    if total_weight > 0:
        overall_f1 = sum(f * w for f, w in zip(f1_scores, weights, strict=False)) / total_weight
        overall_p = sum(p * w for p, w in zip(precisions, weights, strict=False)) / total_weight
        overall_r = sum(r * w for r, w in zip(recalls, weights, strict=False)) / total_weight
    else:
        overall_f1 = overall_p = overall_r = 0.0

    # ── Eval gate ────────────────────────────────────────────────────────
    gate_passed = overall_f1 >= min_overall_f1

    metrics = {
        "overall_f1": overall_f1,
        "overall_precision": overall_p,
        "overall_recall": overall_r,
        "per_axis": per_axis,
        "total_samples": len(predictions),
        "eval_gate_passed": gate_passed,
    }

    passed = "true" if gate_passed else "false"
    return (passed, json.dumps(metrics))


@dsl.component(base_image=_PIPELINE_BASE_IMAGE)
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


@dsl.component(base_image=_PIPELINE_BASE_IMAGE)
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
