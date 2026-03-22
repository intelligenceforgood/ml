"""XGBoost tabular feature classification training container.

Entry point for training XGBoost models on tabular features from BigQuery.
1. Loads training config from GCS
2. Loads tabular features from BigQuery
3. Trains XGBoost classifier
4. Logs metrics to Vertex AI Experiments
5. Uploads model artifacts to GCS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from google.cloud import aiplatform, storage
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost training."""
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--config", required=True, help="GCS path to training config YAML")
    parser.add_argument("--dataset", required=True, help="GCS path to dataset directory")
    parser.add_argument("--experiment", required=True, help="Vertex AI Experiment name")
    parser.add_argument("--output", default=None, help="GCS path for model artifacts")
    return parser.parse_args()


def download_from_gcs(gcs_path: str, local_path: str) -> str:
    """Download a GCS object to a local path and return it."""
    client = storage.Client()
    parts = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    blob = bucket.blob(parts[1])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    return local_path


def load_config(gcs_path: str) -> dict[str, Any]:
    """Load training config YAML from GCS."""
    local = download_from_gcs(gcs_path, "/tmp/training_config.yaml")
    with open(local) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load records from a local JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_features(records: list[dict], feature_cols: list[str]) -> pd.DataFrame:
    """Extract tabular features into a DataFrame."""
    rows = []
    for r in records:
        features = r.get("features", {})
        if isinstance(features, str):
            features = json.loads(features)
        row = {col: features.get(col, 0) for col in feature_cols}
        rows.append(row)
    return pd.DataFrame(rows)


def train(config: dict, train_data: list[dict], eval_data: list[dict]) -> Path:
    """Train XGBoost classifier on tabular features."""
    feature_cols = [
        "text_length",
        "word_count",
        "entity_count",
        "has_crypto_wallet",
        "has_bank_account",
        "has_phone",
        "has_email",
        "classification_axis_count",
        "current_classification_conf",
    ]

    X_train = prepare_features(train_data, feature_cols)
    X_eval = prepare_features(eval_data, feature_cols)

    # Use the first axis for labels
    label_schema = config.get("label_schema", {})
    first_axis = next(iter(label_schema)) if label_schema else "INTENT"

    y_train_raw = [r.get("labels", {}).get(first_axis, "UNKNOWN") for r in train_data]
    y_eval_raw = [r.get("labels", {}).get(first_axis, "UNKNOWN") for r in eval_data]

    le = LabelEncoder()
    le.fit(y_train_raw + y_eval_raw)
    y_train = le.transform(y_train_raw)
    y_eval = le.transform(y_eval_raw)

    hyperparams = config.get("hyperparameters", {})
    params = {
        "objective": "multi:softprob",
        "num_class": len(le.classes_),
        "max_depth": hyperparams.get("max_depth", 6),
        "learning_rate": hyperparams.get("learning_rate", 0.1),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_cols)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=hyperparams.get("num_boost_round", 100),
        evals=[(deval, "eval")],
        early_stopping_rounds=10,
        verbose_eval=10,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(deval), axis=1)
    f1 = f1_score(y_eval, y_pred, average="weighted")
    logger.info("Eval weighted F1: %.4f", f1)
    logger.info("\n%s", classification_report(y_eval, y_pred, target_names=le.classes_))

    # Save
    output_dir = Path("/tmp/model_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "model.json"))
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"weighted_f1": float(f1)}, f)

    return output_dir


def upload_artifacts(local_dir: Path, gcs_output: str) -> None:
    """Upload model artifacts from a local directory to GCS."""
    client = storage.Client()
    parts = gcs_output.replace("gs://", "").split("/", 1)
    bucket = client.bucket(parts[0])
    prefix = parts[1].rstrip("/")

    for local_file in local_dir.rglob("*"):
        if local_file.is_file():
            blob_path = f"{prefix}/{local_file.relative_to(local_dir)}"
            bucket.blob(blob_path).upload_from_filename(str(local_file))
            logger.info("Uploaded %s", blob_path)


def main() -> None:
    """XGBoost training container entry point."""
    args = parse_args()

    project_id = os.environ.get("CLOUD_ML_PROJECT_ID", "i4g-ml")
    aiplatform.init(project=project_id, location="us-central1")

    config = load_config(args.config)
    logger.info("Config loaded: %s", config.get("model_id", "unknown"))

    # Load JSONL splits
    for split in ("train", "eval", "test"):
        download_from_gcs(f"{args.dataset}/{split}.jsonl", f"/tmp/data/{split}.jsonl")

    train_data = load_jsonl("/tmp/data/train.jsonl")
    eval_data = load_jsonl("/tmp/data/eval.jsonl")
    logger.info("Loaded %d train, %d eval records", len(train_data), len(eval_data))

    model_dir = train(config, train_data, eval_data)

    output = args.output or f"gs://i4g-ml-data/models/{args.experiment}/"
    upload_artifacts(model_dir, output)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
