"""XGBoost tabular feature classification training container.

Entry point for training XGBoost models on tabular features.
1. Loads training config from a local path or GCS
2. Loads JSONL dataset splits (train/eval/test)
3. Trains XGBoost classifier
4. Logs metrics to Vertex AI Experiments (when running on GCP)
5. Saves model artifacts locally or uploads to GCS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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


def _is_gcs(path: str) -> bool:
    """Return True if the path is a GCS URI."""
    return path.startswith("gs://")


def _resolve_path(path: str, local_dest: str) -> str:
    """Download from GCS or return a local path as-is."""
    if _is_gcs(path):
        return download_from_gcs(path, local_dest)
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XGBoost training."""
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--config", required=True, help="Path to training config YAML (local or gs://)")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory (local or gs://)")
    parser.add_argument("--experiment", default="local", help="Vertex AI Experiment name")
    parser.add_argument("--output", default=None, help="Path for model artifacts (local or gs://)")
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


def load_config(path: str) -> dict[str, Any]:
    """Load training config YAML from a local path or GCS."""
    local = _resolve_path(path, "/tmp/training_config.yaml")
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
    df = pd.DataFrame(rows)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0)


def train(config: dict, train_data: list[dict], eval_data: list[dict]) -> Path:
    """Train XGBoost model on tabular features.

    Dispatches between classifier (multi:softprob) and regressor (reg:squarederror)
    based on ``config["capability"]``.
    """
    capability = config.get("capability", "classification")

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

    hyperparams = config.get("hyperparameters", {})

    if capability == "risk_scoring":
        return _train_regressor(config, X_train, X_eval, train_data, eval_data, feature_cols, hyperparams)
    return _train_classifier(config, X_train, X_eval, train_data, eval_data, feature_cols, hyperparams)


def _train_classifier(
    config: dict,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    train_data: list[dict],
    eval_data: list[dict],
    feature_cols: list[str],
    hyperparams: dict,
) -> Path:
    """Train an XGBoost multi-class classifier."""
    label_schema = config.get("label_schema", {})
    first_axis = next(iter(label_schema)) if label_schema else "INTENT"

    y_train_raw = [r.get("labels", {}).get(first_axis, "UNKNOWN") for r in train_data]
    y_eval_raw = [r.get("labels", {}).get(first_axis, "UNKNOWN") for r in eval_data]

    le = LabelEncoder()
    le.fit(y_train_raw + y_eval_raw)
    y_train = le.transform(y_train_raw)
    y_eval = le.transform(y_eval_raw)

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

    y_pred = np.argmax(model.predict(deval), axis=1)
    f1 = f1_score(y_eval, y_pred, average="weighted")
    logger.info("Eval weighted F1: %.4f", f1)
    logger.info(
        "\n%s",
        classification_report(
            y_eval, y_pred, labels=list(range(len(le.classes_))), target_names=le.classes_, zero_division=0
        ),
    )

    output_dir = Path("/tmp/model_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "xgboost_model.json"))
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({first_axis: le.classes_.tolist()}, f)
    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"weighted_f1": float(f1)}, f)

    return output_dir


def _train_regressor(
    config: dict,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    train_data: list[dict],
    eval_data: list[dict],
    feature_cols: list[str],
    hyperparams: dict,
) -> Path:
    """Train an XGBoost regressor for risk scoring."""
    from scipy.stats import spearmanr

    label_key = config.get("label_key", "severity")
    y_train = np.array([float(r.get("labels", {}).get(label_key, 0.0)) for r in train_data])
    y_eval = np.array([float(r.get("labels", {}).get(label_key, 0.0)) for r in eval_data])

    params = {
        "objective": "reg:squarederror",
        "max_depth": hyperparams.get("max_depth", 6),
        "learning_rate": hyperparams.get("learning_rate", 0.05),
        "eval_metric": "rmse",
        "tree_method": "hist",
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_cols)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=hyperparams.get("num_boost_round", 200),
        evals=[(deval, "eval")],
        early_stopping_rounds=15,
        verbose_eval=10,
    )

    y_pred = model.predict(deval)
    mse = float(np.mean((y_pred - y_eval) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_eval)))
    rho, _ = spearmanr(y_pred, y_eval) if len(y_eval) >= 2 else (0.0, 1.0)
    logger.info("Eval MSE: %.4f  MAE: %.4f  Spearman: %.4f", mse, mae, rho)

    output_dir = Path("/tmp/model_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_dir / "xgboost_model.json"))
    with open(output_dir / "label_map.json", "w") as f:
        json.dump({}, f)
    with open(output_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"mse": mse, "mae": mae, "spearman_rho": float(rho)}, f)

    return output_dir


def save_artifacts(local_dir: Path, output: str) -> None:
    """Save model artifacts to a local directory or upload to GCS."""
    if _is_gcs(output):
        client = storage.Client()
        parts = output.replace("gs://", "").split("/", 1)
        bucket = client.bucket(parts[0])
        prefix = parts[1].rstrip("/")

        for local_file in local_dir.rglob("*"):
            if local_file.is_file():
                blob_path = f"{prefix}/{local_file.relative_to(local_dir)}"
                bucket.blob(blob_path).upload_from_filename(str(local_file))
                logger.info("Uploaded %s", blob_path)
    else:
        import shutil

        dest = Path(output)
        dest.mkdir(parents=True, exist_ok=True)
        for src_file in local_dir.rglob("*"):
            if src_file.is_file():
                target = dest / src_file.relative_to(local_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, target)
                logger.info("Saved %s", target)


def main() -> None:
    """XGBoost training container entry point."""
    args = parse_args()

    is_local = not _is_gcs(args.config)
    if not is_local:
        project_id = os.environ.get("CLOUD_ML_PROJECT_ID", "i4g-ml")
        aiplatform.init(project=project_id, location="us-central1")

    config = load_config(args.config)
    logger.info("Config loaded: %s", config.get("model_id", "unknown"))

    # Load JSONL splits
    for split in ("train", "eval", "test"):
        _resolve_path(f"{args.dataset}/{split}.jsonl", f"/tmp/data/{split}.jsonl")

    data_dir = args.dataset if is_local else "/tmp/data"
    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    eval_data = load_jsonl(f"{data_dir}/eval.jsonl")
    logger.info("Loaded %d train, %d eval records", len(train_data), len(eval_data))

    model_dir = train(config, train_data, eval_data)

    output = args.output or f"gs://i4g-ml-data/models/{args.experiment}/"
    save_artifacts(model_dir, output)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
