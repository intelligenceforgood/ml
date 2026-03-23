"""Prediction inference logic.

Supports two model types detected from the artifact directory:
- **PyTorch** — HuggingFace Transformers model (``model/`` subdir)
- **XGBoost** — serialized booster (``xgboost_model.json``)

Model artifacts must include a ``label_map.json`` mapping integer indices
to taxonomy label codes, grouped by axis.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_STATE: dict[str, Any] = {}

# Sentinel for "model load was attempted but failed"
_LOAD_FAILED = False


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _download_artifacts(artifact_uri: str, dest: Path) -> None:
    """Download model artifacts from GCS to a local directory."""
    from google.cloud import storage as gcs

    if not artifact_uri.startswith("gs://"):
        raise ValueError(f"artifact_uri must be a gs:// path, got: {artifact_uri}")

    # Parse gs://bucket/prefix
    without_scheme = artifact_uri[5:]
    bucket_name, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/")

    client = gcs.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No artifacts found at {artifact_uri}")

    for blob in blobs:
        rel = blob.name[len(prefix) :].lstrip("/")
        if not rel:
            continue
        local_path = dest / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    logger.info("Downloaded %d artifact files from %s", len(blobs), artifact_uri)


def _detect_model_type(artifact_dir: Path) -> str:
    """Detect whether artifacts are PyTorch or XGBoost."""
    if (artifact_dir / "xgboost_model.json").exists():
        return "xgboost"
    if (artifact_dir / "model").is_dir() or (artifact_dir / "config.json").exists():
        return "pytorch"
    raise ValueError(f"Cannot detect model type from artifacts in {artifact_dir}")


def _load_label_map(artifact_dir: Path) -> dict[str, list[str]]:
    """Load label_map.json → ``{axis: [label_code, ...]}``."""
    path = artifact_dir / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"label_map.json not found in {artifact_dir}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# PyTorch loading + inference
# ---------------------------------------------------------------------------


def _load_pytorch(artifact_dir: Path) -> None:
    """Load a HuggingFace Transformers model + tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = artifact_dir / "model" if (artifact_dir / "model").is_dir() else artifact_dir

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    _MODEL_STATE["tokenizer"] = tokenizer
    _MODEL_STATE["model"] = model
    _MODEL_STATE["framework"] = "pytorch"
    logger.info("Loaded PyTorch model from %s", model_dir)


def _predict_pytorch(text: str) -> dict[str, dict[str, Any]]:
    """Run PyTorch inference: tokenize → forward → softmax → per-axis labels."""
    import torch

    tokenizer = _MODEL_STATE["tokenizer"]
    model = _MODEL_STATE["model"]
    label_map: dict[str, list[str]] = _MODEL_STATE["label_map"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits  # shape: (1, num_labels)

    probs = torch.softmax(logits, dim=-1).squeeze(0)

    # Distribute logits across axes based on label_map ordering.
    # label_map values are concatenated in axis order → contiguous index ranges.
    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for axis, labels in label_map.items():
        n = len(labels)
        axis_probs = probs[offset : offset + n]
        best_idx = int(torch.argmax(axis_probs).item())
        result[axis] = {
            "code": labels[best_idx],
            "confidence": round(float(axis_probs[best_idx].item()), 4),
        }
        offset += n

    return result


# ---------------------------------------------------------------------------
# XGBoost loading + inference
# ---------------------------------------------------------------------------


def _load_xgboost(artifact_dir: Path) -> None:
    """Load an XGBoost booster from JSON."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "xgboost_model.json"))

    _MODEL_STATE["booster"] = booster
    _MODEL_STATE["framework"] = "xgboost"
    logger.info("Loaded XGBoost model from %s", artifact_dir)


def _predict_xgboost(text: str, features: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Run XGBoost inference on tabular features extracted from text."""
    import numpy as np
    import xgboost as xgb

    from ml.serving.features import compute_inline_features

    feat = features if features else compute_inline_features(text)
    label_map: dict[str, list[str]] = _MODEL_STATE["label_map"]

    # Build feature vector in a stable order
    feature_keys = sorted(feat.keys())
    values = [float(feat.get(k, 0)) for k in feature_keys]
    dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)

    raw_pred = _MODEL_STATE["booster"].predict(dmat)  # shape depends on objective

    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for axis, labels in label_map.items():
        n = len(labels)
        axis_probs = raw_pred[0][offset : offset + n] if raw_pred.ndim > 1 else raw_pred[offset : offset + n]
        best_idx = int(np.argmax(axis_probs))
        result[axis] = {
            "code": labels[best_idx],
            "confidence": round(float(axis_probs[best_idx]), 4),
        }
        offset += n

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(artifact_uri: str) -> None:
    """Load model artifacts from the given GCS URI into memory.

    Detects model type (PyTorch or XGBoost) automatically from artifact
    contents. Sets ``_LOAD_FAILED`` on error so the server can serve 503.
    """
    global _LOAD_FAILED

    _MODEL_STATE["artifact_uri"] = artifact_uri
    _MODEL_STATE["model_id"] = artifact_uri.rstrip("/").split("/")[-2] if "/" in artifact_uri else "unknown"

    # If the URI is empty, stay in stub mode (no model to load)
    if not artifact_uri:
        _MODEL_STATE["version"] = 0
        _MODEL_STATE["stage"] = "stub"
        return

    try:
        # Download artifacts to a temp directory
        dest = Path(tempfile.mkdtemp(prefix="ml_model_"))
        _download_artifacts(artifact_uri, dest)

        # Load label map
        _MODEL_STATE["label_map"] = _load_label_map(dest)

        # Detect and load model
        model_type = _detect_model_type(dest)
        if model_type == "pytorch":
            _load_pytorch(dest)
        else:
            _load_xgboost(dest)

        # Parse version from URI path (e.g. gs://bucket/models/name/v3/ → 3)
        parts = artifact_uri.rstrip("/").split("/")
        version_str = next((p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()), "v1")
        _MODEL_STATE["version"] = int(version_str[1:])
        _MODEL_STATE["stage"] = "candidate"
        _LOAD_FAILED = False
        logger.info("Model loaded successfully: type=%s, uri=%s", model_type, artifact_uri)

    except Exception:
        _LOAD_FAILED = True
        _MODEL_STATE["version"] = 0
        _MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load model from %s", artifact_uri)


def is_model_ready() -> bool:
    """Return True if a model was loaded and is ready for inference."""
    return not _LOAD_FAILED and _MODEL_STATE.get("framework") is not None


def classify_text(
    text: str,
    case_id: str,
    *,
    features: dict | None = None,
) -> dict[str, Any]:
    """Run classification on input text. Returns prediction dict.

    Falls back to stub predictions if no model is loaded.
    Raises ``RuntimeError`` if model load was attempted but failed.
    """
    start = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    framework = _MODEL_STATE.get("framework")

    if _LOAD_FAILED:
        raise RuntimeError("Model failed to load — serving unavailable")

    if framework == "pytorch":
        prediction = _predict_pytorch(text)
    elif framework == "xgboost":
        prediction = _predict_xgboost(text, features)
    else:
        # No model loaded — stub mode
        prediction = {
            "INTENT": {"code": "INTENT.UNKNOWN", "confidence": 0.5},
            "CHANNEL": {"code": "CHANNEL.UNKNOWN", "confidence": 0.5},
        }

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "prediction_id": prediction_id,
        "prediction": prediction,
        "risk_score": None,
        "model_info": {
            "model_id": _MODEL_STATE.get("model_id", "stub"),
            "version": _MODEL_STATE.get("version", 0),
            "stage": _MODEL_STATE.get("stage", "experimental"),
        },
        "latency_ms": elapsed_ms,
    }
