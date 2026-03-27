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
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_STATE: dict[str, Any] = {}
_SHADOW_MODEL_STATE: dict[str, Any] = {}
_CHALLENGER_MODEL_STATE: dict[str, Any] = {}
_RISK_MODEL_STATE: dict[str, Any] = {}

# Sentinel for "model load was attempted but failed"
_LOAD_FAILED = False
_SHADOW_LOAD_FAILED = False
_CHALLENGER_LOAD_FAILED = False
_RISK_LOAD_FAILED = False


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

    # Use training feature order if available, else fall back to sorted keys
    feature_keys = _MODEL_STATE.get("feature_cols") or sorted(feat.keys())
    values = [float(feat.get(k) or 0) for k in feature_keys]
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
    parts = artifact_uri.rstrip("/").split("/")
    last = parts[-1] if parts else "unknown"
    # Two-level path (models/name/vN) → use name; single-level (models/name) → use last
    if last.startswith("v") and last[1:].isdigit() and len(parts) >= 2:
        _MODEL_STATE["model_id"] = parts[-2]
    else:
        _MODEL_STATE["model_id"] = last if last else "unknown"

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
            # Load feature column order used during training
            feature_cols_path = dest / "feature_cols.json"
            if feature_cols_path.exists():
                with open(feature_cols_path) as f:
                    _MODEL_STATE["feature_cols"] = json.load(f)

        # Parse version from URI path (e.g. gs://bucket/models/name/v3/ → 3)
        parts = artifact_uri.rstrip("/").split("/")
        version_str = next((p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()), "v1")
        _MODEL_STATE["version"] = int(version_str[1:])
        _MODEL_STATE["stage"] = "candidate"
        _LOAD_FAILED = False
        logger.info("Model loaded successfully: type=%s, uri=%s", model_type, artifact_uri)

    except Exception:  # noqa: BLE001 — catch-all for model loading (GCS, pickle, etc.)
        _LOAD_FAILED = True
        _MODEL_STATE["version"] = 0
        _MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load model from %s", artifact_uri)


def is_model_ready() -> bool:
    """Return True if a model was loaded and is ready for inference."""
    return not _LOAD_FAILED and _MODEL_STATE.get("framework") is not None


def load_shadow_model(artifact_uri: str) -> None:
    """Load shadow model artifacts for A/B comparison.

    The shadow model runs inference asynchronously alongside the champion but
    its results never affect the returned response.  If loading fails or
    memory exceeds the safety threshold the shadow is silently disabled.
    """
    global _SHADOW_LOAD_FAILED

    if not artifact_uri:
        return

    _SHADOW_MODEL_STATE["artifact_uri"] = artifact_uri
    parts = artifact_uri.rstrip("/").split("/")
    last = parts[-1] if parts else "unknown"
    if last.startswith("v") and last[1:].isdigit() and len(parts) >= 2:
        _SHADOW_MODEL_STATE["model_id"] = parts[-2]
    else:
        _SHADOW_MODEL_STATE["model_id"] = last if last else "unknown"

    try:
        dest = Path(tempfile.mkdtemp(prefix="ml_shadow_"))
        _download_artifacts(artifact_uri, dest)

        _SHADOW_MODEL_STATE["label_map"] = _load_label_map(dest)

        model_type = _detect_model_type(dest)
        if model_type == "pytorch":
            _load_shadow_pytorch(dest)
        else:
            _load_shadow_xgboost(dest)
            feature_cols_path = dest / "feature_cols.json"
            if feature_cols_path.exists():
                with open(feature_cols_path) as f:
                    _SHADOW_MODEL_STATE["feature_cols"] = json.load(f)

        parts = artifact_uri.rstrip("/").split("/")
        version_str = next(
            (p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()),
            "v1",
        )
        _SHADOW_MODEL_STATE["version"] = int(version_str[1:])
        _SHADOW_MODEL_STATE["stage"] = "shadow"
        _SHADOW_LOAD_FAILED = False

        # Memory guard — check RSS after loading both models
        _check_memory_guard()

        logger.info("Shadow model loaded successfully: type=%s, uri=%s", model_type, artifact_uri)

    except Exception:  # noqa: BLE001 — catch-all for model loading
        _SHADOW_LOAD_FAILED = True
        _SHADOW_MODEL_STATE["version"] = 0
        _SHADOW_MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load shadow model from %s", artifact_uri)


def _check_memory_guard() -> None:
    """Log memory usage after loading models; disable shadow if RSS > 80%."""
    global _SHADOW_LOAD_FAILED

    try:
        import resource

        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports KB
        import sys

        if sys.platform == "darwin":
            rss_mb = rss_bytes / (1024 * 1024)
        else:
            rss_mb = rss_bytes / 1024

        # Cloud Run instance memory from env (e.g. "2Gi" → 2048 MB)
        instance_memory_str = os.environ.get("CLOUD_RUN_MEMORY_LIMIT", "")
        instance_memory_mb = _parse_memory_limit(instance_memory_str)

        logger.info("Memory after model loading: RSS=%.0fMB, limit=%sMB", rss_mb, instance_memory_mb or "unknown")

        if instance_memory_mb and rss_mb > instance_memory_mb * 0.8:
            logger.warning(
                "RSS (%.0fMB) exceeds 80%% of instance memory (%dMB) — disabling shadow model",
                rss_mb,
                instance_memory_mb,
            )
            _SHADOW_LOAD_FAILED = True
            _SHADOW_MODEL_STATE.clear()
    except Exception:  # noqa: BLE001 — best-effort memory check
        logger.debug("Could not check memory usage", exc_info=True)


def _parse_memory_limit(limit_str: str) -> int | None:
    """Parse memory limit string like '2Gi' or '512Mi' to MB."""
    if not limit_str:
        return None
    limit_str = limit_str.strip()
    if limit_str.endswith("Gi"):
        return int(float(limit_str[:-2]) * 1024)
    if limit_str.endswith("Mi"):
        return int(float(limit_str[:-2]))
    if limit_str.endswith("G"):
        return int(float(limit_str[:-1]) * 1000)
    if limit_str.endswith("M"):
        return int(float(limit_str[:-1]))
    return None


def is_shadow_ready() -> bool:
    """Return True if a shadow model was loaded and is ready."""
    return not _SHADOW_LOAD_FAILED and _SHADOW_MODEL_STATE.get("framework") is not None


# ---------------------------------------------------------------------------
# Shadow model loading helpers
# ---------------------------------------------------------------------------


def _load_shadow_pytorch(artifact_dir: Path) -> None:
    """Load a HuggingFace Transformers model into the shadow state."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = artifact_dir / "model" if (artifact_dir / "model").is_dir() else artifact_dir
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    _SHADOW_MODEL_STATE["tokenizer"] = tokenizer
    _SHADOW_MODEL_STATE["model"] = model
    _SHADOW_MODEL_STATE["framework"] = "pytorch"
    logger.info("Loaded shadow PyTorch model from %s", model_dir)


def _load_shadow_xgboost(artifact_dir: Path) -> None:
    """Load an XGBoost booster into the shadow state."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(artifact_dir / "xgboost_model.json"))

    _SHADOW_MODEL_STATE["booster"] = booster
    _SHADOW_MODEL_STATE["framework"] = "xgboost"
    logger.info("Loaded shadow XGBoost model from %s", artifact_dir)


def _fetch_features_with_fallback(text: str, case_id: str) -> dict[str, Any]:
    """Try Feature Store first, fall back to inline feature computation.

    When ``FEATURE_STORE_ID`` is set, attempts a sub-100ms online read from
    Vertex AI Feature Store via ``fetch_online_features()``.  On miss or
    error, falls back to ``compute_inline_features()``.
    """
    if os.environ.get("FEATURE_STORE_ID"):
        from ml.data.feature_store import fetch_online_features

        fs_features = fetch_online_features(case_id)
        if fs_features is not None:
            return fs_features

    from ml.serving.features import compute_inline_features

    return compute_inline_features(text)


def classify_text(
    text: str,
    case_id: str,
    *,
    features: dict | None = None,
    variant: str = "champion",
) -> dict[str, Any]:
    """Run classification on input text. Returns prediction dict.

    Falls back to stub predictions if no model is loaded.
    Raises ``RuntimeError`` if model load was attempted but failed.

    Args:
        text: Input text to classify.
        case_id: Case identifier.
        features: Pre-computed features (optional).
        variant: Which model to use: "champion" or "challenger".
    """
    start = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    # Select model state based on variant
    if variant == "challenger" and is_challenger_ready():
        model_state = _CHALLENGER_MODEL_STATE
    else:
        model_state = _MODEL_STATE
        variant = "champion"

    framework = model_state.get("framework")

    if variant == "champion" and _LOAD_FAILED:
        raise RuntimeError("Model failed to load — serving unavailable")
    if variant == "challenger" and _CHALLENGER_LOAD_FAILED:
        raise RuntimeError("Challenger model failed to load")

    # Enrich features with Feature Store when configured (Sprint 3.3)
    if features is None:
        features = _fetch_features_with_fallback(text, case_id)

    if framework == "pytorch":
        prediction = _predict_with_state(model_state, text, features, framework="pytorch")
    elif framework == "xgboost":
        prediction = _predict_with_state(model_state, text, features, framework="xgboost")
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
            "model_id": model_state.get("model_id", "stub"),
            "version": model_state.get("version", 0),
            "stage": model_state.get("stage", "experimental"),
        },
        "variant": variant,
        "latency_ms": elapsed_ms,
    }


def _predict_with_state(
    state: dict[str, Any],
    text: str,
    features: dict[str, Any] | None,
    *,
    framework: str,
) -> dict[str, dict[str, Any]]:
    """Run inference using the given model state dict."""
    if framework == "pytorch":
        import torch

        tokenizer = state["tokenizer"]
        model = state["model"]
        label_map: dict[str, list[str]] = state["label_map"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0)

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
    else:
        import numpy as np
        import xgboost as xgb

        from ml.serving.features import compute_inline_features

        feat = features if features else compute_inline_features(text)
        label_map_xgb: dict[str, list[str]] = state["label_map"]

        feature_keys = state.get("feature_cols") or sorted(feat.keys())
        values = [float(feat.get(k) or 0) for k in feature_keys]
        dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)

        raw_pred = state["booster"].predict(dmat)

        result_xgb: dict[str, dict[str, Any]] = {}
        offset = 0
        for axis, labels in label_map_xgb.items():
            n = len(labels)
            axis_probs = raw_pred[0][offset : offset + n] if raw_pred.ndim > 1 else raw_pred[offset : offset + n]
            best_idx = int(np.argmax(axis_probs))
            result_xgb[axis] = {
                "code": labels[best_idx],
                "confidence": round(float(axis_probs[best_idx]), 4),
            }
            offset += n
        return result_xgb


def classify_text_shadow(
    text: str,
    case_id: str,
    champion_prediction_id: str,
    *,
    features: dict | None = None,
) -> dict[str, Any] | None:
    """Run shadow model inference.  Returns None if shadow is unavailable."""
    if not is_shadow_ready():
        return None

    start = time.perf_counter()
    prediction_id = f"{champion_prediction_id}-shadow"

    framework = _SHADOW_MODEL_STATE.get("framework")

    if framework == "pytorch":
        prediction = _predict_shadow_pytorch(text)
    elif framework == "xgboost":
        prediction = _predict_shadow_xgboost(text, features)
    else:
        return None

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "prediction_id": prediction_id,
        "prediction": prediction,
        "risk_score": None,
        "model_info": {
            "model_id": _SHADOW_MODEL_STATE.get("model_id", "shadow"),
            "version": _SHADOW_MODEL_STATE.get("version", 0),
            "stage": "shadow",
        },
        "latency_ms": elapsed_ms,
        "is_shadow": True,
    }


def _predict_shadow_pytorch(text: str) -> dict[str, dict[str, Any]]:
    """Run PyTorch inference using the shadow model state."""
    import torch

    tokenizer = _SHADOW_MODEL_STATE["tokenizer"]
    model = _SHADOW_MODEL_STATE["model"]
    label_map: dict[str, list[str]] = _SHADOW_MODEL_STATE["label_map"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze(0)

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


def _predict_shadow_xgboost(text: str, features: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Run XGBoost inference using the shadow model state."""
    import numpy as np
    import xgboost as xgb

    from ml.serving.features import compute_inline_features

    feat = features if features else compute_inline_features(text)
    label_map: dict[str, list[str]] = _SHADOW_MODEL_STATE["label_map"]

    feature_keys = _SHADOW_MODEL_STATE.get("feature_cols") or sorted(feat.keys())
    values = [float(feat.get(k) or 0) for k in feature_keys]
    dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)

    raw_pred = _SHADOW_MODEL_STATE["booster"].predict(dmat)

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
# Challenger model loading
# ---------------------------------------------------------------------------


def load_challenger_model(artifact_uri: str) -> None:
    """Load challenger model artifacts for A/B routing.

    The challenger model receives a fraction of live traffic and its results
    ARE returned to users (unlike shadow, which is silent).
    """
    global _CHALLENGER_LOAD_FAILED

    if not artifact_uri:
        return

    _CHALLENGER_MODEL_STATE["artifact_uri"] = artifact_uri
    parts = artifact_uri.rstrip("/").split("/")
    last = parts[-1] if parts else "unknown"
    if last.startswith("v") and last[1:].isdigit() and len(parts) >= 2:
        _CHALLENGER_MODEL_STATE["model_id"] = parts[-2]
    else:
        _CHALLENGER_MODEL_STATE["model_id"] = last if last else "unknown"

    try:
        dest = Path(tempfile.mkdtemp(prefix="ml_challenger_"))
        _download_artifacts(artifact_uri, dest)

        _CHALLENGER_MODEL_STATE["label_map"] = _load_label_map(dest)

        model_type = _detect_model_type(dest)
        if model_type == "pytorch":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_dir = dest / "model" if (dest / "model").is_dir() else dest
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
            model.eval()
            _CHALLENGER_MODEL_STATE["tokenizer"] = tokenizer
            _CHALLENGER_MODEL_STATE["model"] = model
            _CHALLENGER_MODEL_STATE["framework"] = "pytorch"
        else:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(dest / "xgboost_model.json"))
            _CHALLENGER_MODEL_STATE["booster"] = booster
            _CHALLENGER_MODEL_STATE["framework"] = "xgboost"
            feature_cols_path = dest / "feature_cols.json"
            if feature_cols_path.exists():
                with open(feature_cols_path) as f:
                    _CHALLENGER_MODEL_STATE["feature_cols"] = json.load(f)

        parts = artifact_uri.rstrip("/").split("/")
        version_str = next(
            (p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()),
            "v1",
        )
        _CHALLENGER_MODEL_STATE["version"] = int(version_str[1:])
        _CHALLENGER_MODEL_STATE["stage"] = "challenger"
        _CHALLENGER_LOAD_FAILED = False
        _check_memory_guard()
        logger.info("Challenger model loaded: type=%s, uri=%s", model_type, artifact_uri)

    except Exception:  # noqa: BLE001 — catch-all for model loading
        _CHALLENGER_LOAD_FAILED = True
        _CHALLENGER_MODEL_STATE["version"] = 0
        _CHALLENGER_MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load challenger model from %s", artifact_uri)


def is_challenger_ready() -> bool:
    """Return True if a challenger model is loaded and ready."""
    return not _CHALLENGER_LOAD_FAILED and _CHALLENGER_MODEL_STATE.get("framework") is not None


# ---------------------------------------------------------------------------
# Risk Scoring Model (Sprint 4)
# ---------------------------------------------------------------------------


def load_risk_model(artifact_uri: str) -> None:
    """Load risk scoring model artifacts from GCS.

    Risk scoring uses XGBoost regression to produce a continuous 0.0–1.0 score.
    """
    global _RISK_LOAD_FAILED

    if not artifact_uri:
        return

    _RISK_MODEL_STATE["artifact_uri"] = artifact_uri
    parts = artifact_uri.rstrip("/").split("/")
    last = parts[-1] if parts else "risk-unknown"
    if last.startswith("v") and last[1:].isdigit() and len(parts) >= 2:
        _RISK_MODEL_STATE["model_id"] = parts[-2]
    else:
        _RISK_MODEL_STATE["model_id"] = last if last else "risk-unknown"

    try:
        dest = Path(tempfile.mkdtemp(prefix="ml_risk_"))
        _download_artifacts(artifact_uri, dest)

        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(dest / "xgboost_model.json"))

        _RISK_MODEL_STATE["booster"] = booster
        _RISK_MODEL_STATE["framework"] = "xgboost"

        feature_cols_path = dest / "feature_cols.json"
        if feature_cols_path.exists():
            with open(feature_cols_path) as f:
                _RISK_MODEL_STATE["feature_cols"] = json.load(f)

        parts = artifact_uri.rstrip("/").split("/")
        version_str = next(
            (p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()),
            "v1",
        )
        _RISK_MODEL_STATE["version"] = int(version_str[1:])
        _RISK_MODEL_STATE["stage"] = "candidate"
        _RISK_LOAD_FAILED = False
        logger.info("Risk scoring model loaded: uri=%s", artifact_uri)

    except Exception:  # noqa: BLE001 — catch-all for model loading
        _RISK_LOAD_FAILED = True
        _RISK_MODEL_STATE["version"] = 0
        _RISK_MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load risk model from %s", artifact_uri)


def is_risk_ready() -> bool:
    """Return True if the risk scoring model is loaded and ready."""
    return not _RISK_LOAD_FAILED and _RISK_MODEL_STATE.get("framework") is not None


def predict_risk_score(text: str, features: dict[str, Any] | None = None) -> float:
    """Compute a 0.0–1.0 risk score for a case.

    Args:
        text: Case text (used for inline feature extraction if no features).
        features: Pre-computed case features.

    Returns:
        Risk score clamped to [0.0, 1.0].
    """
    import numpy as np
    import xgboost as xgb

    from ml.serving.features import compute_inline_features

    if not is_risk_ready():
        raise RuntimeError("Risk scoring model not loaded")

    feat = features if features else compute_inline_features(text)
    feature_keys = _RISK_MODEL_STATE.get("feature_cols") or sorted(feat.keys())
    values = [float(feat.get(k) or 0) for k in feature_keys]
    dmat = xgb.DMatrix(np.array([values], dtype=np.float32), feature_names=feature_keys)

    raw_pred = _RISK_MODEL_STATE["booster"].predict(dmat)
    score = float(raw_pred[0])
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# NER Model State
# ---------------------------------------------------------------------------

_NER_MODEL_STATE: dict[str, Any] = {}
_NER_LOAD_FAILED = False


def load_ner_model(artifact_uri: str) -> None:
    """Load NER model artifacts from GCS.

    Sets ``_NER_LOAD_FAILED`` on error so the server can serve 503 for NER.
    """
    global _NER_LOAD_FAILED  # noqa: PLW0603

    if not artifact_uri:
        return

    _NER_MODEL_STATE["artifact_uri"] = artifact_uri
    parts = artifact_uri.rstrip("/").split("/")
    last = parts[-1] if parts else "ner-unknown"
    if last.startswith("v") and last[1:].isdigit() and len(parts) >= 2:
        _NER_MODEL_STATE["model_id"] = parts[-2]
    else:
        _NER_MODEL_STATE["model_id"] = last if last else "ner-unknown"

    try:
        dest = Path(tempfile.mkdtemp(prefix="ml_ner_"))
        _download_artifacts(artifact_uri, dest)

        # Load label map (NER uses label2id/id2label format)
        label_map_path = dest / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                ner_label_map = json.load(f)
            _NER_MODEL_STATE["label2id"] = ner_label_map.get("label2id", {})
            _NER_MODEL_STATE["id2label"] = {int(k): v for k, v in ner_label_map.get("id2label", {}).items()}

        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_dir = dest / "model" if (dest / "model").is_dir() else dest
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        model.eval()

        _NER_MODEL_STATE["tokenizer"] = tokenizer
        _NER_MODEL_STATE["model"] = model
        _NER_MODEL_STATE["framework"] = "pytorch"

        parts = artifact_uri.rstrip("/").split("/")
        version_str = next(
            (p for p in reversed(parts) if p.startswith("v") and p[1:].isdigit()),
            "v1",
        )
        _NER_MODEL_STATE["version"] = int(version_str[1:])
        _NER_MODEL_STATE["stage"] = "candidate"
        _NER_LOAD_FAILED = False
        logger.info("NER model loaded: uri=%s", artifact_uri)

    except Exception:  # noqa: BLE001 — catch-all for model loading
        _NER_LOAD_FAILED = True
        _NER_MODEL_STATE["version"] = 0
        _NER_MODEL_STATE["stage"] = "error"
        logger.exception("Failed to load NER model from %s", artifact_uri)


def is_ner_ready() -> bool:
    """Return True if the NER model is loaded and ready."""
    return not _NER_LOAD_FAILED and _NER_MODEL_STATE.get("framework") is not None


@dataclass
class EntitySpan:
    """A detected entity span."""

    text: str
    label: str
    start: int
    end: int
    confidence: float


def extract_entities(text: str, case_id: str) -> dict[str, Any]:
    """Run NER token classification on input text.

    Returns prediction dict with entity spans. Raises ``RuntimeError`` if NER
    model load was attempted but failed.
    """
    start_time = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    if _NER_LOAD_FAILED:
        raise RuntimeError("NER model failed to load — NER serving unavailable")

    if not is_ner_ready():
        raise RuntimeError("NER model not loaded")

    import torch

    tokenizer = _NER_MODEL_STATE["tokenizer"]
    model = _NER_MODEL_STATE["model"]
    id2label: dict[int, str] = _NER_MODEL_STATE.get("id2label", {})

    encoding = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model(**encoding).logits

    probs = torch.softmax(logits, dim=-1)
    predictions = logits.argmax(dim=-1)[0].tolist()
    max_probs = probs.max(dim=-1).values[0].tolist()

    # Decode BIO tags to entity spans
    entities: list[dict[str, Any]] = []
    current_entity: dict[str, Any] | None = None

    for pred_id, (char_start, char_end), conf in zip(predictions, offset_mapping, max_probs, strict=False):
        if char_start == 0 and char_end == 0:
            # Special token
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue

        tag = id2label.get(pred_id, "O")

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            label = tag[2:]
            current_entity = {
                "text": text[char_start:char_end],
                "label": label,
                "start": char_start,
                "end": char_end,
                "confidence": round(conf, 4),
            }
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["label"]:
            current_entity["end"] = char_end
            current_entity["text"] = text[current_entity["start"] : char_end]
            # Average confidence
            current_entity["confidence"] = round((current_entity["confidence"] + conf) / 2, 4)
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return {
        "prediction_id": prediction_id,
        "prediction": {"entities": entities},
        "model_info": {
            "model_id": _NER_MODEL_STATE.get("model_id", "ner-stub"),
            "version": _NER_MODEL_STATE.get("version", 0),
            "stage": _NER_MODEL_STATE.get("stage", "experimental"),
        },
        "latency_ms": elapsed_ms,
    }
