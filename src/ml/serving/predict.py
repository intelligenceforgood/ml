"""Prediction inference logic."""

from __future__ import annotations

import time
import uuid
from typing import Any

_MODEL_STATE: dict = {}


def load_model(artifact_uri: str) -> None:
    """Load model artifacts from the given URI into memory."""
    # Phase 0: placeholder — real implementation loads PyTorch/XGBoost model
    _MODEL_STATE["artifact_uri"] = artifact_uri
    _MODEL_STATE["model_id"] = artifact_uri.split("/")[-2] if "/" in artifact_uri else "unknown"
    _MODEL_STATE["version"] = 1
    _MODEL_STATE["stage"] = "experimental"


def classify_text(
    text: str,
    case_id: str,
    *,
    features: dict | None = None,
) -> dict[str, Any]:
    """Run classification on input text. Returns prediction dict."""
    start = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    # Phase 0 stub: return placeholder predictions
    # Real impl: tokenize text, run model forward pass, decode outputs
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
