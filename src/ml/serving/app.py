"""FastAPI prediction server application."""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from ml.serving.logging import log_outcome, log_prediction
from ml.serving.predict import classify_text, load_model

logger = logging.getLogger(__name__)

app = FastAPI(title="I4G ML Serving", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    text: str
    case_id: str
    features: dict | None = None


class ModelInfo(BaseModel):
    model_id: str
    version: int
    stage: str


class ClassifyResponse(BaseModel):
    prediction: dict[str, dict]
    risk_score: float | None = None
    model_info: ModelInfo
    prediction_id: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    case_id: str
    correction: dict[str, str]
    analyst_id: str


class FeedbackResponse(BaseModel):
    outcome_id: str
    status: str = "recorded"


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_id: str | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Load model artifacts on server startup."""
    artifact_uri = os.environ.get("MODEL_ARTIFACT_URI", "")
    if artifact_uri:
        load_model(artifact_uri)
        logger.info("Model loaded from %s", artifact_uri)
    else:
        logger.warning("MODEL_ARTIFACT_URI not set — serving will use stub predictions")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health status and loaded model identifier."""
    from ml.serving.predict import _MODEL_STATE

    return HealthResponse(
        status="healthy",
        model_id=_MODEL_STATE.get("model_id"),
    )


@app.post("/predict/classify")
async def predict_classify(request: Request) -> dict[str, Any]:
    """Handle both Vertex AI format ({"instances": [...]}) and direct format."""
    body = await request.json()

    # Vertex AI wraps payload as {"instances": [...]}, direct format otherwise
    instances = body.get("instances", [body])

    predictions = []
    for instance in instances:
        req = ClassifyRequest(**instance)
        try:
            result = classify_text(req.text, req.case_id, features=req.features)
        except Exception as exc:
            logger.exception("Prediction failed for case %s", req.case_id)
            raise HTTPException(status_code=500, detail="Prediction failed") from exc

        log_prediction(
            prediction_id=result["prediction_id"],
            case_id=req.case_id,
            model_id=result["model_info"]["model_id"],
            model_version=result["model_info"]["version"],
            prediction=result["prediction"],
            features=req.features,
            latency_ms=result.get("latency_ms", 0),
        )

        predictions.append(
            ClassifyResponse(
                prediction=result["prediction"],
                risk_score=result.get("risk_score"),
                model_info=ModelInfo(**result["model_info"]),
                prediction_id=result["prediction_id"],
            ).model_dump()
        )

    return {"predictions": predictions}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record analyst correction for a prediction."""
    outcome_id = log_outcome(
        prediction_id=req.prediction_id,
        case_id=req.case_id,
        correction=req.correction,
        analyst_id=req.analyst_id,
    )
    return FeedbackResponse(outcome_id=outcome_id)
