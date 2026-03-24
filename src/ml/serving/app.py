"""FastAPI prediction server application."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from ml.serving.logging import log_outcome, log_prediction
from ml.serving.predict import (
    _LOAD_FAILED,
    _NER_LOAD_FAILED,
    classify_text,
    classify_text_shadow,
    extract_entities,
    is_ner_ready,
    is_shadow_ready,
    load_model,
    load_ner_model,
    load_shadow_model,
)

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
    shadow_active: bool = False
    ner_active: bool = False


class ExtractEntitiesRequest(BaseModel):
    text: str
    case_id: str


class EntitySpanResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class ExtractEntitiesResponse(BaseModel):
    prediction_id: str
    entities: list[EntitySpanResponse]
    model_info: ModelInfo


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

    shadow_uri = os.environ.get("SHADOW_MODEL_ARTIFACT_URI", "")
    if shadow_uri:
        load_shadow_model(shadow_uri)
        logger.info("Shadow model loaded from %s", shadow_uri)
    else:
        logger.info("SHADOW_MODEL_ARTIFACT_URI not set — shadow mode disabled")

    ner_uri = os.environ.get("NER_MODEL_ARTIFACT_URI", "")
    if ner_uri:
        load_ner_model(ner_uri)
        logger.info("NER model loaded from %s", ner_uri)
    else:
        logger.info("NER_MODEL_ARTIFACT_URI not set — NER serving disabled")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return server health status and loaded model identifier."""
    from ml.serving.predict import _MODEL_STATE

    status = "healthy" if not _LOAD_FAILED else "degraded"
    return HealthResponse(
        status=status,
        model_id=_MODEL_STATE.get("model_id"),
        shadow_active=is_shadow_ready(),
        ner_active=is_ner_ready(),
    )


@app.post("/predict/classify")
async def predict_classify(request: Request) -> dict[str, Any]:
    """Handle both Vertex AI format ({"instances": [...]}) and direct format."""
    body = await request.json()

    # Vertex AI wraps payload as {"instances": [...]}, direct format otherwise
    if _LOAD_FAILED:
        raise HTTPException(status_code=503, detail="Model failed to load — serving unavailable")

    instances = body.get("instances", [body])

    predictions = []
    for instance in instances:
        req = ClassifyRequest(**instance)
        try:
            result = classify_text(req.text, req.case_id, features=req.features)
        except Exception as exc:  # noqa: BLE001 — return HTTP 500 for any prediction error
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

        # Fire shadow inference asynchronously — never blocks champion response
        if is_shadow_ready():
            asyncio.create_task(_run_shadow_inference(req.text, req.case_id, result["prediction_id"], req.features))

        predictions.append(
            ClassifyResponse(
                prediction=result["prediction"],
                risk_score=result.get("risk_score"),
                model_info=ModelInfo(**result["model_info"]),
                prediction_id=result["prediction_id"],
            ).model_dump()
        )

    return {"predictions": predictions}


async def _run_shadow_inference(
    text: str,
    case_id: str,
    champion_prediction_id: str,
    features: dict | None,
) -> None:
    """Run shadow model inference and log result.  All exceptions are caught."""
    try:
        shadow_result = await asyncio.to_thread(
            classify_text_shadow, text, case_id, champion_prediction_id, features=features
        )
        if shadow_result is None:
            return

        log_prediction(
            prediction_id=shadow_result["prediction_id"],
            case_id=case_id,
            model_id=shadow_result["model_info"]["model_id"],
            model_version=shadow_result["model_info"]["version"],
            prediction=shadow_result["prediction"],
            features=features,
            latency_ms=shadow_result.get("latency_ms", 0),
            is_shadow=True,
        )
    except Exception:  # noqa: BLE001 — shadow must never affect champion
        logger.exception("Shadow inference failed for champion prediction %s", champion_prediction_id)


@app.post("/predict/extract-entities", response_model=ExtractEntitiesResponse)
async def predict_extract_entities(req: ExtractEntitiesRequest) -> ExtractEntitiesResponse:
    """Extract named entities from text using the NER model."""
    if _NER_LOAD_FAILED:
        raise HTTPException(status_code=503, detail="NER model failed to load — NER serving unavailable")
    if not is_ner_ready():
        raise HTTPException(status_code=503, detail="NER model not loaded")

    try:
        result = extract_entities(req.text, req.case_id)
    except Exception as exc:  # noqa: BLE001 — return HTTP 500 for any extraction error
        logger.exception("Entity extraction failed for case %s", req.case_id)
        raise HTTPException(status_code=500, detail="Entity extraction failed") from exc

    log_prediction(
        prediction_id=result["prediction_id"],
        case_id=req.case_id,
        model_id=result["model_info"]["model_id"],
        model_version=result["model_info"]["version"],
        prediction=result["prediction"],
        features=None,
        latency_ms=result.get("latency_ms", 0),
        capability="ner",
    )

    return ExtractEntitiesResponse(
        prediction_id=result["prediction_id"],
        entities=[EntitySpanResponse(**e) for e in result["prediction"]["entities"]],
        model_info=ModelInfo(**result["model_info"]),
    )


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
