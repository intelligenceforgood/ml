"""FastAPI prediction server application."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
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
    is_challenger_ready,
    is_ner_ready,
    is_risk_ready,
    is_shadow_ready,
    load_challenger_model,
    load_model,
    load_ner_model,
    load_risk_model,
    load_shadow_model,
    predict_risk_score,
)
from ml.serving.routing import load_traffic_config, route_prediction_cost_aware

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
    challenger_active: bool = False
    ner_active: bool = False
    risk_active: bool = False


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


class RiskScoreRequest(BaseModel):
    text: str
    case_id: str
    features: dict | None = None


class RiskScoreResponse(BaseModel):
    case_id: str
    risk_score: float
    model_info: ModelInfo
    prediction_id: str


class SimilarCasesRequest(BaseModel):
    text: str
    case_id: str
    k: int = 10


class SimilarCaseResult(BaseModel):
    case_id: str
    distance: float
    score: float


class SimilarCasesResponse(BaseModel):
    case_id: str
    similar_cases: list[SimilarCaseResult]
    prediction_id: str


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

    challenger_uri = os.environ.get("CHALLENGER_MODEL_ARTIFACT_URI", "")
    if challenger_uri:
        load_challenger_model(challenger_uri)
        logger.info("Challenger model loaded from %s", challenger_uri)
    else:
        logger.info("CHALLENGER_MODEL_ARTIFACT_URI not set — challenger routing disabled")

    ner_uri = os.environ.get("NER_MODEL_ARTIFACT_URI", "")
    if ner_uri:
        load_ner_model(ner_uri)
        logger.info("NER model loaded from %s", ner_uri)
    else:
        logger.info("NER_MODEL_ARTIFACT_URI not set — NER serving disabled")

    risk_uri = os.environ.get("RISK_MODEL_ARTIFACT_URI", "")
    if risk_uri:
        load_risk_model(risk_uri)
        logger.info("Risk scoring model loaded from %s", risk_uri)
    else:
        logger.info("RISK_MODEL_ARTIFACT_URI not set — risk scoring disabled")

    # Initialize similarity index if embeddings are available
    if os.environ.get("SIMILARITY_ENABLED", "").lower() == "true":
        try:
            from ml.serving.similarity import rebuild_index_from_bq

            rebuild_index_from_bq()
            logger.info("Similarity index initialized")
        except Exception:  # noqa: BLE001 — non-critical
            logger.warning("Failed to initialize similarity index", exc_info=True)


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
        challenger_active=is_challenger_ready(),
        ner_active=is_ner_ready(),
        risk_active=is_risk_ready(),
    )


@app.post("/predict/classify")
async def predict_classify(request: Request) -> dict[str, Any]:
    """Handle both Vertex AI format ({"instances": [...]}) and direct format."""
    body = await request.json()

    # Vertex AI wraps payload as {"instances": [...]}, direct format otherwise
    if _LOAD_FAILED:
        raise HTTPException(status_code=503, detail="Model failed to load — serving unavailable")

    instances = body.get("instances", [body])

    # Load routing config once per request batch
    traffic_config = load_traffic_config()

    predictions = []
    for instance in instances:
        req = ClassifyRequest(**instance)

        # Route to champion or challenger
        routing = route_prediction_cost_aware(
            req.case_id,
            capability="classification",
            challenger_ready=is_challenger_ready(),
            traffic_config=traffic_config,
        )

        try:
            result = classify_text(req.text, req.case_id, features=req.features, variant=routing.variant)
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
            variant=result.get("variant", "champion"),
            routing_reason=routing.routing_reason,
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


# ---------------------------------------------------------------------------
# Risk Scoring (Sprint 4)
# ---------------------------------------------------------------------------


@app.post("/predict/risk-score", response_model=RiskScoreResponse)
async def predict_risk(req: RiskScoreRequest) -> RiskScoreResponse:
    """Compute a fraud risk score for a case."""
    if not is_risk_ready():
        raise HTTPException(status_code=503, detail="Risk scoring model not loaded")

    prediction_id = str(uuid.uuid4())
    try:
        score = predict_risk_score(req.text, req.features)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Risk scoring failed for case %s", req.case_id)
        raise HTTPException(status_code=500, detail="Risk scoring failed") from exc

    from ml.serving.predict import _RISK_MODEL_STATE

    model_info = ModelInfo(
        model_id=_RISK_MODEL_STATE.get("model_id", "risk-stub"),
        version=_RISK_MODEL_STATE.get("version", 0),
        stage=_RISK_MODEL_STATE.get("stage", "experimental"),
    )

    log_prediction(
        prediction_id=prediction_id,
        case_id=req.case_id,
        model_id=model_info.model_id,
        model_version=model_info.version,
        prediction={"risk_score": score},
        features=req.features,
        capability="risk_scoring",
    )

    return RiskScoreResponse(
        case_id=req.case_id,
        risk_score=score,
        model_info=model_info,
        prediction_id=prediction_id,
    )


# ---------------------------------------------------------------------------
# Document Similarity (Sprint 5)
# ---------------------------------------------------------------------------


@app.post("/predict/similar-cases", response_model=SimilarCasesResponse)
async def predict_similar_cases(req: SimilarCasesRequest) -> SimilarCasesResponse:
    """Find cases similar to the input text using embedding similarity."""
    import numpy as np

    prediction_id = str(uuid.uuid4())

    try:
        from ml.serving.embeddings import compute_embedding
        from ml.serving.similarity import get_similarity_index

        idx = get_similarity_index()
        if idx.size == 0:
            raise HTTPException(status_code=503, detail="Similarity index not built")

        embedding = compute_embedding(req.text)
        results = idx.search(np.array(embedding, dtype=np.float32), k=req.k)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Similarity search failed for case %s", req.case_id)
        raise HTTPException(status_code=500, detail="Similarity search failed") from exc

    log_prediction(
        prediction_id=prediction_id,
        case_id=req.case_id,
        model_id=os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
        model_version=0,
        prediction={"similar_cases": [{"case_id": r.case_id, "score": r.score} for r in results]},
        features=None,
        capability="document_similarity",
    )

    return SimilarCasesResponse(
        case_id=req.case_id,
        similar_cases=[SimilarCaseResult(case_id=r.case_id, distance=r.distance, score=r.score) for r in results],
        prediction_id=prediction_id,
    )
