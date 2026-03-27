"""Champion/Challenger A/B routing for ML serving.

Supports three routing strategies:
- **Random**: weight-based random traffic split
- **Deterministic**: hash-based split on case_id for reproducible assignments
- **Cost-aware**: picks cheapest model meeting a quality bar
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TrafficSplitConfig(BaseModel):
    """Traffic split between champion and challenger models."""

    champion_weight: float = 1.0
    challenger_weight: float = 0.0
    challenger_artifact_uri: str | None = None
    split_strategy: Literal["random", "deterministic"] = "random"

    @field_validator("champion_weight", "challenger_weight")
    @classmethod
    def _weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("challenger_weight")
    @classmethod
    def _weights_sum(cls, v: float, info: Any) -> float:
        champion = info.data.get("champion_weight", 1.0)
        if abs(champion + v - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got champion={champion} + challenger={v} = {champion + v}")
        return v


def load_traffic_config() -> TrafficSplitConfig:
    """Load traffic split configuration from environment variables."""
    challenger_uri = os.environ.get("CHALLENGER_MODEL_ARTIFACT_URI", "")
    challenger_weight = float(os.environ.get("CHALLENGER_TRAFFIC_WEIGHT", "0.0"))
    strategy = os.environ.get("TRAFFIC_SPLIT_STRATEGY", "random")

    if not challenger_uri or challenger_weight <= 0:
        return TrafficSplitConfig()

    return TrafficSplitConfig(
        champion_weight=1.0 - challenger_weight,
        challenger_weight=challenger_weight,
        challenger_artifact_uri=challenger_uri,
        split_strategy=strategy,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    variant: str  # "champion" or "challenger"
    model_state_key: str  # key into the model state dict to use
    routing_reason: str  # why this variant was selected


def route_prediction(
    case_id: str,
    config: TrafficSplitConfig | None = None,
    *,
    challenger_ready: bool = False,
) -> RoutingDecision:
    """Select champion or challenger based on traffic split config.

    Args:
        case_id: Case identifier (used for deterministic hashing).
        config: Traffic split config. If None, loads from env.
        challenger_ready: Whether the challenger model is loaded and ready.

    Returns:
        RoutingDecision with the selected variant.
    """
    if config is None:
        config = load_traffic_config()

    # If challenger is not configured or not loaded, always return champion
    if not challenger_ready or config.challenger_weight <= 0 or not config.challenger_artifact_uri:
        return RoutingDecision(
            variant="champion",
            model_state_key="champion",
            routing_reason="champion_only",
        )

    # Determine variant assignment
    if config.split_strategy == "deterministic":
        # Hash case_id to get a deterministic float in [0, 1)
        hash_val = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10000 / 10000
        use_challenger = hash_val < config.challenger_weight
    else:
        # Random split
        use_challenger = random.random() < config.challenger_weight

    if use_challenger:
        return RoutingDecision(
            variant="challenger",
            model_state_key="challenger",
            routing_reason=f"ab_split_{config.split_strategy}",
        )

    return RoutingDecision(
        variant="champion",
        model_state_key="champion",
        routing_reason=f"ab_split_{config.split_strategy}",
    )


# ---------------------------------------------------------------------------
# Cost-aware routing
# ---------------------------------------------------------------------------


@dataclass
class ModelCostProfile:
    """Cost and quality profile for a loaded model."""

    model_id: str
    capability: str
    cost_per_prediction: float
    avg_latency_ms: float
    f1_score: float


def load_cost_profiles() -> list[ModelCostProfile]:
    """Load cost profiles from BigQuery analytics_cost_summary.

    Returns cached profiles (refreshed hourly in production).
    """
    try:
        from google.cloud import bigquery

        from ml.config import get_settings

        settings = get_settings()
        client = bigquery.Client(project=settings.platform.project_id)
        query = f"""
            SELECT model_id, capability, cost_per_prediction, avg_latency_ms, f1_score
            FROM `{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_cost_summary`
            WHERE _snapshot_date = (
                SELECT MAX(_snapshot_date)
                FROM `{settings.platform.project_id}.{settings.bigquery.dataset_id}.analytics_cost_summary`
            )
        """
        rows = list(client.query(query).result())
        return [
            ModelCostProfile(
                model_id=row.model_id,
                capability=row.capability,
                cost_per_prediction=row.cost_per_prediction,
                avg_latency_ms=row.avg_latency_ms,
                f1_score=row.f1_score,
            )
            for row in rows
        ]
    except Exception:
        logger.exception("Failed to load cost profiles — falling back to empty list")
        return []


def select_cheapest_model(
    capability: str,
    profiles: list[ModelCostProfile] | None = None,
    *,
    quality_bar: float = 0.8,
) -> ModelCostProfile | None:
    """Select the cheapest model that meets the quality bar for a capability.

    Args:
        capability: ML capability (classification, ner, risk_scoring).
        profiles: Pre-loaded cost profiles. If None, loads from BQ.
        quality_bar: Minimum F1 score required.

    Returns:
        The cheapest qualifying model, or None if no model meets the bar.
    """
    if profiles is None:
        profiles = load_cost_profiles()

    candidates = [p for p in profiles if p.capability == capability and p.f1_score >= quality_bar]
    if not candidates:
        return None

    return min(candidates, key=lambda p: p.cost_per_prediction)


def route_prediction_cost_aware(
    case_id: str,
    capability: str = "classification",
    *,
    quality_bar: float = 0.8,
    cost_profiles: list[ModelCostProfile] | None = None,
    challenger_ready: bool = False,
    traffic_config: TrafficSplitConfig | None = None,
) -> RoutingDecision:
    """Route prediction using cost-aware strategy when enabled.

    Falls back to standard A/B routing when COST_AWARE_ROUTING is not enabled
    or when cost profiles are unavailable.
    """
    cost_aware_enabled = os.environ.get("COST_AWARE_ROUTING", "false").lower() == "true"

    if cost_aware_enabled:
        cheapest = select_cheapest_model(capability, cost_profiles, quality_bar=quality_bar)
        if cheapest:
            # Map the cheapest model to the appropriate variant
            variant = "champion"  # default
            routing_reason = f"cost_aware:model={cheapest.model_id},cost={cheapest.cost_per_prediction:.4f}"

            # If the cheapest is the challenger model, route there
            if (
                challenger_ready
                and traffic_config
                and traffic_config.challenger_artifact_uri
                and cheapest.model_id != "champion"
            ):
                variant = "challenger"

            return RoutingDecision(
                variant=variant,
                model_state_key=variant,
                routing_reason=routing_reason,
            )

    # Fall back to standard A/B routing
    return route_prediction(case_id, traffic_config, challenger_ready=challenger_ready)
