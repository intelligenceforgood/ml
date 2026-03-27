"""Unit tests for champion/challenger A/B routing (Sprint 1)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from ml.serving.routing import (
    ModelCostProfile,
    TrafficSplitConfig,
    load_traffic_config,
    route_prediction,
    route_prediction_cost_aware,
    select_cheapest_model,
)


class TestTrafficSplitConfig:
    """Traffic split config validation."""

    def test_default_champion_only(self):
        config = TrafficSplitConfig()
        assert config.champion_weight == 1.0
        assert config.challenger_weight == 0.0

    def test_valid_split(self):
        config = TrafficSplitConfig(champion_weight=0.8, challenger_weight=0.2)
        assert config.champion_weight == 0.8
        assert config.challenger_weight == 0.2

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            TrafficSplitConfig(champion_weight=0.8, challenger_weight=0.3)

    def test_weight_out_of_range_negative(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TrafficSplitConfig(champion_weight=-0.1, challenger_weight=1.1)

    def test_weight_out_of_range_above_one(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TrafficSplitConfig(champion_weight=1.5, challenger_weight=-0.5)

    def test_fifty_fifty_split(self):
        config = TrafficSplitConfig(
            champion_weight=0.5,
            challenger_weight=0.5,
            challenger_artifact_uri="gs://bucket/models/challenger/v1",
        )
        assert config.challenger_artifact_uri is not None


class TestLoadTrafficConfig:
    """Loading config from environment variables."""

    def test_default_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            config = load_traffic_config()
        assert config.champion_weight == 1.0
        assert config.challenger_weight == 0.0

    def test_from_env_vars(self):
        env = {
            "CHALLENGER_MODEL_ARTIFACT_URI": "gs://bucket/challenger/v1",
            "CHALLENGER_TRAFFIC_WEIGHT": "0.2",
            "TRAFFIC_SPLIT_STRATEGY": "deterministic",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_traffic_config()
        assert config.challenger_weight == 0.2
        assert config.champion_weight == 0.8
        assert config.split_strategy == "deterministic"


class TestRoutePredicitionLogic:
    """Routing decision logic."""

    def test_champion_only_when_no_challenger(self):
        config = TrafficSplitConfig()
        decision = route_prediction("case-1", config, challenger_ready=False)
        assert decision.variant == "champion"
        assert decision.routing_reason == "champion_only"

    def test_champion_only_when_challenger_not_ready(self):
        config = TrafficSplitConfig(
            champion_weight=0.8,
            challenger_weight=0.2,
            challenger_artifact_uri="gs://bucket/challenger/v1",
        )
        decision = route_prediction("case-1", config, challenger_ready=False)
        assert decision.variant == "champion"

    def test_deterministic_same_case_same_result(self):
        config = TrafficSplitConfig(
            champion_weight=0.5,
            challenger_weight=0.5,
            challenger_artifact_uri="gs://bucket/challenger/v1",
            split_strategy="deterministic",
        )
        results = {route_prediction("stable-case", config, challenger_ready=True).variant for _ in range(100)}
        assert len(results) == 1  # Same case always gets same variant

    def test_random_split_distribution(self):
        """Statistical test: over 10K samples, ≈20% should be challenger."""
        config = TrafficSplitConfig(
            champion_weight=0.8,
            challenger_weight=0.2,
            challenger_artifact_uri="gs://bucket/challenger/v1",
            split_strategy="random",
        )
        challenger_count = sum(
            1
            for i in range(10000)
            if route_prediction(f"case-{i}", config, challenger_ready=True).variant == "challenger"
        )
        # Allow ±3% tolerance
        assert 1700 < challenger_count < 2300, f"Expected ~2000 challenger, got {challenger_count}"


class TestCostAwareRouting:
    """Cost-aware routing selection."""

    def test_cheapest_model_above_quality_bar(self):
        profiles = [
            ModelCostProfile("model-a", "classification", 0.01, 50, 0.9),
            ModelCostProfile("model-b", "classification", 0.005, 100, 0.85),
            ModelCostProfile("model-c", "classification", 0.002, 200, 0.7),
        ]
        best = select_cheapest_model("classification", profiles, quality_bar=0.8)
        assert best is not None
        assert best.model_id == "model-b"

    def test_no_model_above_bar(self):
        profiles = [
            ModelCostProfile("model-a", "classification", 0.01, 50, 0.6),
        ]
        best = select_cheapest_model("classification", profiles, quality_bar=0.8)
        assert best is None

    def test_filters_by_capability(self):
        profiles = [
            ModelCostProfile("ner-model", "ner", 0.001, 10, 0.95),
            ModelCostProfile("cls-model", "classification", 0.01, 50, 0.9),
        ]
        best = select_cheapest_model("classification", profiles, quality_bar=0.8)
        assert best is not None
        assert best.model_id == "cls-model"

    def test_falls_back_to_ab_when_disabled(self):
        with patch.dict(os.environ, {"COST_AWARE_ROUTING": "false"}, clear=True):
            decision = route_prediction_cost_aware("case-1", challenger_ready=False)
        assert decision.variant == "champion"
