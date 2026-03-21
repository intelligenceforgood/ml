"""Tests for TrainingConfig and related models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ml.training.config import EvalGateConfig, LoraConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LABEL_SCHEMA = {
    "scam_type": ["phishing", "romance", "investment"],
    "severity": ["low", "medium", "high"],
}


def _minimal_config(**overrides) -> TrainingConfig:
    defaults = {
        "model_id": "classification-v1",
        "capability": "classification",
        "base_model": "google/gemma-2b",
        "framework": "pytorch",
        "training_type": "lora",
        "label_schema": _LABEL_SCHEMA,
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_minimal_construction(self):
        cfg = _minimal_config()
        assert cfg.model_id == "classification-v1"
        assert cfg.capability == "classification"
        assert cfg.framework == "pytorch"

    def test_default_hyperparameters(self):
        cfg = _minimal_config()
        assert cfg.epochs == 3
        assert cfg.batch_size == 8
        assert cfg.learning_rate == pytest.approx(2e-4)
        assert cfg.warmup_ratio == pytest.approx(0.1)

    def test_default_resources(self):
        cfg = _minimal_config()
        assert cfg.data_bucket == "i4g-ml-data"  # GCP bucket name stays unchanged
        assert cfg.machine_type == "n1-standard-4"
        assert cfg.gpu_type == "NVIDIA_TESLA_T4"
        assert cfg.gpu_count == 1

    def test_custom_hyperparameters(self):
        cfg = _minimal_config(epochs=10, batch_size=32, learning_rate=1e-3)
        assert cfg.epochs == 10
        assert cfg.batch_size == 32
        assert cfg.learning_rate == pytest.approx(1e-3)

    def test_lora_config(self):
        cfg = _minimal_config(lora=LoraConfig(r=8, alpha=16, dropout=0.05))
        assert cfg.lora is not None
        assert cfg.lora.r == 8
        assert cfg.lora.alpha == 16
        assert cfg.lora.dropout == pytest.approx(0.05)

    def test_lora_none_for_tabular(self):
        cfg = _minimal_config(framework="xgboost", training_type="tabular", lora=None)
        assert cfg.lora is None

    def test_label_schema_roundtrip(self):
        cfg = _minimal_config()
        assert "scam_type" in cfg.label_schema
        assert cfg.label_schema["severity"] == ["low", "medium", "high"]

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            TrainingConfig(capability="classification")  # missing model_id, base_model, etc.

    def test_serialization_roundtrip(self):
        cfg = _minimal_config(lora=LoraConfig())
        data = cfg.model_dump()
        restored = TrainingConfig(**data)
        assert restored == cfg


# ---------------------------------------------------------------------------
# EvalGateConfig
# ---------------------------------------------------------------------------


class TestEvalGateConfig:
    def test_defaults(self):
        gate = EvalGateConfig()
        assert gate.min_overall_f1 == 0.0
        assert gate.max_per_axis_regression == pytest.approx(0.05)

    def test_custom_thresholds(self):
        gate = EvalGateConfig(min_overall_f1=0.75, max_per_axis_regression=0.02)
        assert gate.min_overall_f1 == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# LoraConfig
# ---------------------------------------------------------------------------


class TestLoraConfig:
    def test_defaults(self):
        lora = LoraConfig()
        assert lora.r == 16
        assert lora.alpha == 32
        assert lora.dropout == pytest.approx(0.1)
        assert lora.target_modules == ["q_proj", "v_proj"]

    def test_custom_target_modules(self):
        lora = LoraConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        assert len(lora.target_modules) == 4
