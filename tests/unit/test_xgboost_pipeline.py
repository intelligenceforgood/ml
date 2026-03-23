"""Tests for XGBoost pipeline integration — Sprint 2 task 2.4."""

from __future__ import annotations

from pathlib import Path

import yaml

from ml.training.config import TrainingConfig

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "pipelines" / "configs"


class TestXGBoostConfig:
    """2.4.1 / 2.4.2 — Verify XGBoost training config YAML is valid."""

    def test_xgboost_config_exists(self):
        path = CONFIGS_DIR / "classification_xgboost.yaml"
        assert path.exists(), f"XGBoost config not found at {path}"

    def test_xgboost_config_loads_as_training_config(self):
        """The YAML should deserialize into a valid TrainingConfig."""
        path = CONFIGS_DIR / "classification_xgboost.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        # TrainingConfig expects flat hyperparams, but the YAML nests them
        flat = {
            "model_id": raw["model_id"],
            "capability": raw["capability"],
            "base_model": raw["base_model"],
            "framework": raw["framework"],
            "training_type": raw["training_type"],
            "epochs": raw["hyperparameters"]["epochs"],
            "batch_size": raw["hyperparameters"]["batch_size"],
            "learning_rate": raw["hyperparameters"]["learning_rate"],
            "warmup_ratio": raw["hyperparameters"]["warmup_ratio"],
            "label_schema": raw["label_schema"],
            "eval_gate": raw["eval_gate"],
            "data_bucket": raw["data_bucket"],
            "machine_type": raw["resources"]["machine_type"],
            "gpu_type": raw["resources"].get("gpu_type") or None,
            "gpu_count": raw["resources"]["gpu_count"],
        }
        cfg = TrainingConfig(**flat)
        assert cfg.framework == "xgboost"
        assert cfg.gpu_count == 0
        assert cfg.lora is None

    def test_xgboost_config_has_correct_label_schema(self):
        path = CONFIGS_DIR / "classification_xgboost.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        assert "INTENT" in raw["label_schema"]
        assert "CHANNEL" in raw["label_schema"]
        assert len(raw["label_schema"]["INTENT"]) == 7
        assert len(raw["label_schema"]["CHANNEL"]) == 6

    def test_xgboost_config_no_gpu(self):
        """XGBoost should not request GPU resources."""
        path = CONFIGS_DIR / "classification_xgboost.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        assert raw["resources"]["gpu_count"] == 0
        assert raw["resources"]["gpu_type"] == ""

    def test_xgboost_specific_params(self):
        """Verify XGBoost-specific hyperparameters are present."""
        path = CONFIGS_DIR / "classification_xgboost.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        xgb = raw["xgboost_params"]
        assert xgb["max_depth"] == 6
        assert xgb["objective"] == "multi:softprob"
        assert xgb["early_stopping_rounds"] == 10


class TestPyTorchConfig:
    """Cross-check: ensure PyTorch config is also valid."""

    def test_pytorch_config_exists(self):
        path = CONFIGS_DIR / "classification_gemma2b.yaml"
        assert path.exists()

    def test_pytorch_config_has_lora(self):
        path = CONFIGS_DIR / "classification_gemma2b.yaml"
        with open(path) as f:
            raw = yaml.safe_load(f)

        assert raw["training_type"] == "lora"
        assert raw["lora"]["r"] == 16


class TestPipelineXGBoostCompatibility:
    """2.4.2 — Verify the pipeline train_model component accepts XGBoost config."""

    def test_training_pipeline_accepts_xgboost_container_uri(self):
        """The training_pipeline function should accept XGBoost container URI."""
        from ml.training.pipeline import training_pipeline

        # KFP @dsl.pipeline wraps the function; check the underlying pipeline_func
        pipeline_func = getattr(training_pipeline, "pipeline_func", training_pipeline)
        import inspect

        sig = inspect.signature(pipeline_func)
        assert "container_uri" in sig.parameters
        assert "config_path" in sig.parameters

    def test_evaluate_model_handles_xgboost(self):
        """The evaluate_model component should handle XGBoost model type."""
        from ml.training.pipeline import evaluate_model

        # Verify the component is defined and callable
        assert callable(evaluate_model)


class TestFrameworkComparison:
    """2.4.4 — Framework selection criteria documented and testable."""

    def test_both_configs_share_label_schema(self):
        """XGBoost and PyTorch configs must use the same label schema."""
        xgb_path = CONFIGS_DIR / "classification_xgboost.yaml"
        pt_path = CONFIGS_DIR / "classification_gemma2b.yaml"

        with open(xgb_path) as f:
            xgb = yaml.safe_load(f)
        with open(pt_path) as f:
            pt = yaml.safe_load(f)

        assert xgb["label_schema"] == pt["label_schema"], "XGBoost and PyTorch configs must use identical label schemas"

    def test_both_configs_share_eval_gate(self):
        """Both frameworks should use the same eval gate thresholds."""
        xgb_path = CONFIGS_DIR / "classification_xgboost.yaml"
        pt_path = CONFIGS_DIR / "classification_gemma2b.yaml"

        with open(xgb_path) as f:
            xgb = yaml.safe_load(f)
        with open(pt_path) as f:
            pt = yaml.safe_load(f)

        assert xgb["eval_gate"] == pt["eval_gate"]
