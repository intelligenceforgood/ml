"""Training configuration schema — Pydantic models for experiment configs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    """LoRA adapter hyperparameters for parameter-efficient fine-tuning."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class EvalGateConfig(BaseModel):
    """Minimum quality thresholds a trained model must pass before promotion."""

    min_overall_f1: float = 0.0
    max_per_axis_regression: float = 0.05
    # Risk scoring eval gate (Sprint 4)
    max_mse: float | None = None  # Max mean squared error
    min_spearman: float | None = None  # Min Spearman correlation (e.g. 0.6)


class TrainingConfig(BaseModel):
    """Immutable experiment descriptor: what to train, how, and where.

    All training runs — whether kicked off from the CLI, the KFP pipeline,
    or a notebook — should be driven by a ``TrainingConfig`` instance so
    that every run is fully reproducible and auditable.
    """

    model_id: str = Field(..., description="Unique run identifier (e.g. 'classification-v3')")
    capability: str = Field(..., description="ML capability being trained (e.g. 'classification')")
    base_model: str = Field(..., description="HuggingFace model hub ID or GCS path")
    framework: str = Field(..., description="Training framework: pytorch | xgboost | tensorflow")
    training_type: str = Field(..., description="Training strategy: lora | full | tabular")

    # Hyperparameters
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1

    # LoRA (only for fine-tuning)
    lora: LoraConfig | None = None

    # Label schema: axis → list of valid label codes
    label_schema: dict[str, list[str]]

    # Eval gate
    eval_gate: EvalGateConfig = Field(default_factory=EvalGateConfig)

    # Vizier
    enable_vizier: bool = False

    # Resources
    data_bucket: str = "i4g-ml-data"
    machine_type: str = "n1-standard-4"
    gpu_type: str | None = "NVIDIA_TESLA_T4"
    gpu_count: int = 1
