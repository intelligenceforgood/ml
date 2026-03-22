"""ML Platform configuration.

Settings are loaded from TOML files in ``ml/config/`` and can be overridden
via environment variables prefixed with ``I4G_ML_``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Settings section models
# ---------------------------------------------------------------------------


class PlatformSettings(BaseModel):
    """Top-level platform identification."""

    project_id: str = "i4g-ml"
    region: str = "us-central1"


class BigQuerySettings(BaseModel):
    """BigQuery dataset and table references."""

    dataset_id: str = "i4g_ml"
    prediction_log_table: str = "predictions_prediction_log"
    outcome_log_table: str = "predictions_outcome_log"


class StorageSettings(BaseModel):
    """Cloud Storage layout."""

    data_bucket: str = "i4g-ml-data"
    datasets_prefix: str = "datasets"
    models_prefix: str = "models"


class ServingSettings(BaseModel):
    """Vertex AI Endpoint configuration."""

    dev_endpoint_name: str = "serving-dev"
    prod_endpoint_name: str = "serving-prod"
    min_replicas: int = 0
    max_replicas: int = 2
    machine_type: str = "n1-standard-4"


class TrainingSettings(BaseModel):
    """Default training resource configuration."""

    default_machine_type: str = "n1-standard-4"
    gpu_machine_type: str = "n1-standard-4"
    gpu_type: str = "NVIDIA_TESLA_T4"
    gpu_count: int = 1


class EtlSettings(BaseModel):
    """ETL job configuration."""

    source_db_connection: str = ""
    source_instance: str = ""
    source_db_name: str = ""
    source_db_user: str = ""
    source_enable_iam_auth: bool = True
    batch_size: int = 1000


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """Aggregated ML Platform settings."""

    platform: PlatformSettings = Field(default_factory=PlatformSettings)
    bigquery: BigQuerySettings = Field(default_factory=BigQuerySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    etl: EtlSettings = Field(default_factory=EtlSettings)


def _project_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``)."""
    path = Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "pyproject.toml").exists():
            return path
        path = path.parent
    return Path.cwd()


def _load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file and return its contents as a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


@lru_cache
def get_settings() -> Settings:
    """Load settings from config files with environment variable overrides.

    Environment variables use the prefix ``I4G_ML_`` with double underscores
    for section nesting.  For example, ``I4G_ML_ETL__SOURCE_INSTANCE`` maps to
    ``settings.etl.source_instance``.
    """
    root = Path(os.environ.get("I4G_ML_PROJECT_ROOT", str(_project_root())))
    config_dir = root / "config"

    data: dict[str, dict[str, Any]] = {}
    for name in ("settings.default.toml", "settings.dev.toml", "settings.local.toml"):
        path = config_dir / name
        if path.exists():
            loaded = _load_toml(path)
            for section, values in loaded.items():
                data.setdefault(section, {}).update(values)

    # Apply environment variable overrides (I4G_ML_<SECTION>__<KEY>)
    prefix = "I4G_ML_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :].lower()
        if "__" not in remainder:
            continue
        section, setting = remainder.split("__", 1)
        data.setdefault(section, {})[setting] = value

    return Settings(**data)
