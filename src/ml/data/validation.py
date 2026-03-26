"""Data quality validation rules for datasets."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of a dataset validation check."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_dataset(
    df: pd.DataFrame,
    *,
    label_column: str = "labels",
    min_samples_per_class: int = 50,
    max_null_rate: float = 0.1,
    max_class_imbalance_ratio: float = 20.0,
) -> ValidationResult:
    """Validate a training DataFrame before export.

    Checks:
    1. Minimum sample count per class
    2. Null rate per column
    3. Class balance ratio (majority / minority)
    4. Duplicate ``case_id`` detection
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        return ValidationResult(passed=False, errors=["Dataset is empty"])

    # --- Duplicate check ---
    if "case_id" in df.columns:
        dup_count = df["case_id"].duplicated().sum()
        if dup_count > 0:
            errors.append(f"{dup_count} duplicate case_id values")

    # --- Null rate check ---
    for col in df.columns:
        null_rate = df[col].isna().mean()
        if null_rate > max_null_rate:
            errors.append(f"Column '{col}' has {null_rate:.1%} null rate (max {max_null_rate:.1%})")

    # --- Label distribution check ---
    if label_column in df.columns:
        labels = df[label_column]
        if labels.dtype == object:
            # String labels: count directly
            counts = labels.value_counts()
        else:
            # Dict labels (multi-axis): flatten
            flat_labels = []
            for val in labels:
                if isinstance(val, dict):
                    for axis, code in val.items():
                        flat_labels.append(f"{axis}:{code}")
            counts = pd.Series(flat_labels).value_counts()

        for label, count in counts.items():
            if count < min_samples_per_class:
                warnings.append(f"Class '{label}' has only {count} samples (min {min_samples_per_class})")

        if len(counts) >= 2:
            ratio = counts.max() / counts.min()
            if ratio > max_class_imbalance_ratio:
                warnings.append(f"Class imbalance ratio is {ratio:.1f}x (max {max_class_imbalance_ratio}x)")

    stats = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "null_rates": {col: float(df[col].isna().mean()) for col in df.columns},
    }

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
