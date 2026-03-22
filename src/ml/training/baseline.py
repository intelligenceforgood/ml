"""Baseline benchmark — few-shot LLM classifier evaluation.

Runs the current few-shot LLM classifier against a golden test set and
records per-axis F1 as the baseline to beat.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from ml.training.evaluation import EvalResult, compute_metrics

logger = logging.getLogger(__name__)


def load_golden_set(path: str | Path) -> list[dict]:
    """Load a golden test set from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_baseline(
    golden_path: str | Path,
    classifier_fn: Callable[[str], dict[str, str]],
    *,
    label_key: str = "labels",
) -> EvalResult:
    """Run a classifier function against the golden set and return metrics.

    *classifier_fn* should accept a text string and return a dict mapping
    axis names to predicted label codes.
    """
    records = load_golden_set(golden_path)
    predictions: list[dict[str, str]] = []
    ground_truth: list[dict[str, str]] = []

    for record in records:
        text = record.get("text", "")
        gt = record.get(label_key, {})
        if not gt:
            continue

        pred = classifier_fn(text)
        predictions.append(pred)
        ground_truth.append(gt)

    result = compute_metrics(predictions, ground_truth)
    logger.info("Baseline evaluation:\n%s", result.summary())
    return result


def save_baseline_result(result: EvalResult, output_path: str | Path) -> None:
    """Save baseline evaluation results to a JSON file."""
    data = {
        "overall_f1": result.overall_f1,
        "overall_precision": result.overall_precision,
        "overall_recall": result.overall_recall,
        "total_samples": result.total_samples,
        "per_axis": {
            axis: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
            }
            for axis, m in result.per_axis.items()
        },
    }
    Path(output_path).write_text(json.dumps(data, indent=2))
