"""Run majority-class baseline benchmark against the golden test set.

Phase 0: No narrative text is available in BigQuery yet, so we cannot run the
LLM few-shot classifier.  Instead, this establishes a majority-class baseline
— always predict the most common label per axis from the training set.
This gives a minimum F1 floor that any trained model must exceed.

Usage:
    conda run -n ml python scripts/run_baseline.py
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from google.cloud import storage

from ml.training.evaluation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "i4g-ml-data"
TRAIN_PATH = "datasets/classification/v1/train.jsonl"
GOLDEN_PATH = "datasets/classification/golden/test.jsonl"
OUTPUT_PATH = Path("data/baseline_result.json")


def load_jsonl_from_gcs(bucket_name: str, gcs_path: str) -> list[dict]:
    """Load JSONL records from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    content = blob.download_as_text()
    return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def compute_majority_labels(records: list[dict]) -> dict[str, str]:
    """Find the most common label per axis across all records."""
    axis_counts: dict[str, Counter] = {}
    for rec in records:
        labels = rec.get("labels", {})
        for axis, code in labels.items():
            if axis not in axis_counts:
                axis_counts[axis] = Counter()
            axis_counts[axis][code] += 1

    majority: dict[str, str] = {}
    for axis, counts in axis_counts.items():
        most_common = counts.most_common(1)[0]
        majority[axis] = most_common[0]
        logger.info("Majority %s: %s (%d/%d)", axis, most_common[0], most_common[1], sum(counts.values()))

    return majority


def main() -> None:
    """Compute majority-class baseline and save results locally."""
    # Load training data to compute majority labels
    logger.info("Loading training data from gs://%s/%s", BUCKET, TRAIN_PATH)
    train_records = load_jsonl_from_gcs(BUCKET, TRAIN_PATH)
    logger.info("Training records: %d", len(train_records))

    majority_labels = compute_majority_labels(train_records)
    logger.info("Majority labels: %s", majority_labels)

    # Load golden test set
    logger.info("Loading golden test set from gs://%s/%s", BUCKET, GOLDEN_PATH)
    golden_records = load_jsonl_from_gcs(BUCKET, GOLDEN_PATH)
    logger.info("Golden test records: %d", len(golden_records))

    # Run majority-class "classifier" against golden set
    predictions: list[dict[str, str]] = []
    ground_truth: list[dict[str, str]] = []

    for rec in golden_records:
        gt = rec.get("labels", {})
        if not gt:
            continue
        # Predict majority label for each axis present in ground truth
        pred = {axis: majority_labels.get(axis, "") for axis in gt}
        predictions.append(pred)
        ground_truth.append(gt)

    result = compute_metrics(predictions, ground_truth)
    logger.info("Majority-class baseline:\n%s", result.summary())

    # Save result
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "baseline_type": "majority_class",
        "overall_f1": result.overall_f1,
        "overall_precision": result.overall_precision,
        "overall_recall": result.overall_recall,
        "total_samples": result.total_samples,
        "majority_labels": majority_labels,
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
    OUTPUT_PATH.write_text(json.dumps(data, indent=2))
    logger.info("Saved baseline result → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
