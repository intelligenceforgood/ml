"""Bootstrap training dataset from existing LLM classifications.

Phase 0: Since we have no human analyst labels yet, we use the LLM-generated
classification_result from raw_cases as pseudo ground-truth labels. This script:

1. Queries raw_cases + features_case_features from BigQuery
2. Parses classification_result JSON to extract top label per axis
3. Filters to cases with at least one classified axis
4. Creates a golden test set (~100 diverse cases) and v1 training dataset
5. Exports as JSONL to GCS
6. Registers v1 in training_dataset_registry

Usage:
    conda run -n ml python scripts/bootstrap_dataset.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

import pandas as pd
from google.cloud import bigquery, storage

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "i4g-ml"
DATASET = "i4g_ml"
BUCKET = "i4g-ml-data"
AXES = ["intent", "channel", "techniques", "actions", "persona"]


def _extract_top_labels(classification_result: str | None) -> dict[str, str]:
    """Extract top-confidence label per axis from classification_result JSON."""
    if not classification_result:
        return {}
    try:
        cr = json.loads(classification_result)
    except (json.JSONDecodeError, TypeError):
        return {}

    labels: dict[str, str] = {}
    for axis in AXES:
        items = cr.get(axis, [])
        if not items:
            continue
        # Pick highest confidence label
        top = max(items, key=lambda x: x.get("confidence", 0.0))
        label = top.get("label", "")
        if label:
            labels[axis.upper()] = label
    return labels


def query_data(bq: bigquery.Client) -> pd.DataFrame:
    """Query raw_cases + features, parse labels, filter to labeled cases."""
    sql = f"""
        SELECT
            c.case_id,
            c.classification_result,
            c.status,
            c.risk_score,
            f.text_length,
            f.word_count,
            f.avg_sentence_length,
            f.entity_count,
            f.unique_entity_types,
            f.has_crypto_wallet,
            f.has_bank_account,
            f.has_phone,
            f.has_email,
            f.indicator_count,
            f.indicator_diversity,
            f.max_indicator_confidence,
            f.classification_axis_count,
            f.current_classification_conf,
            f.case_age_days
        FROM `{PROJECT}.{DATASET}.raw_cases` c
        JOIN `{PROJECT}.{DATASET}.features_case_features` f
          ON c.case_id = f.case_id
    """
    df = bq.query(sql).to_dataframe()
    logger.info("Queried %d rows from BQ", len(df))

    # Parse labels from classification_result
    df["labels"] = df["classification_result"].apply(_extract_top_labels)
    df["label_count"] = df["labels"].apply(len)

    # Filter to cases with at least one classified axis
    labeled = df[df["label_count"] > 0].copy()
    logger.info("Cases with labels: %d / %d", len(labeled), len(df))

    # Drop raw classification_result (already parsed into labels)
    labeled = labeled.drop(columns=["classification_result"])
    return labeled


def stratified_split(
    df: pd.DataFrame,
    golden_size: int = 100,
    train_ratio: float = 0.70,
    eval_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into golden test set + train/eval/test for v1 dataset."""
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Golden test set: first N cases (diverse sample after shuffle)
    golden = shuffled[:golden_size]
    remaining = shuffled[golden_size:]

    # Split remaining into train/eval/test
    n = len(remaining)
    train_end = int(n * train_ratio)
    eval_end = int(n * (train_ratio + eval_ratio))
    train = remaining[:train_end]
    eval_set = remaining[train_end:eval_end]
    test = remaining[eval_end:]

    return golden, train, eval_set, test


def _row_to_record(row: pd.Series) -> dict:
    """Convert a DataFrame row to a JSON-serializable dict."""
    record: dict = {"case_id": row["case_id"]}

    # Features dict
    feature_cols = [
        "text_length",
        "word_count",
        "avg_sentence_length",
        "entity_count",
        "unique_entity_types",
        "has_crypto_wallet",
        "has_bank_account",
        "has_phone",
        "has_email",
        "indicator_count",
        "indicator_diversity",
        "max_indicator_confidence",
        "classification_axis_count",
        "current_classification_conf",
        "case_age_days",
    ]
    features = {}
    for col in feature_cols:
        val = row.get(col)
        if pd.isna(val):
            features[col] = None
        elif isinstance(val, bool | int | float):
            features[col] = val
        else:
            features[col] = val
    record["features"] = features

    # Labels
    record["labels"] = row["labels"]

    # Risk score
    record["risk_score"] = float(row["risk_score"]) if pd.notna(row.get("risk_score")) else None

    return record


def export_jsonl(df: pd.DataFrame, bucket_name: str, gcs_path: str) -> str:
    """Write DataFrame as JSONL to GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    lines = [json.dumps(_row_to_record(row), default=str) for _, row in df.iterrows()]
    blob.upload_from_string("\n".join(lines), content_type="application/jsonl")
    uri = f"gs://{bucket_name}/{gcs_path}"
    logger.info("Exported %d records → %s", len(df), uri)
    return uri


def register_dataset(bq: bigquery.Client, metadata: dict) -> None:
    """Register dataset version in training_dataset_registry."""
    table_ref = f"{PROJECT}.{DATASET}.training_dataset_registry"
    errors = bq.insert_rows_json(table_ref, [metadata])
    if errors:
        logger.error("Failed to register dataset: %s", errors)
    else:
        logger.info("Registered dataset: %s", metadata["dataset_id"])


def main() -> None:
    """Query BQ, split data, export to GCS, and register dataset v1."""
    bq = bigquery.Client(project=PROJECT)

    # Query and prepare data
    df = query_data(bq)
    if len(df) < 10:
        logger.error("Not enough labeled data (%d rows). Aborting.", len(df))
        sys.exit(1)

    # Log label distribution
    intent_counts: dict[str, int] = {}
    for labels in df["labels"]:
        intent = labels.get("INTENT", "NONE")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    logger.info("Intent distribution: %s", json.dumps(intent_counts, indent=2))

    # Split
    golden_size = min(100, len(df) // 5)  # At most 20% for golden set
    golden, train, eval_set, test = stratified_split(df, golden_size=golden_size)
    logger.info(
        "Split: golden=%d, train=%d, eval=%d, test=%d",
        len(golden),
        len(train),
        len(eval_set),
        len(test),
    )

    # Export golden test set
    export_jsonl(golden, BUCKET, "datasets/classification/golden/test.jsonl")

    # Export v1 training dataset
    export_jsonl(train, BUCKET, "datasets/classification/v1/train.jsonl")
    export_jsonl(eval_set, BUCKET, "datasets/classification/v1/eval.jsonl")
    export_jsonl(test, BUCKET, "datasets/classification/v1/test.jsonl")

    # Register v1 in training_dataset_registry
    label_counts = {}
    for labels in train["labels"]:
        for axis, code in labels.items():
            key = f"{axis}:{code}"
            label_counts[key] = label_counts.get(key, 0) + 1

    metadata = {
        "dataset_id": "classification-v1",
        "version": 1,
        "capability": "classification",
        "gcs_path": f"gs://{BUCKET}/datasets/classification/v1",
        "train_count": len(train),
        "eval_count": len(eval_set),
        "test_count": len(test),
        "label_distribution": json.dumps(label_counts),
        "config": json.dumps(
            {
                "source": "llm_classification_bootstrap",
                "golden_size": golden_size,
                "min_label_axes": 1,
            }
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "bootstrap",
    }
    register_dataset(bq, metadata)

    logger.info("✓ Bootstrap complete")


if __name__ == "__main__":
    main()
