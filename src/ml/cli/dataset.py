"""Dataset management commands."""

from __future__ import annotations

import logging

import typer

logger = logging.getLogger(__name__)

dataset_app = typer.Typer(help="Dataset creation, versioning, and export.")


@dataset_app.command("bootstrap")
def bootstrap(
    project: str = typer.Option("i4g-ml", "--project", "-p", help="GCP project ID."),
    bucket: str = typer.Option("i4g-ml-data", "--bucket", "-b", help="GCS bucket for datasets."),
    dataset: str = typer.Option("i4g_ml", "--dataset", "-d", help="BigQuery dataset ID."),
) -> None:
    """Bootstrap training dataset from existing LLM classifications.

    Queries raw_cases + features, parses labels, creates golden test set
    and v1 train/eval/test splits, exports as JSONL to GCS, and registers
    in the training_dataset_registry.
    """
    import json
    from datetime import UTC, datetime

    import pandas as pd
    from google.cloud import bigquery, storage

    axes = ["intent", "channel", "techniques", "actions", "persona"]

    def _extract_top_labels(classification_result: str | None) -> dict[str, str]:
        if not classification_result:
            return {}
        try:
            cr = json.loads(classification_result)
        except (json.JSONDecodeError, TypeError):
            return {}
        labels: dict[str, str] = {}
        for axis in axes:
            items = cr.get(axis, [])
            if not items:
                continue
            top = max(items, key=lambda x: x.get("confidence", 0.0))
            label = top.get("label", "")
            if label:
                labels[axis.upper()] = label
        return labels

    bq_client = bigquery.Client(project=project)

    sql = f"""
        SELECT c.case_id, c.classification_result, c.status, c.risk_score,
               f.text_length, f.word_count, f.avg_sentence_length, f.entity_count,
               f.unique_entity_types, f.has_crypto_wallet, f.has_bank_account,
               f.has_phone, f.has_email, f.indicator_count, f.indicator_diversity,
               f.max_indicator_confidence, f.classification_axis_count,
               f.current_classification_conf, f.case_age_days
        FROM `{project}.{dataset}.raw_cases` c
        JOIN `{project}.{dataset}.features_case_features` f ON c.case_id = f.case_id
    """
    df = bq_client.query(sql).to_dataframe()
    typer.echo(f"Queried {len(df)} rows from BQ")

    df["labels"] = df["classification_result"].apply(_extract_top_labels)
    df["label_count"] = df["labels"].apply(len)
    labeled = df[df["label_count"] > 0].copy().drop(columns=["classification_result"])
    typer.echo(f"Cases with labels: {len(labeled)} / {len(df)}")

    if len(labeled) < 10:
        typer.echo("Not enough labeled data. Aborting.", err=True)
        raise typer.Exit(code=1)

    shuffled = labeled.sample(frac=1, random_state=42).reset_index(drop=True)
    golden_size = min(100, len(labeled) // 5)
    golden = shuffled[:golden_size]
    remaining = shuffled[golden_size:]
    n = len(remaining)
    train = remaining[: int(n * 0.70)]
    eval_set = remaining[int(n * 0.70) : int(n * 0.85)]
    test = remaining[int(n * 0.85) :]

    typer.echo(f"Split: golden={len(golden)}, train={len(train)}, eval={len(eval_set)}, test={len(test)}")

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

    def _row_to_record(row: pd.Series) -> dict:
        record: dict = {"case_id": row["case_id"]}
        features = {}
        for col in feature_cols:
            val = row.get(col)
            features[col] = None if pd.isna(val) else val
        record["features"] = features
        record["labels"] = row["labels"]
        record["risk_score"] = float(row["risk_score"]) if pd.notna(row.get("risk_score")) else None
        return record

    def _export(df_part: pd.DataFrame, gcs_path: str) -> str:
        gcs_client = storage.Client()
        bkt = gcs_client.bucket(bucket)
        blob = bkt.blob(gcs_path)
        lines = [json.dumps(_row_to_record(row), default=str) for _, row in df_part.iterrows()]
        blob.upload_from_string("\n".join(lines), content_type="application/jsonl")
        uri = f"gs://{bucket}/{gcs_path}"
        typer.echo(f"Exported {len(df_part)} records → {uri}")
        return uri

    _export(golden, "datasets/classification/golden/test.jsonl")
    _export(train, "datasets/classification/v1/train.jsonl")
    _export(eval_set, "datasets/classification/v1/eval.jsonl")
    _export(test, "datasets/classification/v1/test.jsonl")

    label_counts: dict[str, int] = {}
    for labels in train["labels"]:
        for axis, code in labels.items():
            key = f"{axis}:{code}"
            label_counts[key] = label_counts.get(key, 0) + 1

    metadata = {
        "dataset_id": "classification-v1",
        "version": 1,
        "capability": "classification",
        "gcs_path": f"gs://{bucket}/datasets/classification/v1",
        "train_count": len(train),
        "eval_count": len(eval_set),
        "test_count": len(test),
        "label_distribution": json.dumps(label_counts),
        "config": json.dumps(
            {"source": "llm_classification_bootstrap", "golden_size": golden_size, "min_label_axes": 1}
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "bootstrap",
    }
    table_ref = f"{project}.{dataset}.training_dataset_registry"
    errors = bq_client.insert_rows_json(table_ref, [metadata])
    if errors:
        typer.echo(f"Failed to register dataset: {errors}", err=True)
    else:
        typer.echo(f"Registered dataset: {metadata['dataset_id']}")

    typer.echo("Bootstrap complete.")


@dataset_app.command("create")
def create(
    capability: str = typer.Option(
        "classification", "--capability", "-c", help="ML capability (classification, ner, risk_scoring)."
    ),
    version: int | None = typer.Option(None, "--version", "-v", help="Version number (auto-increments if omitted)."),
    redact: bool = typer.Option(True, "--redact/--no-redact", help="Redact PII before export."),
    min_samples: int = typer.Option(50, "--min-samples", help="Minimum samples per class."),
) -> None:
    """Create a new versioned dataset: query BQ, validate, split, export to GCS."""
    from ml.data.datasets import create_dataset_version

    result = create_dataset_version(
        capability=capability,
        version=version,
        redact=redact,
        min_samples_per_class=min_samples,
    )
    typer.echo(f"Dataset created: {result['dataset_id']} → {result['gcs_path']}")
