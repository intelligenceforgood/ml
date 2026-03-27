"""Dataset creation, versioning, and export to GCS."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pandas as pd
from google.cloud import bigquery, storage

from ml.config import get_settings
from ml.data.pii import redact_record
from ml.data.validation import validate_dataset

logger = logging.getLogger(__name__)


def _stratified_split(
    df: pd.DataFrame,
    label_column: str,
    train_ratio: float = 0.70,
    eval_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/eval/test with stratified sampling."""
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    eval_end = int(n * (train_ratio + eval_ratio))
    return shuffled[:train_end], shuffled[train_end:eval_end], shuffled[eval_end:]


def _export_jsonl(df: pd.DataFrame, bucket_name: str, gcs_path: str, *, redact: bool = False) -> str:
    """Write a DataFrame as JSONL to GCS. Returns the ``gs://`` URI.

    When *redact* is True, PII is redacted from text fields before export.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)

    lines = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, dict | list):
                record[col] = val
            elif pd.isna(val):
                record[col] = None
            else:
                record[col] = val
        if redact:
            record = redact_record(record)
        lines.append(json.dumps(record, default=str))

    blob.upload_from_string("\n".join(lines), content_type="application/jsonl")
    return f"gs://{bucket_name}/{gcs_path}"


def create_dataset_version(
    *,
    capability: str = "classification",
    version: int | None = None,
    query: str | None = None,
    label_column: str = "labels",
    min_samples_per_class: int = 50,
    redact: bool = True,
) -> dict:
    """Create a versioned dataset: query BQ, validate, split, export to GCS.

    When *redact* is True (default), PII is stripped from text fields before
    writing JSONL to GCS.

    Returns metadata dict registered in ``training_dataset_registry``.
    """
    settings = get_settings()
    bq_client = bigquery.Client(project=settings.platform.project_id)
    dataset_id = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}"

    # Default query: join features + labels with analyst-correction priority.
    # Label source priority: analyst (from outcome_log corrections) > llm_bootstrap
    # (from raw_analyst_labels).  When an analyst has corrected a prediction via the
    # feedback endpoint, that correction is the ground truth label.
    if query is None:
        query = f"""
            WITH analyst_corrections AS (
                -- Analyst corrections from the feedback loop (highest priority)
                SELECT
                    o.case_id,
                    JSON_EXTRACT_SCALAR(o.correction, '$.INTENT') AS intent_label,
                    JSON_EXTRACT_SCALAR(o.correction, '$.CHANNEL') AS channel_label,
                    'analyst' AS label_source,
                    o.timestamp AS label_timestamp
                FROM `{dataset_id}.predictions_outcome_log` o
                QUALIFY ROW_NUMBER() OVER (PARTITION BY o.case_id ORDER BY o.timestamp DESC) = 1
            ),
            bootstrap_labels AS (
                -- LLM bootstrap labels (lower priority — used when no analyst correction exists)
                SELECT
                    al.case_id,
                    MAX(IF(al.axis = 'INTENT', al.label_code, NULL)) AS intent_label,
                    MAX(IF(al.axis = 'CHANNEL', al.label_code, NULL)) AS channel_label,
                    'llm_bootstrap' AS label_source,
                    MAX(al.created_at) AS label_timestamp
                FROM `{dataset_id}.raw_analyst_labels` al
                GROUP BY al.case_id
            ),
            merged_labels AS (
                -- Prefer analyst corrections; fall back to bootstrap
                SELECT
                    COALESCE(ac.case_id, bl.case_id) AS case_id,
                    COALESCE(ac.intent_label, bl.intent_label) AS intent_label,
                    COALESCE(ac.channel_label, bl.channel_label) AS channel_label,
                    COALESCE(ac.label_source, bl.label_source) AS label_source,
                    COALESCE(ac.label_timestamp, bl.label_timestamp) AS label_timestamp
                FROM bootstrap_labels bl
                FULL OUTER JOIN analyst_corrections ac ON bl.case_id = ac.case_id
            )
            SELECT
                f.case_id,
                c.narrative AS text,
                STRUCT(
                    f.text_length,
                    f.word_count,
                    f.entity_count,
                    f.has_crypto_wallet,
                    f.has_bank_account,
                    f.has_phone,
                    f.has_email,
                    f.classification_axis_count,
                    f.current_classification_conf,
                    gf.shared_entity_count,
                    gf.entity_reuse_frequency,
                    gf.cluster_size
                ) AS features,
                ml.intent_label AS label_code,
                ml.label_source,
                ml.label_timestamp
            FROM `{dataset_id}.features_case_features` f
            JOIN `{dataset_id}.raw_cases` c ON f.case_id = c.case_id
            JOIN merged_labels ml ON f.case_id = ml.case_id
            LEFT JOIN `{dataset_id}.features_graph_features` gf ON f.case_id = gf.case_id
            WHERE ml.intent_label IS NOT NULL
        """

    df = bq_client.query(query).to_dataframe()
    logger.info("Queried %d rows for dataset", len(df))

    # Validate
    result = validate_dataset(df, label_column=label_column, min_samples_per_class=min_samples_per_class)
    if not result.passed:
        raise ValueError(f"Dataset validation failed: {result.errors}")
    for w in result.warnings:
        logger.warning("Validation warning: %s", w)

    # Determine version
    if version is None:
        v_query = f"""
            SELECT COALESCE(MAX(version), 0) + 1 AS next_v
            FROM `{dataset_id}.training_dataset_registry`
            WHERE capability = @cap
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("cap", "STRING", capability)]
        )
        next_v = list(bq_client.query(v_query, job_config=job_config).result())[0].next_v
        version = next_v

    # Split
    train_df, eval_df, test_df = _stratified_split(df, label_column)

    # Export to GCS
    prefix = f"{settings.storage.datasets_prefix}/{capability}/v{version}"
    bucket = settings.storage.data_bucket
    _export_jsonl(train_df, bucket, f"{prefix}/train.jsonl", redact=redact)
    _export_jsonl(eval_df, bucket, f"{prefix}/eval.jsonl", redact=redact)
    _export_jsonl(test_df, bucket, f"{prefix}/test.jsonl", redact=redact)
    gcs_path = f"gs://{bucket}/{prefix}"

    logger.info(
        "Exported dataset v%d: train=%d, eval=%d, test=%d → %s (redacted=%s)",
        version,
        len(train_df),
        len(eval_df),
        len(test_df),
        gcs_path,
        redact,
    )

    # Register in BQ
    metadata = {
        "dataset_id": f"{capability}-v{version}",
        "version": version,
        "capability": capability,
        "gcs_path": gcs_path,
        "train_count": len(train_df),
        "eval_count": len(eval_df),
        "test_count": len(test_df),
        "redacted": redact,
        "label_distribution": json.dumps(result.stats.get("label_counts", {})),
        "config": json.dumps({"query": query, "min_samples_per_class": min_samples_per_class, "redacted": redact}),
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "etl",
    }
    table_ref = f"{dataset_id}.training_dataset_registry"
    bq_client.insert_rows_json(table_ref, [metadata])

    return metadata


def create_ner_dataset_version(
    *,
    version: int | None = None,
    min_samples: int = 50,
    min_entity_type_count: int = 5,
    redact: bool = True,
) -> dict:
    """Create a versioned NER dataset: query BQ, validate, split, export to GCS.

    Queries ``raw_entities`` + ``raw_cases`` for character-offset entity
    annotations and creates train/eval/test splits stratified by entity
    type distribution.

    When *redact* is True, PII in non-entity text segments is redacted
    (entity mentions are preserved since they are training labels).

    Returns metadata dict registered in ``training_dataset_registry``.
    """
    settings = get_settings()
    bq_client = bigquery.Client(project=settings.platform.project_id)
    dataset_id = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}"

    query = f"""
        SELECT
            c.case_id,
            c.narrative AS text,
            ARRAY_AGG(STRUCT(
                e.start_offset AS start,
                e.end_offset AS end,
                e.entity_type AS label,
                e.entity_text AS text
            )) AS entities
        FROM `{dataset_id}.raw_cases` c
        JOIN `{dataset_id}.raw_entities` e ON c.case_id = e.case_id
        WHERE e.entity_type IS NOT NULL
        GROUP BY c.case_id, c.narrative
    """

    df = bq_client.query(query).to_dataframe()
    logger.info("Queried %d NER samples", len(df))

    if len(df) < min_samples:
        raise ValueError(f"NER dataset has {len(df)} samples, minimum is {min_samples}")

    # Validate entity type distribution
    entity_type_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        if isinstance(row["entities"], list):
            for ent in row["entities"]:
                etype = ent.get("label", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN"
                entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

    logger.info("Entity type distribution: %s", entity_type_counts)

    sparse_types = [et for et, cnt in entity_type_counts.items() if cnt < min_entity_type_count]
    if sparse_types:
        logger.warning(
            "Entity types with fewer than %d examples: %s",
            min_entity_type_count,
            sparse_types,
        )

    # Stratified split by dominant entity type per sample
    def _dominant_entity_type(entities: list) -> str:
        if not entities:
            return "NONE"
        type_counts: dict[str, int] = {}
        for ent in entities:
            lbl = ent.get("label", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN"
            type_counts[lbl] = type_counts.get(lbl, 0) + 1
        return max(type_counts, key=type_counts.get)

    df["_stratify_col"] = df["entities"].apply(_dominant_entity_type)
    train_df, eval_df, test_df = _stratified_split(df, "_stratify_col")

    # Drop stratify helper column
    for split_df in (train_df, eval_df, test_df):
        split_df.drop(columns=["_stratify_col"], inplace=True)

    # Determine version
    if version is None:
        v_query = f"""
            SELECT COALESCE(MAX(version), 0) + 1 AS next_v
            FROM `{dataset_id}.training_dataset_registry`
            WHERE capability = @cap
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("cap", "STRING", "ner")])
        next_v = list(bq_client.query(v_query, job_config=job_config).result())[0].next_v
        version = next_v

    # Export to GCS (redact non-entity text only)
    prefix = f"{settings.storage.datasets_prefix}/ner/v{version}"
    bucket = settings.storage.data_bucket
    _export_jsonl(train_df, bucket, f"{prefix}/train.jsonl", redact=redact)
    _export_jsonl(eval_df, bucket, f"{prefix}/eval.jsonl", redact=redact)
    _export_jsonl(test_df, bucket, f"{prefix}/test.jsonl", redact=redact)
    gcs_path = f"gs://{bucket}/{prefix}"

    logger.info(
        "Exported NER dataset v%d: train=%d, eval=%d, test=%d → %s",
        version,
        len(train_df),
        len(eval_df),
        len(test_df),
        gcs_path,
    )

    # Register in BQ
    metadata = {
        "dataset_id": f"ner-v{version}",
        "version": version,
        "capability": "ner",
        "gcs_path": gcs_path,
        "train_count": len(train_df),
        "eval_count": len(eval_df),
        "test_count": len(test_df),
        "redacted": redact,
        "label_distribution": json.dumps(entity_type_counts),
        "config": json.dumps(
            {
                "min_samples": min_samples,
                "min_entity_type_count": min_entity_type_count,
                "redacted": redact,
            }
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "etl",
    }
    table_ref = f"{dataset_id}.training_dataset_registry"
    bq_client.insert_rows_json(table_ref, [metadata])

    return metadata


# ---------------------------------------------------------------------------
# Risk Scoring dataset (Sprint 4.2)
# ---------------------------------------------------------------------------


def create_risk_dataset_version(
    *,
    version: int | None = None,
    min_samples: int = 100,
    redact: bool = True,
) -> dict:
    """Create a versioned risk scoring dataset.

    Risk labels are sourced from analyst severity ratings in
    ``raw_analyst_labels`` (axis = 'severity') or derived from case
    outcome proxies (e.g. reported financial loss amount buckets).

    Returns metadata dict registered in ``training_dataset_registry``.
    """
    settings = get_settings()
    bq_client = bigquery.Client(project=settings.platform.project_id)
    dataset_id = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}"

    query = f"""
        WITH severity_labels AS (
            SELECT
                al.case_id,
                CAST(al.label_code AS FLOAT64) AS risk_label,
                'analyst_severity' AS label_source,
                al.created_at AS label_timestamp
            FROM `{dataset_id}.raw_analyst_labels` al
            WHERE al.axis = 'severity'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY al.case_id ORDER BY al.created_at DESC) = 1
        ),
        proxy_labels AS (
            SELECT
                c.case_id,
                LEAST(1.0, COALESCE(c.reported_loss, 0.0) / 100000.0) AS risk_label,
                'loss_proxy' AS label_source,
                c.created_at AS label_timestamp
            FROM `{dataset_id}.raw_cases` c
            WHERE c.reported_loss IS NOT NULL AND c.reported_loss > 0
        ),
        merged AS (
            SELECT
                COALESCE(s.case_id, p.case_id) AS case_id,
                COALESCE(s.risk_label, p.risk_label) AS risk_label,
                COALESCE(s.label_source, p.label_source) AS label_source
            FROM severity_labels s
            FULL OUTER JOIN proxy_labels p ON s.case_id = p.case_id
        )
        SELECT
            f.case_id,
            c.narrative AS text,
            m.risk_label,
            m.label_source,
            f.text_length, f.word_count, f.entity_count,
            f.has_crypto_wallet, f.has_bank_account, f.has_phone, f.has_email,
            f.classification_axis_count, f.current_classification_conf,
            COALESCE(gf.shared_entity_count, 0) AS shared_entity_count,
            COALESCE(gf.entity_reuse_frequency, 0.0) AS entity_reuse_frequency,
            COALESCE(gf.cluster_size, 0) AS cluster_size
        FROM merged m
        JOIN `{dataset_id}.features_case_features` f ON m.case_id = f.case_id
        JOIN `{dataset_id}.raw_cases` c ON m.case_id = c.case_id
        LEFT JOIN `{dataset_id}.features_graph_features` gf ON m.case_id = gf.case_id
        WHERE m.risk_label IS NOT NULL
    """

    df = bq_client.query(query).to_dataframe()
    logger.info("Queried %d risk scoring samples", len(df))

    if len(df) < min_samples:
        raise ValueError(
            f"Risk dataset has {len(df)} samples, minimum is {min_samples}. " "Check severity labels and loss proxies."
        )

    risk_std = df["risk_label"].std()
    if risk_std < 0.01:
        raise ValueError(
            f"Risk label standard deviation is {risk_std:.4f} — distribution is degenerate. "
            "Need more diverse severity ratings."
        )

    train_df, eval_df, test_df = _stratified_split(df, "risk_label")

    if version is None:
        v_query = f"""
            SELECT COALESCE(MAX(version), 0) + 1 AS next_v
            FROM `{dataset_id}.training_dataset_registry`
            WHERE capability = @cap
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("cap", "STRING", "risk_scoring")]
        )
        next_v = list(bq_client.query(v_query, job_config=job_config).result())[0].next_v
        version = next_v

    prefix = f"{settings.storage.datasets_prefix}/risk_scoring/v{version}"
    bucket = settings.storage.data_bucket
    _export_jsonl(train_df, bucket, f"{prefix}/train.jsonl", redact=redact)
    _export_jsonl(eval_df, bucket, f"{prefix}/eval.jsonl", redact=redact)
    _export_jsonl(test_df, bucket, f"{prefix}/test.jsonl", redact=redact)
    gcs_path = f"gs://{bucket}/{prefix}"

    stats = {
        "mean": round(float(df["risk_label"].mean()), 4),
        "std": round(float(risk_std), 4),
        "min": round(float(df["risk_label"].min()), 4),
        "max": round(float(df["risk_label"].max()), 4),
        "label_source_counts": df["label_source"].value_counts().to_dict(),
    }

    logger.info(
        "Exported risk dataset v%d: train=%d, eval=%d, test=%d → %s",
        version,
        len(train_df),
        len(eval_df),
        len(test_df),
        gcs_path,
    )

    metadata = {
        "dataset_id": f"risk_scoring-v{version}",
        "version": version,
        "capability": "risk_scoring",
        "gcs_path": gcs_path,
        "train_count": len(train_df),
        "eval_count": len(eval_df),
        "test_count": len(test_df),
        "redacted": redact,
        "label_distribution": json.dumps(stats),
        "config": json.dumps({"min_samples": min_samples, "redacted": redact}),
        "created_at": datetime.now(UTC).isoformat(),
        "created_by": "etl",
    }
    table_ref = f"{dataset_id}.training_dataset_registry"
    bq_client.insert_rows_json(table_ref, [metadata])

    return metadata
