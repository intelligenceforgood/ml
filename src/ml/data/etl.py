"""ETL pipeline: Cloud SQL → BigQuery incremental sync.

Watermark-based incremental ingest from I4G Cloud SQL to BigQuery ``raw.*``
tables.  Each source table is synced independently using MERGE on the primary
key for idempotency.
"""

from __future__ import annotations

import json as _json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from google.cloud import bigquery
from sqlalchemy import create_engine, text

from ml.config import EtlSettings, get_settings

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingest configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestConfig:
    """Describes how one source table maps to a BigQuery raw table."""

    source_table: str
    target_table: str
    primary_key: str
    watermark_column: str
    columns: list[str] = field(default_factory=list)


TABLE_CONFIGS: list[IngestConfig] = [
    IngestConfig(
        source_table="cases",
        target_table="raw_cases",
        primary_key="case_id",
        watermark_column="updated_at",
        columns=[
            "case_id",
            "classification_result",
            "status",
            "risk_score",
            "taxonomy_version",
            "created_at",
            "updated_at",
        ],
    ),
    IngestConfig(
        source_table="entities",
        target_table="raw_entities",
        primary_key="entity_id",
        watermark_column="created_at",
        columns=[
            "entity_id",
            "case_id",
            "entity_type",
            "canonical_value",
            "confidence",
            "created_at",
        ],
    ),
    IngestConfig(
        source_table="analyst_labels",
        target_table="raw_analyst_labels",
        primary_key="id",
        watermark_column="created_at",
        columns=["id", "case_id", "axis", "label_code", "analyst_id", "confidence", "notes", "created_at"],
    ),
]


# ---------------------------------------------------------------------------
# Watermark helpers
# ---------------------------------------------------------------------------


def _get_watermark(bq_client: bigquery.Client, dataset_id: str, config: IngestConfig) -> datetime | None:
    """Return the max ``_ingested_at`` from the BigQuery target table.

    Falls back to ``None`` (full ingest) if the table is empty or doesn't
    exist yet.
    """
    query = f"""
        SELECT MAX(_ingested_at) AS wm
        FROM `{dataset_id}.{config.target_table}`
    """
    try:
        rows = list(bq_client.query(query).result())
    except Exception:  # noqa: BLE001 — table may not exist yet on first run
        logger.info("Table %s.%s not queryable — doing full ingest", dataset_id, config.target_table)
        return None
    if rows and rows[0].wm is not None:
        return rows[0].wm
    return None


def _extract_rows(
    engine,
    config: IngestConfig,
    watermark: datetime | None,
    batch_size: int,
) -> list[dict]:
    """Extract rows from the source database newer than *watermark*."""
    col_list = ", ".join(config.columns)
    sql = f"SELECT {col_list} FROM {config.source_table}"  # noqa: S608
    params: dict = {}
    if watermark is not None:
        sql += f" WHERE {config.watermark_column} > :wm"
        params["wm"] = watermark
    sql += f" ORDER BY {config.watermark_column} LIMIT :batch_size"
    params["batch_size"] = batch_size

    with engine.connect() as conn:
        result = conn.execute(text(sql), params)
        return [_coerce_row(dict(row._mapping)) for row in result]


def _coerce_row(row: dict) -> dict:
    """Convert types that are not natively JSON-serializable."""
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, Decimal):
            out[k] = float(v)
        elif isinstance(v, datetime | date):
            out[k] = v.isoformat()
        elif isinstance(v, dict | list):
            out[k] = _json.dumps(v, default=str)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# BigQuery MERGE
# ---------------------------------------------------------------------------


def _merge_rows(
    bq_client: bigquery.Client,
    dataset_id: str,
    config: IngestConfig,
    rows: list[dict],
) -> int:
    """MERGE *rows* into the BigQuery target table. Returns rows affected."""
    if not rows:
        return 0

    staging_table = f"{dataset_id}._staging_{config.target_table}_{uuid.uuid4().hex[:8]}"

    # Use the target table's schema (minus _ingested_at) for the staging table
    # so types match exactly and MERGE doesn't hit type-mismatch errors.
    target_schema = bq_client.get_table(f"{dataset_id}.{config.target_table}").schema
    staging_schema = [f for f in target_schema if f.name != "_ingested_at"]

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=staging_schema,
    )
    bq_client.load_table_from_json(rows, staging_table, job_config=job_config).result()

    set_clause = ", ".join(f"T.{c} = S.{c}" for c in config.columns if c != config.primary_key)
    insert_cols = config.columns + ["_ingested_at"]
    insert_vals = [f"S.{c}" for c in config.columns] + [
        "CURRENT_TIMESTAMP()",
    ]

    merge_sql = f"""
        MERGE `{dataset_id}.{config.target_table}` T
        USING `{staging_table}` S
        ON T.{config.primary_key} = S.{config.primary_key}
        WHEN MATCHED THEN
            UPDATE SET {set_clause},
                       T._ingested_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT ({', '.join(insert_cols)})
            VALUES ({', '.join(insert_vals)})
    """
    bq_client.query(merge_sql).result()

    bq_client.delete_table(staging_table, not_found_ok=True)
    return len(rows)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _build_engine(etl: EtlSettings) -> Engine:
    """Create a SQLAlchemy engine from ETL settings.

    When ``source_instance`` is set (Cloud Run environment), uses the Cloud SQL
    Python Connector with IAM authentication.  Otherwise falls back to
    ``source_db_connection`` as a plain SQLAlchemy URL (local dev).
    """
    if etl.source_instance:
        from google.cloud.sql.connector import Connector

        connector = Connector()

        def _getconn():
            return connector.connect(
                etl.source_instance,
                "pg8000",
                user=etl.source_db_user,
                db=etl.source_db_name,
                enable_iam_auth=etl.source_enable_iam_auth,
            )

        return create_engine("postgresql+pg8000://", creator=_getconn)

    if etl.source_db_connection:
        return create_engine(etl.source_db_connection)

    raise ValueError(
        "ETL source is not configured. Set either source_instance " "(Cloud SQL) or source_db_connection (direct URL)."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_incremental_ingest(
    *,
    table_configs: list[IngestConfig] | None = None,
    source_connection: str | None = None,
) -> dict[str, int]:
    """Run watermark-based incremental ETL for all configured tables.

    Returns a mapping of ``{target_table: rows_merged}``.
    """
    settings = get_settings()
    configs = table_configs or TABLE_CONFIGS

    engine = create_engine(source_connection) if source_connection else _build_engine(settings.etl)

    bq_client = bigquery.Client(project=settings.platform.project_id)
    dataset_id = f"{settings.platform.project_id}.{settings.bigquery.dataset_id}"

    results: dict[str, int] = {}
    for config in configs:
        try:
            watermark = _get_watermark(bq_client, dataset_id, config)
            logger.info(
                "Ingesting %s → %s (watermark=%s)",
                config.source_table,
                config.target_table,
                watermark,
            )
            rows = _extract_rows(engine, config, watermark, settings.etl.batch_size)
            merged = _merge_rows(bq_client, dataset_id, config, rows)
            results[config.target_table] = merged
            logger.info("Merged %d rows into %s", merged, config.target_table)
        except Exception:  # noqa: BLE001 — one table failure should not abort the entire ETL run
            logger.exception("Failed to ingest %s", config.source_table)
            results[config.target_table] = -1

    return results


if __name__ == "__main__":
    results = run_incremental_ingest()
    failed = {k: v for k, v in results.items() if v < 0}
    if failed:
        logger.error("ETL completed with failures: %s", list(failed.keys()))
        raise SystemExit(1)
    logger.info("ETL completed: %s", results)
