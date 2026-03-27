# Exercise 1: Data Flow Walkthrough

> **Objective:** Trace a case from Cloud SQL through the entire ML data pipeline to a training-ready dataset on GCS.
> **Prerequisites:** conda env `ml` activated, `pip install -e ".[dev]"`, access to `i4g-ml` GCP project
> **Time:** ~30 minutes

> **Requires GCP access.** Most steps use `bq` and `gsutil` against live BigQuery/GCS data.
> Without access, read through the commands and their expected outputs to understand the flow,
> then run `pytest tests/unit/test_etl.py tests/unit/test_feature_definitions.py -v`
> to validate the underlying logic locally.

---

## Overview

The ML platform pulls case data from the core application's Cloud SQL database, transforms it through several stages, and produces versioned JSONL datasets on GCS for model training:

```
Cloud SQL (core DB) → ETL (incremental sync) → BigQuery raw tables
                                                      ↓
                                              Feature materialization
                                                      ↓
                                          BigQuery features table
                                                      ↓
                                          Dataset export (stratified split)
                                                      ↓
                                          GCS: train/eval/test JSONL files
```

---

## Step 1: Understand the ETL configuration

Open `src/ml/data/etl.py` and find the `TABLE_CONFIGS` list. Each `IngestConfig` maps a source table in Cloud SQL to a BigQuery raw table:

```bash
grep -A 8 "TABLE_CONFIGS" src/ml/data/etl.py | head -30
```

**What to notice:**

- `source_table`: the Cloud SQL table name (e.g., `cases`, `entities`, `analyst_labels`)
- `target_table`: the BigQuery destination (e.g., `raw_cases`, `raw_entities`, `raw_analyst_labels`)
- `primary_key`: used for MERGE-based idempotent upserts
- `watermark_column`: timestamp column for incremental ingest (only new/updated rows)

**What just happened:** You've seen how the ETL identifies what data to sync. The watermark pattern means each run only processes rows newer than the previous run — keeping syncs fast even as tables grow.

---

## Step 2: Explore BigQuery raw tables

Use the BigQuery CLI to check what tables exist:

```bash
bq ls i4g-ml:i4g_ml | head -20
```

Query a raw table to see the synced data:

```bash
bq query --use_legacy_sql=false \
  'SELECT case_id, classification_result, status, updated_at
   FROM `i4g-ml.i4g_ml.raw_cases`
   ORDER BY updated_at DESC
   LIMIT 5'
```

**What to notice:** The BigQuery tables mirror the Cloud SQL structure. Each row has an `_ingested_at` timestamp added by the ETL pipeline showing when it was synced.

---

## Step 3: Understand feature materialization

Open `src/ml/data/features.py` and examine the `FEATURE_CATALOG`:

```bash
grep -A 5 "FeatureDefinition(" src/ml/data/features.py | head -40
```

Features are computed from the raw tables via a SQL view. Check the view definition:

```bash
cat pipelines/sql/v_case_features.sql
```

Then check the materialized features table:

```bash
bq query --use_legacy_sql=false \
  'SELECT case_id, text_length, word_count, entity_count
   FROM `i4g-ml.i4g_ml.features_case_features`
   LIMIT 5'
```

**What just happened:** Features are defined declaratively in Python (`FEATURE_CATALOG`), computed via BigQuery SQL views, and materialized into a features table. This separation means you can add new features by adding a `FeatureDefinition` entry and the corresponding SQL.

---

## Step 4: Understand dataset creation

Open `src/ml/data/datasets.py` and read the `create_dataset_version()` function signature:

```bash
head -80 src/ml/data/datasets.py
```

Key behaviors:

- Queries BigQuery to join raw cases with features and analyst labels
- Applies PII redaction (via `redact_record()`) before export
- Stratified splitting: train (70%) / eval (15%) / test (15%)
- Exports as JSONL to GCS with versioned paths
- Registers the version in `training_dataset_registry` BigQuery table

Check existing dataset versions:

```bash
bq query --use_legacy_sql=false \
  'SELECT dataset_id, capability, version, eval_count, test_count, created_at
   FROM `i4g-ml.i4g_ml.training_dataset_registry`
   ORDER BY created_at DESC
   LIMIT 5'
```

List the exported files on GCS:

```bash
gsutil ls gs://i4g-ml-data/datasets/ | head -10
```

**What just happened:** You traced the full data path from raw ingestion to training-ready JSONL. Each dataset version is immutable and registered — you can always reproduce which data a model was trained on.

---

## Step 5: Inspect a dataset file

Download and inspect a few records from a JSONL dataset:

```bash
gsutil cat gs://i4g-ml-data/datasets/classification/v1/train.jsonl | head -3 | jq . || true
```

**What to notice:**

- `label_source` field: `analyst` (human-labeled) or `llm_bootstrap` (LLM-generated labels for early training)
- Text fields are PII-redacted
- Feature columns from the features table are included
- Label columns match the fraud taxonomy axes

---

## Summary

| Stage                   | Code                                | Output                                            |
| ----------------------- | ----------------------------------- | ------------------------------------------------- |
| ETL sync                | `data/etl.py`                       | BigQuery `raw_*` tables                           |
| Feature materialization | `pipelines/sql/v_case_features.sql` | BigQuery `features_case_features`                 |
| Dataset export          | `data/datasets.py`                  | GCS JSONL files + `training_dataset_registry` row |

**Next exercise:** [02 — Train a Model Locally](02-train-locally.md), where you use one of these datasets to train an XGBoost model.
