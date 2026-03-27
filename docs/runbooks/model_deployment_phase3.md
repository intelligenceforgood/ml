# Model Deployment Runbook — Phase 3 Capabilities

Deployment procedures for Phase 3 models: risk scoring, similarity search, and challenger models.

## Prerequisites

- Model artifacts uploaded to GCS
- Serving container image built with Phase 3 code
- BigQuery schema migrations applied

## Pre-Deployment Checklist

1. Run unit tests: `pytest tests/unit -x`
2. Build serving container: `make build-serve-dev`
3. Apply BQ schema migrations (see below)

## BigQuery Schema Migrations

```bash
# Sprint 1: Add variant column
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/alter_prediction_log_add_variant.sql

# Sprint 1.3: Variant comparison table
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/analytics_variant_comparison.sql

# Sprint 2: Batch predictions table
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/batch_predictions.sql

# Sprint 5: Case embeddings table
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/case_embeddings.sql
```

## Deploy Serving Container with New Capabilities

### Risk Scoring

```bash
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --set-env-vars="RISK_MODEL_ARTIFACT_URI=gs://i4g-ml-data/models/risk-scoring-xgboost-v1/v1"
```

Verification:

```bash
curl -s https://ml-serving-<hash>.run.app/health | jq '.risk_active'
# Should return: true

curl -X POST https://ml-serving-<hash>.run.app/predict/risk-score \
  -H "Content-Type: application/json" \
  -d '{"text": "test case", "case_id": "test-001"}'
```

### Similarity Search

```bash
# 1. Generate embeddings for existing cases
i4g-ml serve batch \
  --capability embedding \
  --source-query "SELECT case_id, narrative FROM \`i4g-ml.i4g_ml.raw_cases\`" \
  --dest-table i4g-ml.i4g_ml.case_embeddings \
  --batch-size 50

# 2. Enable similarity on serving container
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --set-env-vars="SIMILARITY_ENABLED=true,EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2"
```

Estimated startup time: ~3 minutes (rebuilds FAISS index from BQ).

Verification:

```bash
curl -X POST https://ml-serving-<hash>.run.app/predict/similar-cases \
  -H "Content-Type: application/json" \
  -d '{"text": "test case about romance scam", "case_id": "test-001", "k": 5}'
```

### Champion/Challenger Routing

See [ab_routing.md](ab_routing.md) for detailed A/B routing procedures.

### Cost-Aware Routing

```bash
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --set-env-vars="COST_AWARE_ROUTING=true"
```

Requires `analytics_cost_summary` table populated by the cost monitoring job.

## Rollback Procedure

```bash
# Revert to previous revision
gcloud run services update-traffic ml-serving \
  --project=i4g-ml --region=us-central1 \
  --to-revisions=<previous-revision>=100
```

## Troubleshooting

| Symptom                         | Cause                                 | Fix                                            |
| ------------------------------- | ------------------------------------- | ---------------------------------------------- |
| 503 on `/predict/risk-score`    | Risk model not loaded                 | Check `RISK_MODEL_ARTIFACT_URI` and logs       |
| 503 on `/predict/similar-cases` | Similarity index empty                | Run embedding batch job first                  |
| Slow startup                    | Loading multiple models + FAISS index | Increase Cloud Run startup timeout             |
| Memory errors                   | Too many models loaded                | Increase instance memory or use smaller models |
