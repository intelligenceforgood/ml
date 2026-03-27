# Batch Prediction Runbook

Operations guide for running batch prediction jobs to backfill historical cases.

## Prerequisites

- `gcloud` authenticated with `i4g-ml` project
- Model artifact available in GCS
- Source data available in BigQuery

## Run Batch Prediction (Dev)

### Classification

```bash
i4g-ml serve batch \
  --capability classification \
  --model-artifact-uri gs://i4g-ml-data/models/classification-xgboost-v1/v3 \
  --source-query "SELECT case_id, narrative FROM \`i4g-ml.i4g_ml.raw_cases\` LIMIT 1000" \
  --dest-table i4g-ml.i4g_ml.batch_predictions \
  --batch-size 100
```

Estimated runtime: ~5 minutes per 1,000 cases (classification), ~15 minutes per 1,000 (NER).

### NER

```bash
i4g-ml serve batch \
  --capability ner \
  --model-artifact-uri gs://i4g-ml-data/models/ner-bert/v1 \
  --source-query "SELECT case_id, narrative FROM \`i4g-ml.i4g_ml.raw_cases\`" \
  --dest-table i4g-ml.i4g_ml.batch_predictions_ner \
  --batch-size 50
```

### Risk Scoring

```bash
i4g-ml serve batch \
  --capability risk_scoring \
  --model-artifact-uri gs://i4g-ml-data/models/risk-scoring-xgboost-v1/v1 \
  --source-query "SELECT case_id, narrative FROM \`i4g-ml.i4g_ml.raw_cases\`" \
  --dest-table i4g-ml.i4g_ml.batch_predictions_risk \
  --batch-size 200
```

### Embedding Generation

```bash
i4g-ml serve batch \
  --capability embedding \
  --source-query "SELECT case_id, narrative FROM \`i4g-ml.i4g_ml.raw_cases\`" \
  --dest-table i4g-ml.i4g_ml.case_embeddings \
  --batch-size 50
```

Estimated runtime: ~20 minutes per 1,000 cases (embeddings).

## Run as Cloud Run Job (Production)

```bash
gcloud run jobs execute batch-prediction-job \
  --project=i4g-ml --region=us-central1 \
  --set-env-vars="CAPABILITY=classification,MODEL_ARTIFACT_URI=gs://..."
```

## Verify Results

```sql
SELECT capability, COUNT(*) as count, MIN(predicted_at), MAX(predicted_at)
FROM `i4g-ml.i4g_ml.batch_predictions`
GROUP BY capability;
```

## Troubleshooting

| Symptom              | Cause                                     | Fix                                                    |
| -------------------- | ----------------------------------------- | ------------------------------------------------------ |
| Job fails at startup | Model artifact not found                  | Verify GCS path exists                                 |
| Slow throughput      | Batch size too small or GPU not available | Increase `--batch-size`                                |
| BQ write errors      | Schema mismatch                           | Run DDL in `pipelines/sql/batch_predictions.sql` first |
| OOM errors           | Batch size too large for NER/embeddings   | Reduce `--batch-size` to 25                            |
