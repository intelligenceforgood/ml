# Feature Store Runbook

Operations guide for syncing features to Vertex AI Feature Store and serving online features.

## Prerequisites

- `gcloud` authenticated with `i4g-ml` project
- Vertex AI Feature Store created (one-time Terraform setup)
- Feature tables populated in BigQuery (`features_case_features`, `features_graph_features`)

## Initial Setup (One-Time)

Create the Feature Store and entity type via Terraform or CLI:

```bash
# Via Terraform (preferred)
cd infra/environments/ml
terraform apply -target=module.feature_store

# Or via gcloud
gcloud ai featurestores create i4g-features \
  --project=i4g-ml --region=us-central1 \
  --online-serving-config-fixed-node-count=1
```

## Sync Features to Feature Store

### Full Sync (Dev)

```bash
i4g-ml dataset sync-features \
  --env dev \
  --full-sync
```

Estimated runtime: ~10 minutes for 10K cases.

### Incremental Sync

```bash
# Uses watermark to sync only new/changed features since last run
i4g-ml dataset sync-features --env dev
```

### Makefile Target

```bash
make sync-features-dev
```

## Set Feature Store Environment Variable

```bash
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --set-env-vars="FEATURE_STORE_ID=i4g-features"
```

## Verify Online Serving

```bash
# Test feature retrieval
curl -s https://ml-serving-<hash>.run.app/health | jq .

# Check Feature Store entity count
gcloud ai featurestores entity-types describe case_features \
  --featurestore=i4g-features \
  --project=i4g-ml --region=us-central1
```

## Monitor Cache Performance

The serving container caches Feature Store reads with a 60-second TTL and 128-entry LRU.
Monitor cache hit rate in Cloud Run logs:

```bash
gcloud logging read 'resource.type="cloud_run_revision" AND textPayload=~"feature_cache"' \
  --project=i4g-ml --limit=50
```

## Scheduled Sync (Production)

Set up Cloud Scheduler to run incremental sync daily:

```bash
gcloud scheduler jobs create http sync-features-daily \
  --project=i4g-ml \
  --schedule="0 6 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/i4g-ml/jobs/feature-sync-job:run" \
  --time-zone="UTC" \
  --http-method=POST
```

## Troubleshooting

| Symptom                              | Cause                           | Fix                               |
| ------------------------------------ | ------------------------------- | --------------------------------- |
| `fetch_online_features` returns None | `FEATURE_STORE_ID` not set      | Set env var on Cloud Run          |
| Stale features returned              | Cache TTL (60s)                 | Wait or restart container         |
| Sync job fails                       | BQ source tables empty          | Run feature materialization first |
| High latency on feature reads        | Feature Store under-provisioned | Increase node count in Terraform  |
