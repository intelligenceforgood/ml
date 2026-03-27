# A/B Routing Runbook

Operations guide for champion/challenger A/B routing on the ML serving endpoint.

## Prerequisites

- `gcloud` authenticated with `i4g-ml` project
- Challenger model artifact uploaded to GCS
- Serving container deployed with routing support

## Enable Challenger Routing

1. **Set environment variables** on the Cloud Run service:

```bash
gcloud run services update ml-serving \
  --project=i4g-ml \
  --region=us-central1 \
  --set-env-vars="CHALLENGER_MODEL_ARTIFACT_URI=gs://i4g-ml-data/models/challenger/v1,CHALLENGER_TRAFFIC_WEIGHT=0.1,TRAFFIC_SPLIT_STRATEGY=random"
```

Estimated wait: ~2 minutes for new revision to deploy.

2. **Verify deployment:**

```bash
gcloud run services describe ml-serving \
  --project=i4g-ml --region=us-central1 \
  --format="value(spec.template.spec.containers[0].env)"
```

3. **Check health endpoint:**

```bash
curl -s https://ml-serving-<hash>.run.app/health | jq .
# Should show: "challenger_active": true
```

## Adjust Traffic Split

```bash
# Increase challenger traffic to 20%
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --update-env-vars="CHALLENGER_TRAFFIC_WEIGHT=0.2"
```

## Switch to Deterministic Routing

Deterministic routing hashes on `case_id`, so the same case always gets the same variant.

```bash
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --update-env-vars="TRAFFIC_SPLIT_STRATEGY=deterministic"
```

## Monitor Variant Performance

```sql
-- Check traffic distribution
SELECT variant, COUNT(*) as count
FROM `i4g-ml.i4g_ml.predictions_prediction_log`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY variant;
```

```bash
# Run variant comparison via CLI
i4g-ml eval variant-comparison --lookback-days 7
```

## Disable Challenger (Rollback)

```bash
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --update-env-vars="CHALLENGER_TRAFFIC_WEIGHT=0.0"
```

## Troubleshooting

| Symptom                              | Cause                                              | Fix                                                 |
| ------------------------------------ | -------------------------------------------------- | --------------------------------------------------- |
| `challenger_active: false` in health | Model failed to load                               | Check Cloud Run logs for load errors                |
| All traffic goes to champion         | `CHALLENGER_TRAFFIC_WEIGHT=0.0` or model not ready | Verify env var and logs                             |
| Uneven distribution                  | Small sample size or deterministic mode            | Wait for more traffic or switch to `random`         |
| OOM on startup                       | Champion + challenger too large                    | Increase Cloud Run memory or use smaller challenger |
