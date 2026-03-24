# Retraining Runbook

This runbook covers the automated and manual retraining pipeline for the I4G ML Platform.

## Architecture

```
Cloud Scheduler (daily/monthly)
  └── Cloud Run Job: retrain-trigger
        ├── evaluate_retraining_conditions()
        │     ├── Data volume: ≥200 new analyst labels since last training
        │     ├── Drift: any axis PSI > 0.2 (from drift materialization)
        │     └── Time: >30 days since last training
        ├── If should_retrain → submit_pipeline()
        │     └── Vertex AI Pipeline Job
        └── record_trigger_event() → analytics_trigger_log (BigQuery)
```

## Automated Retraining

Two Cloud Scheduler triggers run automatically:

| Schedule             | Target                                                | Description                                        |
| -------------------- | ----------------------------------------------------- | -------------------------------------------------- |
| Daily 6 AM UTC       | `retrain-trigger --capability classification`         | Evaluates conditions; submits pipeline only if met |
| Monthly 1st 7 AM UTC | `retrain-trigger --capability classification --force` | Unconditional monthly retraining                   |

## Manual Trigger (Dev)

```bash
# Evaluate conditions and submit pipeline if warranted
make trigger-retrain-dev

# Force retraining regardless of conditions
conda run -n ml python scripts/trigger_retraining.py --capability classification --force
```

## E2E Test Procedure (Dev)

1. **Insert synthetic analyst labels** (≥200 rows to exceed the data volume threshold):

   ```sql
   INSERT INTO `i4g-ml.i4g_ml.raw_analyst_labels`
     (case_id, label_source, label_value, _ingested_at)
   SELECT
     CONCAT('test-', FORMAT('%03d', n)),
     'analyst',
     IF(MOD(n, 3) = 0, 'fraud', 'legitimate'),
     CURRENT_TIMESTAMP()
   FROM UNNEST(GENERATE_ARRAY(1, 250)) AS n;
   ```

2. **Run drift materialization** to populate drift metrics:

   ```bash
   conda run -n ml python -m ml.monitoring.drift --model-id classification-v1 --window-days 7
   ```

3. **Run retrain trigger** manually:

   ```bash
   make trigger-retrain-dev
   ```

4. **Verify trigger event** in BigQuery:

   ```sql
   SELECT * FROM `i4g-ml.i4g_ml.analytics_trigger_log`
   ORDER BY triggered_at DESC LIMIT 5;
   ```

   Confirm `should_retrain = true` and `reasons` includes `data_volume`.

5. **Verify pipeline submitted** — check Vertex AI Pipelines console or:

   ```bash
   gcloud ai pipelines list --project=i4g-ml --region=us-central1 --limit=5
   ```

6. **Wait for pipeline completion** — monitor in Vertex AI console. Confirm:
   - Model registered in Model Registry
   - Eval gate ran: model either promoted or rejection logged

## Monitoring

- **Structured log**: `action=retrain_submitted` — emitted on successful pipeline submission
- **Cloud Monitoring alert**: triggers notification on `action=retrain_submitted` (see Terraform)
- **BigQuery**: `analytics_trigger_log` — full history of trigger evaluations

## Troubleshooting

| Symptom                                | Cause                                          | Fix                                                      |
| -------------------------------------- | ---------------------------------------------- | -------------------------------------------------------- |
| Trigger runs but no pipeline submitted | Conditions not met                             | Check `analytics_trigger_log` for `should_retrain=false` |
| Pipeline submitted but fails           | Training container error                       | Check Vertex AI Pipeline logs                            |
| `retrain-trigger` Cloud Run Job fails  | Script error (should be rare — exits 0 always) | Check Cloud Run Job logs                                 |
| Trigger evaluates but no drift data    | Drift materialization not running              | Verify `drift-materialization` Cloud Run Job schedule    |
