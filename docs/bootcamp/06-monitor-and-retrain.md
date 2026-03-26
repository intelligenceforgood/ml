# Exercise 6: Monitor and Retrain

> **Objective:** Trigger drift computation, read monitoring data, and manually invoke the retraining pipeline.
> **Prerequisites:** Exercise 5 completed (predictions and outcomes in BigQuery)
> **Time:** ~30 minutes

> **Requires GCP access.** Steps use `bq` queries and submit jobs to the live project.
> Without access, read through the exercise, then run
> `conda run -n ml pytest tests/unit/test_drift.py tests/unit/test_triggers.py tests/unit/test_trigger_retraining.py -v`
> to validate the monitoring logic locally.

---

## Overview

The monitoring layer tracks three things: prediction drift (are model outputs changing?), feature drift (are model inputs changing?), and accuracy (is the model still correct?). When drift or data volume thresholds are met, the retraining trigger fires automatically.

```
Scheduled jobs (daily)
├── drift.py → analytics_drift_metrics
├── accuracy.py → analytics_model_performance
└── cost.py → analytics_cost_summary

triggers.py evaluates:
├── New analyst labels ≥ 200? → retrain
├── Any axis PSI > 0.2? → retrain
├── > 30 days since last training? → retrain
└── force flag? → retrain
```

---

## Step 1: Understand drift detection

```bash
head -60 src/ml/monitoring/drift.py
```

**What to notice:**

- **PSI (Population Stability Index):** measures how much a distribution has shifted from baseline
- Threshold: PSI > 0.2 = drifted (standard practice for categorical distributions)
- Two types of drift: prediction drift (label distributions) and feature drift (input feature distributions)
- `DriftReport` wraps per-axis `PredictionDrift` and per-feature `FeatureDrift` results

---

## Step 2: Check existing drift metrics

```bash
bq query --use_legacy_sql=false \
  'SELECT model_id, report_type, axis_or_feature, psi, is_drifted, computed_at
   FROM `i4g-ml.i4g_ml.analytics_drift_metrics`
   ORDER BY computed_at DESC
   LIMIT 10'
```

If the table is empty, drift computation hasn't run yet. That's what you'll trigger next.

---

## Step 3: Understand retraining triggers

```bash
head -60 src/ml/monitoring/triggers.py
```

**What to notice:**

- `MIN_ANALYST_LABELS = 200`: retrain when 200+ new analyst labels are available
- `DRIFT_PSI_THRESHOLD = 0.2`: retrain when any axis drifts above this threshold
- `MAX_DAYS_SINCE_TRAINING = 30`: retrain if the model is older than 30 days
- `force=True`: bypasses all conditions (useful for manual retraining)
- `RetrainingTrigger`: captures the decision + reasons for auditability

---

## Step 4: Check the trigger entry point

```bash
head -80 scripts/trigger_retraining.py
```

**What to notice:**

- Always exits with code 0 (Cloud Run Jobs treat exit code 1 as failure)
- Uses structured JSON logging for alerting (not exit codes)
- Calls `evaluate_retraining_conditions()` → if `should_retrain`, calls `submit_pipeline()`
- Records every evaluation in `analytics_trigger_log` for audit trail

---

## Step 5: Run the trigger manually (dry run)

```bash
conda run -n ml python scripts/trigger_retraining.py --capability classification
```

**Expected output:** Structured JSON log with either:

- `"action": "retrain_submitted"` — conditions met, pipeline submitted
- `"action": "retrain_skipped"` — conditions not met, reasons listed

Check the trigger log:

```bash
bq query --use_legacy_sql=false \
  'SELECT event_id, capability, should_retrain, reasons, pipeline_job_name, triggered_at
   FROM `i4g-ml.i4g_ml.analytics_trigger_log`
   ORDER BY triggered_at DESC
   LIMIT 5'
```

---

## Step 6: Force a retraining

To bypass conditions and force a retrain:

```bash
conda run -n ml python scripts/trigger_retraining.py --capability classification --force
```

This simulates the monthly forced retraining (Cloud Scheduler runs this on the 1st of each month).

---

## Step 7: Check accuracy monitoring

The accuracy pipeline computes model performance from predictions vs. outcomes:

```bash
bq query --use_legacy_sql=false \
  'SELECT model_id, model_version, capability, computed_at,
          accuracy, correction_rate, f1
   FROM `i4g-ml.i4g_ml.analytics_model_performance`
   ORDER BY computed_at DESC, model_version DESC
   LIMIT 5'
```

---

## Step 8: Check cost monitoring

The cost pipeline compares ML platform cost against LLM API baseline:

```bash
bq query --use_legacy_sql=false \
  'SELECT model_id, capability, ml_cost_per_prediction, llm_cost_per_prediction, savings_pct
   FROM `i4g-ml.i4g_ml.analytics_cost_summary`
   ORDER BY computed_at DESC
   LIMIT 5'
```

---

## Step 9: Understand the Cloud Scheduler setup

In production, these jobs run automatically:

| Job                | Schedule              | What it does                                                |
| ------------------ | --------------------- | ----------------------------------------------------------- |
| Drift computation  | Daily 6 AM UTC        | Computes drift metrics, writes to `analytics_drift_metrics` |
| Accuracy metrics   | Daily 5 AM UTC        | Computes accuracy from prediction+outcome logs              |
| Cost summary       | Daily 5:30 AM UTC     | Computes per-prediction cost comparison                     |
| Retraining trigger | Daily 6 AM UTC        | Evaluates conditions, submits pipeline if warranted         |
| Forced retrain     | Monthly 1st, 7 AM UTC | `--force` flag ensures at least monthly retraining          |

These are defined in Terraform: `infra/stacks/ml/main.tf`.

---

## Summary

| Component               | What it does                             | BigQuery table                       |
| ----------------------- | ---------------------------------------- | ------------------------------------ |
| `drift.py`              | Detects input/output distribution shifts | `analytics_drift_metrics`            |
| `triggers.py`           | Evaluates retraining conditions          | `analytics_trigger_log`              |
| `accuracy.py`           | Tracks model accuracy over time          | `analytics_model_performance`        |
| `cost.py`               | Compares ML vs. LLM costs                | `analytics_cost_summary`             |
| `trigger_retraining.py` | Cloud Run Job entry point                | Submits pipeline when conditions met |

**Next exercise:** [07 — Add a New Capability](07-add-new-capability.md), where you add a toy third ML capability end-to-end.
