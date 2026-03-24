# Exercise 9: Looker Studio Dashboard

> **Objective:** Connect BigQuery analytics tables to Looker Studio and build a two-page monitoring dashboard.
> **Prerequisites:** BigQuery `analytics_*` tables populated (Exercises 5–6), Google account with Looker Studio access
> **Time:** ~45 minutes

---

## Overview

The ML platform materializes monitoring data into four BigQuery tables. This exercise connects them to Looker Studio to build a dashboard with accuracy trends and cost comparison views.

**Data sources:**

| BigQuery table | Content | Refreshed |
|---------------|---------|-----------|
| `analytics_model_performance` | Per-model per-axis accuracy, override rate | Daily 5 AM UTC |
| `analytics_drift_metrics` | PSI scores for prediction and feature drift | Daily 6 AM UTC |
| `analytics_cost_summary` | ML vs. LLM cost per prediction | Daily 5:30 AM UTC |
| `predictions_prediction_log` | Raw prediction records | Real-time |

---

## Step 1: Verify data availability

Confirm the analytics tables have data:

```bash
# Accuracy metrics
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as rows FROM `i4g-ml.i4g_ml.analytics_model_performance`'

# Drift metrics
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as rows FROM `i4g-ml.i4g_ml.analytics_drift_metrics`'

# Cost summary
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as rows FROM `i4g-ml.i4g_ml.analytics_cost_summary`'

# Prediction log
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as rows FROM `i4g-ml.i4g_ml.predictions_prediction_log`'
```

---

## Step 2: Review table schemas

Understand what columns are available for dashboard charts:

```bash
# Accuracy table
bq show --format=prettyjson i4g-ml:i4g_ml.analytics_model_performance | python -c "
import json, sys
schema = json.load(sys.stdin)['schema']['fields']
for f in schema: print(f'{f[\"name\"]:30s} {f[\"type\"]:10s} {f.get(\"description\", \"\")}')"

# Cost table
cat pipelines/sql/analytics_cost_summary.sql

# Drift table
cat pipelines/sql/analytics_drift_metrics.sql
```

---

## Step 3: Create a Looker Studio report

1. Open [Looker Studio](https://lookerstudio.google.com/)
2. Click **Create** → **Report**
3. Add data source: **BigQuery** → Project: `i4g-ml` → Dataset: `i4g_ml`

Add these four tables as data sources:
- `analytics_model_performance`
- `analytics_drift_metrics`
- `analytics_cost_summary`
- `predictions_prediction_log`

---

## Step 4: Build Page 1 — Accuracy Dashboard

**Charts to create:**

### Rolling F1 per model per axis (Line chart)
- **Dimension:** `computed_at` (Date)
- **Breakdown:** `model_id`
- **Metric:** `overall_f1`
- **Filter:** Date range selector (last 30 days default)

### Override rate trend (Bar chart)
- **Dimension:** `computed_at` (Date)
- **Metric:** Calculate override rate from `predictions_prediction_log` + `predictions_outcome_log`
- Shows how often analysts override model predictions

### Confusion matrix (Table - latest period)
- **Data source:** Custom query or calculated field
- **Shows:** Predicted vs. actual label distribution for the latest evaluation period

### Drift indicators (Scorecard cards)
- **Data source:** `analytics_drift_metrics`
- **Metric:** Latest PSI scores per axis
- **Conditional formatting:** Green (PSI < 0.1), Yellow (0.1–0.2), Red (> 0.2)

---

## Step 5: Build Page 2 — Cost Dashboard

### Per-prediction cost comparison (Bar chart by capability)
- **Dimension:** `capability`
- **Metrics:** `ml_cost_per_prediction`, `llm_cost_per_prediction`
- Side-by-side comparison showing ML platform savings

### Cumulative savings (Line chart)
- **Dimension:** `period_end` (Date)
- **Metric:** Calculate `(llm_total - ml_total)` as cumulative savings
- Shows total cost savings over time

### Cost breakdown by GCP component (Pie chart)
- **Data source:** Cost data broken down by component (Vertex AI, Cloud Run, BigQuery, Storage)
- Shows where ML platform costs are allocated

---

## Step 6: Add filters and controls

1. **Date range filter:** Add to both pages, default to last 30 days
2. **Model version filter:** Drop-down for `model_id` to compare specific models
3. **Capability filter:** For cost dashboard, filter by capability (classification, ner)

---

## Step 7: Configure scheduled email delivery

1. In Looker Studio, go to **File** → **Schedule email delivery**
2. Set frequency: weekly on Monday
3. Recipients: ML team distribution list
4. This ensures the team reviews metrics regularly without logging in

---

## Reference: BigQuery queries for dashboard

These queries can be used as custom data sources in Looker Studio:

**Latest accuracy per model:**
```sql
SELECT model_id, overall_f1, overall_precision, overall_recall, computed_at
FROM `i4g-ml.i4g_ml.analytics_model_performance`
WHERE computed_at = (
  SELECT MAX(computed_at) FROM `i4g-ml.i4g_ml.analytics_model_performance`
  WHERE model_id = t.model_id
)
```

**Drift status:**
```sql
SELECT model_id, axis_or_feature, psi, is_drifted, report_type, computed_at
FROM `i4g-ml.i4g_ml.analytics_drift_metrics`
WHERE computed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY psi DESC
```

**Cost savings:**
```sql
SELECT capability,
       SUM(ml_total) as total_ml_cost,
       SUM(llm_total) as total_llm_cost,
       SUM(llm_total - ml_total) as total_savings,
       AVG(savings_pct) as avg_savings_pct
FROM `i4g-ml.i4g_ml.analytics_cost_summary`
GROUP BY capability
```

---

## Summary

| Page | Charts | Data source |
|------|--------|-------------|
| Accuracy | Rolling F1 line, override rate bar, confusion matrix, drift scores | `analytics_model_performance`, `analytics_drift_metrics` |
| Cost | Per-prediction cost bar, cumulative savings line, component pie | `analytics_cost_summary` |

This dashboard provides at-a-glance visibility into ML platform health. Combined with Cloud Monitoring alerts (configured in Terraform), the team is notified within 24 hours of any model regression.

---

## Congratulations!

You've completed all 9 bootcamp exercises. You now understand the full ML platform lifecycle:

1. **Data flow:** Cloud SQL → BigQuery → GCS datasets
2. **Training:** Containerized training on Vertex AI
3. **Evaluation:** Per-axis metrics with eval gate promotion
4. **Serving:** Multi-capability FastAPI on Cloud Run
5. **Monitoring:** Drift, accuracy, and cost tracking
6. **Retraining:** Automated trigger → pipeline → promotion loop
7. **Extensibility:** Adding new capabilities follows a systematic pattern
8. **Graph features:** Dataflow/Beam for cross-case entity analysis
9. **Dashboards:** Looker Studio connected to BigQuery analytics tables
