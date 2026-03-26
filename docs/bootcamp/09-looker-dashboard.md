# Exercise 9: Looker Studio Dashboard

> **Objective:** Connect BigQuery analytics tables to Looker Studio and build a two-page monitoring dashboard.
> **Prerequisites:** BigQuery `analytics_*` tables populated (Exercises 5–6), Google account with Looker Studio access
> **Time:** ~45 minutes

> **Requires GCP access.** This exercise is entirely GUI-based in Looker Studio and BigQuery.
> Without access, read through the exercise to understand the monitoring dashboard design
> and which BigQuery tables feed each chart.

---

## Overview

The ML platform materializes monitoring data into four BigQuery tables. This exercise connects them to Looker Studio to build a dashboard with accuracy trends and cost comparison views.

**Data sources:**

| BigQuery table                | Content                                     | Refreshed         |
| ----------------------------- | ------------------------------------------- | ----------------- |
| `analytics_model_performance` | Per-model per-axis accuracy, override rate  | Daily 5 AM UTC    |
| `analytics_drift_metrics`     | PSI scores for prediction and feature drift | Daily 6 AM UTC    |
| `analytics_cost_summary`      | ML vs. LLM cost per prediction              | Daily 5:30 AM UTC |
| `predictions_prediction_log`  | Raw prediction records                      | Real-time         |

---

## Step 1: Verify data availability

Confirm the analytics tables have data:

```bash
# Accuracy metrics
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as row_count FROM `i4g-ml.i4g_ml.analytics_model_performance`'

# Drift metrics
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as row_count FROM `i4g-ml.i4g_ml.analytics_drift_metrics`'

# Cost summary
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as row_count FROM `i4g-ml.i4g_ml.analytics_cost_summary`'

# Prediction log
bq query --use_legacy_sql=false \
  'SELECT COUNT(*) as row_count FROM `i4g-ml.i4g_ml.predictions_prediction_log`'
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

> You will also add custom BigQuery queries as data sources in Steps 4 and 5.
> Those are added inline when you need them.

---

## Step 4: Build Page 1 — Accuracy Dashboard

This page uses two data sources: `analytics_model_performance` and `analytics_drift_metrics`.

### Chart 1: Rolling F1 per model (Line chart)

1. Click **Add a chart** → **Time series** (the line-chart icon)
2. Draw a rectangle in the upper half of the page
3. In the **Setup** panel on the right:
   - **Data source:** `analytics_model_performance`
   - **Date range dimension:** `computed_at`
   - **Dimension:** `computed_at` (this should auto-populate)
   - **Breakdown dimension:** click **Add dimension** → select `model_id`
   - **Metric:** click the default metric, change it to `f1` → set **Aggregation** to `Average`
4. In the **Style** panel:
   - Set **Missing data** to "Line to zero" (avoids gaps on days with no data)

**Add a date-range control** so viewers can adjust the window:

1. Click **Add a control** → **Date range control**
2. Place it above the chart
3. Click the control → in **Setup**, set **Default date range** to "Last 30 days"

### Chart 2: Override rate trend (Bar chart)

The override rate is already pre-computed in `analytics_model_performance` as the
`correction_rate` column (fraction of predictions that analysts corrected). No custom
query or table join is needed.

1. Click **Add a chart** → **Bar** → **Column chart**
2. Draw a rectangle below the F1 line chart
3. In the **Setup** panel:
   - **Data source:** `analytics_model_performance`
   - **Dimension:** `computed_at`
   - **Breakdown dimension:** `model_id`
   - **Metric:** `correction_rate` → set **Aggregation** to `Average`
4. In the **Style** panel:
   - **Axes** → set Y-axis label to "Override Rate"
   - Optionally set **Number format** to "Percent" (values are 0–1 decimals)

### Chart 3: Confusion matrix (Table — latest period)

This chart uses a custom BigQuery query as its data source so it only shows the
most recent evaluation period.

1. Click **Add data** (the cylinder icon in the toolbar) → **BigQuery** → **Custom query**
2. Select project `i4g-ml` and paste this query:

   ```sql
   SELECT model_id, model_version, f1, accuracy, correction_rate, computed_at
   FROM `i4g-ml.i4g_ml.analytics_model_performance` t
   WHERE computed_at = (
     SELECT MAX(computed_at)
     FROM `i4g-ml.i4g_ml.analytics_model_performance`
     WHERE model_id = t.model_id
   )
   ```

3. Click **Add** → then **Add to report**
4. Click **Add a chart** → **Table**
5. Draw a rectangle in the lower-left area
6. In **Setup**:
   - **Data source:** select the custom query you just added
   - **Dimensions:** `model_id`, `model_version`
   - **Metrics:** `f1`, `accuracy`, `correction_rate`

### Chart 4: Drift indicators (Scorecard cards)

Scorecards display a single number. You will create one per axis (e.g., two
scorecards if you have two axes).

1. Click **Add a chart** → **Scorecard**
2. Draw a small rectangle in the lower-right area
3. In **Setup**:
   - **Data source:** `analytics_drift_metrics`
   - **Metric:** `psi` → set **Aggregation** to `Max`
   - **Add a filter:** click **Add a filter** → create filter where
     `axis_or_feature` **equals** the axis name you want (e.g., `fraud_type`)
4. Duplicate the scorecard (Ctrl/Cmd+D) for each additional axis, updating the
   filter value
5. To add conditional formatting, in the **Style** panel:
   - Click **Conditional formatting** → **Add**
   - **Color if value:** is less than 0.1 → green background
   - Add another rule: is between 0.1 and 0.2 → yellow background
   - Add another rule: is greater than 0.2 → red background

---

## Step 5: Build Page 2 — Cost Dashboard

Add a second page: in the toolbar click **Page** → **New page**. All charts on this
page use the `analytics_cost_summary` data source.

### Chart 1: Per-prediction cost comparison (Bar chart)

1. Click **Add a chart** → **Bar** → **Column chart**
2. Draw a rectangle in the upper half of the page
3. In **Setup**:
   - **Data source:** `analytics_cost_summary`
   - **Dimension:** `capability`
   - **Metrics:** add both `ml_cost_per_prediction` and `llm_cost_per_prediction`
     → set **Aggregation** to `Average` for each
4. In **Style**:
   - The two metrics appear as side-by-side bars automatically
   - Set bar colors to distinguish ML (e.g., blue) vs. LLM (e.g., orange)

### Chart 2: Cumulative savings (Line chart)

This chart needs a calculated field because the savings column doesn't exist
in the raw table.

1. In the **Data** panel (bottom-left), click **analytics_cost_summary** →
   **Add a field** (pencil icon or "+" button)
2. Name: `savings`, Formula: `llm_total - ml_total` → click **Save**
3. Click **Add a chart** → **Time series**
4. In **Setup**:
   - **Date range dimension:** `period_end`
   - **Dimension:** `period_end`
   - **Metric:** `savings` → set **Aggregation** to **Sum**
   - Check **Cumulative** under the metric options (or toggle "Running total"
     in the **Style** tab if available in your Looker Studio version)

### Chart 3: Cost breakdown by GCP component (Pie chart)

> This chart requires GCP billing export data. If you haven't set up
> [Cloud Billing export to BigQuery](https://cloud.google.com/billing/docs/how-to/export-data-bigquery),
> skip this chart — the data source won't exist yet.

1. Click **Add data** → **BigQuery** → **Custom query** → project `i4g-ml`
2. Paste:

   ```sql
   SELECT service.description AS component,
          SUM(cost) AS total_cost
   FROM `i4g-ml.billing_export.gcp_billing_export_v1_*`
   WHERE project.id = 'i4g-ml'
     AND usage_start_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
   GROUP BY component
   ORDER BY total_cost DESC
   ```

3. Click **Add** → **Add to report**
4. Click **Add a chart** → **Pie chart**
5. In **Setup**:
   - **Dimension:** `component`
   - **Metric:** `total_cost` → **Aggregation:** `Sum`

---

## Step 6: Add filters and controls

Add interactive controls so viewers can slice the data without editing charts.

1. **Date range filter** (both pages):
   - Click **Add a control** → **Date range control** → place at top of each page
   - Set **Default date range** to "Last 30 days"
2. **Model version filter** (Accuracy page):
   - Click **Add a control** → **Drop-down list**
   - **Data source:** `analytics_model_performance`
   - **Control field:** `model_id`
   - Place near the top of Page 1
3. **Capability filter** (Cost page):
   - Click **Add a control** → **Drop-down list**
   - **Data source:** `analytics_cost_summary`
   - **Control field:** `capability`
   - Place near the top of Page 2

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
SELECT model_id, model_version, f1, accuracy, correction_rate, computed_at
FROM `i4g-ml.i4g_ml.analytics_model_performance` t
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

| Page     | Charts                                                             | Data source                                                           |
| -------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Accuracy | Rolling F1 line, override rate bar, confusion matrix, drift scores | `analytics_model_performance`, `analytics_drift_metrics`              |
| Cost     | Per-prediction cost bar, cumulative savings line, component pie    | `analytics_cost_summary`, GCP billing export (optional for pie chart) |

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
