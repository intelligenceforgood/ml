# ML Platform Monitoring

## BigQuery Tables

All monitoring data lives in the `i4g_ml` BigQuery dataset in the `i4g-ml` project.

| Table                         | Purpose                      | Written by         |
| ----------------------------- | ---------------------------- | ------------------ |
| `predictions_prediction_log`  | Every prediction request     | Serving app        |
| `predictions_outcome_log`     | Analyst feedback/corrections | Serving app        |
| `analytics_model_performance` | Periodic eval metrics        | Evaluation harness |
| `training_dataset_registry`   | Dataset version metadata     | Dataset creation   |

## Key Monitoring Queries

### Prediction Volume (last 7 days)

```sql
SELECT
  DATE(timestamp) AS day,
  COUNT(*) AS predictions,
  COUNT(DISTINCT case_id) AS unique_cases
FROM `i4g-ml.i4g_ml.predictions_prediction_log`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY day
ORDER BY day DESC;
```

### Confidence Distribution

```sql
SELECT
  CASE
    WHEN confidence >= 0.9 THEN 'high (≥0.9)'
    WHEN confidence >= 0.7 THEN 'medium (0.7-0.9)'
    ELSE 'low (<0.7)'
  END AS confidence_band,
  COUNT(*) AS count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
FROM `i4g-ml.i4g_ml.predictions_prediction_log`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
GROUP BY confidence_band
ORDER BY confidence_band;
```

### Analyst Override Rate (accuracy proxy)

```sql
WITH predictions AS (
  SELECT prediction_id, case_id, predicted_label, axis
  FROM `i4g-ml.i4g_ml.predictions_prediction_log`
  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
),
outcomes AS (
  SELECT prediction_id, corrected_label
  FROM `i4g-ml.i4g_ml.predictions_outcome_log`
)
SELECT
  p.axis,
  COUNT(*) AS total,
  COUNTIF(o.corrected_label IS NOT NULL AND o.corrected_label != p.predicted_label) AS overridden,
  ROUND(COUNTIF(o.corrected_label IS NOT NULL AND o.corrected_label != p.predicted_label) * 100.0 / COUNT(*), 1) AS override_rate_pct
FROM predictions p
LEFT JOIN outcomes o USING (prediction_id)
GROUP BY p.axis
ORDER BY override_rate_pct DESC;
```

### Per-Axis F1 Over Time

```sql
SELECT
  model_id,
  model_version,
  computed_at,
  f1,
  accuracy,
  correction_rate,
  per_axis_metrics
FROM `i4g-ml.i4g_ml.analytics_model_performance`
ORDER BY computed_at DESC, model_version DESC;
```

### Latency Percentiles (last 24h)

```sql
SELECT
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)] AS p50,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(90)] AS p90,
  APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)] AS p99,
  COUNT(*) AS total_requests
FROM `i4g-ml.i4g_ml.predictions_prediction_log`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
```

## Alerting Thresholds

| Metric                     | Warning           | Critical          | Action                           |
| -------------------------- | ----------------- | ----------------- | -------------------------------- |
| Override rate              | > 20% over 7 days | > 30% over 7 days | Investigate label drift, retrain |
| Low-confidence predictions | > 40% of volume   | > 60% of volume   | Check input distribution shift   |
| Prod endpoint latency p90  | > 2 000 ms        | > 5 000 ms        | Scale replicas, check cold start |
| Prod endpoint error rate   | > 1%              | > 5%              | Check model load, container logs |

---

## Data Quality — Scheduled Queries

The following queries run as BigQuery scheduled queries for continuous data
observability. They power the Data Quality Dashboard (see below).

### Label Distribution per Axis (daily)

Scheduled: every day at 6 AM UTC.

```sql
-- Scheduled query: label_distribution_daily
SELECT
  CURRENT_DATE() AS report_date,
  al.axis,
  al.label_code,
  COUNT(*) AS label_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY al.axis), 1) AS pct_of_axis
FROM `i4g-ml.i4g_ml.raw_analyst_labels` al
GROUP BY al.axis, al.label_code
ORDER BY al.axis, label_count DESC;
```

### ETL Ingestion Freshness (daily)

Scheduled: every day at 6:15 AM UTC.

```sql
-- Scheduled query: etl_freshness_daily
SELECT
  'raw_cases' AS table_name,
  MAX(_ingested_at) AS last_ingest,
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(_ingested_at), HOUR) AS hours_since_ingest,
  COUNT(*) AS total_rows
FROM `i4g-ml.i4g_ml.raw_cases`
UNION ALL
SELECT
  'raw_entities',
  MAX(_ingested_at),
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(_ingested_at), HOUR),
  COUNT(*)
FROM `i4g-ml.i4g_ml.raw_entities`
UNION ALL
SELECT
  'raw_analyst_labels',
  MAX(_ingested_at),
  TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(_ingested_at), HOUR),
  COUNT(*)
FROM `i4g-ml.i4g_ml.raw_analyst_labels`;
```

### Feature Null Rates and Distribution Stats (daily)

Scheduled: every day at 6:30 AM UTC.

```sql
-- Scheduled query: feature_quality_daily
SELECT
  CURRENT_DATE() AS report_date,
  COUNT(*) AS total_rows,
  -- Null rates
  ROUND(COUNTIF(text_length IS NULL) * 100.0 / COUNT(*), 2) AS text_length_null_pct,
  ROUND(COUNTIF(word_count IS NULL) * 100.0 / COUNT(*), 2) AS word_count_null_pct,
  ROUND(COUNTIF(entity_count IS NULL) * 100.0 / COUNT(*), 2) AS entity_count_null_pct,
  ROUND(COUNTIF(current_classification_conf IS NULL) * 100.0 / COUNT(*), 2) AS classification_conf_null_pct,
  -- Distribution stats
  ROUND(AVG(text_length), 1) AS avg_text_length,
  ROUND(STDDEV(text_length), 1) AS stddev_text_length,
  ROUND(AVG(word_count), 1) AS avg_word_count,
  ROUND(AVG(entity_count), 1) AS avg_entity_count,
  COUNTIF(has_crypto_wallet) AS cases_with_crypto,
  COUNTIF(has_email) AS cases_with_email,
  COUNTIF(has_phone) AS cases_with_phone,
  COUNTIF(has_bank_account) AS cases_with_bank_account
FROM `i4g-ml.i4g_ml.features_case_features`;
```

---

## Data Quality Dashboard Specification

**Platform:** Looker Studio (or equivalent notebook-based dashboard)

### Dashboard Pages

#### Page 1 — Model Performance

| Panel                        | Source                        | Visualization       |
| ---------------------------- | ----------------------------- | ------------------- |
| Overall accuracy over time   | `analytics_model_performance` | Line chart (daily)  |
| Override rate over time      | `analytics_model_performance` | Line chart (daily)  |
| Per-axis F1 heatmap          | `analytics_model_performance` | Heatmap / bar chart |
| Active model versions        | `analytics_model_performance` | Scorecard           |
| Prediction volume (7d trend) | `predictions_prediction_log`  | Line chart (daily)  |

#### Page 2 — Data Quality

| Panel                   | Source                      | Visualization        |
| ----------------------- | --------------------------- | -------------------- |
| ETL freshness           | `etl_freshness_daily`       | Table (RAG status)   |
| Feature null rates      | `feature_quality_daily`     | Bar chart            |
| Label distribution      | `label_distribution`        | Stacked bar / pie    |
| Dataset version history | `training_dataset_registry` | Table                |
| Class imbalance ratio   | `label_distribution`        | Scorecard (per axis) |

#### Page 3 — Cost & Latency

| Panel                             | Source                       | Visualization      |
| --------------------------------- | ---------------------------- | ------------------ |
| Total ML platform cost (30d)      | GCP Billing Export           | Scorecard          |
| Cost by GCP service               | GCP Billing Export           | Pie / bar chart    |
| Cost per prediction               | Billing + prediction_log     | Time series        |
| ML vs LLM cost comparison         | Cost monitoring module       | Comparison bar     |
| Latency percentiles (p50/p90/p99) | `predictions_prediction_log` | Line chart (daily) |
| Confidence distribution           | `predictions_prediction_log` | Histogram          |

### Filters

- Date range (default: last 30 days)
- Model version
- Taxonomy axis

### Alerting Integration

Dashboard tiles link to the Cloud Monitoring alert policies:

- Override rate > 20% / 30%
- Outcome logging dead-letter rate > 5%
- Prod endpoint latency p90 > 2 s
- Prod endpoint error rate > 5%
  | Prediction volume drop | < 50% of 7-day avg | < 25% of 7-day avg | Check ETL, serving health |
  | P99 latency | > 2000ms | > 5000ms | Scale endpoint, check model size |
  | Per-axis F1 drop | > 5% vs champion | > 10% vs champion | Block promotion, investigate |

## Drift Metrics Queries

Drift detection runs daily at 6 AM UTC via a Cloud Run Job. Results are materialized to `analytics_drift_metrics`.

### Latest Drift Report

```sql
SELECT
  report_id,
  model_id,
  report_type,
  axis_or_feature,
  baseline_rate,
  current_rate,
  psi,
  is_drifted,
  window_start,
  window_end
FROM `i4g-ml.i4g_ml.analytics_drift_metrics`
WHERE computed_at = (
  SELECT MAX(computed_at) FROM `i4g-ml.i4g_ml.analytics_drift_metrics`
)
ORDER BY psi DESC;
```

### Drift Trend (last 30 days)

```sql
SELECT
  DATE(computed_at) AS report_date,
  report_type,
  axis_or_feature,
  psi,
  is_drifted
FROM `i4g-ml.i4g_ml.analytics_drift_metrics`
WHERE computed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY computed_at DESC, psi DESC;
```

### Retraining Trigger Log

```sql
SELECT
  event_id,
  capability,
  should_retrain,
  reasons,
  new_label_count,
  max_drift_psi,
  pipeline_job_name,
  triggered_at
FROM `i4g-ml.i4g_ml.analytics_trigger_log`
ORDER BY triggered_at DESC
LIMIT 20;
```

## Dashboard Access

The Looker Studio dashboard is accessible to all project members. To view:

1. Open Looker Studio at https://lookerstudio.google.com
2. Find the "ML Platform Monitoring" dashboard under the I4G workspace
3. Use the date range filter (default: last 30 days) and model version filter

Data freshness: tables are updated by Cloud Scheduler jobs at 5:00–6:30 AM UTC daily.

## Vertex AI Model Monitoring

Vertex AI provides built-in model monitoring. Configure via Terraform or console:

```bash
gcloud ai model-monitoring-jobs create \
  --project=i4g-ml \
  --region=us-central1 \
  --display-name=classification-monitor \
  --endpoint=<endpoint-id> \
  --prediction-sampling-rate=1.0 \
  --monitor-feature-attribution \
  --notification-channels=<channel-id>
```

## Dashboard Setup

Create a BigQuery-backed Looker Studio dashboard with:

1. **Prediction volume** — daily time series
2. **Confidence distribution** — histogram per axis
3. **Override rate** — trend line per axis
4. **Model performance** — F1 over time per axis
5. **Latency** — P50/P90/P99 time series
