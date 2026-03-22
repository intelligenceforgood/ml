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
  evaluation_date,
  axis,
  f1_score,
  precision_score,
  recall_score,
  model_version
FROM `i4g-ml.i4g_ml.analytics_model_performance`
ORDER BY evaluation_date DESC, axis;
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

| Metric                     | Warning            | Critical           | Action                           |
| -------------------------- | ------------------ | ------------------ | -------------------------------- |
| Override rate              | > 20% over 7 days  | > 30% over 7 days  | Investigate label drift, retrain |
| Low-confidence predictions | > 40% of volume    | > 60% of volume    | Check input distribution shift   |
| Prediction volume drop     | < 50% of 7-day avg | < 25% of 7-day avg | Check ETL, serving health        |
| P99 latency                | > 2000ms           | > 5000ms           | Scale endpoint, check model size |
| Per-axis F1 drop           | > 5% vs champion   | > 10% vs champion  | Block promotion, investigate     |

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
