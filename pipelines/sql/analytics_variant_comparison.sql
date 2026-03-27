-- Sprint 1.3: Analytics table for champion vs challenger variant comparison
-- Materialized by monitoring/accuracy.py compute_variant_comparison()

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.analytics_variant_comparison` (
  computed_at         DATE          NOT NULL,
  variant             STRING        NOT NULL,   -- 'champion' or 'challenger'
  model_id            STRING        NOT NULL,
  total_outcomes      INT64         NOT NULL,
  correct             INT64         NOT NULL,
  accuracy            FLOAT64       NOT NULL,
  override_rate       FLOAT64       NOT NULL,
  f1                  FLOAT64       NOT NULL,
  per_axis_metrics    STRING,                   -- JSON blob with per-axis breakdown
  lookback_days       INT64         NOT NULL
)
PARTITION BY computed_at
CLUSTER BY variant, model_id;
