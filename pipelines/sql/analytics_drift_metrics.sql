-- DDL: analytics_drift_metrics
-- Stores per-axis and per-feature drift metrics (PSI) computed daily.

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.analytics_drift_metrics` (
    report_id       STRING    NOT NULL,
    model_id        STRING    NOT NULL,
    report_type     STRING    NOT NULL,   -- 'prediction' or 'feature'
    axis_or_feature STRING    NOT NULL,
    baseline_rate   FLOAT64,
    current_rate    FLOAT64,
    psi             FLOAT64   NOT NULL,
    is_drifted      BOOL      NOT NULL,
    window_start    TIMESTAMP NOT NULL,
    window_end      TIMESTAMP NOT NULL,
    computed_at     TIMESTAMP NOT NULL
);
