-- DDL: analytics_trigger_log
-- Records each retraining trigger evaluation — whether retrain was triggered and why.

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.analytics_trigger_log` (
    event_id          STRING    NOT NULL,
    capability        STRING    NOT NULL,
    should_retrain    BOOL      NOT NULL,
    reasons           STRING,             -- JSON array of reason strings
    new_label_count   INT64,
    max_drift_psi     FLOAT64,
    pipeline_job_name STRING,             -- NULL if no retrain submitted
    triggered_at      TIMESTAMP NOT NULL
);
