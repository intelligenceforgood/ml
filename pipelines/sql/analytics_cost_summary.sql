-- DDL: analytics_cost_summary
-- Stores daily ML platform vs LLM cost comparison metrics.

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.analytics_cost_summary` (
    summary_id            STRING    NOT NULL,
    model_id              STRING,
    capability            STRING    NOT NULL,
    prediction_count      INT64     NOT NULL,
    ml_cost_per_prediction FLOAT64  NOT NULL,
    llm_cost_per_prediction FLOAT64 NOT NULL,
    ml_total              FLOAT64   NOT NULL,
    llm_total             FLOAT64   NOT NULL,
    savings_pct           FLOAT64   NOT NULL,
    period_start          TIMESTAMP NOT NULL,
    period_end            TIMESTAMP NOT NULL,
    computed_at           TIMESTAMP NOT NULL
);
