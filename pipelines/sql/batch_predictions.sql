-- Sprint 2: Batch predictions destination table
-- Created automatically by batch.py, but defined here for reference/manual creation.

CREATE TABLE IF NOT EXISTS `i4g-ml.i4g_ml.batch_predictions` (
  case_id         STRING        NOT NULL,
  prediction      STRING        NOT NULL,   -- JSON blob
  confidence      FLOAT64,
  model_id        STRING        NOT NULL,
  model_version   INT64         NOT NULL,
  predicted_at    TIMESTAMP     NOT NULL,
  capability      STRING        NOT NULL     -- classification, ner, risk_scoring, embedding
)
PARTITION BY DATE(predicted_at)
CLUSTER BY capability, model_id;
