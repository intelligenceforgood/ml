-- Sprint 1: Add variant and routing_reason columns to prediction_log
-- Run once per environment (dev, then prod).
-- Safe to re-run: ALTER TABLE ADD COLUMN IF NOT EXISTS

ALTER TABLE `i4g-ml.i4g_ml.predictions_prediction_log`
  ADD COLUMN IF NOT EXISTS variant STRING DEFAULT 'champion';

ALTER TABLE `i4g-ml.i4g_ml.predictions_prediction_log`
  ADD COLUMN IF NOT EXISTS routing_reason STRING DEFAULT '';
