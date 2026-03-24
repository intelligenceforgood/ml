-- Add is_shadow column to prediction_log for shadow mode inference tracking.
-- Shadow predictions are logged alongside champion predictions with
-- prediction_id = "{champion_prediction_id}-shadow".
ALTER TABLE `i4g_ml.predictions_prediction_log`
ADD COLUMN IF NOT EXISTS is_shadow BOOL DEFAULT FALSE;
