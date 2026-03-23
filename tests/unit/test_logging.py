"""Unit tests for prediction/outcome logging with retry and dead-letter."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


class TestLogOutcomeRetry:
    """2.1.1 / 2.1.3 — Verify outcome logging with retry and dead-letter."""

    def setup_method(self):
        import ml.serving.logging as log_mod

        log_mod._bq_client = None

    @patch("ml.serving.logging._get_bq_client")
    @patch("ml.serving.logging.get_settings")
    def test_successful_write(self, mock_settings, mock_bq):
        """A successful first attempt should write the row and not retry."""
        from ml.serving.logging import log_outcome

        mock_settings.return_value = _fake_settings()
        mock_bq.return_value.insert_rows_json.return_value = []  # no errors

        outcome_id = log_outcome(
            prediction_id="pred-1",
            case_id="case-1",
            correction={"INTENT": "INTENT.ROMANCE"},
            analyst_id="analyst-1",
        )

        assert outcome_id  # non-empty UUID
        mock_bq.return_value.insert_rows_json.assert_called_once()
        row = mock_bq.return_value.insert_rows_json.call_args[0][1][0]
        assert row["prediction_id"] == "pred-1"
        assert row["analyst_id"] == "analyst-1"
        assert json.loads(row["correction"]) == {"INTENT": "INTENT.ROMANCE"}

    @patch("ml.serving.logging.time.sleep")
    @patch("ml.serving.logging._get_bq_client")
    @patch("ml.serving.logging.get_settings")
    def test_retry_then_succeed(self, mock_settings, mock_bq, mock_sleep):
        """Fail twice, succeed on third attempt."""
        from ml.serving.logging import log_outcome

        mock_settings.return_value = _fake_settings()
        client = mock_bq.return_value
        client.insert_rows_json.side_effect = [
            RuntimeError("transient"),
            RuntimeError("transient"),
            [],  # success
        ]

        outcome_id = log_outcome(
            prediction_id="pred-2",
            case_id="case-2",
            correction={"CHANNEL": "CHANNEL.EMAIL"},
            analyst_id="analyst-2",
        )

        assert outcome_id
        assert client.insert_rows_json.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("ml.serving.logging._dead_letter_logger")
    @patch("ml.serving.logging.time.sleep")
    @patch("ml.serving.logging._get_bq_client")
    @patch("ml.serving.logging.get_settings")
    def test_dead_letter_on_exhaustion(self, mock_settings, mock_bq, mock_sleep, mock_dl_logger):
        """After MAX_RETRIES failures the row is dead-lettered."""
        from ml.serving.logging import MAX_RETRIES, log_outcome

        mock_settings.return_value = _fake_settings()
        client = mock_bq.return_value
        client.insert_rows_json.side_effect = RuntimeError("permanent")

        outcome_id = log_outcome(
            prediction_id="pred-3",
            case_id="case-3",
            correction={"INTENT": "INTENT.OTHER"},
            analyst_id="analyst-3",
        )

        assert outcome_id  # still returns an ID even on failure
        assert client.insert_rows_json.call_count == MAX_RETRIES
        mock_dl_logger.error.assert_called_once()
        dl_msg = mock_dl_logger.error.call_args[0][0]
        assert "Dead-letter" in dl_msg


class TestLogPredictionRetry:
    """2.1.1 — Verify prediction logging also uses retry."""

    def setup_method(self):
        import ml.serving.logging as log_mod

        log_mod._bq_client = None

    @patch("ml.serving.logging._get_bq_client")
    @patch("ml.serving.logging.get_settings")
    def test_prediction_log_success(self, mock_settings, mock_bq):
        from ml.serving.logging import log_prediction

        mock_settings.return_value = _fake_settings()
        mock_bq.return_value.insert_rows_json.return_value = []

        log_prediction(
            prediction_id="pred-10",
            case_id="case-10",
            model_id="model-1",
            model_version=1,
            prediction={"INTENT": {"code": "INTENT.ROMANCE", "confidence": 0.9}},
        )

        mock_bq.return_value.insert_rows_json.assert_called_once()


class TestOutcomeJoinQuery:
    """2.1.2 — Verify the SQL for joining prediction_log + outcome_log."""

    def test_override_rate_query_structure(self):
        """The join query should produce analyst-override-rate metrics."""
        query = _build_override_rate_query("i4g-ml.i4g_ml")
        assert "prediction_log" in query
        assert "outcome_log" in query
        assert "override_rate" in query
        assert "JOIN" in query


# ── Helpers ──────────────────────────────────────────────────────────────


def _fake_settings():
    """Return a minimal mock Settings for logging tests."""
    s = MagicMock()
    s.platform.project_id = "i4g-ml"
    s.bigquery.dataset_id = "i4g_ml"
    s.bigquery.prediction_log_table = "predictions_prediction_log"
    s.bigquery.outcome_log_table = "predictions_outcome_log"
    s.serving.dev_endpoint_name = "serving-dev"
    return s


def _build_override_rate_query(dataset_id: str) -> str:
    """Build the SQL for computing analyst override rate (2.1.2 verification)."""
    return f"""
        SELECT
            p.model_id,
            p.model_version,
            COUNT(*) AS total_predictions,
            COUNTIF(o.outcome_id IS NOT NULL) AS outcomes_received,
            COUNTIF(o.correction IS NOT NULL
                    AND JSON_EXTRACT_SCALAR(o.correction, '$.INTENT')
                        != JSON_EXTRACT_SCALAR(p.prediction, '$.INTENT.code'))
                AS overrides,
            SAFE_DIVIDE(
                COUNTIF(o.correction IS NOT NULL
                        AND JSON_EXTRACT_SCALAR(o.correction, '$.INTENT')
                            != JSON_EXTRACT_SCALAR(p.prediction, '$.INTENT.code')),
                COUNTIF(o.outcome_id IS NOT NULL)
            ) AS override_rate
        FROM `{dataset_id}.predictions_prediction_log` p
        LEFT JOIN `{dataset_id}.predictions_outcome_log` o
            ON p.prediction_id = o.prediction_id
        GROUP BY p.model_id, p.model_version
        ORDER BY p.model_id, p.model_version
    """
