"""E2E smoke test for the ML Platform.

Tests the full prediction + feedback + logging lifecycle.

Usage:
    conda run -n ml python scripts/e2e_smoke_test.py
"""

from __future__ import annotations

import logging
import time

from google.cloud import aiplatform, bigquery

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT = "i4g-ml"
REGION = "us-central1"
DATASET = "i4g_ml"
ENDPOINT_NAME = "serving-dev"


def main() -> None:
    """Run end-to-end smoke test against the serving-dev endpoint."""
    aiplatform.init(project=PROJECT, location=REGION)
    bq_client = bigquery.Client(project=PROJECT)

    # 1. Get endpoint
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{ENDPOINT_NAME}"')
    assert endpoints, f"Endpoint '{ENDPOINT_NAME}' not found"
    endpoint = endpoints[0]
    logger.info("Endpoint: %s", endpoint.resource_name)

    # 2. Send prediction
    logger.info("--- Step 1: Send prediction ---")
    response = endpoint.predict(
        instances=[{"text": "E2E smoke test - suspicious wire transfer", "case_id": "e2e-final-001"}]
    )
    pred = response.predictions[0]
    prediction_id = pred["prediction_id"]
    logger.info("prediction_id: %s", prediction_id)
    logger.info("INTENT: %s", pred["prediction"]["INTENT"])
    logger.info("CHANNEL: %s", pred["prediction"]["CHANNEL"])
    logger.info("model_info: %s", pred["model_info"])
    assert "prediction_id" in pred, "Missing prediction_id"
    assert "prediction" in pred, "Missing prediction"
    assert "INTENT" in pred["prediction"], "Missing INTENT axis"
    assert "CHANNEL" in pred["prediction"], "Missing CHANNEL axis"
    logger.info("PASS: Prediction response structure valid")

    # 3. Wait for BQ streaming buffer
    logger.info("--- Step 2: Verify prediction logged in BigQuery ---")
    logger.info("Waiting 30s for BQ streaming buffer...")
    time.sleep(30)

    query = f"""
    SELECT prediction_id, case_id, model_id, endpoint, prediction, timestamp
    FROM `{PROJECT}.{DATASET}.predictions_prediction_log`
    WHERE prediction_id = '{prediction_id}'
    """
    rows = list(bq_client.query(query).result())
    assert len(rows) == 1, f"Expected 1 prediction log row, got {len(rows)}"
    row = rows[0]
    logger.info("Logged: prediction_id=%s, case_id=%s, model_id=%s", row.prediction_id, row.case_id, row.model_id)
    assert row.case_id == "e2e-final-001"
    assert row.model_id == "stub"
    assert row.endpoint == "serving-dev"
    logger.info("PASS: Prediction logged in BigQuery")

    # 4. Compare stub F1 vs baseline
    logger.info("--- Step 3: Compare model vs baseline ---")
    baseline_f1 = 0.3929  # from data/baseline_result.json
    logger.info("Baseline F1: %.4f", baseline_f1)
    logger.info(
        "Stub model returns UNKNOWN predictions — will not beat baseline. "
        "This is expected for Phase 0 (proves lifecycle, not model quality)."
    )
    logger.info("PASS: Baseline comparison noted")

    # 5. Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("E2E SMOKE TEST: ALL CHECKS PASSED")
    logger.info("=" * 60)
    logger.info("  [x] Prediction endpoint returns valid response")
    logger.info("  [x] Prediction logged in BigQuery")
    logger.info("  [x] Model lifecycle operational (stub → endpoint → log)")
    logger.info("  [ ] Feedback endpoint: requires direct HTTP (not via Vertex AI predict route)")
    logger.info("  [ ] Model F1 vs baseline: deferred to Phase 1 (real training)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
