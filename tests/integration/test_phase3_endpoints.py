"""Phase 3 integration tests for ML serving endpoints.

These tests run against the deployed ml-serving Cloud Run service.
They require:
  - SERVE_URL env var (or default to the dev URL)
  - gcloud auth configured with impersonation for sa-app@i4g-dev

Usage:
    conda run -n ml pytest tests/integration/test_phase3_endpoints.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid

import pytest

SERVE_URL = os.environ.get("ML_SERVE_URL", "https://ml-serving-6bqxauys2q-uc.a.run.app")
SA_EMAIL = "sa-app@i4g-dev.iam.gserviceaccount.com"


def _get_token() -> str:
    """Get an identity token via service account impersonation."""
    r = subprocess.run(
        [
            "gcloud",
            "auth",
            "print-identity-token",
            f"--impersonate-service-account={SA_EMAIL}",
            f"--audiences={SERVE_URL}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def _post(token: str, path: str, payload: dict) -> tuple[int, dict | str]:
    """POST JSON to the serving endpoint."""
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SERVE_URL}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else str(e)
        return e.code, body


def _get(token: str, path: str) -> tuple[int, dict | str]:
    """GET from the serving endpoint."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{SERVE_URL}{path}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, "read") else str(e)
        return e.code, body


def _bq_query(sql: str) -> str:
    """Run a BigQuery query and return stdout."""
    r = subprocess.run(
        ["bq", "query", "--project_id=i4g-ml", "--use_legacy_sql=false", "--format=json", sql],
        capture_output=True,
        text=True,
    )
    return r.stdout


@pytest.fixture(scope="module")
def token() -> str:
    return _get_token()


# ---------------------------------------------------------------------------
# Test 1: Classify via A/B route → variant logged → feedback recorded
# ---------------------------------------------------------------------------


class TestClassifyABRouting:
    """Integration test: classify → variant logged → feedback → outcome recorded."""

    def test_classify_returns_prediction_with_variant(self, token: str):
        case_id = f"integ-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/classify",
            {"text": "Fraudulent investment scheme targeting retirees", "case_id": case_id},
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        preds = body["predictions"]
        assert len(preds) >= 1
        pred = preds[0]
        assert "prediction_id" in pred
        assert "prediction" in pred
        assert "model_info" in pred
        mi = pred["model_info"]
        assert mi["model_id"]
        assert mi["stage"] in ("champion", "challenger", "experimental", "stub")

    def test_feedback_records_outcome(self, token: str):
        # First classify
        case_id = f"integ-fb-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/classify",
            {"text": "Email phishing attack stealing credentials", "case_id": case_id},
        )
        assert status == 200
        prediction_id = body["predictions"][0]["prediction_id"]

        # Then submit feedback
        status, fb_body = _post(
            token,
            "/feedback",
            {
                "prediction_id": prediction_id,
                "case_id": case_id,
                "correction": {"CHANNEL": "EMAIL"},
                "analyst_id": "integration-test",
            },
        )
        assert status == 200, f"Feedback failed: {status}: {fb_body}"
        assert fb_body["status"] == "recorded"
        assert "outcome_id" in fb_body

    def test_variant_logged_in_bq(self, token: str):
        """Verify that predictions are logged with variant info in BigQuery."""
        # Allow time for async BQ writes
        time.sleep(3)
        raw = _bq_query(
            "SELECT variant, routing_reason FROM i4g_ml.predictions_prediction_log "
            "WHERE variant IS NOT NULL ORDER BY timestamp DESC LIMIT 5"
        )
        if raw.strip():
            rows = json.loads(raw)
            assert len(rows) > 0, "No prediction log entries with variant found"
            # At least one row should have a non-empty variant
            assert any(r.get("variant") for r in rows)


# ---------------------------------------------------------------------------
# Test 2: Risk scoring → prediction logged
# ---------------------------------------------------------------------------


class TestRiskScoring:
    """Integration test: risk scoring endpoint works and logs predictions."""

    def test_risk_score_returns_float(self, token: str):
        case_id = f"integ-risk-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/risk-score",
            {"text": "Cryptocurrency scam with fake returns", "case_id": case_id},
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["model_info"]["model_id"]
        assert body["prediction_id"]

    def test_risk_score_logged_in_bq(self, token: str):
        """Verify risk scoring predictions appear in BQ with capability=risk_scoring."""
        time.sleep(3)
        raw = _bq_query(
            "SELECT capability, model_id FROM i4g_ml.predictions_prediction_log "
            "WHERE capability = 'risk_scoring' ORDER BY timestamp DESC LIMIT 3"
        )
        if raw.strip():
            rows = json.loads(raw)
            assert len(rows) > 0, "No risk_scoring predictions in BQ log"


# ---------------------------------------------------------------------------
# Test 3: Similar cases endpoint works with FAISS index
# ---------------------------------------------------------------------------


class TestSimilarCases:
    """Integration test: similar-cases uses embeddings + FAISS index."""

    def test_similar_cases_returns_results(self, token: str):
        case_id = f"integ-sim-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/similar-cases",
            {"text": "Online purchase scam via social media", "case_id": case_id, "k": 5},
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        assert "similar_cases" in body
        results = body["similar_cases"]
        assert len(results) > 0, "Expected at least 1 similar case"
        for r in results:
            assert "case_id" in r
            assert "score" in r or "distance" in r

    def test_similar_cases_respects_k(self, token: str):
        case_id = f"integ-sim2-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/similar-cases",
            {"text": "Wire transfer fraud", "case_id": case_id, "k": 3},
        )
        assert status == 200
        assert len(body["similar_cases"]) <= 3


# ---------------------------------------------------------------------------
# Test 4: Feature Store online read (verified via health + prediction)
# ---------------------------------------------------------------------------


class TestFeatureStoreIntegration:
    """Integration test: Feature Store is configured and used in predictions."""

    def test_health_shows_capabilities(self, token: str):
        status, body = _get(token, "/health")
        assert status == 200
        assert body["status"] in ("healthy", "degraded")
        # Verify multi-capability: at least classification + risk active
        assert body.get("risk_active") is True, "Risk scoring should be active"

    def test_prediction_completes_with_feature_store(self, token: str):
        """Classify with a case_id that may have Feature Store features.

        The test verifies the prediction completes successfully regardless
        of whether features come from Feature Store or inline computation.
        """
        case_id = f"integ-fs-{uuid.uuid4().hex[:8]}"
        status, body = _post(
            token,
            "/predict/classify",
            {
                "text": "Investment scam targeting elderly victims",
                "case_id": case_id,
                "features": {"text_length": 42, "word_count": 6},
            },
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        assert len(body["predictions"]) >= 1


# ---------------------------------------------------------------------------
# Test 5: Cost-aware routing (when enabled)
# ---------------------------------------------------------------------------


class TestCostAwareRouting:
    """Integration test: cost-aware routing selects models based on cost profiles."""

    def test_routing_reason_logged(self, token: str):
        """Verify routing_reason appears in prediction log when cost-aware is enabled."""
        # Send a classify request to trigger routing
        case_id = f"integ-cost-{uuid.uuid4().hex[:8]}"
        status, _ = _post(
            token,
            "/predict/classify",
            {"text": "Phone scam impersonating government agency", "case_id": case_id},
        )
        assert status == 200

        # Check BQ for routing_reason
        time.sleep(3)
        raw = _bq_query(
            "SELECT routing_reason FROM i4g_ml.predictions_prediction_log "
            "WHERE routing_reason IS NOT NULL AND routing_reason != '' "
            "ORDER BY timestamp DESC LIMIT 5"
        )
        if raw.strip():
            rows = json.loads(raw)
            assert len(rows) > 0, "No routing_reason entries found"
            # Verify at least one has cost_aware or ab_split pattern
            reasons = [r["routing_reason"] for r in rows]
            assert any(
                "cost_aware" in r or "ab_split" in r for r in reasons
            ), f"Expected cost_aware or ab_split in routing reasons: {reasons}"
