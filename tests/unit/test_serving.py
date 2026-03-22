"""Unit tests for serving layer."""

from __future__ import annotations

from ml.serving.features import compute_inline_features
from ml.serving.predict import _MODEL_STATE, classify_text, load_model


class TestComputeInlineFeatures:
    def test_basic_text(self):
        features = compute_inline_features("Hello world test text")
        assert features["text_length"] == len("Hello world test text")
        assert features["word_count"] == 4
        assert 0 < features["lexical_diversity"] <= 1.0

    def test_email_detection(self):
        features = compute_inline_features("Contact me at user@example.com")
        assert features["has_email"] is True

    def test_phone_detection(self):
        features = compute_inline_features("Call 555-123-4567 now")
        assert features["has_phone"] is True

    def test_no_pii(self):
        features = compute_inline_features("A simple sentence without any PII")
        assert features["has_email"] is False
        assert features["has_phone"] is False
        assert features["has_crypto_wallet"] is False

    def test_empty_text(self):
        features = compute_inline_features("")
        assert features["text_length"] == 0
        assert features["word_count"] == 0


class TestClassifyText:
    def test_returns_required_keys(self):
        result = classify_text("some text", "case-1")
        assert "prediction_id" in result
        assert "prediction" in result
        assert "model_info" in result

    def test_prediction_structure(self):
        result = classify_text("some text", "case-1")
        pred = result["prediction"]
        assert "INTENT" in pred
        assert "code" in pred["INTENT"]
        assert "confidence" in pred["INTENT"]

    def test_model_info_structure(self):
        result = classify_text("test", "case-1")
        info = result["model_info"]
        assert "model_id" in info
        assert "version" in info
        assert "stage" in info


class TestLoadModel:
    def test_load_sets_state(self):
        load_model("gs://bucket/models/test-model/v1")
        assert _MODEL_STATE["artifact_uri"] == "gs://bucket/models/test-model/v1"
        assert _MODEL_STATE["model_id"] == "test-model"


class TestPII:
    def test_redact_email(self):
        from ml.data.pii import redact_pii

        result = redact_pii("Contact user@example.com for help")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_redact_phone(self):
        from ml.data.pii import redact_pii

        result = redact_pii("Call 555-123-4567")
        assert "[PHONE]" in result

    def test_redact_ssn(self):
        from ml.data.pii import redact_pii

        result = redact_pii("SSN: 123-45-6789")
        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_redact_record(self):
        from ml.data.pii import redact_record

        record = {"text": "Email: a@b.com", "case_id": "c1"}
        result = redact_record(record)
        assert "[EMAIL]" in result["text"]
        assert result["case_id"] == "c1"
