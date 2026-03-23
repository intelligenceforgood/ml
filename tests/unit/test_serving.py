"""Unit tests for serving layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from ml.serving.features import compute_inline_features
from ml.serving.predict import (
    _MODEL_STATE,
    _detect_model_type,
    _load_label_map,
    classify_text,
    is_model_ready,
    load_model,
)


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
    def setup_method(self):
        """Reset model state before each test."""
        _MODEL_STATE.clear()

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

    def test_stub_mode_returns_unknown(self):
        """When no model is loaded, returns UNKNOWN stub predictions."""
        result = classify_text("some text", "case-1")
        assert result["prediction"]["INTENT"]["code"] == "INTENT.UNKNOWN"
        assert result["prediction"]["CHANNEL"]["code"] == "CHANNEL.UNKNOWN"


class TestLoadModel:
    def setup_method(self):
        _MODEL_STATE.clear()

    def test_load_empty_uri_stays_stub(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = False
        load_model("")
        assert _MODEL_STATE["stage"] == "stub"
        assert _MODEL_STATE["version"] == 0

    def test_load_sets_state(self):
        """load_model with empty URI preserves model_id parsing."""
        load_model("gs://bucket/models/test-model/v1")
        # Will fail on download (no GCS) but model_id should be set
        assert _MODEL_STATE["model_id"] == "test-model"


class TestDetectModelType:
    def test_detect_xgboost(self, tmp_path: Path):
        (tmp_path / "xgboost_model.json").write_text("{}")
        assert _detect_model_type(tmp_path) == "xgboost"

    def test_detect_pytorch_model_dir(self, tmp_path: Path):
        (tmp_path / "model").mkdir()
        assert _detect_model_type(tmp_path) == "pytorch"

    def test_detect_pytorch_config_json(self, tmp_path: Path):
        (tmp_path / "config.json").write_text("{}")
        assert _detect_model_type(tmp_path) == "pytorch"

    def test_detect_unknown_raises(self, tmp_path: Path):
        import pytest

        with pytest.raises(ValueError, match="Cannot detect model type"):
            _detect_model_type(tmp_path)


class TestLoadLabelMap:
    def test_load_label_map(self, tmp_path: Path):
        label_map = {
            "INTENT": ["INTENT.ROMANCE", "INTENT.INVESTMENT", "INTENT.OTHER"],
            "CHANNEL": ["CHANNEL.EMAIL", "CHANNEL.PHONE"],
        }
        (tmp_path / "label_map.json").write_text(json.dumps(label_map))
        result = _load_label_map(tmp_path)
        assert result == label_map

    def test_missing_label_map_raises(self, tmp_path: Path):
        import pytest

        with pytest.raises(FileNotFoundError):
            _load_label_map(tmp_path)


class TestIsModelReady:
    def setup_method(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = False
        _MODEL_STATE.clear()

    def test_not_ready_when_no_framework(self):
        assert is_model_ready() is False

    def test_ready_when_framework_set(self):
        _MODEL_STATE["framework"] = "pytorch"
        assert is_model_ready() is True

    def test_not_ready_when_load_failed(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = True
        _MODEL_STATE["framework"] = "pytorch"
        assert is_model_ready() is False


class TestClassifyWithLoadFailure:
    def setup_method(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = True
        _MODEL_STATE.clear()

    def teardown_method(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = False

    def test_raises_runtime_error(self):
        import pytest

        with pytest.raises(RuntimeError, match="Model failed to load"):
            classify_text("text", "case-1")


class TestPyTorchInference:
    """Test PyTorch inference path with mocked model."""

    def setup_method(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = False
        _MODEL_STATE.clear()

    def test_pytorch_classify(self):
        """Verify PyTorch path selects the correct label per axis."""
        # We need to mock torch since it's not installed in the test env.
        # _predict_pytorch imports torch at call time so we can patch sys.modules.
        import sys
        import types

        mock_torch = types.ModuleType("torch")

        class _Tensor:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                import numpy as np

                return _Tensor(np.array(self._data)[key])

            def squeeze(self, dim=0):
                import numpy as np

                return _Tensor(np.array(self._data).squeeze(dim))

            def item(self):
                import numpy as np

                return float(np.array(self._data))

        def _softmax(t, dim=-1):
            import numpy as np

            arr = np.array(t._data)
            e = np.exp(arr - np.max(arr, axis=dim, keepdims=True))
            return _Tensor((e / e.sum(axis=dim, keepdims=True)).tolist())

        def _argmax(t):
            import numpy as np

            return _Tensor(int(np.argmax(np.array(t._data))))

        def _no_grad():
            class _ctx:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    pass

            return _ctx()

        mock_torch.softmax = _softmax
        mock_torch.argmax = _argmax
        mock_torch.no_grad = _no_grad
        mock_torch.tensor = lambda data, **kw: _Tensor(data)

        sys.modules["torch"] = mock_torch
        try:
            label_map = {
                "INTENT": ["INTENT.ROMANCE", "INTENT.INVESTMENT", "INTENT.OTHER"],
                "CHANNEL": ["CHANNEL.EMAIL", "CHANNEL.PHONE"],
            }
            _MODEL_STATE["label_map"] = label_map
            _MODEL_STATE["framework"] = "pytorch"
            _MODEL_STATE["model_id"] = "test-model"
            _MODEL_STATE["version"] = 1
            _MODEL_STATE["stage"] = "candidate"

            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": _Tensor([[1, 2, 3]])}

            # Mock model — 5 total labels (3 INTENT + 2 CHANNEL)
            mock_output = MagicMock()
            mock_output.logits = _Tensor([[2.0, 0.5, 0.1, 3.0, 0.2]])
            mock_model = MagicMock(return_value=mock_output)

            _MODEL_STATE["tokenizer"] = mock_tokenizer
            _MODEL_STATE["model"] = mock_model

            result = classify_text("suspicious romance scam", "case-1")
            pred = result["prediction"]

            assert pred["INTENT"]["code"] == "INTENT.ROMANCE"
            assert pred["CHANNEL"]["code"] == "CHANNEL.EMAIL"
            assert 0 < pred["INTENT"]["confidence"] <= 1.0
            assert result["model_info"]["model_id"] == "test-model"
        finally:
            del sys.modules["torch"]


class TestXGBoostInference:
    """Test XGBoost inference path with mocked booster."""

    def setup_method(self):
        import ml.serving.predict as pred_mod

        pred_mod._LOAD_FAILED = False
        _MODEL_STATE.clear()

    def test_xgboost_classify(self):
        import sys
        import types

        import numpy as np

        # Mock xgboost module
        mock_xgb = types.ModuleType("xgboost")

        class MockDMatrix:
            def __init__(self, data, feature_names=None):
                self.data = data
                self.feature_names = feature_names

        mock_xgb.DMatrix = MockDMatrix
        sys.modules["xgboost"] = mock_xgb

        try:
            label_map = {
                "INTENT": ["INTENT.ROMANCE", "INTENT.INVESTMENT"],
                "CHANNEL": ["CHANNEL.EMAIL", "CHANNEL.PHONE"],
            }
            _MODEL_STATE["label_map"] = label_map
            _MODEL_STATE["framework"] = "xgboost"
            _MODEL_STATE["model_id"] = "xgb-model"
            _MODEL_STATE["version"] = 2
            _MODEL_STATE["stage"] = "candidate"

            # Mock booster — returns 4 probabilities (2 INTENT + 2 CHANNEL)
            mock_booster = MagicMock()
            mock_booster.predict.return_value = np.array([[0.3, 0.7, 0.9, 0.1]])
            _MODEL_STATE["booster"] = mock_booster

            result = classify_text(
                "investment fraud text",
                "case-2",
                features={
                    "text_length": 22,
                    "word_count": 3,
                    "lexical_diversity": 1.0,
                    "has_email": False,
                    "has_phone": False,
                    "has_crypto_wallet": False,
                    "has_bank_account": False,
                },
            )
            pred = result["prediction"]

            assert pred["INTENT"]["code"] == "INTENT.INVESTMENT"  # 0.7 > 0.3
            assert pred["CHANNEL"]["code"] == "CHANNEL.EMAIL"  # 0.9 > 0.1
        finally:
            del sys.modules["xgboost"]


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
