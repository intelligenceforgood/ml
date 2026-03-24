"""Tests for NER serving — multi-capability predict + app endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml.serving.predict import _NER_MODEL_STATE, EntitySpan, extract_entities, is_ner_ready, load_ner_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_ner_state():
    """Clear NER model state between tests."""
    import ml.serving.predict as pred

    pred._NER_LOAD_FAILED = False
    _NER_MODEL_STATE.clear()


@pytest.fixture(autouse=True)
def _ner_cleanup():
    """Reset NER state before and after each test."""
    _reset_ner_state()
    yield
    _reset_ner_state()


# ---------------------------------------------------------------------------
# is_ner_ready
# ---------------------------------------------------------------------------


class TestIsNerReady:
    def test_not_ready_initially(self):
        assert is_ner_ready() is False

    def test_ready_when_loaded(self):
        _NER_MODEL_STATE["framework"] = "pytorch"
        assert is_ner_ready() is True

    def test_not_ready_when_load_failed(self):
        import ml.serving.predict as pred

        _NER_MODEL_STATE["framework"] = "pytorch"
        pred._NER_LOAD_FAILED = True
        assert is_ner_ready() is False


# ---------------------------------------------------------------------------
# load_ner_model
# ---------------------------------------------------------------------------


class TestLoadNerModel:
    def test_empty_uri_does_nothing(self):
        load_ner_model("")
        assert _NER_MODEL_STATE == {}

    @patch("ml.serving.predict._download_artifacts")
    def test_load_success(self, mock_download, tmp_path):
        """Successful load sets framework and version."""
        # Create fake model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create label_map.json in NER format
        import json

        label_map = {
            "label2id": {"O": 0, "B-PERSON": 1, "I-PERSON": 2},
            "id2label": {"0": "O", "1": "B-PERSON", "2": "I-PERSON"},
        }
        (tmp_path / "label_map.json").write_text(json.dumps(label_map))

        mock_download.side_effect = lambda _uri, dest: None

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with (
            patch("ml.serving.predict.tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
            patch("transformers.AutoModelForTokenClassification.from_pretrained", return_value=mock_model),
        ):
            load_ner_model("gs://bucket/models/ner-bert/v2/artifacts")

        assert _NER_MODEL_STATE["framework"] == "pytorch"
        assert _NER_MODEL_STATE["version"] == 2
        assert _NER_MODEL_STATE["tokenizer"] is mock_tokenizer
        assert is_ner_ready()

    @patch("ml.serving.predict._download_artifacts", side_effect=Exception("download failed"))
    def test_load_failure_sets_flag(self, _mock_download):
        import ml.serving.predict as pred

        load_ner_model("gs://bucket/bad")
        assert pred._NER_LOAD_FAILED is True
        assert not is_ner_ready()


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_raises_when_load_failed(self):
        import ml.serving.predict as pred

        pred._NER_LOAD_FAILED = True
        with pytest.raises(RuntimeError, match="NER model failed to load"):
            extract_entities("test text", "case-1")

    def test_raises_when_not_loaded(self):
        with pytest.raises(RuntimeError, match="NER model not loaded"):
            extract_entities("test text", "case-1")

    def test_extract_entities_returns_spans(self):
        """Test BIO decoding produces entity spans."""
        import torch

        # id2label for 5 tags: O, B-PERSON, I-PERSON, B-ORG, I-ORG
        id2label = {0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-ORG", 4: "I-ORG"}

        # Simulate tokenizer output: [CLS] John Smith works at Acme [SEP]
        # Offsets: CLS=(0,0), John=(0,4), Smith=(5,10), works=(11,16), at=(17,19), Acme=(20,24), SEP=(0,0)
        mock_tokenizer = MagicMock()
        offset_mapping = torch.tensor(
            [
                [0, 0],  # CLS
                [0, 4],  # John
                [5, 10],  # Smith
                [11, 16],  # works
                [17, 19],  # at
                [20, 24],  # Acme
                [0, 0],  # SEP
            ]
        )
        encoding = {
            "input_ids": torch.tensor([[101, 1, 2, 3, 4, 5, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]]),
            "offset_mapping": offset_mapping.unsqueeze(0),
        }
        mock_tokenizer.return_value = encoding

        # Logits: 7 tokens x 5 labels
        # [CLS]=O, John=B-PERSON, Smith=I-PERSON, works=O, at=O, Acme=B-ORG, [SEP]=O
        logits = torch.zeros(1, 7, 5)
        logits[0, 0, 0] = 10.0  # CLS -> O
        logits[0, 1, 1] = 10.0  # John -> B-PERSON
        logits[0, 2, 2] = 10.0  # Smith -> I-PERSON
        logits[0, 3, 0] = 10.0  # works -> O
        logits[0, 4, 0] = 10.0  # at -> O
        logits[0, 5, 3] = 10.0  # Acme -> B-ORG
        logits[0, 6, 0] = 10.0  # SEP -> O

        mock_model = MagicMock()
        mock_model.return_value.logits = logits

        _NER_MODEL_STATE.update(
            {
                "framework": "pytorch",
                "tokenizer": mock_tokenizer,
                "model": mock_model,
                "id2label": id2label,
                "model_id": "ner-bert",
                "version": 1,
                "stage": "candidate",
            }
        )

        text = "John Smith works at Acme"
        result = extract_entities(text, "case-1")

        assert "prediction_id" in result
        entities = result["prediction"]["entities"]
        assert len(entities) == 2

        # First entity: John Smith (PERSON)
        assert entities[0]["label"] == "PERSON"
        assert entities[0]["start"] == 0
        assert entities[0]["end"] == 10
        assert entities[0]["text"] == "John Smith"

        # Second entity: Acme (ORG)
        assert entities[1]["label"] == "ORG"
        assert entities[1]["start"] == 20
        assert entities[1]["end"] == 24

    def test_503_when_ner_disabled(self):
        """App endpoint returns 503 when NER model not loaded."""
        from fastapi.testclient import TestClient

        from ml.serving.app import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/predict/extract-entities", json={"text": "hello", "case_id": "c1"})
        assert resp.status_code == 503

    def test_entity_span_dataclass(self):
        span = EntitySpan(text="test", label="PERSON", start=0, end=4, confidence=0.95)
        assert span.text == "test"
        assert span.label == "PERSON"
        assert span.confidence == 0.95
