"""Unit tests for NER evaluation harness."""

from __future__ import annotations

from ml.training.evaluation import (
    NerEvalResult,
    NerGolden,
    NerPrediction,
    align_labels_with_tokens,
    evaluate_ner,
    spans_to_bio_tags,
)


class TestSpansToBioTags:
    def test_no_entities(self):
        text = "hello world"
        tags = spans_to_bio_tags(text, [], ["hello", "world"])
        assert tags == ["O", "O"]

    def test_single_entity(self):
        text = "John Doe is here"
        spans = [{"start": 0, "end": 8, "label": "PERSON"}]
        tags = spans_to_bio_tags(text, spans, ["John", "Doe", "is", "here"])
        assert tags == ["B-PERSON", "I-PERSON", "O", "O"]

    def test_adjacent_different_entities(self):
        text = "John Doe ACME Corp"
        spans = [
            {"start": 0, "end": 8, "label": "PERSON"},
            {"start": 9, "end": 18, "label": "ORG"},
        ]
        tags = spans_to_bio_tags(text, spans, ["John", "Doe", "ACME", "Corp"])
        assert tags == ["B-PERSON", "I-PERSON", "B-ORG", "I-ORG"]

    def test_entity_at_end(self):
        text = "call 555-1234"
        spans = [{"start": 5, "end": 13, "label": "PHONE"}]
        tags = spans_to_bio_tags(text, spans, ["call", "555-1234"])
        assert tags == ["O", "B-PHONE"]

    def test_entity_at_start(self):
        text = "john@test.com sent a message"
        spans = [{"start": 0, "end": 13, "label": "EMAIL"}]
        tags = spans_to_bio_tags(text, spans, ["john@test.com", "sent", "a", "message"])
        assert tags == ["B-EMAIL", "O", "O", "O"]

    def test_empty_text(self):
        tags = spans_to_bio_tags("", [], [])
        assert tags == []


class TestAlignLabelsWithTokens:
    def test_simple_alignment(self):
        labels = ["B-PERSON", "I-PERSON", "O"]
        word_ids = [None, 0, 1, 2, None]  # [CLS] John Doe is [SEP]
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "B-PERSON", "I-PERSON", "O", "O"]

    def test_subword_continuation(self):
        # "cryptocurrency" split into ["crypto", "##currency"]
        labels = ["O", "B-CRYPTO_WALLET"]
        word_ids = [None, 0, 0, 1, None]  # [CLS] he ##llo wallet [SEP]
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "O", "O", "B-CRYPTO_WALLET", "O"]

    def test_b_becomes_i_for_continuation(self):
        # First subword gets B-, continuation subwords get I-
        labels = ["B-PERSON"]
        word_ids = [None, 0, 0, 0, None]
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "B-PERSON", "I-PERSON", "I-PERSON", "O"]

    def test_all_special_tokens(self):
        labels = []
        word_ids = [None, None]  # [CLS] [SEP]
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "O"]

    def test_multi_token_entity(self):
        labels = ["B-ORG", "I-ORG", "I-ORG", "O"]
        word_ids = [None, 0, 1, 2, 3, None]
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "B-ORG", "I-ORG", "I-ORG", "O", "O"]

    def test_word_id_beyond_labels(self):
        """Edge case: word_id exceeds label count — should get 'O'."""
        labels = ["B-PERSON"]
        word_ids = [None, 0, 5, None]  # word_id=5 has no label
        result = align_labels_with_tokens(labels, word_ids)
        assert result == ["O", "B-PERSON", "O", "O"]


class TestEvaluateNer:
    def test_perfect_score(self):
        text = "John Doe works at ACME"
        entities = [
            {"start": 0, "end": 8, "label": "PERSON"},
            {"start": 18, "end": 22, "label": "ORG"},
        ]
        predictions = [NerPrediction(text=text, entities=entities)]
        golden = [NerGolden(text=text, entities=entities)]

        result = evaluate_ner(predictions, golden)

        assert result.micro_f1 == 1.0
        assert result.micro_precision == 1.0
        assert result.micro_recall == 1.0
        assert result.total_samples == 1

    def test_zero_score(self):
        text = "John Doe works at ACME"
        golden_entities = [{"start": 0, "end": 8, "label": "PERSON"}]
        pred_entities: list[dict] = []  # No predictions

        predictions = [NerPrediction(text=text, entities=pred_entities)]
        golden = [NerGolden(text=text, entities=golden_entities)]

        result = evaluate_ner(predictions, golden)

        assert result.micro_f1 == 0.0
        assert result.micro_recall == 0.0

    def test_partial_match(self):
        text = "John Doe and Jane Smith work at ACME"
        golden_entities = [
            {"start": 0, "end": 8, "label": "PERSON"},
            {"start": 13, "end": 23, "label": "PERSON"},
            {"start": 32, "end": 36, "label": "ORG"},
        ]
        # Only predicted one of two persons
        pred_entities = [
            {"start": 0, "end": 8, "label": "PERSON"},
        ]
        predictions = [NerPrediction(text=text, entities=pred_entities)]
        golden = [NerGolden(text=text, entities=golden_entities)]

        result = evaluate_ner(predictions, golden)

        assert 0.0 < result.micro_f1 < 1.0
        assert "PERSON" in result.per_entity_type

    def test_empty_inputs(self):
        result = evaluate_ner([], [])
        assert isinstance(result, NerEvalResult)
        assert result.micro_f1 == 0.0
        assert result.total_samples == 0

    def test_summary_output(self):
        text = "test entity"
        entities = [{"start": 5, "end": 11, "label": "ORG"}]
        predictions = [NerPrediction(text=text, entities=entities)]
        golden = [NerGolden(text=text, entities=entities)]
        result = evaluate_ner(predictions, golden)
        summary = result.summary()
        assert "Micro" in summary
        assert "Macro" in summary

    def test_multiple_samples(self):
        samples = [
            ("John works here", [{"start": 0, "end": 4, "label": "PERSON"}]),
            ("ACME is big", [{"start": 0, "end": 4, "label": "ORG"}]),
        ]
        predictions = [NerPrediction(text=t, entities=e) for t, e in samples]
        golden = [NerGolden(text=t, entities=e) for t, e in samples]

        result = evaluate_ner(predictions, golden)
        assert result.total_samples == 2
        assert result.micro_f1 == 1.0
