"""Unit tests for evaluation harness."""

from __future__ import annotations

import pytest

from ml.training.evaluation import AxisMetrics, EvalResult, compute_metrics


class TestComputeMetrics:
    def test_perfect_predictions(self):
        predictions = [
            {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
            {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.PHONE"},
        ]
        ground_truth = [
            {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
            {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.PHONE"},
        ]
        result = compute_metrics(predictions, ground_truth)
        assert result.overall_f1 == pytest.approx(1.0)
        assert result.overall_precision == pytest.approx(1.0)
        assert result.overall_recall == pytest.approx(1.0)

    def test_all_wrong(self):
        predictions = [
            {"INTENT": "INTENT.INVESTMENT"},
            {"INTENT": "INTENT.ROMANCE"},
        ]
        ground_truth = [
            {"INTENT": "INTENT.ROMANCE"},
            {"INTENT": "INTENT.INVESTMENT"},
        ]
        result = compute_metrics(predictions, ground_truth)
        assert result.overall_f1 == pytest.approx(0.0)

    def test_partial_correct(self):
        predictions = [
            {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
            {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.WRONG"},
        ]
        ground_truth = [
            {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.EMAIL"},
            {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.PHONE"},
        ]
        result = compute_metrics(predictions, ground_truth)
        # INTENT: F1=1.0, CHANNEL: F1=0.5
        assert 0.5 < result.overall_f1 < 1.0

    def test_per_axis_metrics(self):
        predictions = [{"INTENT": "A"}, {"INTENT": "B"}]
        ground_truth = [{"INTENT": "A"}, {"INTENT": "B"}]
        result = compute_metrics(predictions, ground_truth)
        assert "INTENT" in result.per_axis
        assert result.per_axis["INTENT"].f1 == pytest.approx(1.0)
        assert result.per_axis["INTENT"].support == 2

    def test_empty_predictions(self):
        result = compute_metrics([], [])
        assert result.overall_f1 == 0.0
        assert result.total_samples == 0

    def test_missing_axis_in_prediction(self):
        predictions = [{}]
        ground_truth = [{"INTENT": "A"}]
        result = compute_metrics(predictions, ground_truth)
        assert result.overall_f1 == 0.0


class TestEvalResult:
    def test_summary(self):
        result = EvalResult(
            overall_f1=0.85,
            overall_precision=0.90,
            overall_recall=0.80,
            per_axis={
                "INTENT": AxisMetrics(axis="INTENT", precision=0.9, recall=0.8, f1=0.85, support=100),
            },
            total_samples=100,
        )
        summary = result.summary()
        assert "0.8500" in summary
        assert "INTENT" in summary


class TestEvalGate:
    def test_passes_no_champion(self):
        from ml.registry.promotion import _passes_eval_gate

        candidate = EvalResult(overall_f1=0.5, overall_precision=0.5, overall_recall=0.5)
        passed, reason = _passes_eval_gate(candidate, None)
        assert passed
        assert "first model" in reason.lower()

    def test_passes_better_f1(self):
        from ml.registry.promotion import _passes_eval_gate

        candidate = EvalResult(
            overall_f1=0.9,
            overall_precision=0.9,
            overall_recall=0.9,
            per_axis={"INTENT": AxisMetrics("INTENT", 0.9, 0.9, 0.9, 100)},
        )
        champion = EvalResult(
            overall_f1=0.8,
            overall_precision=0.8,
            overall_recall=0.8,
            per_axis={"INTENT": AxisMetrics("INTENT", 0.8, 0.8, 0.8, 100)},
        )
        passed, _ = _passes_eval_gate(candidate, champion)
        assert passed

    def test_fails_lower_f1(self):
        from ml.registry.promotion import _passes_eval_gate

        candidate = EvalResult(overall_f1=0.7, overall_precision=0.7, overall_recall=0.7)
        champion = EvalResult(overall_f1=0.8, overall_precision=0.8, overall_recall=0.8)
        passed, reason = _passes_eval_gate(candidate, champion)
        assert not passed
        assert "overall F1" in reason

    def test_fails_axis_regression(self):
        from ml.registry.promotion import _passes_eval_gate

        candidate = EvalResult(
            overall_f1=0.85,
            overall_precision=0.85,
            overall_recall=0.85,
            per_axis={"INTENT": AxisMetrics("INTENT", 0.7, 0.7, 0.7, 100)},
        )
        champion = EvalResult(
            overall_f1=0.8,
            overall_precision=0.8,
            overall_recall=0.8,
            per_axis={"INTENT": AxisMetrics("INTENT", 0.9, 0.9, 0.9, 100)},
        )
        passed, reason = _passes_eval_gate(candidate, champion, max_regression=0.05)
        assert not passed
        assert "regressed" in reason.lower()
