"""Unit tests for regression evaluation (Sprint 4 — Risk Scoring)."""

from __future__ import annotations

import pytest

from ml.training.evaluation import RegressionResult, evaluate_regression


class TestEvaluateRegression:
    """Regression metrics computation."""

    def test_perfect_prediction(self):
        preds = [0.1, 0.5, 0.9]
        truth = [0.1, 0.5, 0.9]
        result = evaluate_regression(preds, truth)
        assert result.mse == pytest.approx(0.0, abs=1e-6)
        assert result.mae == pytest.approx(0.0, abs=1e-6)
        assert result.spearman_rho == pytest.approx(1.0, abs=1e-6)
        assert result.total_samples == 3

    def test_imperfect_prediction(self):
        preds = [0.2, 0.6, 0.8]
        truth = [0.1, 0.5, 0.9]
        result = evaluate_regression(preds, truth)
        assert result.mse > 0
        assert result.mae > 0
        assert result.rmse == pytest.approx(result.mse**0.5)
        # Order is preserved → Spearman should be 1.0
        assert result.spearman_rho == pytest.approx(1.0, abs=1e-6)

    def test_inverse_ranking(self):
        preds = [0.9, 0.5, 0.1]
        truth = [0.1, 0.5, 0.9]
        result = evaluate_regression(preds, truth)
        assert result.spearman_rho < 0  # Inverse ranking → negative Spearman

    def test_empty_input(self):
        result = evaluate_regression([], [])
        assert result.total_samples == 0
        assert result.mse == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_regression([0.1, 0.2], [0.1])

    def test_summary_string(self):
        result = RegressionResult(mse=0.05, mae=0.1, rmse=0.22, spearman_rho=0.85, total_samples=100)
        s = result.summary()
        assert "MSE=" in s
        assert "Spearman=" in s


class TestSpearmanEdgeCases:
    """Edge cases for Spearman rank correlation."""

    def test_single_sample(self):
        result = evaluate_regression([0.5], [0.5])
        assert result.spearman_rho == 0.0  # Cannot compute with n=1

    def test_tied_values(self):
        preds = [0.5, 0.5, 0.5]
        truth = [0.1, 0.5, 0.9]
        result = evaluate_regression(preds, truth)
        # All preds tied → rank correlation is 0
        assert result.spearman_rho == pytest.approx(0.0, abs=0.1)
