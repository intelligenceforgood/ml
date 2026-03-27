"""Unit tests for risk scoring promotion eval gate (Sprint 4)."""

from __future__ import annotations

from ml.registry.promotion import _passes_risk_scoring_eval_gate
from ml.training.evaluation import RegressionResult


class TestRiskScoringEvalGate:
    """Risk scoring promotion gate checks."""

    def test_first_model_passes_with_good_spearman(self):
        candidate = RegressionResult(mse=0.03, mae=0.1, rmse=0.17, spearman_rho=0.7, total_samples=100)
        passed, reason = _passes_risk_scoring_eval_gate(candidate, None)
        assert passed
        assert "first model" in reason.lower()

    def test_first_model_rejected_low_spearman(self):
        candidate = RegressionResult(mse=0.03, mae=0.1, rmse=0.17, spearman_rho=0.4, total_samples=100)
        passed, reason = _passes_risk_scoring_eval_gate(candidate, None)
        assert not passed
        assert "Spearman" in reason

    def test_candidate_better_than_champion(self):
        champion = RegressionResult(mse=0.05, mae=0.15, rmse=0.22, spearman_rho=0.7, total_samples=100)
        candidate = RegressionResult(mse=0.03, mae=0.1, rmse=0.17, spearman_rho=0.8, total_samples=100)
        passed, reason = _passes_risk_scoring_eval_gate(candidate, champion)
        assert passed

    def test_candidate_mse_regressed(self):
        champion = RegressionResult(mse=0.03, mae=0.1, rmse=0.17, spearman_rho=0.8, total_samples=100)
        candidate = RegressionResult(mse=0.06, mae=0.2, rmse=0.24, spearman_rho=0.75, total_samples=100)
        passed, reason = _passes_risk_scoring_eval_gate(candidate, champion)
        assert not passed
        assert "MSE" in reason

    def test_candidate_low_spearman(self):
        champion = RegressionResult(mse=0.05, mae=0.15, rmse=0.22, spearman_rho=0.7, total_samples=100)
        candidate = RegressionResult(mse=0.04, mae=0.12, rmse=0.2, spearman_rho=0.5, total_samples=100)
        passed, reason = _passes_risk_scoring_eval_gate(candidate, champion)
        assert not passed
        assert "Spearman" in reason
