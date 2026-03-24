"""Unit tests for cost monitoring with mock billing data."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ml.monitoring.cost import CostComparison, CostSummary, compare_to_llm_cost, compute_cost_summary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings():
    return SimpleNamespace(
        platform=SimpleNamespace(project_id="test-project"),
        bigquery=SimpleNamespace(
            dataset_id="test_dataset",
            prediction_log_table="predictions_prediction_log",
            outcome_log_table="predictions_outcome_log",
        ),
    )


# ---------------------------------------------------------------------------
# Tests — compute_cost_summary
# ---------------------------------------------------------------------------


class TestComputeCostSummary:
    @patch("ml.monitoring.cost.get_settings")
    def test_no_billing_table(self, mock_settings):
        """Without billing table, returns zero-cost summary with prediction count."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        # prediction count query returns 100
        mock_client.query.return_value.result.return_value = [SimpleNamespace(total=100)]

        summary = compute_cost_summary(period_days=30, billing_table="", client=mock_client)

        assert isinstance(summary, CostSummary)
        assert summary.total_cost_usd == 0.0
        assert summary.total_predictions == 100
        assert summary.cost_per_prediction_usd == 0.0

    @patch("ml.monitoring.cost.get_settings")
    def test_with_billing_data(self, mock_settings):
        """With billing data, aggregates costs correctly."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()

        # First call: prediction count
        count_result = MagicMock()
        count_result.result.return_value = [SimpleNamespace(total=1000)]

        # Second call: billing export
        billing_result = MagicMock()
        billing_result.result.return_value = [
            SimpleNamespace(service_name="Vertex AI", total_cost=15.50),
            SimpleNamespace(service_name="Cloud Run", total_cost=8.20),
            SimpleNamespace(service_name="BigQuery", total_cost=2.30),
            SimpleNamespace(service_name="Cloud Storage", total_cost=1.00),
        ]

        mock_client.query.side_effect = [count_result, billing_result]

        summary = compute_cost_summary(
            period_days=30,
            billing_table="billing-project.billing_dataset.gcp_billing_export",
            client=mock_client,
        )

        assert summary.total_cost_usd == 27.0
        assert summary.total_predictions == 1000
        assert summary.cost_per_prediction_usd == 0.027
        assert len(summary.components) == 4

    @patch("ml.monitoring.cost.get_settings")
    def test_zero_predictions(self, mock_settings):
        """Zero predictions should produce zero cost per prediction."""
        mock_settings.return_value = _fake_settings()
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = [SimpleNamespace(total=0)]

        summary = compute_cost_summary(period_days=7, billing_table="", client=mock_client)

        assert summary.total_predictions == 0
        assert summary.cost_per_prediction_usd == 0.0


# ---------------------------------------------------------------------------
# Tests — compare_to_llm_cost
# ---------------------------------------------------------------------------


class TestCompareToLlmCost:
    @patch("ml.monitoring.cost.compute_cost_summary")
    def test_savings_calculation(self, mock_summary):
        """ML platform should show savings vs LLM at typical volumes."""
        mock_summary.return_value = CostSummary(
            period_days=30,
            total_cost_usd=27.0,
            total_predictions=1000,
            cost_per_prediction_usd=0.027,
            computed_at="2026-03-23",
        )

        comparison = compare_to_llm_cost(
            period_days=30,
            llm_cost_per_prediction=0.03,
        )

        assert isinstance(comparison, CostComparison)
        assert comparison.ml_cost_per_prediction == 0.027
        assert comparison.llm_cost_per_prediction == 0.03
        assert comparison.savings_per_prediction == 0.003
        assert comparison.ml_total_cost == 27.0
        assert comparison.llm_equivalent_cost == 30.0
        assert comparison.total_predictions == 1000

    @patch("ml.monitoring.cost.compute_cost_summary")
    def test_ml_more_expensive(self, mock_summary):
        """When ML is more expensive, savings should be negative."""
        mock_summary.return_value = CostSummary(
            period_days=30,
            total_cost_usd=50.0,
            total_predictions=1000,
            cost_per_prediction_usd=0.05,
            computed_at="2026-03-23",
        )

        comparison = compare_to_llm_cost(
            period_days=30,
            llm_cost_per_prediction=0.03,
        )

        assert comparison.savings_per_prediction < 0
        assert comparison.savings_percentage < 0

    @patch("ml.monitoring.cost.compute_cost_summary")
    def test_zero_predictions(self, mock_summary):
        """Zero predictions should not cause division errors."""
        mock_summary.return_value = CostSummary(
            period_days=7,
            total_cost_usd=0.0,
            total_predictions=0,
            cost_per_prediction_usd=0.0,
            computed_at="2026-03-23",
        )

        comparison = compare_to_llm_cost(period_days=7)

        assert comparison.total_predictions == 0
        assert comparison.savings_percentage == 0.0
