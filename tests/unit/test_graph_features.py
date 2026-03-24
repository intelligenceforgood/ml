"""Unit tests for graph features Beam pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from ml.data.graph_features import (
    ComputeConnectedComponents,
    ComputePerCaseFeatures,
    EmitCoOccurrencePairs,
    MergeAndEmitRows,
)


class TestEmitCoOccurrencePairs:
    def test_two_cases_emit_one_pair(self):
        dofn = EmitCoOccurrencePairs()
        result = list(dofn.process(("wallet123", ["case_a", "case_b"])))
        assert result == [("case_a", "case_b")]

    def test_three_cases_emit_three_pairs(self):
        dofn = EmitCoOccurrencePairs()
        result = list(dofn.process(("email@test.com", ["case_a", "case_b", "case_c"])))
        assert len(result) == 3
        assert ("case_a", "case_b") in result
        assert ("case_a", "case_c") in result
        assert ("case_b", "case_c") in result

    def test_single_case_emits_nothing(self):
        dofn = EmitCoOccurrencePairs()
        result = list(dofn.process(("unique_entity", ["case_x"])))
        assert result == []

    def test_duplicate_case_ids_deduplicated(self):
        dofn = EmitCoOccurrencePairs()
        result = list(dofn.process(("entity", ["case_a", "case_a", "case_b"])))
        assert result == [("case_a", "case_b")]


class TestComputePerCaseFeatures:
    def test_basic_computation(self):
        dofn = ComputePerCaseFeatures()
        entity_counts = [("wallet123", 3), ("email@x.com", 2)]
        result = list(dofn.process(("case_a", entity_counts)))
        assert len(result) == 1
        case_id, features = result[0]
        assert case_id == "case_a"
        assert features["shared_entity_count"] == 2
        assert features["entity_reuse_frequency"] == 2.5  # (3 + 2) / 2

    def test_empty_entities(self):
        dofn = ComputePerCaseFeatures()
        result = list(dofn.process(("case_x", [])))
        assert len(result) == 1
        _, features = result[0]
        assert features["shared_entity_count"] == 0
        assert features["entity_reuse_frequency"] == 0.0

    def test_single_entity(self):
        dofn = ComputePerCaseFeatures()
        result = list(dofn.process(("case_a", [("wallet", 5)])))
        _, features = result[0]
        assert features["shared_entity_count"] == 1
        assert features["entity_reuse_frequency"] == 5.0


class TestComputeConnectedComponents:
    def test_single_component(self):
        dofn = ComputeConnectedComponents()
        edges = [("a", "b"), ("b", "c")]
        result = dict(dofn.process(edges))
        assert result["a"] == 3
        assert result["b"] == 3
        assert result["c"] == 3

    def test_two_components(self):
        dofn = ComputeConnectedComponents()
        edges = [("a", "b"), ("c", "d")]
        result = dict(dofn.process(edges))
        assert result["a"] == 2
        assert result["b"] == 2
        assert result["c"] == 2
        assert result["d"] == 2

    def test_empty_edges(self):
        dofn = ComputeConnectedComponents()
        result = list(dofn.process([]))
        assert result == []

    def test_triangle(self):
        dofn = ComputeConnectedComponents()
        edges = [("a", "b"), ("b", "c"), ("a", "c")]
        result = dict(dofn.process(edges))
        assert result["a"] == 3
        assert result["b"] == 3
        assert result["c"] == 3


class TestMergeAndEmitRows:
    @patch("ml.data.graph_features.datetime")
    def test_full_merge(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 1, 15, tzinfo=UTC)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        dofn = MergeAndEmitRows()
        grouped = {
            "features": [{"shared_entity_count": 3, "entity_reuse_frequency": 2.5}],
            "clusters": [5],
        }
        result = list(dofn.process(("case_a", grouped)))
        assert len(result) == 1
        row = result[0]
        assert row["case_id"] == "case_a"
        assert row["shared_entity_count"] == 3
        assert row["entity_reuse_frequency"] == 2.5
        assert row["cluster_size"] == 5

    def test_missing_features_defaults(self):
        dofn = MergeAndEmitRows()
        grouped = {"features": [], "clusters": [2]}
        result = list(dofn.process(("case_x", grouped)))
        row = result[0]
        assert row["shared_entity_count"] == 0
        assert row["entity_reuse_frequency"] == 0.0
        assert row["cluster_size"] == 2

    def test_missing_cluster_defaults_to_one(self):
        dofn = MergeAndEmitRows()
        grouped = {
            "features": [{"shared_entity_count": 1, "entity_reuse_frequency": 3.0}],
            "clusters": [],
        }
        result = list(dofn.process(("case_y", grouped)))
        row = result[0]
        assert row["cluster_size"] == 1

    def test_output_schema(self):
        dofn = MergeAndEmitRows()
        grouped = {
            "features": [{"shared_entity_count": 2, "entity_reuse_frequency": 1.5}],
            "clusters": [3],
        }
        result = list(dofn.process(("case_z", grouped)))
        row = result[0]
        expected_keys = {"case_id", "shared_entity_count", "entity_reuse_frequency", "cluster_size", "_computed_at"}
        assert set(row.keys()) == expected_keys
