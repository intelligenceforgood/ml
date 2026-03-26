"""Beam pipeline for graph-based features — entity co-occurrence and connected components.

Computes per-case graph features from entity co-occurrence across cases:
- ``shared_entity_count``: distinct entities shared with other cases
- ``entity_reuse_frequency``: average number of cases each entity appears in
- ``cluster_size``: connected component size via NetworkX

Runs on ``DirectRunner`` locally or ``DataflowRunner`` in production.

Usage::

    python -m ml.data.graph_features \
        --project i4g-ml \
        --dataset i4g_ml \
        --temp-location gs://i4g-ml-data/dataflow/temp

"""

from __future__ import annotations

import argparse
import itertools
import logging
from datetime import UTC, datetime

import apache_beam as beam
from apache_beam import io as beam_io
from apache_beam.options.pipeline_options import PipelineOptions

logger = logging.getLogger(__name__)

# ── BigQuery output schema ───────────────────────────────────────────────────

OUTPUT_TABLE_SCHEMA = {
    "fields": [
        {"name": "case_id", "type": "STRING", "mode": "REQUIRED"},
        {"name": "shared_entity_count", "type": "INT64", "mode": "REQUIRED"},
        {"name": "entity_reuse_frequency", "type": "FLOAT64", "mode": "REQUIRED"},
        {"name": "cluster_size", "type": "INT64", "mode": "REQUIRED"},
        {"name": "_computed_at", "type": "TIMESTAMP", "mode": "REQUIRED"},
    ]
}


# ── DoFns and transforms ────────────────────────────────────────────────────


class EmitCoOccurrencePairs(beam.DoFn):
    """For each entity shared by ≥ 2 cases, emit all (case_i, case_j) pairs."""

    def process(self, element):
        _entity_value, case_ids = element
        unique_cases = sorted(set(case_ids))
        if len(unique_cases) < 2:
            return
        yield from itertools.combinations(unique_cases, 2)


class ComputePerCaseFeatures(beam.DoFn):
    """Compute shared_entity_count and entity_reuse_frequency per case."""

    def process(self, element):
        case_id, entity_case_counts = element
        # entity_case_counts is a list of (entity_value, total_cases_for_entity)
        shared_count = len(entity_case_counts)
        if shared_count == 0:
            yield (case_id, {"shared_entity_count": 0, "entity_reuse_frequency": 0.0})
        else:
            avg_reuse = sum(c for _, c in entity_case_counts) / shared_count
            yield (case_id, {"shared_entity_count": shared_count, "entity_reuse_frequency": avg_reuse})


class ComputeConnectedComponents(beam.DoFn):
    """Build a NetworkX graph from all co-occurrence edges and emit (case_id, cluster_size)."""

    def process(self, elements):
        import networkx as nx

        edges = list(elements)
        if not edges:
            return

        g = nx.Graph()
        g.add_edges_from(edges)

        for component in nx.connected_components(g):
            size = len(component)
            for case_id in component:
                yield (case_id, size)


class MergeAndEmitRows(beam.DoFn):
    """Join per-case features with cluster_size and emit final BQ rows."""

    def process(self, element):
        case_id, grouped = element
        features_list = grouped.get("features", [])
        cluster_list = grouped.get("clusters", [])

        features = features_list[0] if features_list else {"shared_entity_count": 0, "entity_reuse_frequency": 0.0}
        cluster_size = cluster_list[0] if cluster_list else 1  # isolated node = cluster of 1

        yield {
            "case_id": case_id,
            "shared_entity_count": features["shared_entity_count"],
            "entity_reuse_frequency": round(features["entity_reuse_frequency"], 4),
            "cluster_size": cluster_size,
            "_computed_at": datetime.now(UTC).isoformat(),
        }


def build_pipeline(pipeline, project: str, dataset: str, output_table: str | None = None):
    """Construct the Beam pipeline graph.

    Parameters
    ----------
    pipeline : beam.Pipeline
        The pipeline to attach transforms to.
    project : str
        GCP project ID.
    dataset : str
        BigQuery dataset ID.
    output_table : str | None
        Full BQ table reference. Defaults to ``{project}.{dataset}.features_graph_features``.
    """
    if output_table is None:
        output_table = f"{project}.{dataset}.features_graph_features"

    source_query = f"""
        SELECT
            canonical_value AS entity_value,
            case_id
        FROM `{project}.{dataset}.raw_entities`
        WHERE canonical_value IS NOT NULL
          AND entity_type IS NOT NULL
    """

    # Stage 1: Read entity-case pairs from BigQuery
    entity_case_pairs = (
        pipeline
        | "ReadEntities" >> beam_io.ReadFromBigQuery(query=source_query, use_standard_sql=True, project=project)
        | "ExtractPairs" >> beam.Map(lambda row: (row["entity_value"], row["case_id"]))
    )

    # Stage 2: Group by entity_value → list of case_ids
    entities_grouped = entity_case_pairs | "GroupByEntity" >> beam.GroupByKey()

    # Stage 3: Emit co-occurrence pairs for entities shared by ≥ 2 cases
    co_occurrence_pairs = entities_grouped | "EmitPairs" >> beam.ParDo(EmitCoOccurrencePairs())

    # Stage 4: Per-case aggregation — shared entities and reuse frequency
    # First, compute per-entity case count for entities with ≥ 2 cases
    entity_with_counts = entities_grouped | "EntityCounts" >> beam.Map(lambda kv: (kv[0], len(set(kv[1]))))

    # Filter to shared entities only (count ≥ 2)
    shared_entities = entity_with_counts | "FilterShared" >> beam.Filter(lambda kv: kv[1] >= 2)

    # Re-join with case_ids to get (case_id, (entity_value, total_cases))
    entity_case_with_count = (
        {
            "pairs": entity_case_pairs,
            "counts": shared_entities,
        }
        | "CoGroupEntityCounts" >> beam.CoGroupByKey()
        | "FlattenCaseCounts"
        >> beam.FlatMap(
            lambda kv: [(case_id, (kv[0], kv[1]["counts"][0])) for case_id in set(kv[1]["pairs"]) if kv[1]["counts"]]
        )
    )

    per_case_features = (
        entity_case_with_count
        | "GroupByCase" >> beam.GroupByKey()
        | "ComputeFeatures" >> beam.ParDo(ComputePerCaseFeatures())
        | "UnpackFeatures" >> beam.Map(lambda kv: (kv[0], kv[1]))
    )

    # Stage 5: Connected components — collect all edges, compute in one DoFn
    cluster_sizes = (
        co_occurrence_pairs
        | "CollectEdges" >> beam.combiners.ToList()
        | "ConnectedComponents" >> beam.ParDo(ComputeConnectedComponents())
    )

    # Stage 6: Merge features + cluster sizes → write to BigQuery
    merged = (
        {"features": per_case_features, "clusters": cluster_sizes}
        | "CoGroupResults" >> beam.CoGroupByKey()
        | "MergeAndEmit" >> beam.ParDo(MergeAndEmitRows())
    )

    _ = merged | "WriteToBigQuery" >> beam_io.WriteToBigQuery(
        output_table,
        schema=OUTPUT_TABLE_SCHEMA,
        write_disposition=beam_io.BigQueryDisposition.WRITE_TRUNCATE,
        create_disposition=beam_io.BigQueryDisposition.CREATE_IF_NEEDED,
    )

    return pipeline


def parse_args(argv=None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Graph features Beam pipeline")
    parser.add_argument("--project", default="i4g-ml", help="GCP project ID")
    parser.add_argument("--dataset", default="i4g_ml", help="BigQuery dataset ID")
    parser.add_argument("--runner", default="DirectRunner", help="Beam runner")
    parser.add_argument("--temp-location", default="gs://i4g-ml-data/dataflow/temp", help="GCS temp location")
    parser.add_argument("--staging-location", default="gs://i4g-ml-data/dataflow/staging", help="GCS staging location")
    parser.add_argument("--region", default="us-central1", help="Dataflow region")
    parser.add_argument("--service-account-email", default=None, help="Service account for Dataflow workers")
    parser.add_argument("--requirements-file", default=None, help="Pip requirements file for Dataflow workers")
    return parser.parse_known_args(argv)


def main(argv=None):
    args, pipeline_args = parse_args(argv)

    options_kwargs = dict(
        runner=args.runner,
        project=args.project,
        temp_location=args.temp_location,
        staging_location=args.staging_location,
        region=args.region,
    )
    if args.service_account_email:
        options_kwargs["service_account_email"] = args.service_account_email
    if args.requirements_file:
        options_kwargs["requirements_file"] = args.requirements_file

    pipeline_options = PipelineOptions(pipeline_args, **options_kwargs)

    with beam.Pipeline(options=pipeline_options) as p:
        build_pipeline(p, project=args.project, dataset=args.dataset)

    logger.info("Graph features pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
