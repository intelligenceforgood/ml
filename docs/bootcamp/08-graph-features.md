# Exercise 8: Graph Features Pipeline

> **Objective:** Understand the Dataflow/Beam pipeline for graph-based features and run it locally with `DirectRunner`.
> **Prerequisites:** Exercises 1–4 completed, `pip install -e ".[graph]"` (installs `apache-beam` and `networkx`)
> **Time:** ~30 minutes

> **Partially offline.** Steps 1–4 and 6–7 are code reading (no GCP needed). Step 5’s
> `DirectRunner` requires a live BigQuery connection; use the unit test fallback instead:
> `pytest tests/unit/test_graph_features.py -v`

---

## Overview

The graph features pipeline computes per-case features from entity co-occurrence across fraud cases. If two cases share the same entity (phone number, crypto wallet, etc.), they're likely related. The pipeline answers: "How connected is this case to other cases?"

```
BigQuery raw_entities → Entity co-occurrence pairs → Per-case aggregation
                              ↓                              ↓
                     Connected components          shared_entity_count
                       (NetworkX)                  entity_reuse_frequency
                              ↓
                        cluster_size
                              ↓
                  BigQuery features_graph_features
```

**Architecture decision:** Dataflow/Beam was chosen over Spark/Dataproc because the computation is co-occurrence aggregations + connected components — not heavy iterative graph algorithms. Beam handles this with native BigQuery I/O, pay-per-use autoscaling, and zero cluster configuration.

---

## Step 1: Install graph dependencies

```bash
pip install -e ".[graph]"
```

This installs `apache-beam[gcp]` and `networkx`.

---

## Step 2: Understand the pipeline stages

```bash
head -60 src/ml/data/graph_features.py
```

The pipeline has 6 stages:

1. **Read:** Query `raw_entities` from BigQuery → `(entity_value, case_id)` pairs
2. **Group:** `GroupByKey` on `entity_value` → collect all case IDs per entity
3. **Co-occurrence:** For entities appearing in ≥ 2 cases, emit all `(case_i, case_j)` pairs
4. **Per-case features:** For each case, compute:
   - `shared_entity_count`: number of distinct entities shared with other cases
   - `entity_reuse_frequency`: average number of cases each entity appears in
5. **Connected components:** Build a NetworkX graph from co-occurrence edges, compute `connected_components()`, emit `(case_id, cluster_size)`
6. **Write:** Join features + cluster sizes, write to `features_graph_features` BigQuery table

---

## Step 3: Examine the DoFns

```bash
grep -A 15 "class.*DoFn" src/ml/data/graph_features.py
```

**What to notice:**

- `EmitCoOccurrencePairs`: generates all `(case_i, case_j)` combinations for entities shared across cases
- `ComputePerCaseFeatures`: aggregates co-occurrence data per case
- Connected components use NetworkX `connected_components()` — viable at our scale (~10K cases, sparse graph)

---

## Step 4: Check the BigQuery output schema

```bash
cat pipelines/sql/features_graph_features.sql
```

The output table has:

- `case_id`: the case identifier
- `shared_entity_count`: how many entities this case shares with other cases
- `entity_reuse_frequency`: average reuse count of this case's entities
- `cluster_size`: size of the connected component this case belongs to
- `_computed_at`: timestamp of computation

---

## Step 5: Run locally with DirectRunner

The pipeline supports `DirectRunner` for local testing:

```bash
# Note: DirectRunner with BigQuery I/O requires a live GCP connection.
# For fully offline testing, the unit tests mock BigQuery reads.

# If you have GCP access:
python -m ml.data.graph_features \
  --project i4g-ml \
  --dataset i4g_ml \
  --runner DirectRunner
```

If you don't have GCP access, run the unit tests instead:

```bash
pytest tests/unit/data/test_graph_features.py -v
```

---

## Step 6: Check feature registration

The graph features are registered in the feature catalog:

```bash
grep -A 8 "shared_entity_count\|entity_reuse_frequency\|cluster_size" src/ml/data/features.py
```

**What to notice:** These features use `compute_method=ComputeMethod.DATAFLOW`, distinguishing them from the BigQuery SQL features.

---

## Step 7: Understand dataset integration

The dataset export LEFT JOINs graph features when available:

```bash
grep -A 5 "graph_features\|LEFT JOIN.*features_graph" src/ml/data/datasets.py
```

**What to notice:** Graph features are nullable — models handle missing graph features gracefully (e.g., for cases processed before the first graph features run).

---

## Step 8: Production deployment

In production, the pipeline runs on `DataflowRunner` via a Cloud Run Job:

```bash
# Cloud Run Job (weekly, Sunday 4 AM UTC):
make submit-graph-features-dev
```

The Makefile target shows the full `DataflowRunner` invocation:

```bash
grep -A 7 "submit-graph-features-dev" Makefile
```

---

## Summary

| Feature                        | Value                           | Interpretation                   |
| ------------------------------ | ------------------------------- | -------------------------------- |
| `shared_entity_count = 0`      | No shared entities              | Case is isolated                 |
| `shared_entity_count = 5`      | 5 shared entities               | Moderate connections             |
| `entity_reuse_frequency = 3.2` | Each entity appears in ~3 cases | Entities are reused across cases |
| `cluster_size = 1`             | Standalone case                 | Not connected to other cases     |
| `cluster_size = 50`            | Part of a 50-case cluster       | Potentially a fraud ring         |

**Next exercise:** [09 — Looker Studio Dashboard](09-looker-dashboard.md), where you build a monitoring dashboard from BigQuery analytics tables.
