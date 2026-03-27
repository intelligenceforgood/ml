# ML Bootcamp

Hands-on exercises that walk you through the I4G ML platform end-to-end — from raw data
to a deployed, monitored model.

## Prerequisites

Before starting, make sure you have:

| Requirement                       | How to verify                                                    | Used in               |
| --------------------------------- | ---------------------------------------------------------------- | --------------------- |
| **Conda env `ml`**                | `conda activate ml && python --version`                          | All exercises         |
| **Editable install**              | `pip install -e ".[dev]"`                                        | All exercises         |
| **Docker**                        | `docker --version`                                               | Exercise 2            |
| **GCP project access** (`i4g-ml`) | `gcloud projects describe i4g-ml`                                | Exercises 1, 3–6, 8–9 |
| **`gcloud` CLI authenticated**    | `gcloud auth application-default login`                          | Exercises 1, 3–6, 8–9 |
| **Looker Studio access**          | Open [lookerstudio.google.com](https://lookerstudio.google.com/) | Exercise 9            |

> **No GCP access yet?** Exercises 2, 4 (Steps 4–5), and 7 work fully offline.
> For GCP-dependent exercises, each one marks which steps need GCP and suggests unit tests
> you can run locally instead.

## Exercises

Work through these in order — each builds on the previous.

### Fundamentals (start here)

| #   | Exercise                                             | What you learn                                                        | Time    |
| --- | ---------------------------------------------------- | --------------------------------------------------------------------- | ------- |
| 1   | [Data Flow Walkthrough](01-data-flow-walkthrough.md) | How case data flows from Cloud SQL → BigQuery → GCS training datasets | ~30 min |
| 2   | [Train a Model Locally](02-train-locally.md)         | Run the XGBoost training container on synthetic data with Docker      | ~30 min |
| 3   | [Submit a Pipeline](03-submit-pipeline.md)           | Compile a KFP v2 pipeline and run it on Vertex AI                     | ~45 min |

### Model lifecycle

| #   | Exercise                                           | What you learn                                                 | Time    |
| --- | -------------------------------------------------- | -------------------------------------------------------------- | ------- |
| 4   | [Evaluate and Promote](04-evaluate-and-promote.md) | Eval metrics, the promotion gate, and model lifecycle stages   | ~20 min |
| 5   | [Deploy to Serving](05-deploy-to-serving.md)       | Deploy to Cloud Run, send predictions, verify BigQuery logging | ~25 min |
| 6   | [Monitor and Retrain](06-monitor-and-retrain.md)   | Drift detection, retraining triggers, cost tracking            | ~30 min |

### Advanced topics

| #   | Exercise                                          | What you learn                                                        | Time    |
| --- | ------------------------------------------------- | --------------------------------------------------------------------- | ------- |
| 7   | [Add a New Capability](07-add-new-capability.md)  | The multi-capability pattern — add a toy capability across all layers | ~60 min |
| 8   | [Graph Features Pipeline](08-graph-features.md)   | Dataflow/Beam pipeline for cross-case entity analysis                 | ~30 min |
| 9   | [Looker Studio Dashboard](09-looker-dashboard.md) | Build a monitoring dashboard from BigQuery analytics tables           | ~45 min |

**Total time:** ~5.5 hours (spread across multiple sessions is recommended)

## Running unit tests locally

If you want to validate the code without GCP access, the unit test suite covers all core logic:

```bash
pytest tests/unit -v
```

For a specific module (e.g., evaluation):

```bash
pytest tests/unit/test_evaluation.py -v
```

## Quick reference

| What                | Where                         |
| ------------------- | ----------------------------- |
| Source code         | `src/ml/`                     |
| Training containers | `containers/`                 |
| Pipeline configs    | `pipelines/configs/`          |
| SQL views           | `pipelines/sql/`              |
| Platform settings   | `config/settings.*.toml`      |
| Architecture doc    | `docs/design/architecture.md` |
