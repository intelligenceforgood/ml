# Exercise 3: Submit a Pipeline

> **Objective:** Compile a KFP v2 pipeline and submit it to Vertex AI on the dev environment. Monitor it to completion.
> **Prerequisites:** Exercise 2 completed, `gcloud auth application-default login`, access to `i4g-ml` GCP project
> **Time:** ~45 minutes (including pipeline execution wait)

> **Requires GCP access.** Step 2 (compile) works offline, but submission and monitoring need
> a live `i4g-ml` project. Without access, complete Steps 1–2, read the remaining steps, then
> run `pytest tests/unit/test_submit_pipeline.py -v` to validate the submission logic.

---

## Overview

The training pipeline is defined with Kubeflow Pipelines (KFP) v2 and runs on Vertex AI Pipelines. The pipeline orchestrates these steps:

```
Prepare data → Train model → Evaluate → Register in Model Registry → (optional) Deploy
```

Each step runs in its own container. The pipeline YAML is compiled from Python and submitted via a script.

---

## Step 1: Examine the pipeline definition

```bash
head -80 src/ml/training/pipeline.py
```

**What to notice:**

- The pipeline is defined as a Python function decorated with `@kfp.dsl.pipeline`
- Each step is a KFP component (containerized function)
- Steps are connected via inputs/outputs — KFP handles data passing between containers
- The `training_pipeline` function accepts parameters like `capability`, `config_path`, `model_id`

---

## Step 2: Compile the pipeline

The pipeline Python definition must be compiled to a YAML file before submission:

```bash
make compile-pipeline
```

Verify the compiled output:

```bash
ls -la pipelines/training_pipeline.yaml
head -20 pipelines/training_pipeline.yaml
```

**What just happened:** KFP's compiler converted the Python pipeline graph into a YAML spec that Vertex AI Pipelines understands. This YAML is what gets submitted to GCP.

---

## Step 3: Understand the submission logic

```bash
head -80 src/ml/cli/pipeline.py
```

**What to notice:**

- `_auto_compile_if_stale()`: automatically recompiles if the source `.py` is newer than the compiled `.yaml`
- `submit_pipeline()`: initializes Vertex AI, loads config, creates a `PipelineJob`, and submits it
- Tags each run with: capability, trigger reason, dataset version, timestamp — for tracking
- The same function is called by both manual runs and the automated retraining trigger

---

## Step 4: Review the pipeline config

```bash
cat pipelines/configs/classification_xgboost.yaml
```

This YAML configures the pipeline run: which framework, hyperparameters, eval metrics, and search spaces.

---

## Step 5: Submit the pipeline

```bash
i4g-ml pipeline submit \
  --config pipelines/configs/classification_xgboost.yaml
```

**Expected output:**

- "Compiled to pipelines/training_pipeline.yaml" (if recompiled)
- "Submitting pipeline job..." with a Vertex AI pipeline job resource name
- A URL to the Vertex AI Pipelines console

Copy the pipeline job URL and open it in your browser.

---

## Step 6: Monitor the pipeline

In the Vertex AI console, you can see:

- Each step's status (pending → running → succeeded/failed)
- Logs from each container
- Input/output artifacts for each step
- Metrics logged during evaluation

You can also check status from the CLI:

```bash
gcloud ai custom-jobs list --project=i4g-ml --region=us-central1 --limit=3
```

Wait for the pipeline to complete (typically 10–20 minutes for XGBoost).

---

## Step 7: Check the pipeline outputs

After the pipeline completes, open **Vertex AI Pipelines** in the console and click into
your pipeline run.

**7a. Check the evaluate_model node.** Click it and inspect the `metrics_json` output.
You should see `"eval_gate_passed": true` along with `overall_f1`, `per_axis` breakdowns,
and `total_samples`. If `eval_gate_passed` is `false`, look for an `"error"` field —
common causes:

- `"Empty golden test set"` — the golden JSONL at
  `gs://i4g-ml-data/datasets/classification/golden/test.jsonl` has no data
- `"label_map.json not found in model artifacts"` — the training container
  didn't write `label_map.json` to the output directory
- `"Unknown model type in artifacts"` — neither `xgboost_model.json` nor a
  PyTorch `model/` dir was found at the artifact path

**7b. Check the register_model node.** If `eval_gate_passed` was `false`, this step
returns `"SKIPPED"` silently — the pipeline still shows as succeeded, but no model
was registered. You can verify:

```bash
gcloud ai models list --project=i4g-ml --region=us-central1 --limit=3
```

If your model doesn't appear, the eval gate blocked registration. Fix the underlying
issue (missing golden data or artifacts) and resubmit.

**7c. Confirm model artifacts in GCS** regardless of registration status:

```bash
# The pipeline writes artifacts to gs://i4g-ml-data/models/<experiment_name>/
# experiment_name = <model_id>-<timestamp>, e.g. classification-xgboost-v1-20260325-2151
gsutil ls gs://i4g-ml-data/models/ | tail -5
```

**7d. Check the training dataset registry** for the dataset version used:

```bash
bq query --use_legacy_sql=false \
  'SELECT dataset_id, capability, version, eval_count, test_count, created_at
   FROM `i4g-ml.i4g_ml.training_dataset_registry`
   ORDER BY created_at DESC
   LIMIT 3'
```

---

## Summary

| Step                     | What you did                 | Key file                           |
| ------------------------ | ---------------------------- | ---------------------------------- |
| Read pipeline definition | Understood the KFP DAG       | `training/pipeline.py`             |
| Compiled pipeline        | Generated YAML from Python   | `pipelines/training_pipeline.yaml` |
| Submitted pipeline       | Sent job to Vertex AI        | `i4g-ml pipeline submit`           |
| Monitored execution      | Tracked steps in console     | Vertex AI Pipelines UI             |
| Verified outputs         | Confirmed model registration | Model Registry + BigQuery          |

**Next exercise:** [04 — Evaluate and Promote](04-evaluate-and-promote.md), where you interpret evaluation metrics and promote the trained model.
