# Exercise 3: Submit a Pipeline

> **Objective:** Compile a KFP v2 pipeline and submit it to Vertex AI on the dev environment. Monitor it to completion.
> **Prerequisites:** Exercise 2 completed, `gcloud auth application-default login`, access to `i4g-ml` GCP project
> **Time:** ~45 minutes (including pipeline execution wait)

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
conda run -n ml make compile-pipeline
```

Verify the compiled output:

```bash
ls -la pipelines/training_pipeline.yaml
head -20 pipelines/training_pipeline.yaml
```

**What just happened:** KFP's compiler converted the Python pipeline graph into a YAML spec that Vertex AI Pipelines understands. This YAML is what gets submitted to GCP.

---

## Step 3: Understand the submission script

```bash
head -80 scripts/submit_pipeline.py
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
conda run -n ml python scripts/submit_pipeline.py \
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

After the pipeline completes, verify the model was registered:

```bash
gcloud ai models list --project=i4g-ml --region=us-central1 --limit=3
```

Check the training dataset registry for the dataset version used:

```bash
bq query --use_legacy_sql=false \
  'SELECT dataset_id, capability, version, sample_count, created_at
   FROM `i4g-ml.i4g_ml.training_dataset_registry`
   ORDER BY created_at DESC
   LIMIT 3'
```

---

## Summary

| Step | What you did | Key file |
|------|-------------|----------|
| Read pipeline definition | Understood the KFP DAG | `training/pipeline.py` |
| Compiled pipeline | Generated YAML from Python | `pipelines/training_pipeline.yaml` |
| Submitted pipeline | Sent job to Vertex AI | `scripts/submit_pipeline.py` |
| Monitored execution | Tracked steps in console | Vertex AI Pipelines UI |
| Verified outputs | Confirmed model registration | Model Registry + BigQuery |

**Next exercise:** [04 — Evaluate and Promote](04-evaluate-and-promote.md), where you interpret evaluation metrics and promote the trained model.
