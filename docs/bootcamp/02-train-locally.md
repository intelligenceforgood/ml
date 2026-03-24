# Exercise 2: Train a Model Locally

> **Objective:** Run the XGBoost training container on a sample dataset and produce a model artifact.
> **Prerequisites:** Exercise 1 completed, `conda activate ml`, Docker installed
> **Time:** ~30 minutes

---

## Overview

Training on the ML platform uses containerized training jobs. Each framework (XGBoost, PyTorch NER) has its own container with a `train.py` entry point. Locally, you run the container with Docker; in production, Vertex AI runs the same container.

```
Dataset (GCS JSONL) → Training Container → Model Artifact (GCS)
                                              ↓
                                        label_map.json
                                        xgboost_model.json (or model.safetensors)
                                        training_config.json
```

---

## Step 1: Examine the training container

Look at the XGBoost training code:

```bash
cat containers/train-xgboost/train.py
```

**What to notice:**

- Reads training config from environment variables or a config YAML
- Downloads JSONL dataset from GCS
- Extracts tabular features and label columns
- Trains an XGBoost classifier
- Exports: `xgboost_model.json`, `label_map.json`, `training_config.json`
- Logs metrics to Vertex AI Experiments (when running on Vertex AI)

Check the container's dependencies:

```bash
cat containers/train-xgboost/requirements.txt
```

---

## Step 2: Prepare sample training data

Create a small synthetic dataset for local training:

```bash
mkdir -p data/local-training

# Create a minimal synthetic training set
python -c "
import json

samples = []
for i in range(100):
    samples.append(json.dumps({
        'case_id': f'case-{i:04d}',
        'text': f'Sample case narrative number {i} about financial fraud',
        'text_length': 50 + i,
        'word_count': 10 + (i % 20),
        'entity_count': i % 5,
        'unique_entity_types': i % 3,
        'lexical_diversity': 0.5 + (i % 50) / 100,
        'labels': {'INTENT': 'INTENT.ROMANCE' if i % 2 == 0 else 'INTENT.INVESTMENT'},
        'label_source': 'analyst'
    }))
print('\n'.join(samples))
" > data/local-training/train.jsonl

echo "Created $(wc -l < data/local-training/train.jsonl) training samples"
```

---

## Step 3: Build the training container

```bash
docker build -f docker/train-xgboost.Dockerfile -t train-xgboost:local .
```

**What just happened:** Docker built the training image using the Dockerfile, which installs Python dependencies and copies the training script.

---

## Step 4: Run training locally

Run the container with the synthetic dataset mounted:

```bash
docker run --rm \
  -v "$(pwd)/data/local-training:/data" \
  -e TRAIN_DATA_PATH=/data/train.jsonl \
  -e OUTPUT_DIR=/data/output \
  -e MODEL_ID=xgboost-local-test \
  -e CAPABILITY=classification \
  train-xgboost:local
```

**Expected output:**

- Training logs showing XGBoost iterations
- Final metrics (accuracy, F1) printed to stdout
- Model artifacts written to `data/local-training/output/`

---

## Step 5: Inspect the model artifact

```bash
ls -la data/local-training/output/

# Check the label map
cat data/local-training/output/label_map.json | python -m json.tool

# Check the training config
cat data/local-training/output/training_config.json | python -m json.tool
```

**What to notice:**

- `xgboost_model.json`: the serialized XGBoost booster (the actual model weights)
- `label_map.json`: maps integer predictions back to taxonomy labels, grouped by axis
- `training_config.json`: records hyperparameters, dataset version, and training metadata for reproducibility

---

## Step 6: Understand the pipeline config

Training in production is configured via YAML files:

```bash
cat pipelines/configs/classification_xgboost.yaml
```

**What to notice:**

- `capability`: which ML task this config is for
- `framework`: which training container to use
- `model_id`: identifier for the trained model
- Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, etc.
- `vizier_search_space`: ranges for automated hyperparameter tuning (Exercise 6)

---

## Step 7: Clean up

```bash
rm -rf data/local-training
```

---

## Summary

| Step                   | What you did                        | Key file                               |
| ---------------------- | ----------------------------------- | -------------------------------------- |
| Examined training code | Understood how the container trains | `containers/train-xgboost/train.py`    |
| Created synthetic data | Produced a minimal JSONL dataset    | —                                      |
| Built container        | Packaged training code into Docker  | `docker/train-xgboost.Dockerfile`      |
| Ran training           | Produced model artifacts locally    | `containers/train-xgboost/train.py`    |
| Inspected artifacts    | Understood output format            | `label_map.json`, `xgboost_model.json` |

**Next exercise:** [03 — Submit a Pipeline](03-submit-pipeline.md), where you send this training job to Vertex AI Pipelines on GCP.
