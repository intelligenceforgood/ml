# Exercise 2: Train a Model Locally

> **Objective:** Run the XGBoost training container on a sample dataset and produce a model artifact.
> **Prerequisites:** Exercise 1 completed, conda env `ml` activated, Docker installed
> **Time:** ~30 minutes

---

## Overview

Training on the ML platform uses containerized training jobs. Each framework (XGBoost, PyTorch NER) has its own container with a `train.py` entry point. Locally, you run the container with Docker; in production, Vertex AI runs the same container.

The container accepts CLI args pointing to a config YAML, a dataset directory, and an output path. All three can be local paths (for development) or `gs://` URIs (for production on Vertex AI).

```
Config YAML + Dataset (JSONL splits) → Training Container → Model Artifacts
                                                                ↓
                                                          model.json
                                                          label_encoder.pkl
                                                          feature_cols.json
                                                          metrics.json
```

---

## Step 1: Examine the training container

Look at the XGBoost training code:

```bash
cat containers/train-xgboost/train.py
```

**What to notice:**

- Uses `argparse` with four flags: `--config`, `--dataset`, `--experiment`, `--output`
- Accepts both local paths and GCS URIs (auto-detects by `gs://` prefix)
- Reads features from a nested `features` dict in each JSONL record
- Labels come from a nested `labels` dict keyed by taxonomy axis (e.g., `INTENT`)
- Outputs: `model.json` (XGBoost booster weights), `label_encoder.pkl` (label mapping), `feature_cols.json`, `metrics.json`
- Logs metrics to Vertex AI Experiments when running on GCP

Check the container's dependencies:

```bash
cat containers/train-xgboost/requirements.txt
```

---

## Step 2: Create a local training config

The container reads training parameters from a YAML config. Create one for local testing:

```bash
mkdir -p data/local-training

cat > data/local-training/config.yaml << 'EOF'
model_id: "xgboost-local-test"
capability: "classification"
framework: "xgboost"

hyperparameters:
  max_depth: 4
  learning_rate: 0.1
  num_boost_round: 50

label_schema:
  INTENT:
    - INTENT.ROMANCE
    - INTENT.INVESTMENT
EOF
```

**What just happened:** You created a minimal version of the production config at `pipelines/configs/classification_xgboost.yaml`. The `label_schema` tells the trainer which axes and labels to expect. The `hyperparameters` section becomes XGBoost booster params.

---

## Step 3: Prepare sample training data

Create synthetic JSONL files matching the format `train.py` expects. The container needs `train.jsonl` and `eval.jsonl` splits in the dataset directory. Each record has a `features` dict (tabular features) and a `labels` dict (taxonomy labels):

```bash
python -c "
import json, random

random.seed(42)

def make_record(i, intent):
    return json.dumps({
        'case_id': f'case-{i:04d}',
        'features': {
            'text_length': random.randint(50, 500),
            'word_count': random.randint(10, 100),
            'entity_count': random.randint(0, 8),
            'has_crypto_wallet': random.randint(0, 1),
            'has_bank_account': random.randint(0, 1),
            'has_phone': random.randint(0, 1),
            'has_email': random.randint(0, 1),
            'classification_axis_count': random.randint(1, 3),
            'current_classification_conf': round(random.uniform(0.3, 0.95), 2),
        },
        'labels': {'INTENT': intent},
    })

intents = ['INTENT.ROMANCE', 'INTENT.INVESTMENT']

# 80 train + 20 eval
train = [make_record(i, intents[i % 2]) for i in range(80)]
eval_ = [make_record(i + 80, intents[i % 2]) for i in range(20)]

with open('data/local-training/train.jsonl', 'w') as f:
    f.write('\n'.join(train) + '\n')
with open('data/local-training/eval.jsonl', 'w') as f:
    f.write('\n'.join(eval_) + '\n')
# test split required by the loader but not used for training
with open('data/local-training/test.jsonl', 'w') as f:
    f.write('\n'.join(eval_) + '\n')

print(f'Created {len(train)} train + {len(eval_)} eval records')
"

# Verify the data format
head -1 data/local-training/train.jsonl | python -m json.tool
```

**What to notice:** Each record's `features` dict matches the `feature_cols` list hard-coded in `train.py`. Missing features default to 0, so the schema is tolerant — but matching it exactly gives the best results.

---

## Step 4: Build the training container

```bash
docker build -f docker/train-xgboost.Dockerfile -t train-xgboost:local .
```

**What just happened:** Docker built the training image using the Dockerfile, which installs Python dependencies and copies the training script. This is the same image that runs on Vertex AI — the only difference is how you pass paths (local vs `gs://`).

---

## Step 5: Run training locally

Run the container with the config and data volume-mounted:

```bash
docker run --rm \
  -v "$(pwd)/data/local-training:/data" \
  train-xgboost:local \
  --config /data/config.yaml \
  --dataset /data \
  --output /data/output
```

**Expected output:**

- `Config loaded: xgboost-local-test`
- XGBoost iteration logs (`[0] eval-mlogloss:...`)
- `Eval weighted F1: 0.XXXX` (exact value depends on synthetic data)
- `Training complete`
- Model artifacts written to `data/local-training/output/`

> **Troubleshooting:** If you see `error: the following arguments are required`, make sure the
> `--config`, `--dataset`, and `--output` flags come _after_ the image name (`train-xgboost:local`).
> Docker passes anything after the image name as arguments to the container's `ENTRYPOINT`.

---

## Step 6: Inspect the model artifacts

```bash
ls -la data/local-training/output/
```

You should see four files:

```bash
# The trained XGBoost booster (model weights)
python -c "import xgboost as xgb; m = xgb.Booster(); m.load_model('data/local-training/output/model.json'); print(f'Feature names: {m.feature_names}')"

# Label mapping (sklearn LabelEncoder)
python -c "import pickle; le = pickle.load(open('data/local-training/output/label_encoder.pkl','rb')); print(f'Classes: {list(le.classes_)}')"

# Feature column names used during training
cat data/local-training/output/feature_cols.json | python -m json.tool

# Training metrics
cat data/local-training/output/metrics.json | python -m json.tool
```

**What to notice:**

- `model.json`: the serialized XGBoost booster — this is what the serving container loads at startup
- `label_encoder.pkl`: maps integer predictions back to taxonomy labels like `INTENT.ROMANCE`
- `feature_cols.json`: records which features the model was trained on (important for serving parity)
- `metrics.json`: evaluation metrics from the held-out eval split

---

## Step 7: Understand the production pipeline config

In production, training is configured via more comprehensive YAML files:

```bash
cat pipelines/configs/classification_xgboost.yaml
```

**What to notice:**

- `label_schema`: defines all taxonomy axes and their labels — more axes than our simplified local config
- `hyperparameters` + `xgboost_params`: full set of tuning knobs
- `vizier_search_space`: ranges for automated hyperparameter tuning (Exercise 3 runs this on Vertex AI)
- `eval_gate`: thresholds for the promotion gate (Exercise 4 covers this)

---

## Step 8: Clean up

```bash
rm -rf data/local-training
docker rmi train-xgboost:local    # optional: reclaim ~500MB disk space
```

> **Tip:** Training container images accumulate over time. Run `docker system prune` periodically
> to reclaim disk space from unused images and build cache.

---

## Summary

| Step                   | What you did                       | Key file                            |
| ---------------------- | ---------------------------------- | ----------------------------------- |
| Examined training code | Understood the argparse interface  | `containers/train-xgboost/train.py` |
| Created config         | Defined hyperparams + label schema | `data/local-training/config.yaml`   |
| Created synthetic data | Produced JSONL train/eval splits   | —                                   |
| Built container        | Packaged training code into Docker | `docker/train-xgboost.Dockerfile`   |
| Ran training           | Produced model artifacts locally   | `containers/train-xgboost/train.py` |
| Inspected artifacts    | Understood output format           | `model.json`, `label_encoder.pkl`   |

**Next exercise:** [03 — Submit a Pipeline](03-submit-pipeline.md), where you send this training job to Vertex AI Pipelines on GCP.
