# Exercise 7: Add a New Capability

> **Objective:** Add a toy third ML capability (binary spam classifier) end-to-end: config, container, pipeline, serving route, and core client integration.
> **Prerequisites:** Exercises 1–6 completed, understanding of the multi-capability architecture
> **Time:** ~60 minutes

---

## Overview

The ML platform is designed for multi-capability serving. Classification and NER already exist. This exercise walks you through adding a third capability — a binary spam classifier — touching every layer of the stack. This is intentionally a toy model to teach the pattern without the complexity of a production model.

**Layers you'll touch:**

```
Pipeline config (YAML) → Training container → Eval harness
    → Serving route → Core client → Terraform env vars
```

---

## Step 1: Define the pipeline config

Create a new config file for the spam classifier:

```bash
cat > pipelines/configs/spam_xgboost.yaml << 'EOF'
capability: spam
framework: xgboost
model_id: spam-xgboost-v1
eval_metric: macro_f1

# Hyperparameters
n_estimators: 200
max_depth: 4
learning_rate: 0.1
subsample: 0.8

# Vizier search space (for automated tuning)
vizier_search_space:
  n_estimators:
    min: 100
    max: 400
  max_depth:
    min: 3
    max: 6
  learning_rate:
    min: 0.01
    max: 0.3
    scale: log

enable_vizier: false
EOF
```

**What just happened:** You defined a new capability's training configuration. The config schema matches `TrainingConfig` in `training/config.py`.

---

## Step 2: Add the serving route

The serving container needs to know about the new capability. Open `src/ml/serving/predict.py` and understand the multi-model pattern:

```bash
grep -n "SPAM\|_MODELS\|NER_MODEL\|MODEL_ARTIFACT" src/ml/serving/predict.py | head -20
```

To add a new capability, you need:

1. **Environment variable:** `SPAM_MODEL_ARTIFACT_URI` (empty = disabled)
2. **Startup loading:** call `load_model("spam", artifact_uri)` if env var is set
3. **Inference function:** `classify_spam(text) -> dict` reading from `_MODELS["spam"]`
4. **API route:** `POST /predict/classify-spam` in `app.py`

Here's a sketch of what the serving route would look like in `app.py`:

```python
# In serving/app.py — add this route
@app.post("/predict/classify-spam", response_model=SpamResponse)
async def classify_spam_route(request: SpamRequest) -> SpamResponse:
    if not is_spam_ready():
        raise HTTPException(503, "Spam model not loaded")

    result = classify_spam(request.text)
    prediction_id = str(uuid.uuid4())

    # Log prediction
    asyncio.ensure_future(log_prediction(
        prediction_id=prediction_id,
        case_id=request.case_id,
        model_id=result["model_id"],
        capability="spam",
        prediction=result["prediction"],
    ))

    return SpamResponse(
        prediction=result["prediction"],
        model_info=ModelInfo(**result["model_info"]),
        prediction_id=prediction_id,
    )
```

**Important pattern:** Don't modify this file yet — just understand the pattern. The key insight is that each capability follows the same structure:
1. Load model at startup via env var
2. Add inference function in `predict.py`
3. Add route in `app.py`
4. Log predictions to BigQuery with `capability="spam"`

---

## Step 3: Update the eval gate

The promotion module dispatches on capability to select the right eval metric:

```bash
grep -A 10 "def promote_model" src/ml/registry/promotion.py
```

For the spam classifier, the eval gate would use macro F1 (same as classification). You'd add a case to the dispatch:

```python
# In registry/promotion.py — capability dispatch pattern:
if capability == "ner":
    passed, reason = _passes_ner_eval_gate(candidate_eval, champion_eval)
elif capability in ("classification", "spam"):
    passed, reason = _passes_eval_gate(candidate_eval, champion_eval)
```

---

## Step 4: Add the dataset export

In `data/datasets.py`, the `create_dataset_version()` function accepts a `capability` parameter. For spam, you'd need:

1. A BigQuery query that produces text + spam/not-spam labels
2. A dataset validation rule (min samples per class)

```python
# Example: how to call dataset creation for a new capability
from ml.data.datasets import create_dataset_version

metadata = create_dataset_version(
    capability="spam",
    label_column="is_spam",
    min_samples_per_class=50,
)
```

---

## Step 5: Add the core client method

In the core repo, `MLPlatformClient` needs a method to call the new endpoint:

```python
# In core/src/i4g/ml/client.py — following the existing pattern:
async def classify_spam(self, text: str, case_id: str) -> dict:
    """POST to /predict/classify-spam."""
    response = await self._client.post(
        f"{self._base_url}/predict/classify-spam",
        json={"text": text, "case_id": case_id},
        timeout=self._timeout,
    )
    response.raise_for_status()
    return response.json()
```

---

## Step 6: Add Terraform env var

In `infra/stacks/ml/main.tf`, add the new env var to the Cloud Run service:

```hcl
env {
  name  = "SPAM_MODEL_ARTIFACT_URI"
  value = ""  # Empty = disabled until model is trained
}
```

---

## Step 7: Write a unit test

Following the testing pattern in `tests/unit/serving/`:

```python
# tests/unit/serving/test_spam_prediction.py
def test_classify_spam_returns_binary_prediction(mock_model):
    """Spam classification returns is_spam boolean."""
    result = classify_spam("Win a free iPhone! Click here now!")
    assert "is_spam" in result["prediction"]
    assert isinstance(result["prediction"]["is_spam"], bool)

def test_classify_spam_503_when_not_loaded(client):
    """Returns 503 when spam model is not loaded."""
    response = client.post("/predict/classify-spam", json={"text": "test", "case_id": "c1"})
    assert response.status_code == 503
```

---

## Checklist: Adding a New Capability

When adding any new ML capability, follow this checklist:

- [ ] Pipeline config YAML in `pipelines/configs/`
- [ ] Training container (or reuse existing framework container)
- [ ] Eval harness: metric computation + eval gate dispatch
- [ ] Serving: env var + model loading + inference function + API route
- [ ] BigQuery: prediction logging with `capability = "<new>"` column value
- [ ] Core client: new method in `MLPlatformClient`
- [ ] Core settings: backend routing config
- [ ] Terraform: env var on Cloud Run service
- [ ] Unit tests: model loading, inference, 503 when disabled, eval gate
- [ ] Docs: update `docs/design/architecture.md` with new capability

---

## Summary

This exercise demonstrated the pattern — not a production implementation. The key takeaway is that the platform is designed for extensibility: each capability follows the same layered structure, and adding a new one is a systematic process through all layers.

**Next exercise:** [08 — Graph Features Pipeline](08-graph-features.md), where you run the Dataflow/Beam pipeline to compute graph-based features.
