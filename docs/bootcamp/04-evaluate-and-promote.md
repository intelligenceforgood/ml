# Exercise 4: Evaluate and Promote

> **Objective:** Run the evaluation harness on a model's predictions, interpret the metrics, and promote the model through the eval gate.
> **Prerequisites:** Exercise 3 completed (a model registered in Vertex AI Model Registry)
> **Time:** ~20 minutes

> **Partially offline.** Steps 4–5 (manual eval and gate simulation) run entirely in a local
> Python shell — no GCP needed. Steps 3 and 6 use `bq` and `gcloud` against live data.
> Run `pytest tests/unit/test_evaluation.py -v` to validate eval logic locally.

---

## Overview

Every trained model passes through an evaluation gate before promotion. The pipeline runs evaluation automatically, but understanding the metrics and promotion logic is critical for debugging and manual intervention.

```
Model artifact → Eval harness → EvalResult (per-axis P/R/F1)
                                      ↓
                              Promotion logic (eval gate)
                                      ↓
                        experimental → candidate → champion
```

---

## Step 1: Understand evaluation metrics

Open the evaluation module:

```bash
head -90 src/ml/training/evaluation.py
```

**What to notice:**

- `AxisMetrics`: precision, recall, F1 for a single taxonomy axis (e.g., INTENT, CHANNEL)
- `EvalResult`: overall metrics + per-axis breakdown
- `compute_metrics()`: takes predictions and ground truth (both as `[{axis: label_code}]` dicts), computes TP/FP/FN per axis

The key metric is **weighted F1** — the average F1 across all axes, weighted by support (number of samples per axis). This gives more influence to axes with more data.

---

## Step 2: Understand the promotion workflow

Open the promotion module:

```bash
head -80 src/ml/registry/promotion.py
```

**Model lifecycle stages:**

1. **`experimental`** — freshly trained, unvalidated
2. **`candidate`** — eval gate passed, ready for manual review
3. **`champion`** — approved for production serving

**Eval gate rules (`_passes_eval_gate`):**

- First model always passes (no champion to compare against)
- Candidate's overall F1 must be ≥ champion's overall F1
- No per-axis regression > 5% (configurable via `max_regression`)

---

## Step 3: Check evaluation results

The training pipeline's `evaluate_model` step computes eval metrics and returns them as a
JSON artifact. To see them, open the **Vertex AI Pipelines** console for your Exercise 3 run
and click the **evaluate_model** node — the `metrics_json` output contains
`overall_f1`, `overall_precision`, `overall_recall`, and `per_axis` breakdowns.

> **Note:** The `analytics_model_performance` BigQuery table is populated by the
> **monitoring** module (`ml.monitoring.accuracy`), not the training pipeline. It tracks
> accuracy against real analyst outcomes once a model is deployed and receiving predictions.
> After Exercise 3 alone, this table will be empty.

Once a model is deployed and has received outcomes, you can query it:

```bash
bq query --use_legacy_sql=false \
  'SELECT model_id, model_version, capability, computed_at, total_predictions,
          outcomes_received, correct_predictions, accuracy, correction_rate, f1
   FROM `i4g-ml.i4g_ml.analytics_model_performance`
   ORDER BY computed_at DESC, model_version DESC
   LIMIT 10'
```

---

## Step 4: Run evaluation manually (optional)

You can also run evaluation outside the pipeline. This is useful for debugging:

```python
# In a Python shell (python)
from ml.training.evaluation import compute_metrics

predictions = [
    {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.SOCIAL_MEDIA"},
    {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.EMAIL"},
    {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.SOCIAL_MEDIA"},
]
ground_truth = [
    {"INTENT": "INTENT.ROMANCE", "CHANNEL": "CHANNEL.SOCIAL_MEDIA"},
    {"INTENT": "INTENT.INVESTMENT", "CHANNEL": "CHANNEL.PHONE"},  # wrong CHANNEL
    {"INTENT": "INTENT.TECH_SUPPORT", "CHANNEL": "CHANNEL.SOCIAL_MEDIA"},  # wrong INTENT
]

result = compute_metrics(predictions, ground_truth)
print(result.summary())
```

**Expected output:** Overall F1 plus per-axis breakdown. Notice how CHANNEL has lower F1 because of the misprediction.

---

## Step 5: Simulate the eval gate

```python
from ml.training.evaluation import EvalResult, AxisMetrics
from ml.registry.promotion import _passes_eval_gate

champion = EvalResult(
    overall_f1=0.80, overall_precision=0.82, overall_recall=0.78,
    per_axis={"INTENT": AxisMetrics("INTENT", 0.85, 0.80, 0.82, 100)},
    total_samples=100
)

# Candidate that passes (higher F1):
good_candidate = EvalResult(
    overall_f1=0.85, overall_precision=0.87, overall_recall=0.83,
    per_axis={"INTENT": AxisMetrics("INTENT", 0.88, 0.83, 0.85, 100)},
    total_samples=100
)
passed, reason = _passes_eval_gate(good_candidate, champion)
print(f"Passed: {passed}, Reason: {reason}")

# Candidate that fails (per-axis regression > 5%):
bad_candidate = EvalResult(
    overall_f1=0.82, overall_precision=0.84, overall_recall=0.80,
    per_axis={"INTENT": AxisMetrics("INTENT", 0.70, 0.72, 0.71, 100)},  # 11% drop
    total_samples=100
)
passed, reason = _passes_eval_gate(bad_candidate, champion)
print(f"Passed: {passed}, Reason: {reason}")
```

**What just happened:** You simulated the eval gate decision. The first candidate passed because it improved F1. The second failed because INTENT axis regressed > 5%, even though overall F1 improved.

---

## Step 6: Check model stages in the registry

```bash
gcloud ai models list --project=i4g-ml --region=us-central1 \
  --format="table(displayName,labels.stage,labels.capability)" \
  --limit=5
```

---

## Summary

| Concept        | Key insight                                                            |
| -------------- | ---------------------------------------------------------------------- |
| Weighted F1    | Primary metric — average F1 across axes, weighted by support           |
| Eval gate      | Overall F1 must improve, no axis drops > 5%                            |
| Promotion path | experimental → candidate → champion                                    |
| NER eval gate  | Uses entity micro F1 (not macro F1) + per-entity-type regression check |

**Next exercise:** [05 — Deploy to Serving](05-deploy-to-serving.md), where you deploy a promoted model to the serving endpoint.
