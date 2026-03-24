# Exercise 5: Deploy to Serving

> **Objective:** Deploy a model to the serving endpoint, send a test prediction, and verify BigQuery logging.
> **Prerequisites:** Exercise 4 completed, model promoted to `candidate` or `champion`
> **Time:** ~25 minutes

---

## Overview

The serving layer is a FastAPI application running in a container on Cloud Run. It loads model artifacts from GCS at startup and exposes prediction endpoints:

```
Client → POST /predict/classify → FastAPI → Model inference → Response
                                      ↓
                              BigQuery prediction_log
```

---

## Step 1: Understand the serving architecture

```bash
head -30 src/ml/serving/app.py
head -80 src/ml/serving/predict.py
```

**What to notice:**

- `app.py`: FastAPI app with routes for `/predict/classify`, `/predict/extract-entities`, `/feedback`, `/health`
- `predict.py`: model loading + inference logic
  - `_MODEL_STATE`: holds loaded classification model (weights, tokenizer/booster, label map)
  - `load_model()`: downloads artifacts from GCS, detects framework (PyTorch/XGBoost), initializes model
  - `classify_text()`: runs inference, returns per-axis predictions with confidence
  - Shadow model support: `_SHADOW_MODEL_STATE` for candidate evaluation on live traffic

---

## Step 2: Examine the deployment script

```bash
head -60 scripts/deploy_serving.py
```

**What to notice:**

- Uploads the model to Vertex AI Model Registry with labels (stage, capability)
- Finds the `serving-dev` endpoint
- Deploys with configured machine type and replica count

---

## Step 3: Build and push the serving container

```bash
make build-serve-dev
```

This builds the serving Docker image and pushes it to Artifact Registry.

---

## Step 4: Deploy via Terraform (recommended) or script

The production deployment path uses Terraform to manage the Cloud Run service:

```bash
# Check the current Cloud Run service configuration
gcloud run services describe ml-serving --project=i4g-ml --region=us-central1 \
  --format="yaml(spec.template.spec.containers[0].env)" 2>/dev/null || echo "Service not found — deploy via Terraform"
```

Key environment variables on the Cloud Run service:

- `MODEL_ARTIFACT_URI`: GCS path to the classification model artifact
- `NER_MODEL_ARTIFACT_URI`: GCS path to the NER model (empty = disabled)
- `SHADOW_MODEL_ARTIFACT_URI`: GCS path to shadow model (empty = disabled)

---

## Step 5: Send a test prediction

Once the service is deployed, send a test prediction:

```bash
# Get the Cloud Run service URL (or use the Vertex AI endpoint)
SERVICE_URL=$(gcloud run services describe ml-serving --project=i4g-ml \
  --region=us-central1 --format="value(status.url)" 2>/dev/null)

# Send a classification request
curl -s -X POST "${SERVICE_URL}/predict/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I was contacted on Instagram by someone claiming to be a crypto investment advisor. They asked me to transfer funds via Bitcoin.",
    "case_id": "test-bootcamp-001"
  }' | python -m json.tool
```

**Expected response:**

```json
{
  "prediction": {
    "INTENT": { "label": "INTENT.INVESTMENT", "confidence": 0.87 },
    "CHANNEL": { "label": "CHANNEL.SOCIAL_MEDIA", "confidence": 0.92 }
  },
  "risk_score": 0.85,
  "model_info": {
    "model_id": "classification-xgboost-v1",
    "version": 1,
    "stage": "champion"
  },
  "prediction_id": "pred-abc123..."
}
```

---

## Step 6: Check the health endpoint

```bash
curl -s "${SERVICE_URL}/health" | python -m json.tool
```

**Expected response:**

```json
{
  "status": "healthy",
  "model_id": "classification-xgboost-v1",
  "shadow_active": false,
  "ner_active": false
}
```

---

## Step 7: Verify BigQuery prediction logging

Every prediction is logged to BigQuery for monitoring and feedback:

```bash
bq query --use_legacy_sql=false \
  'SELECT prediction_id, case_id, model_id, is_shadow, created_at
   FROM `i4g-ml.i4g_ml.predictions_prediction_log`
   WHERE case_id = "test-bootcamp-001"
   ORDER BY created_at DESC
   LIMIT 5'
```

**What to notice:** The prediction appears in the log with the correct `model_id`, `case_id`, and `is_shadow = false`. This logging is what makes monitoring, accuracy tracking, and the feedback loop possible.

---

## Step 8: Send feedback (simulating analyst correction)

```bash
PREDICTION_ID="<prediction_id from step 5>"

curl -s -X POST "${SERVICE_URL}/feedback" \
  -H "Content-Type: application/json" \
  -d "{
    \"prediction_id\": \"${PREDICTION_ID}\",
    \"case_id\": \"test-bootcamp-001\",
    \"correction\": {\"INTENT\": \"INTENT.ROMANCE\"},
    \"analyst_id\": \"bootcamp-analyst\"
  }" | python -m json.tool
```

Verify the outcome was recorded:

```bash
bq query --use_legacy_sql=false \
  'SELECT outcome_id, prediction_id, case_id, analyst_id
   FROM `i4g-ml.i4g_ml.predictions_outcome_log`
   WHERE case_id = "test-bootcamp-001"
   LIMIT 5'
```

**What just happened:** You closed the feedback loop. The analyst correction is now in the outcome log, ready to be used for calculating accuracy metrics and triggering retraining.

---

## Summary

| Step            | What you did                 | Verified                    |
| --------------- | ---------------------------- | --------------------------- |
| Built container | Packaged serving code        | Image in Artifact Registry  |
| Deployed        | Model live on Cloud Run      | Health endpoint returns 200 |
| Sent prediction | Got classification result    | Correct JSON response       |
| Checked logging | Prediction in BigQuery       | `prediction_log` row exists |
| Sent feedback   | Simulated analyst correction | `outcome_log` row exists    |

**Next exercise:** [06 — Monitor and Retrain](06-monitor-and-retrain.md), where you trigger drift computation, read monitoring data, and invoke retraining.
