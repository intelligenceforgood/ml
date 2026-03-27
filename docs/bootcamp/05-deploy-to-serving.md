# Exercise 5: Deploy to Serving

> **Objective:** Deploy a model to the serving endpoint, send a test prediction, and verify BigQuery logging.
> **Prerequisites:** Exercise 4 completed, model promoted to `candidate` or `champion`
> **Time:** ~25 minutes

> **Requires GCP access.** All steps interact with Cloud Run and BigQuery. Without access,
> read through the exercise, then run `pytest tests/unit/test_serving.py -v`
> to validate the serving logic locally.

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

## Step 2: Examine the deployment logic

```bash
head -60 src/ml/cli/deploy.py
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

## Step 4: Point the serving container at your model

The Cloud Run `ml-serving` service reads `MODEL_ARTIFACT_URI` on startup. It defaults to
empty — the server runs in stub mode (returning `model_id: null`) until you set it.

**4a. Find your model artifact GCS path.**

The training pipeline writes artifacts to `gs://i4g-ml-data/models/<experiment_name>/`,
where `experiment_name` = `<model_id>-<timestamp>`. List recent artifacts:

```bash
gsutil ls gs://i4g-ml-data/models/ | tail -5
```

Pick the path from your Exercise 3 run. For example:
`gs://i4g-ml-data/models/classification-xgboost-v1-20260325-2151/`

Verify it contains the expected files (`xgboost_model.json` + `label_map.json`):

```bash
gsutil ls gs://i4g-ml-data/models/classification-xgboost-v1-20260325-2151/
```

> **Note:** The training pipeline writes artifacts to GCS regardless of whether the eval
> gate passed. If `register_model` returned `SKIPPED` in Exercise 3 (check Step 7 of that
> exercise), the artifacts still exist — you can use them here for serving.

**4b. Update the Cloud Run env var via `gcloud` (quickest for bootcamp):**

```bash
# Replace the GCS path with your actual artifact URI from 4a
gcloud run services update ml-serving \
  --project=i4g-ml --region=us-central1 \
  --update-env-vars="MODEL_ARTIFACT_URI=gs://i4g-ml-data/models/classification-xgboost-v1-20260325-2151/"
```

Cloud Run redeploys automatically. The serving container downloads artifacts from GCS on
startup.

> **Production path:** For production, set `model_artifact_uri` in
> `infra/environments/ml/terraform.tfvars` and run `terraform apply` instead.

**4c. Verify the model loaded:**

```bash
curl -s "${SERVICE_URL}/health" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" | jq
```

You should now see `"model_id": "classification-xgboost-v1-20260325-2151"` instead of `null`.

Key environment variables on the Cloud Run service:

- `MODEL_ARTIFACT_URI`: GCS path to the classification model artifact (**must be set**)
- `NER_MODEL_ARTIFACT_URI`: GCS path to the NER model (empty = disabled)
- `SHADOW_MODEL_ARTIFACT_URI`: GCS path to shadow model (empty = disabled)

---

## Step 5: Send a test prediction

Once the service is deployed, send a test prediction:

```bash
# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe ml-serving --project=i4g-ml \
  --region=us-central1 --format="value(status.url)" 2>/dev/null)

# Send a classification request
# Note: ml-serving requires an identity token — the service is IAM-protected
curl -s -X POST "${SERVICE_URL}/predict/classify" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -d '{
    "text": "I was contacted on Instagram by someone claiming to be a crypto investment advisor. They asked me to transfer funds via Bitcoin.",
    "case_id": "test-bootcamp-001"
  }' | jq
```

> **Auth note:** If you get an empty response or `403`, your personal gcloud account may not
> have `roles/run.invoker` on `ml-serving`. Ask your admin to add the grant, or test locally
> with `pytest tests/unit/test_serving.py -v` instead.

**Expected response:**

```json
{
  "predictions": [
    {
      "prediction": {
        "INTENT": { "code": "INTENT.IMPOSTER", "confidence": 0.57 }
      },
      "risk_score": null,
      "model_info": {
        "model_id": "classification-xgboost-v1-...",
        "version": 1,
        "stage": "candidate"
      },
      "prediction_id": "93a3d052-..."
    }
  ]
}
```

> **Note:** The exact label and confidence will vary based on your model. The single-axis
> XGBoost model only returns an `INTENT` prediction. `risk_score` is not yet implemented
> (always `null`). The `stage` is `"candidate"` when loaded directly from GCS.

---

## Step 6: Check the health endpoint

```bash
curl -s "${SERVICE_URL}/health" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" | jq
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
  'SELECT prediction_id, case_id, model_id, model_version,
          capability, latency_ms, timestamp
   FROM `i4g-ml.i4g_ml.predictions_prediction_log`
   WHERE case_id = "test-bootcamp-001"
   ORDER BY timestamp DESC
   LIMIT 5'
```

**What to notice:** Your row should show `model_id` matching the artifact directory name
(e.g., `classification-xgboost-v1-...`), `capability = "classification"`, and a non-zero
`latency_ms`. If you see `model_id = "stub"` with `latency_ms = 0`, that row came from an
earlier attempt when no model was loaded — the serving layer falls back to stub predictions
when `MODEL_ARTIFACT_URI` is empty or model loading fails.

---

## Step 8: Send feedback (simulating analyst correction)

```bash
PREDICTION_ID="<prediction_id from step 5>"

curl -s -X POST "${SERVICE_URL}/feedback" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -d "{
    \"prediction_id\": \"${PREDICTION_ID}\",
    \"case_id\": \"test-bootcamp-001\",
    \"correction\": {\"INTENT\": \"INTENT.ROMANCE\"},
    \"analyst_id\": \"bootcamp-analyst\"
  }" | jq
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
