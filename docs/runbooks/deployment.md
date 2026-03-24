# ML Platform Deployment Runbook

## Prerequisites

- `gcloud` CLI authenticated with access to `i4g-ml` project
- Docker installed and authenticated to Artifact Registry
- Conda environment `ml` activated
- Terraform state initialized for `infra/environments/ml/` and `infra/environments/app/dev/`

## 1. Apply Cross-Project IAM (one-time)

The ETL job needs `cloudsql.client` on the Core project. This grant lives in the app stack:

```bash
cd infra/environments/app/dev
make plan    # verify ml_etl_cloudsql_client resource
make apply   # apply the IAM grant (uses local-overrides.tfvars for secrets)
```

## 2. Build and Push Container Images

```bash
cd ml/

# Build all four images
make build-all-dev

# Or build individually
make build-etl-dev
make build-train-pytorch-dev
make build-train-xgboost-dev
make build-serve-dev
```

This pushes to `us-central1-docker.pkg.dev/i4g-ml/containers/<name>:dev`.

## 3. Deploy ETL Cloud Run Job

Deploy the ETL job (runs daily at 2 AM UTC via Cloud Scheduler):

```bash
cd ml/
make deploy-etl-dev
```

Run manually to verify:

```bash
gcloud run jobs execute etl-ingest --project=i4g-ml --region=us-central1
```

Check BigQuery for data:

```sql
SELECT table_id, row_count
FROM `i4g-ml.i4g_ml.__TABLES__`
WHERE table_id LIKE 'raw_%';
```

## 4. Create Feature Materialization

After ETL populates raw tables, create the scheduled query:

```bash
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/v_case_features.sql
bq query --project_id=i4g-ml --use_legacy_sql=false < pipelines/sql/materialize_features.sql
```

Verify features populated:

```sql
SELECT COUNT(*) FROM `i4g-ml.i4g_ml.features_case_features`;
```

## 5. Prepare Training Data

Bootstrap the first training dataset from existing LLM classifications:

```bash
conda run -n ml python -c "
from ml.data.datasets import create_dataset_version
create_dataset_version(
    project='i4g-ml',
    dataset='i4g_ml',
    bucket='i4g-ml-data',
    version='v1',
)
"
```

## 6. Run Baseline Benchmark

Before training, establish the few-shot LLM baseline:

```bash
conda run -n ml python -c "
from ml.training.baseline import run_baseline, save_baseline_result
result = run_baseline(golden_set_path='gs://i4g-ml-data/datasets/classification/golden/test.jsonl')
save_baseline_result(result, project='i4g-ml', dataset='i4g_ml')
print(f'Baseline F1: {result.overall_f1:.4f}')
"
```

## 7. Run Training Pipeline

Submit the KFP v2 pipeline to Vertex AI:

```bash
conda run -n ml python -c "
from ml.training.pipeline import training_pipeline
from kfp import compiler
from google.cloud import aiplatform

compiler.Compiler().compile(training_pipeline, 'pipelines/training_pipeline.yaml')

aiplatform.init(project='i4g-ml', location='us-central1')
job = aiplatform.PipelineJob(
    display_name='classification-gemma2b-v1',
    template_path='pipelines/training_pipeline.yaml',
    parameter_values={
        'project': 'i4g-ml',
        'config_uri': 'gs://i4g-ml-data/configs/classification_gemma2b.yaml',
        'dataset_version': 'v1',
    },
)
job.submit()
print(f'Pipeline job: {job.resource_name}')
"
```

Monitor in the Vertex AI console: https://console.cloud.google.com/vertex-ai/pipelines?project=i4g-ml

## 8. Verify Deployment

After the pipeline completes and the model is deployed:

```bash
# Check model in registry
gcloud ai models list --project=i4g-ml --region=us-central1

# Test serving endpoint
curl -X POST https://serving-dev-<hash>.us-central1.aiplatform.googleapis.com/predict/classify \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test case text", "case_id": "test-001"}'

# Check prediction log
bq query --project_id=i4g-ml --use_legacy_sql=false \
  "SELECT * FROM i4g_ml.predictions_prediction_log ORDER BY timestamp DESC LIMIT 5"
```

## 9. Test Core Integration

From the Core project, verify the `MLPlatformClient` can call the ML endpoint:

```bash
conda run -n ml I4G_ENV=local I4G_ML__INFERENCE_BACKEND=ml_platform \
  python -c "
from i4g.ml.client import MLPlatformClient
client = MLPlatformClient(base_url='https://serving-dev-<hash>.us-central1.aiplatform.googleapis.com')
result = client.classify('Test fraud case text', case_id='smoke-001')
print(result)
"
```

## 10. Run Vizier Hyperparameter Sweep

Vertex AI Vizier automates hyperparameter tuning via Bayesian optimization:

```bash
# XGBoost sweep (10 trials, CPU — ~$5 total)
conda run -n ml make run-vizier-sweep CONFIG=pipelines/configs/classification_xgboost.yaml TRIALS=10

# PyTorch sweep (5 trials, GPU — ~$25-50 total)
conda run -n ml make run-vizier-sweep CONFIG=pipelines/configs/classification_gemma2b.yaml TRIALS=5
```

Vizier manages the study externally — it suggests trial parameters, you submit a pipeline with those
params, report the eval metric back, and Vizier selects the next trial. Results are logged to the
Vizier study and can be queried:

```bash
gcloud ai hp-tuning-jobs list --project=i4g-ml --region=us-central1
```

## 11. Run Dataflow Graph Features Pipeline

The graph features pipeline computes entity co-occurrence features:

```bash
# Local validation with DirectRunner
conda run -n ml python -m ml.data.graph_features --runner DirectRunner

# Submit to Dataflow (production)
conda run -n ml make submit-graph-features-dev
```

Monitor the Dataflow job:

```bash
gcloud dataflow jobs list --project=i4g-ml --region=us-central1 --status=active
```

Verify output:

```sql
SELECT COUNT(*), MIN(_computed_at), MAX(_computed_at)
FROM `i4g-ml.i4g_ml.features_graph_features`;
```

## Rollback

To roll back a model deployment:

```bash
# Undeploy the model from the endpoint
gcloud ai endpoints undeploy-model <endpoint-id> \
  --project=i4g-ml \
  --region=us-central1 \
  --deployed-model-id=<deployed-model-id>

# Redeploy the previous champion
gcloud ai endpoints deploy-model <endpoint-id> \
  --project=i4g-ml \
  --region=us-central1 \
  --model=<previous-model-id> \
  --display-name=champion
```
