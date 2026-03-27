.PHONY: setup install install-dev test test-all lint format clean \
        build-etl-dev build-train-pytorch-dev build-train-xgboost-dev build-serve-dev \
        build-train-ner-dev build-train-ner-prod \
        build-graph-features-dev build-graph-features-prod \
        build-etl-prod build-train-pytorch-prod build-train-xgboost-prod build-serve-prod \
        build-all-dev deploy-etl-dev deploy-serving-dev redeploy-serving-dev \
        rehydrate compile-pipeline run-vizier-sweep \
        trigger-retrain-dev submit-graph-features-dev \
        run-batch-dev run-batch-prod sync-features-dev smoke-e2e \
        dataset-bootstrap dataset-create baseline compare-frameworks

# ---------- Setup ----------
setup: install-dev
	@echo "✅ Setup complete."

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# ---------- Quality ----------
test:
	pytest tests/unit -v

test-all:
	pytest -v

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# ---------- CLI wrappers ----------
# i4g-ml CLI is the primary interface. These Makefile targets are convenience aliases.

dataset-bootstrap:
	i4g-ml dataset bootstrap

dataset-create:
	i4g-ml dataset create --capability $(CAPABILITY)

baseline:
	i4g-ml eval baseline

compare-frameworks:
	i4g-ml eval compare-frameworks --xgboost-metrics $(XGB_METRICS) --pytorch-metrics $(PT_METRICS)

smoke-e2e:
	i4g-ml smoke e2e

# ---------- Docker / Deploy (Dev) ----------
build-etl-dev:
	scripts/build_image.sh etl dev

build-train-pytorch-dev:
	scripts/build_image.sh train-pytorch dev

build-train-xgboost-dev:
	scripts/build_image.sh train-xgboost dev

build-serve-dev:
	scripts/build_image.sh serve dev

build-train-ner-dev:
	scripts/build_image.sh train-ner dev

build-graph-features-dev:
	scripts/build_image.sh graph-features dev

build-graph-features-prod:
	scripts/build_image.sh graph-features prod

build-all-dev: build-etl-dev build-train-pytorch-dev build-train-xgboost-dev build-serve-dev build-train-ner-dev build-graph-features-dev
	@echo "✅ All dev images built and pushed."

deploy-etl-dev: build-etl-dev
	gcloud run jobs deploy etl-ingest \
		--image us-central1-docker.pkg.dev/i4g-ml/containers/etl:dev \
		--region us-central1 \
		--project i4g-ml \
		--service-account=sa-ml-platform@i4g-ml.iam.gserviceaccount.com \
		--set-env-vars="I4G_ML_ETL__SOURCE_INSTANCE=i4g-dev:us-central1:i4g-dev-db,I4G_ML_ETL__SOURCE_DB_NAME=i4g_db,I4G_ML_ETL__SOURCE_DB_USER=sa-ml-platform@i4g-ml.iam" \
		--max-retries=1 \
		--task-timeout=1800s

deploy-serving-dev:
	i4g-ml deploy serving --endpoint serving-dev --image-tag dev

redeploy-serving-dev:
	i4g-ml deploy redeploy --endpoint serving-dev --image-tag dev

# ---------- Docker / Deploy (Prod) ----------
build-etl-prod:
	scripts/build_image.sh etl prod

build-train-pytorch-prod:
	scripts/build_image.sh train-pytorch prod

build-train-xgboost-prod:
	scripts/build_image.sh train-xgboost prod

build-serve-prod:
	scripts/build_image.sh serve prod

build-train-ner-prod:
	scripts/build_image.sh train-ner prod

deploy-serve-prod: build-serve-prod
	@echo "Prod serving image built. Deploy via: cd ../infra/environments/ml && terraform apply"

# ---------- Pipeline ----------
CONFIG ?= pipelines/configs/classification_xgboost.yaml
compile-pipeline:
	i4g-ml pipeline compile

submit-pipeline:
	i4g-ml pipeline submit --config $(CONFIG)

# ---------- Vizier ----------
TRIALS ?= 15
run-vizier-sweep:
	python -m ml.training.vizier --config $(CONFIG) --max-trials $(TRIALS)

# ---------- Retrain Trigger ----------
CAPABILITY ?= classification
trigger-retrain-dev:
	i4g-ml retrain trigger --capability $(CAPABILITY)

# ---------- Batch Prediction ----------
run-batch-dev:
	i4g-ml serve batch --capability $(CAPABILITY)

run-batch-prod:
	i4g-ml serve batch --capability $(CAPABILITY)

# ---------- Feature Store ----------
sync-features-dev:
	python -m ml.data.feature_store --project i4g-ml --feature-store-id i4g_ml_features --entity-type-id case

# ---------- Graph Features ----------
submit-graph-features-dev:
	python -m ml.data.graph_features \
		--project i4g-ml --dataset i4g_ml \
		--runner DataflowRunner \
		--temp-location gs://i4g-ml-data/dataflow/temp \
		--staging-location gs://i4g-ml-data/dataflow/staging \
		--service-account-email sa-ml-platform@i4g-ml.iam.gserviceaccount.com \
		--requirements-file containers/graph-features/requirements.txt \
		--region us-central1

# ---------- Clean ----------
clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ---------- Rehydrate (Copilot session bootstrap) ----------
rehydrate:
	@echo "--- ML Rehydrate ---"
	git status -sb
	@echo "--- Recent changes ---"
	git log --oneline -5 2>/dev/null || true
