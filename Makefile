.PHONY: setup install install-dev test test-all lint format clean \
        build-etl-dev build-train-pytorch-dev build-train-xgboost-dev build-serve-dev \
        build-etl-prod build-train-pytorch-prod build-train-xgboost-prod build-serve-prod \
        build-all-dev deploy-etl-dev rehydrate

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

# ---------- Docker / Deploy (Dev) ----------
build-etl-dev:
	scripts/build_image.sh etl dev

build-train-pytorch-dev:
	scripts/build_image.sh train-pytorch dev

build-train-xgboost-dev:
	scripts/build_image.sh train-xgboost dev

build-serve-dev:
	scripts/build_image.sh serve dev

build-all-dev: build-etl-dev build-train-pytorch-dev build-train-xgboost-dev build-serve-dev
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

# ---------- Docker / Deploy (Prod) ----------
build-etl-prod:
	scripts/build_image.sh etl prod

build-train-pytorch-prod:
	scripts/build_image.sh train-pytorch prod

build-train-xgboost-prod:
	scripts/build_image.sh train-xgboost prod

build-serve-prod:
	scripts/build_image.sh serve prod

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
