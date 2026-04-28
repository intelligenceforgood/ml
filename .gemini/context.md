# Copilot Instructions for i4g/ml

**Unified Workspace Context:** This repository is part of the `i4g` multi-root workspace. Shared coding
standards, routines, and platform context live in the `copilot/` repo. These instructions contain only
repo-specific context.

## Environment

- **Conda env:** `ml` — all commands assume this env is active (`conda activate ml`)
- **Language:** Python 3.11+ (FastAPI, Pydantic v2, KFP v2, Google Cloud SDKs)
- **CLI:** `i4g-ml` (Typer) — primary developer interface

## Build & Test

```bash
pip install -e ".[dev]"                   # install editable with dev deps
pytest tests/unit -x                      # unit tests (stop on first failure)
ruff check src/ tests/                    # lint
black --check src/ tests/                 # format check
```

## Architecture

- **Data layer:** `src/ml/data/` — ETL (Cloud SQL → BigQuery), feature engineering, dataset management
- **Training:** `src/ml/training/` — pipeline definitions, training configs, evaluation harness
- **Serving:** `src/ml/serving/` — FastAPI prediction server deployed to Vertex AI Endpoints
- **Registry:** `src/ml/registry/` — model registration, promotion workflow, eval gates
- **Monitoring:** `src/ml/monitoring/` — drift detection, accuracy tracking, cost, retraining triggers
- **Settings:** `ml.config.get_settings()` — nested sections via `I4G_ML_*` env vars

## GCP Resources

- **Project:** `i4g-ml`
- **Region:** `us-central1`
- **BigQuery dataset:** `i4g_ml`
- **GCS bucket:** `i4g-ml-data`
- **Endpoints:** `serving-dev`, `serving-prod`
- **Artifact Registry:** `containers`

## Coding Conventions

- Python: full type hints, Google-style docstrings, Black/isort at 120-char lines
- Pydantic: `snake_case` internally, `alias_generator = to_camel` for JSON APIs
- Follow `copilot/.github/shared/general-coding.instructions.md` for complete language standards

## Pre-Commit

```bash
pre-commit run --all-files
```
