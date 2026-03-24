# I4G ML Platform

Training, serving, evaluation, and monitoring for machine learning models.

## Quick Start

```bash
pip install -e ".[dev]"        # install with dev dependencies
pytest tests/unit              # run unit tests
```

## Structure

```
src/ml/
├── config.py                  # Settings
├── data/                      # ETL, features, datasets, validation
├── training/                  # Pipeline, config, evaluation
├── serving/                   # FastAPI prediction server
├── registry/                  # Model registry, promotion
└── monitoring/                # Drift, accuracy, cost, triggers
```

See [docs/design/ml_infrastructure_tdd.md](docs/design/ml_infrastructure_tdd.md) for full architecture.
