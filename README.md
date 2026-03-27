# I4G ML Platform

Training, serving, evaluation, and monitoring for machine learning models.

## Prerequisites

- **Conda environment `ml`** — all commands below assume this env is active:

  ```bash
  conda create -n ml python=3.11 -y
  conda activate ml
  ```

## Quick Start

```bash
pip install -e ".[dev]"        # install with dev dependencies
pytest tests/unit              # run unit tests
i4g-ml --help                  # CLI reference
```

## CLI

`i4g-ml` is the primary developer interface. Key commands:

```bash
i4g-ml dataset bootstrap             # bootstrap dataset from LLM labels
i4g-ml pipeline submit --config ...  # submit training pipeline
i4g-ml deploy serving                # deploy to serving endpoint
i4g-ml eval baseline                 # compute baseline metrics
i4g-ml retrain trigger               # evaluate retraining conditions
i4g-ml smoke e2e                     # end-to-end smoke test
i4g-ml settings show                 # display resolved settings
```

Run `i4g-ml <command> --help` for details on any subcommand.

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
