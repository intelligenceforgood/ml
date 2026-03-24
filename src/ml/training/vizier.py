"""Vertex AI Vizier hyperparameter tuning.

Vizier operates OUTSIDE the KFP pipeline.  It manages a study that spawns
multiple training runs with different hyperparameters, evaluates their
results, and selects the best configuration.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google.cloud.aiplatform_v1 import VizierServiceClient
from google.cloud.aiplatform_v1.types import study as study_pb2

from ml.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VizierSearchParam:
    """One parameter in the search space."""

    name: str
    param_type: str  # "DOUBLE", "INTEGER", "CATEGORICAL"
    # For DOUBLE / INTEGER
    min_value: float | None = None
    max_value: float | None = None
    scale: str | None = None  # "LINEAR", "LOG", "REVERSE_LOG"
    # For CATEGORICAL
    values: list[str | int | float] | None = None


def _build_parameter_spec(param: VizierSearchParam) -> study_pb2.StudySpec.ParameterSpec:
    """Convert a VizierSearchParam into a protobuf ParameterSpec."""
    spec = study_pb2.StudySpec.ParameterSpec(parameter_id=param.name)

    scale_map = {
        "linear": study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
        "log": study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
        None: study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
    }

    if param.param_type == "DOUBLE":
        spec.double_value_spec = study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
            min_value=param.min_value,
            max_value=param.max_value,
        )
        if param.scale:
            spec.scale_type = scale_map.get(
                param.scale.lower(),
                study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            )
    elif param.param_type == "INTEGER":
        spec.integer_value_spec = study_pb2.StudySpec.ParameterSpec.IntegerValueSpec(
            min_value=int(param.min_value),
            max_value=int(param.max_value),
        )
        if param.scale:
            spec.scale_type = scale_map.get(
                param.scale.lower(),
                study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
            )
    elif param.param_type == "CATEGORICAL":
        spec.categorical_value_spec = study_pb2.StudySpec.ParameterSpec.CategoricalValueSpec(
            values=[str(v) for v in (param.values or [])],
        )

    return spec


def create_vizier_study(
    capability: str,
    search_space: list[VizierSearchParam],
    *,
    metric_id: str = "macro_f1",
    max_trials: int = 15,
    parallel_trials: int = 2,
    project_id: str | None = None,
    region: str | None = None,
) -> str:
    """Create a Vizier study for hyperparameter optimization.

    Returns the study resource name.
    """
    settings = get_settings()
    project = project_id or settings.platform.project_id
    loc = region or settings.platform.region
    parent = f"projects/{project}/locations/{loc}"

    metric_specs = [
        study_pb2.StudySpec.MetricSpec(
            metric_id=metric_id,
            goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE,
        )
    ]

    parameter_specs = [_build_parameter_spec(p) for p in search_space]

    study_spec = study_pb2.StudySpec(
        metrics=metric_specs,
        parameters=parameter_specs,
        algorithm=study_pb2.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,  # Bayesian by default
    )

    study = study_pb2.Study(
        display_name=f"{capability}-vizier-sweep",
        study_spec=study_spec,
    )

    client = VizierServiceClient(client_options={"api_endpoint": f"{loc}-aiplatform.googleapis.com"})
    created = client.create_study(parent=parent, study=study)
    logger.info("Created Vizier study: %s", created.name)
    return created.name


def run_vizier_sweep(
    study_name: str,
    training_fn: Callable[[dict[str, Any]], float],
    base_config: dict[str, Any],
    *,
    max_trials: int = 15,
    parallel_trials: int = 2,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Run trials: suggest → train → report metric → repeat.

    Args:
        study_name: Resource name of the Vizier study.
        training_fn: Callable that takes a merged config dict and returns the
            evaluation metric (e.g. macro F1).
        base_config: Base configuration dict that trial params override.
        max_trials: Maximum number of trials.
        parallel_trials: Number of parallel trials per suggestion request.
        region: GCP region.

    Returns:
        List of trial result dicts with params and metric values.
    """
    settings = get_settings()
    loc = region or settings.platform.region
    client = VizierServiceClient(client_options={"api_endpoint": f"{loc}-aiplatform.googleapis.com"})

    completed_trials: list[dict[str, Any]] = []
    trials_run = 0

    while trials_run < max_trials:
        suggest_count = min(parallel_trials, max_trials - trials_run)
        suggest_response = client.suggest_trials(
            request={
                "parent": study_name,
                "suggestion_count": suggest_count,
                "client_id": "ml-vizier-sweep",
            }
        )
        # suggest_trials is a long-running operation
        result = suggest_response.result()
        trials = list(result.trials)

        if not trials:
            logger.info("No more trials suggested — study may be complete")
            break

        for trial in trials:
            # Extract suggested parameters
            params = _extract_trial_params(trial)
            logger.info("Trial %s params: %s", trial.name, params)

            # Merge with base config
            merged = {**base_config, **params}

            # Run training and get metric
            try:
                metric_value = training_fn(merged)
            except Exception:
                logger.exception("Trial %s failed", trial.name)
                metric_value = 0.0

            # Report metric back to Vizier
            measurement = study_pb2.Measurement(
                metrics=[
                    study_pb2.Measurement.Metric(
                        metric_id=trial.name.split("/")[-1] if not trial.measurements else "macro_f1",
                        value=metric_value,
                    )
                ]
            )

            # Complete the trial with the measurement
            client.complete_trial(
                request={
                    "name": trial.name,
                    "final_measurement": measurement,
                }
            )

            completed_trials.append(
                {
                    "trial_name": trial.name,
                    "params": params,
                    "metric_value": metric_value,
                }
            )
            trials_run += 1
            logger.info("Trial %s completed: metric=%.4f", trial.name, metric_value)

    logger.info("Vizier sweep complete: %d trials", len(completed_trials))
    return completed_trials


def _extract_trial_params(trial: study_pb2.Trial) -> dict[str, Any]:
    """Extract parameter values from a Vizier trial."""
    params: dict[str, Any] = {}
    for param in trial.parameters:
        pid = param.parameter_id
        if param.value is not None:
            # Protobuf Value — could be number or string
            val = param.value
            if hasattr(val, "number_value") and val.number_value != 0.0:
                params[pid] = val.number_value
            elif hasattr(val, "string_value") and val.string_value:
                params[pid] = val.string_value
            else:
                params[pid] = val.number_value
        else:
            params[pid] = None
    return params


def get_best_config(
    study_name: str,
    *,
    region: str | None = None,
) -> dict[str, Any]:
    """Return the parameter set from the optimal completed trial."""
    settings = get_settings()
    loc = region or settings.platform.region
    client = VizierServiceClient(client_options={"api_endpoint": f"{loc}-aiplatform.googleapis.com"})

    optimal_trials = client.list_optimal_trials(request={"parent": study_name})
    trials = list(optimal_trials.optimal_trials)

    if not trials:
        raise ValueError(f"No optimal trials found for study {study_name}")

    best = trials[0]
    params = _extract_trial_params(best)

    metric_value = None
    if best.final_measurement and best.final_measurement.metrics:
        metric_value = best.final_measurement.metrics[0].value

    logger.info("Best trial: %s, metric=%.4f, params=%s", best.name, metric_value or 0.0, params)
    return {
        "trial_name": best.name,
        "params": params,
        "metric_value": metric_value,
    }


def parse_search_space_from_config(config: dict[str, Any]) -> list[VizierSearchParam]:
    """Parse vizier_search_space from a pipeline config YAML dict."""
    space_def = config.get("vizier_search_space", {})
    params: list[VizierSearchParam] = []

    for name, spec in space_def.items():
        if "values" in spec:
            # Categorical
            params.append(
                VizierSearchParam(
                    name=name,
                    param_type="CATEGORICAL",
                    values=spec["values"],
                )
            )
        elif "min" in spec and "max" in spec:
            # Check if integer or double
            min_val = spec["min"]
            max_val = spec["max"]
            is_int = isinstance(min_val, int) and isinstance(max_val, int) and spec.get("scale") is None
            params.append(
                VizierSearchParam(
                    name=name,
                    param_type="INTEGER" if is_int else "DOUBLE",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    scale=spec.get("scale"),
                )
            )

    return params


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI for running Vizier sweeps."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run Vizier hyperparameter sweep")
    parser.add_argument("--config", required=True, help="Path to pipeline config YAML")
    parser.add_argument("--max-trials", type=int, default=15, help="Maximum number of trials")
    parser.add_argument("--parallel-trials", type=int, default=2, help="Parallel trials per suggest")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    search_space = parse_search_space_from_config(config)
    if not search_space:
        logger.error("No vizier_search_space found in config %s", args.config)
        return

    capability = config.get("capability", "classification")

    study_name = create_vizier_study(
        capability=capability,
        search_space=search_space,
        max_trials=args.max_trials,
        parallel_trials=args.parallel_trials,
    )

    # For CLI runs, the training_fn would submit a pipeline and wait for completion.
    # This is a placeholder that needs to be wired to the actual pipeline submission.
    logger.info("Study created: %s", study_name)
    logger.info(
        "To run the sweep, call run_vizier_sweep() with a training function "
        "that submits a pipeline and returns the evaluation metric."
    )


if __name__ == "__main__":
    main()
