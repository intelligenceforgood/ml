"""Unit tests for Vizier hyperparameter tuning."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ml.training.vizier import (
    VizierSearchParam,
    _build_parameter_spec,
    _extract_trial_params,
    create_vizier_study,
    get_best_config,
    parse_search_space_from_config,
    run_vizier_sweep,
)


class TestBuildParameterSpec:
    def test_double_param(self):
        param = VizierSearchParam(
            name="learning_rate",
            param_type="DOUBLE",
            min_value=0.01,
            max_value=0.3,
            scale="log",
        )
        spec = _build_parameter_spec(param)
        assert spec.parameter_id == "learning_rate"
        assert spec.double_value_spec.min_value == 0.01
        assert spec.double_value_spec.max_value == 0.3

    def test_integer_param(self):
        param = VizierSearchParam(
            name="max_depth",
            param_type="INTEGER",
            min_value=3,
            max_value=8,
        )
        spec = _build_parameter_spec(param)
        assert spec.parameter_id == "max_depth"
        assert spec.integer_value_spec.min_value == 3
        assert spec.integer_value_spec.max_value == 8

    def test_categorical_param(self):
        param = VizierSearchParam(
            name="lora_r",
            param_type="CATEGORICAL",
            values=[4, 8, 16],
        )
        spec = _build_parameter_spec(param)
        assert spec.parameter_id == "lora_r"
        assert list(spec.categorical_value_spec.values) == ["4", "8", "16"]


class TestParseSearchSpace:
    def test_xgboost_config(self):
        config = {
            "vizier_search_space": {
                "n_estimators": {"min": 100, "max": 500},
                "max_depth": {"min": 3, "max": 8},
                "learning_rate": {"min": 0.01, "max": 0.3, "scale": "log"},
                "subsample": {"min": 0.6, "max": 1.0},
            }
        }
        params = parse_search_space_from_config(config)
        assert len(params) == 4

        names = {p.name for p in params}
        assert names == {"n_estimators", "max_depth", "learning_rate", "subsample"}

        # n_estimators and max_depth are INTEGER (int min/max, no scale)
        n_est = next(p for p in params if p.name == "n_estimators")
        assert n_est.param_type == "INTEGER"

        # learning_rate is DOUBLE (has scale)
        lr = next(p for p in params if p.name == "learning_rate")
        assert lr.param_type == "DOUBLE"
        assert lr.scale == "log"

    def test_gemma_config(self):
        config = {
            "vizier_search_space": {
                "learning_rate": {"min": 1e-5, "max": 5e-4, "scale": "log"},
                "lora_r": {"values": [4, 8, 16]},
                "batch_size": {"values": [8, 16, 32]},
            }
        }
        params = parse_search_space_from_config(config)
        assert len(params) == 3

        lr = next(p for p in params if p.name == "learning_rate")
        assert lr.param_type == "DOUBLE"

        lora = next(p for p in params if p.name == "lora_r")
        assert lora.param_type == "CATEGORICAL"
        assert lora.values == [4, 8, 16]

    def test_no_search_space(self):
        config = {"capability": "classification"}
        params = parse_search_space_from_config(config)
        assert params == []


class TestCreateVizierStudy:
    @patch("ml.training.vizier.VizierServiceClient")
    @patch("ml.training.vizier.get_settings")
    def test_creates_study(self, mock_settings, mock_client_cls):
        settings = MagicMock()
        settings.platform.project_id = "test-project"
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_created_study = MagicMock()
        mock_created_study.name = "projects/test-project/locations/us-central1/studies/123"
        mock_client.create_study.return_value = mock_created_study
        mock_client_cls.return_value = mock_client

        search_space = [
            VizierSearchParam(name="lr", param_type="DOUBLE", min_value=0.01, max_value=0.3),
        ]

        result = create_vizier_study("classification", search_space)
        assert result == "projects/test-project/locations/us-central1/studies/123"
        mock_client.create_study.assert_called_once()

    @patch("ml.training.vizier.VizierServiceClient")
    @patch("ml.training.vizier.get_settings")
    def test_study_params(self, mock_settings, mock_client_cls):
        settings = MagicMock()
        settings.platform.project_id = "test-project"
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        mock_client = MagicMock()
        mock_created = MagicMock()
        mock_created.name = "studies/1"
        mock_client.create_study.return_value = mock_created
        mock_client_cls.return_value = mock_client

        search_space = [
            VizierSearchParam(name="lr", param_type="DOUBLE", min_value=0.01, max_value=0.3, scale="log"),
            VizierSearchParam(name="batch_size", param_type="CATEGORICAL", values=[8, 16, 32]),
        ]

        create_vizier_study(
            "classification",
            search_space,
            metric_id="macro_f1",
            max_trials=10,
        )

        call_kwargs = mock_client.create_study.call_args
        study = call_kwargs.kwargs.get("study") or call_kwargs[1].get("study")
        assert study.study_spec.metrics[0].metric_id == "macro_f1"
        assert len(study.study_spec.parameters) == 2


class TestRunVizierSweep:
    @patch("ml.training.vizier.VizierServiceClient")
    @patch("ml.training.vizier.get_settings")
    def test_runs_trials(self, mock_settings, mock_client_cls):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        # Create a trial with parameters
        mock_trial = MagicMock()
        mock_trial.name = "studies/1/trials/1"
        mock_param = MagicMock()
        mock_param.parameter_id = "lr"
        mock_param.value = MagicMock()
        mock_param.value.number_value = 0.05
        mock_param.value.string_value = ""
        mock_trial.parameters = [mock_param]
        mock_trial.measurements = []

        # suggest_trials returns a long-running operation
        mock_suggest_response = MagicMock()
        mock_suggest_result = MagicMock()
        mock_suggest_result.trials = [mock_trial]
        mock_suggest_response.result.return_value = mock_suggest_result

        # Second call returns empty (study done)
        mock_empty_response = MagicMock()
        mock_empty_result = MagicMock()
        mock_empty_result.trials = []
        mock_empty_response.result.return_value = mock_empty_result

        mock_client = MagicMock()
        mock_client.suggest_trials.side_effect = [mock_suggest_response, mock_empty_response]
        mock_client_cls.return_value = mock_client

        # Training function returns a metric
        training_fn = MagicMock(return_value=0.85)

        results = run_vizier_sweep(
            "studies/1",
            training_fn,
            base_config={"epochs": 10},
            max_trials=5,
        )

        assert len(results) == 1
        assert results[0]["metric_value"] == 0.85
        assert results[0]["params"]["lr"] == 0.05
        training_fn.assert_called_once()
        mock_client.complete_trial.assert_called_once()


class TestGetBestConfig:
    @patch("ml.training.vizier.VizierServiceClient")
    @patch("ml.training.vizier.get_settings")
    def test_returns_best(self, mock_settings, mock_client_cls):
        settings = MagicMock()
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        # Create a trial with parameters and measurement
        mock_trial = MagicMock()
        mock_trial.name = "studies/1/trials/best"
        mock_param = MagicMock()
        mock_param.parameter_id = "lr"
        mock_param.value = MagicMock()
        mock_param.value.number_value = 0.1
        mock_param.value.string_value = ""
        mock_trial.parameters = [mock_param]

        mock_metric = MagicMock()
        mock_metric.value = 0.92
        mock_trial.final_measurement = MagicMock()
        mock_trial.final_measurement.metrics = [mock_metric]

        mock_optimal = MagicMock()
        mock_optimal.optimal_trials = [mock_trial]

        mock_client = MagicMock()
        mock_client.list_optimal_trials.return_value = mock_optimal
        mock_client_cls.return_value = mock_client

        result = get_best_config("studies/1")
        assert result["params"]["lr"] == 0.1
        assert result["metric_value"] == 0.92

    @patch("ml.training.vizier.VizierServiceClient")
    @patch("ml.training.vizier.get_settings")
    def test_raises_when_no_optimal(self, mock_settings, mock_client_cls):
        import pytest

        settings = MagicMock()
        settings.platform.region = "us-central1"
        mock_settings.return_value = settings

        mock_optimal = MagicMock()
        mock_optimal.optimal_trials = []

        mock_client = MagicMock()
        mock_client.list_optimal_trials.return_value = mock_optimal
        mock_client_cls.return_value = mock_client

        with pytest.raises(ValueError, match="No optimal trials"):
            get_best_config("studies/1")


class TestExtractTrialParams:
    def test_numeric_params(self):
        mock_trial = MagicMock()
        p1 = MagicMock()
        p1.parameter_id = "lr"
        p1.value = MagicMock()
        p1.value.number_value = 0.05
        p1.value.string_value = ""

        p2 = MagicMock()
        p2.parameter_id = "depth"
        p2.value = MagicMock()
        p2.value.number_value = 6.0
        p2.value.string_value = ""

        mock_trial.parameters = [p1, p2]

        params = _extract_trial_params(mock_trial)
        assert params["lr"] == 0.05
        assert params["depth"] == 6.0

    def test_string_params(self):
        mock_trial = MagicMock()
        p1 = MagicMock()
        p1.parameter_id = "batch_size"
        p1.value = MagicMock()
        p1.value.number_value = 0.0
        p1.value.string_value = "16"

        mock_trial.parameters = [p1]

        params = _extract_trial_params(mock_trial)
        assert params["batch_size"] == "16"
