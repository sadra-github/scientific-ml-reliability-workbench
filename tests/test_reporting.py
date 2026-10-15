"""Tests for reliability reporting and dashboard generation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from reliability_workbench.reporting import (
    create_reliability_dashboard,
    save_json_report,
    write_reliability_reports,
    write_run_metadata,
)


@pytest.fixture
def evaluation_results():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    return {
        "protocol": {
            "split": "random_holdout",
            "test_size_requested": 0.2,
            "random_state": 42,
            "n_samples": 30,
            "n_features": 2,
            "n_train": 24,
            "n_test": 6,
            "feature_names": ["feature_01", "feature_02"],
            "target_name": "target",
            "imputation": "median fitted on training partition only",
        },
        "models": {
            "dummy_mean": {
                "estimator": "DummyRegressor",
                "strategy": "mean",
            },
            "random_forest": {
                "estimator": "RandomForestRegressor",
                "n_estimators": 20,
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "metrics": {
            "dummy_mean": {"mae": 1.5, "rmse": 1.8, "r2": 0.0},
            "random_forest": {"mae": 0.3, "rmse": 0.4, "r2": 0.95},
        },
        "predictions": {
            "y_true": y_true,
            "dummy_mean": [3.5] * len(y_true),
            "random_forest": [1.1, 1.8, 3.2, 3.9, 5.1, 5.8],
        },
    }


@pytest.fixture
def drift_report():
    return {
        "test": "two_sample_kolmogorov_smirnov",
        "correction": "holm",
        "alpha": 0.05,
        "n_reference": 100,
        "n_current": 100,
        "n_features": 2,
        "n_features_with_drift": 1,
        "features": [
            {
                "feature": "feature_01",
                "ks_statistic": 0.42,
                "p_value": 0.001,
                "p_adjusted": 0.002,
                "drift_detected": True,
            },
            {
                "feature": "feature_02",
                "ks_statistic": 0.08,
                "p_value": 0.7,
                "p_adjusted": 0.7,
                "drift_detected": False,
            },
        ],
    }


def test_json_report_round_trip(tmp_path):
    path = tmp_path / "nested" / "report.json"
    expected = {"status": "passed", "count": 3}

    result = save_json_report(expected, path)

    assert result == path
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_json_report_rejects_nonfinite_values(tmp_path):
    path = tmp_path / "invalid.json"

    with pytest.raises(ValueError):
        save_json_report({"value": float("nan")}, path)


def test_dashboard_creates_png(
    tmp_path, evaluation_results, drift_report
):
    path = tmp_path / "plots" / "dashboard.png"

    result = create_reliability_dashboard(
        evaluation_results, drift_report, path
    )

    assert result == path
    assert path.is_file()
    assert path.stat().st_size > 1000
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_dashboard_rejects_mismatched_predictions(
    tmp_path, evaluation_results, drift_report
):
    evaluation_results["predictions"]["random_forest"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="same shape"):
        create_reliability_dashboard(
            evaluation_results, drift_report, tmp_path / "bad.png"
        )


def test_metadata_contains_versions_and_protocol(
    tmp_path, evaluation_results
):
    path = tmp_path / "metadata.json"

    write_run_metadata(path, evaluation_results)
    metadata = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["python_version"]
    assert metadata["numpy_version"]
    assert metadata["scikit_learn_version"]
    assert metadata["protocol"] == evaluation_results["protocol"]
    assert metadata["models"] == evaluation_results["models"]
    assert metadata["created_at_utc"].endswith("+00:00")


def test_full_reporting_creates_five_outputs(
    tmp_path, evaluation_results, drift_report
):
    audit = {"passed": True, "n_samples": 30}
    paths = write_reliability_reports(
        evaluation_results, audit, drift_report, tmp_path / "reports"
    )

    expected = {
        "metrics",
        "audit",
        "drift",
        "metadata",
        "dashboard",
    }
    assert set(paths) == expected
    assert all(path.is_file() for path in paths.values())

    assert json.loads(paths["metrics"].read_text()) == (
        evaluation_results["metrics"]
    )
    assert json.loads(paths["audit"].read_text()) == audit
    assert json.loads(paths["drift"].read_text()) == drift_report
    assert paths["dashboard"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_reporting_does_not_mutate_inputs(
    tmp_path, evaluation_results, drift_report
):
    audit = {"passed": True}
    original_evaluation = copy.deepcopy(evaluation_results)
    original_drift = copy.deepcopy(drift_report)
    original_audit = copy.deepcopy(audit)

    write_reliability_reports(
        evaluation_results, audit, drift_report, tmp_path / "reports"
    )

    assert evaluation_results == original_evaluation
    assert drift_report == original_drift
    assert audit == original_audit


def test_invalid_evaluation_type_is_rejected(tmp_path, drift_report):
    with pytest.raises(TypeError):
        create_reliability_dashboard(
            [], drift_report, tmp_path / "dashboard.png"
        )


def test_invalid_drift_features_are_rejected(
    tmp_path, evaluation_results
):
    invalid_drift = {"features": "not-a-list"}

    with pytest.raises(ValueError, match="must be a list"):
        create_reliability_dashboard(
            evaluation_results, invalid_drift, tmp_path / "dashboard.png"
        )


def test_numpy_nan_is_rejected_by_json_writer(tmp_path):
    with pytest.raises(ValueError):
        save_json_report(
            {"value": np.float64(np.nan)}, tmp_path / "numpy_nan.json"
        )
