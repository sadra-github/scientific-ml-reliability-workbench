"""Tests for data-quality auditing."""

import numpy as np
import pandas as pd

from reliability_workbench.audit import audit_dataset
from reliability_workbench.data import make_synthetic_regression_data


def test_clean_synthetic_data_passes_audit():
    X, y = make_synthetic_regression_data()
    report = audit_dataset(X, y)

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["n_samples"] == 1200
    assert report["n_features"] == 10
    assert report["duplicate_rows"] == 0
    assert report["infinite_values_by_column"] == {}
    assert report["missing_target_values"] == 0


def test_detects_missing_feature_values():
    X, y = make_synthetic_regression_data()
    X.loc[0, "feature_01"] = np.nan

    report = audit_dataset(X, y)

    assert report["missing_values_by_feature"]["feature_01"] == 1
    assert any("Missing values" in warning for warning in report["warnings"])


def test_detects_duplicate_feature_target_rows():
    X = pd.DataFrame(
        {"feature_01": [1.0, 2.0, 1.0], "feature_02": [3.0, 4.0, 3.0]}
    )
    y = pd.Series([5.0, 6.0, 5.0], name="target")

    report = audit_dataset(X, y)

    assert report["duplicate_rows"] == 1
    assert any("Duplicate" in warning for warning in report["warnings"])


def test_detects_infinite_values():
    X, y = make_synthetic_regression_data()
    X.loc[0, "feature_02"] = np.inf

    report = audit_dataset(X, y)

    assert report["infinite_values_by_column"]["feature_02"] == 1
    assert any("Infinite" in warning for warning in report["warnings"])


def test_missing_target_is_a_blocking_error():
    X, _ = make_synthetic_regression_data()

    report = audit_dataset(X, None)

    assert report["passed"] is False
    assert any("Target y is missing" in error for error in report["errors"])


def test_feature_target_length_mismatch_is_a_blocking_error():
    X, y = make_synthetic_regression_data()
    report = audit_dataset(X, y.iloc[:-1])

    assert report["passed"] is False
    assert any("length mismatch" in error for error in report["errors"])


def test_constant_feature_is_reported():
    X, y = make_synthetic_regression_data()
    X["constant_feature"] = 1.0

    report = audit_dataset(X, y)

    assert "constant_feature" in report["constant_features"]
    assert any("Constant" in warning for warning in report["warnings"])


def test_non_numeric_feature_is_a_blocking_error():
    X, y = make_synthetic_regression_data()
    X["invalid_feature"] = "not_numeric"

    report = audit_dataset(X, y)

    assert report["passed"] is False
    assert any("Non-numeric" in error for error in report["errors"])


def test_infinite_target_is_a_blocking_error():
    X, y = make_synthetic_regression_data(n_samples=30)
    y = y.copy()
    y.iloc[0] = np.inf

    report = audit_dataset(X, y)

    assert report["passed"] is False
    assert any("Infinite values" in error for error in report["errors"])


def test_complex_feature_is_a_blocking_error():
    X, y = make_synthetic_regression_data(n_samples=30)
    X = X.copy()
    X["complex_feature"] = X["feature_01"].astype(complex) + 1j

    report = audit_dataset(X, y)

    assert report["passed"] is False
    assert any("Complex-valued feature" in error for error in report["errors"])


def test_complex_target_is_a_blocking_error():
    X, y = make_synthetic_regression_data(n_samples=30)
    y = y.astype(complex) + 1j

    report = audit_dataset(X, y)

    assert report["passed"] is False
    assert any("real-valued" in error for error in report["errors"])
