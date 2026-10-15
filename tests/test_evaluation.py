import numpy as np
import pytest

from reliability_workbench.data import make_synthetic_regression_data
from reliability_workbench.evaluation import evaluate_regression_models


def test_evaluation_returns_expected_models_metrics_and_split():
    X, y = make_synthetic_regression_data()

    result = evaluate_regression_models(X, y)

    assert set(result["metrics"]) == {"dummy_mean", "random_forest"}
    assert set(result["metrics"]["dummy_mean"]) == {"mae", "rmse", "r2"}
    assert set(result["metrics"]["random_forest"]) == {"mae", "rmse", "r2"}
    assert result["protocol"]["n_samples"] == 1200
    assert result["protocol"]["n_features"] == 10
    assert result["protocol"]["n_train"] == 960
    assert result["protocol"]["n_test"] == 240


def test_evaluation_is_reproducible():
    X, y = make_synthetic_regression_data()

    first = evaluate_regression_models(X, y, n_estimators=30)
    second = evaluate_regression_models(X, y, n_estimators=30)

    assert first["metrics"] == second["metrics"]


def test_random_forest_beats_dummy_on_synthetic_data():
    X, y = make_synthetic_regression_data()

    result = evaluate_regression_models(X, y, n_estimators=100)

    assert (
        result["metrics"]["random_forest"]["rmse"]
        < result["metrics"]["dummy_mean"]["rmse"]
    )


def test_missing_feature_values_are_imputed():
    X, y = make_synthetic_regression_data(n_samples=100, random_state=12)
    X = X.copy()
    X.iloc[0, 0] = np.nan

    result = evaluate_regression_models(X, y, n_estimators=20)

    assert np.isfinite(result["metrics"]["random_forest"]["rmse"])


def test_infinite_features_are_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    X = X.copy()
    X.iloc[0, 0] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        evaluate_regression_models(X, y)


def test_non_finite_target_is_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    y = y.copy()
    y[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        evaluate_regression_models(X, y)


def test_mismatched_lengths_are_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)

    with pytest.raises(ValueError, match="same number"):
        evaluate_regression_models(X, y[:-1])


def test_invalid_test_size_is_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)

    with pytest.raises(ValueError, match="test_size"):
        evaluate_regression_models(X, y, test_size=1.0)


def test_duplicate_dataframe_columns_are_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    X = X.copy()
    X.columns = ["duplicate"] * X.shape[1]

    with pytest.raises(ValueError, match="duplicate column"):
        evaluate_regression_models(X, y)


def test_all_missing_feature_is_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    X = X.copy()
    X.iloc[:, 0] = np.nan

    with pytest.raises(ValueError, match="at least one finite"):
        evaluate_regression_models(X, y)


def test_complex_features_are_rejected():
    X = np.array(
        [[1 + 2j, 2], [2 + 1j, 3], [3 + 0j, 4], [4 + 1j, 5]]
    )
    y = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(TypeError, match="Complex-valued"):
        evaluate_regression_models(X, y, n_estimators=5)


def test_complex_dataframe_features_are_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    X = X.copy()
    X["complex_feature"] = X["feature_01"].astype(complex) + 1j

    with pytest.raises(TypeError, match="Complex-valued"):
        evaluate_regression_models(X, y, n_estimators=5)


def test_complex_target_is_rejected():
    X, y = make_synthetic_regression_data(n_samples=30)
    y = y.to_numpy(dtype=complex) + 1j

    with pytest.raises(TypeError, match="real-valued"):
        evaluate_regression_models(X, y, n_estimators=5)


def test_test_split_requires_two_test_samples_for_r2():
    X, y = make_synthetic_regression_data(n_samples=4)

    with pytest.raises(ValueError, match="two test samples"):
        evaluate_regression_models(X, y, test_size=0.2, n_estimators=5)
