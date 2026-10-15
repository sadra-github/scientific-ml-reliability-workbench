"""Tests for deterministic synthetic regression data."""

import numpy as np
import pandas as pd
import pytest

from reliability_workbench.data import make_synthetic_regression_data


def test_default_data_shape_and_schema():
    X, y = make_synthetic_regression_data()

    assert X.shape == (1200, 10)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert y.name == "target"
    assert len(y) == len(X)
    assert X.columns.is_unique
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    assert pd.api.types.is_numeric_dtype(y.dtype)


def test_same_seed_produces_identical_data():
    X1, y1 = make_synthetic_regression_data(random_state=42)
    X2, y2 = make_synthetic_regression_data(random_state=42)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_different_seeds_change_generated_data():
    X1, y1 = make_synthetic_regression_data(random_state=42)
    X2, y2 = make_synthetic_regression_data(random_state=43)

    assert not np.array_equal(X1.to_numpy(), X2.to_numpy())
    assert not np.array_equal(y1.to_numpy(), y2.to_numpy())


@pytest.mark.parametrize("n_samples", [0, 1, -1])
def test_rejects_invalid_sample_counts(n_samples):
    with pytest.raises(ValueError):
        make_synthetic_regression_data(n_samples=n_samples)


def test_rejects_too_few_features():
    with pytest.raises(ValueError):
        make_synthetic_regression_data(n_features=4)


def test_accepts_numpy_integer_parameters():
    X, y = make_synthetic_regression_data(
        n_samples=np.int64(30),
        n_features=np.int64(5),
        random_state=np.int64(42),
    )
    assert X.shape == (30, 5)
    assert len(y) == 30


@pytest.mark.parametrize("parameter", ["n_samples", "n_features", "random_state"])
def test_rejects_boolean_integer_parameters(parameter):
    with pytest.raises(TypeError):
        make_synthetic_regression_data(**{parameter: True})
