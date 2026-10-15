"""Deterministic synthetic regression data generation."""

from numbers import Integral

import numpy as np
import pandas as pd


def make_synthetic_regression_data(
    n_samples: int = 1200,
    n_features: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate reproducible synthetic data for regression experiments.

    The target contains nonlinear effects, an interaction term, and noise.
    The returned feature matrix and target are separate to reduce the risk
    of accidentally passing the target into a model as an input feature.

    Parameters
    ----------
    n_samples:
        Number of observations. Must be at least 2.
    n_features:
        Number of numeric input features. Must be at least 5 because the
        target-generation mechanism uses the first five features.
    random_state:
        Seed for the NumPy random number generator.

    Returns
    -------
    X:
        Numeric feature DataFrame.
    y:
        Numeric target Series named ``target``.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral):
        raise TypeError("n_samples must be an integer.")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")

    if isinstance(n_features, bool) or not isinstance(n_features, Integral):
        raise TypeError("n_features must be an integer.")
    if n_features < 5:
        raise ValueError("n_features must be at least 5.")

    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer.")

    rng = np.random.default_rng(random_state)
    values = rng.normal(size=(n_samples, n_features))

    columns = [f"feature_{index:02d}" for index in range(1, n_features + 1)]
    X = pd.DataFrame(values, columns=columns)

    noise = rng.normal(loc=0.0, scale=0.5, size=n_samples)
    target = (
        2.0 * values[:, 0]
        - 1.2 * values[:, 1] ** 2
        + np.sin(2.0 * values[:, 2])
        + 0.7 * values[:, 3] * values[:, 4]
        + noise
    )

    y = pd.Series(target, name="target", dtype="float64")
    return X, y
