"""Leakage-aware regression evaluation."""

from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def evaluate_regression_models(
    X,
    y,
    *,
    test_size=0.2,
    random_state=42,
    n_estimators=200,
):
    """Compare a mean baseline and random forest on one holdout split."""
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("X must be a pandas DataFrame or NumPy array.")

    if isinstance(X, pd.DataFrame):
        if X.empty:
            raise ValueError("X must not be empty.")
        if X.columns.has_duplicates:
            raise ValueError("X must not have duplicate column names.")
        if not all(pd.api.types.is_numeric_dtype(d) for d in X.dtypes):
            raise TypeError("All features must be numeric.")
        if any(pd.api.types.is_complex_dtype(d) for d in X.dtypes):
            raise TypeError("Complex-valued features are not supported.")
        feature_names = [str(c) for c in X.columns]
        X_array = X.to_numpy(dtype=float, na_value=np.nan)
    else:
        X_array = np.asarray(X)
        if (
            X_array.ndim != 2
            or X_array.shape[0] == 0
            or X_array.shape[1] == 0
        ):
            raise ValueError("X must be a non-empty 2D array.")
        if not np.issubdtype(X_array.dtype, np.number):
            raise TypeError("All features must be numeric.")
        if np.iscomplexobj(X_array):
            raise TypeError("Complex-valued features are not supported.")
        X_array = X_array.astype(float)
        feature_names = [
            f"feature_{i + 1:02d}" for i in range(X_array.shape[1])
        ]

    if np.isinf(X_array).any():
        raise ValueError("X contains infinite values.")
    if np.isnan(X_array).all(axis=0).any():
        raise ValueError("Each feature must contain at least one finite value.")

    y_array = np.asarray(y)
    if np.iscomplexobj(y_array):
        raise TypeError("y must contain real-valued numeric values.")
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if not np.issubdtype(y_array.dtype, np.number):
        raise TypeError("y must be numeric.")
    y_array = y_array.astype(float)

    if len(y_array) != X_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if len(y_array) < 3:
        raise ValueError("At least three samples are required.")
    if not np.isfinite(y_array).all():
        raise ValueError("y must contain only finite values.")

    if isinstance(random_state, bool) or not isinstance(random_state, Integral):
        raise TypeError("random_state must be an integer.")
    if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral):
        raise TypeError("n_estimators must be an integer.")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1.")
    if isinstance(test_size, bool) or not isinstance(test_size, (int, float)):
        raise TypeError("test_size must be numeric.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    n_test = int(np.ceil(len(y_array) * test_size))
    if n_test < 2:
        raise ValueError(
            "test_size must allocate at least two test samples for R2 evaluation."
        )
    if n_test >= len(y_array):
        raise ValueError("test_size must leave train and test samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_array,
        y_array,
        test_size=test_size,
        random_state=int(random_state),
        shuffle=True,
    )

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=int(n_estimators),
                random_state=int(random_state),
                n_jobs=1,
            ),
        ),
    }

    metrics = {}
    predictions_by_model = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions_by_model[name] = [
            float(value) for value in predictions
        ]
        metrics[name] = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
        }

    return {
        "protocol": {
            "split": "random_holdout",
            "test_size_requested": float(test_size),
            "random_state": int(random_state),
            "n_samples": int(len(y_array)),
            "n_features": int(X_array.shape[1]),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "feature_names": feature_names,
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
                "n_estimators": int(n_estimators),
                "random_state": int(random_state),
                "n_jobs": 1,
            },
        },
        "metrics": metrics,
        "predictions": {
            "y_true": [float(value) for value in y_test],
            **predictions_by_model,
        },
    }
