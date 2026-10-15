"""Statistical feature-distribution drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _validate_frame(data: pd.DataFrame, name: str) -> None:
    """Validate one input DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame.")

    if data.empty:
        raise ValueError(f"{name} must not be empty.")

    if data.columns.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate column names.")

    if not all(isinstance(column, str) for column in data.columns):
        raise ValueError(f"{name} column names must be strings.")

    if data.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")

    if data.isna().any().any():
        raise ValueError(f"{name} must not contain missing values.")

    for column in data.columns:
        if not pd.api.types.is_numeric_dtype(data[column].dtype):
            raise ValueError(
                f"{name} feature {column!r} must be numeric."
            )

        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} feature {column!r} must contain only finite values."
            )


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm's step-down adjustment to p-values."""
    count = len(p_values)
    if count == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = (count - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted.tolist()


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> dict:
    """Compare numerical feature distributions using KS tests and Holm correction.

    Each feature is tested independently with a two-sided, two-sample
    Kolmogorov-Smirnov test. Holm adjustment controls the family-wise
    error rate across the tested features.

    A detected distribution difference does not by itself establish
    model-performance degradation.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise ValueError("alpha must be a numeric value between 0 and 1.")

    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    _validate_frame(reference, "reference")
    _validate_frame(current, "current")

    if list(reference.columns) != list(current.columns):
        raise ValueError(
            "reference and current must have identical column names "
            "in the same order."
        )

    if len(reference) < 2 or len(current) < 2:
        raise ValueError(
            "Each dataset must contain at least two samples."
        )

    raw_results = []

    for feature in reference.columns:
        statistic, p_value = ks_2samp(
            reference[feature].to_numpy(dtype=float),
            current[feature].to_numpy(dtype=float),
            alternative="two-sided",
            method="auto",
        )

        raw_results.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )

    adjusted_p_values = _holm_adjust(
        [result["p_value"] for result in raw_results]
    )

    feature_results = []
    for result, adjusted_p in zip(raw_results, adjusted_p_values):
        feature_results.append(
            {
                **result,
                "p_adjusted": float(adjusted_p),
                "drift_detected": bool(adjusted_p < alpha),
            }
        )

    return {
        "test": "two_sample_kolmogorov_smirnov",
        "correction": "Holm",
        "alpha": alpha,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "n_features": int(len(reference.columns)),
        "n_features_with_drift": sum(
            item["drift_detected"] for item in feature_results
        ),
        "features": feature_results,
    }
