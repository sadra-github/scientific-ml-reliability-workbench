"""Numerical residual diagnostics for regression models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def analyze_residuals(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict:
    """Calculate regression residual statistics and flag large residuals.

    Residuals are defined as y_true - y_pred.
    A positive residual indicates underprediction; a negative residual
    indicates overprediction.

    Large residuals are flagged when their absolute deviation from the
    residual mean is greater than three population standard deviations.
    This rule is a diagnostic heuristic, not proof of an outlier.
    """
    raw_actual = np.asarray(y_true)
    raw_predicted = np.asarray(y_pred)

    def contains_complex_values(values: np.ndarray) -> bool:
        if np.iscomplexobj(values):
            return True
        if values.dtype == object:
            return any(
                isinstance(value, (complex, np.complexfloating))
                for value in values.flat
            )
        return False

    if contains_complex_values(raw_actual) or contains_complex_values(raw_predicted):
        raise ValueError("Inputs must contain real-valued numeric values.")

    try:
        actual = np.asarray(raw_actual, dtype=float)
        predicted = np.asarray(raw_predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Inputs must contain numeric values.") from exc

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional arrays.")

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("Inputs must not be empty.")

    if actual.size != predicted.size:
        raise ValueError("Inputs must have the same length.")

    if not np.isfinite(actual).all():
        raise ValueError("y_true must contain only finite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("y_pred must contain only finite values.")

    residuals = actual - predicted
    if not np.isfinite(residuals).all():
        raise ValueError("Residuals overflowed and are not finite.")

    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals, ddof=0))
    absolute_errors = np.abs(residuals)

    if np.std(predicted, ddof=0) == 0 or residual_std == 0:
        residual_prediction_correlation = None
    else:
        correlation = np.corrcoef(residuals, predicted)[0, 1]
        residual_prediction_correlation = (
            float(correlation) if np.isfinite(correlation) else None
        )

    if residual_std == 0:
        large_error_indices: list[int] = []
    else:
        threshold = 3.0 * residual_std
        large_error_indices = np.flatnonzero(
            np.abs(residuals - residual_mean) > threshold
        ).astype(int).tolist()

    return {
        "n_samples": int(actual.size),
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_mae": float(np.mean(absolute_errors)),
        "residual_rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_absolute_error": float(np.max(absolute_errors)),
        "residual_prediction_correlation": residual_prediction_correlation,
        "large_error_threshold_rule": "abs(residual - residual_mean) > 3 * population_std",
        "large_error_count": len(large_error_indices),
        "large_error_indices": large_error_indices,
    }
