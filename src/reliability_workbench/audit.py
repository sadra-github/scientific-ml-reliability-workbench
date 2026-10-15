"""Data-quality auditing for tabular regression datasets."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def audit_dataset(
    X: pd.DataFrame,
    y: pd.Series | None,
    target_name: str = "target",
) -> dict:
    """Audit feature data and a regression target.

    Missing values, infinite values, duplicate rows, and constant features
    are reported. Missing or invalid target data, invalid feature containers,
    non-numeric features, and feature-target length mismatches are blocking
    errors.

    ``passed`` means that no blocking errors were found. Warnings can still
    be present and should be reviewed before model training.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(X, pd.DataFrame):
        return {
            "passed": False,
            "n_samples": None,
            "n_features": None,
            "feature_columns": [],
            "target_name": target_name,
            "missing_values_by_feature": {},
            "missing_target_values": None,
            "duplicate_rows": None,
            "infinite_values_by_column": {},
            "constant_features": [],
            "errors": ["X must be a pandas DataFrame."],
            "warnings": [],
        }

    if X.empty:
        errors.append("Feature data is empty.")

    if X.columns.has_duplicates:
        errors.append("Feature column names must be unique.")

    non_numeric_columns = [
        str(column)
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column].dtype)
    ]
    if non_numeric_columns:
        errors.append(
            "Non-numeric feature columns found: "
            + ", ".join(non_numeric_columns)
        )

    complex_columns = [
        str(column)
        for column in X.columns
        if pd.api.types.is_complex_dtype(X[column].dtype)
    ]
    if complex_columns:
        errors.append(
            "Complex-valued feature columns found: "
            + ", ".join(complex_columns)
        )

    missing_values = {
        str(column): int(count)
        for column, count in X.isna().sum().items()
    }
    missing_features = {
        column: count for column, count in missing_values.items() if count > 0
    }
    if missing_features:
        warnings.append("Missing values were found in feature data.")

    infinite_values: dict[str, int] = {}
    for column in X.columns:
        series = X[column]
        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_complex_dtype(series.dtype)
        ):
            numeric_values = series.to_numpy(dtype=float, na_value=np.nan)
            count = int(np.isinf(numeric_values).sum())
            if count:
                infinite_values[str(column)] = count

    if infinite_values:
        warnings.append("Infinite values were found.")

    constant_features = [
        str(column)
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]
    if constant_features:
        warnings.append("Constant or entirely missing features were found.")

    missing_target_values: int | None = None
    duplicate_rows: int | None = None

    if y is None:
        errors.append("Target y is missing.")
    elif not isinstance(y, pd.Series):
        errors.append("y must be a pandas Series.")
    else:
        missing_target_values = int(y.isna().sum())

        if len(y) != len(X):
            errors.append(
                f"Feature-target length mismatch: len(X)={len(X)}, len(y)={len(y)}."
            )

        if y.empty:
            errors.append("Target data is empty.")

        if not pd.api.types.is_numeric_dtype(y.dtype):
            errors.append("Target must have a numeric data type.")
        elif pd.api.types.is_complex_dtype(y.dtype):
            errors.append("Target must contain real-valued numeric data.")
        else:
            target_values = y.to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(target_values).any():
                errors.append("Infinite values were found in the target.")

        if missing_target_values > 0:
            warnings.append("Missing values were found in the target.")

        if len(y) == len(X):
            combined = pd.concat(
                [
                    X.reset_index(drop=True),
                    y.rename(target_name).reset_index(drop=True),
                ],
                axis=1,
            )
            duplicate_rows = int(combined.duplicated().sum())
            if duplicate_rows > 0:
                warnings.append("Duplicate feature-target rows were found.")

    return {
        "passed": len(errors) == 0,
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "feature_columns": [str(column) for column in X.columns],
        "target_name": target_name,
        "missing_values_by_feature": missing_values,
        "missing_target_values": missing_target_values,
        "duplicate_rows": duplicate_rows,
        "infinite_values_by_column": infinite_values,
        "constant_features": constant_features,
        "errors": errors,
        "warnings": warnings,
    }
