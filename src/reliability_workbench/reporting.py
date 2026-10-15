"""JSON reporting and reliability dashboard generation."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn


def save_json_report(data: dict, output_path: str | Path) -> Path:
    """Save a JSON-compatible report, rejecting NaN and infinity."""
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    return path


def create_reliability_dashboard(
    evaluation_results: dict,
    drift_report: dict,
    output_path: str | Path,
) -> Path:
    """Create a four-panel PNG dashboard from evaluation and drift reports."""
    if not isinstance(evaluation_results, dict):
        raise TypeError("evaluation_results must be a dictionary.")
    if not isinstance(drift_report, dict):
        raise TypeError("drift_report must be a dictionary.")

    try:
        metrics = evaluation_results["metrics"]
        predictions = evaluation_results["predictions"]
        y_true = np.asarray(predictions["y_true"], dtype=float)
        y_dummy = np.asarray(predictions["dummy_mean"], dtype=float)
        y_rf = np.asarray(predictions["random_forest"], dtype=float)
        drift_features = drift_report["features"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Evaluation or drift report has an invalid structure.") from exc

    if y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array.")
    if y_dummy.shape != y_true.shape or y_rf.shape != y_true.shape:
        raise ValueError("Prediction arrays must have the same shape as y_true.")
    if not all(np.isfinite(a).all() for a in (y_true, y_dummy, y_rf)):
        raise ValueError("Prediction arrays must contain only finite values.")
    if not isinstance(drift_features, list):
        raise ValueError("drift_report['features'] must be a list.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_metrics, ax_scatter, ax_residuals, ax_drift = axes.flat

    model_names = ["dummy_mean", "random_forest"]
    labels = ["Mean baseline", "Random forest"]
    x = np.arange(len(model_names))
    width = 0.35
    mae = [float(metrics[name]["mae"]) for name in model_names]
    rmse = [float(metrics[name]["rmse"]) for name in model_names]
    ax_metrics.bar(x - width / 2, mae, width, label="MAE")
    ax_metrics.bar(x + width / 2, rmse, width, label="RMSE")
    ax_metrics.set_xticks(x, labels)
    ax_metrics.set_title("Model error comparison")
    ax_metrics.set_ylabel("Error (lower is better)")
    ax_metrics.legend()

    ax_scatter.scatter(y_true, y_rf, alpha=0.65, label="Random forest")
    lower = float(min(y_true.min(), y_rf.min()))
    upper = float(max(y_true.max(), y_rf.max()))
    ax_scatter.plot([lower, upper], [lower, upper], linestyle="--",
                    label="Ideal prediction")
    ax_scatter.set_title("Observed vs predicted")
    ax_scatter.set_xlabel("Observed target")
    ax_scatter.set_ylabel("Predicted target")
    ax_scatter.legend()

    residuals = y_true - y_rf
    ax_residuals.hist(residuals, bins="auto")
    ax_residuals.axvline(0.0, linestyle="--")
    ax_residuals.set_title("Random forest residual distribution")
    ax_residuals.set_xlabel("Residual (observed - predicted)")
    ax_residuals.set_ylabel("Count")

    if drift_features:
        names = [str(item["feature"]) for item in drift_features]
        statistics = [float(item["ks_statistic"]) for item in drift_features]
        order = np.argsort(statistics)
        ax_drift.barh(
            [names[i] for i in order],
            [statistics[i] for i in order],
        )
        ax_drift.set_xlabel("KS statistic")
    else:
        ax_drift.text(0.5, 0.5, "No feature drift results",
                      ha="center", va="center")
        ax_drift.set_xticks([])
        ax_drift.set_yticks([])
    ax_drift.set_title("Feature distribution drift (KS statistic)")

    fig.suptitle("Scientific ML Reliability Dashboard")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return path


def write_run_metadata(
    output_path: str | Path,
    evaluation_results: dict,
) -> Path:
    """Write reproducibility metadata for an evaluation run."""
    if not isinstance(evaluation_results, dict):
        raise TypeError("evaluation_results must be a dictionary.")
    protocol = evaluation_results.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("evaluation_results must contain a protocol dictionary.")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "matplotlib_version": matplotlib.__version__,
        "protocol": protocol,
        "models": evaluation_results.get("models", {}),
    }
    return save_json_report(metadata, output_path)


def write_reliability_reports(
    evaluation_results: dict,
    audit_report: dict,
    drift_report: dict,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write all JSON reports and the dashboard into output_dir."""
    if not all(isinstance(item, dict) for item in (
        evaluation_results, audit_report, drift_report
    )):
        raise TypeError("All report inputs must be dictionaries.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metrics": save_json_report(
            evaluation_results.get("metrics", {}),
            output_dir / "metrics.json",
        ),
        "audit": save_json_report(
            audit_report,
            output_dir / "audit_report.json",
        ),
        "drift": save_json_report(
            drift_report,
            output_dir / "drift_report.json",
        ),
        "metadata": write_run_metadata(
            output_dir / "run_metadata.json",
            evaluation_results,
        ),
        "dashboard": create_reliability_dashboard(
            evaluation_results,
            drift_report,
            output_dir / "reliability_dashboard.png",
        ),
    }
    return paths
