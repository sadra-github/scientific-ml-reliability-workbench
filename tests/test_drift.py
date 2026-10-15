import json

import numpy as np
import pandas as pd
import pytest

from reliability_workbench.drift import detect_data_drift, _holm_adjust


def test_clear_drift_is_detected():
    ref = pd.DataFrame({"x": np.zeros(30)})
    cur = pd.DataFrame({"x": np.ones(30) * 10})
    result = detect_data_drift(ref, cur)
    assert result["features"][0]["drift_detected"] is True
    assert result["features"][0]["ks_statistic"] == pytest.approx(1.0)


def test_identical_data_has_no_drift():
    ref = pd.DataFrame({"x": np.arange(10, dtype=float)})
    result = detect_data_drift(ref, ref.copy())
    assert result["features"][0]["ks_statistic"] == 0
    assert result["features"][0]["drift_detected"] is False


def test_holm_adjustment():
    assert _holm_adjust([0.01, 0.04, 0.03]) == pytest.approx(
        [0.03, 0.06, 0.06]
    )


def test_multiple_features_reported():
    ref = pd.DataFrame({"a": np.arange(20), "b": np.zeros(20)})
    cur = pd.DataFrame({"a": np.arange(20), "b": np.ones(20) * 5})
    result = detect_data_drift(ref, cur)
    assert result["n_features"] == 2
    assert result["features"][1]["drift_detected"] is True


def test_json_serializable():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = detect_data_drift(frame, frame.copy())
    assert json.loads(json.dumps(result, allow_nan=False))["correction"] == "Holm"


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf, True])
def test_invalid_alpha(alpha):
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        detect_data_drift(frame, frame, alpha=alpha)


def test_mismatched_columns_rejected():
    ref = pd.DataFrame({"x": [1.0, 2.0]})
    cur = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(ValueError):
        detect_data_drift(ref, cur)


def test_missing_values_rejected():
    ref = pd.DataFrame({"x": [1.0, np.nan]})
    cur = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        detect_data_drift(ref, cur)


def test_non_numeric_values_rejected():
    ref = pd.DataFrame({"x": ["a", "b"]})
    cur = pd.DataFrame({"x": ["a", "c"]})
    with pytest.raises(ValueError):
        detect_data_drift(ref, cur)


def test_too_few_samples_rejected():
    ref = pd.DataFrame({"x": [1.0]})
    cur = pd.DataFrame({"x": [2.0]})
    with pytest.raises(ValueError):
        detect_data_drift(ref, cur)
