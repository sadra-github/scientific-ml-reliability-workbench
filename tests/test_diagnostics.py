import json

import numpy as np
import pytest

from reliability_workbench.diagnostics import analyze_residuals


def test_known_residual_metrics():
    result = analyze_residuals(
        y_true=[3, 5, 4],
        y_pred=[2, 7, 4],
    )

    assert result["n_samples"] == 3
    assert result["residual_mean"] == pytest.approx(-1 / 3)
    assert result["residual_mae"] == pytest.approx(1)
    assert result["residual_rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result["max_absolute_error"] == pytest.approx(2)


def test_residual_signs_indicate_under_and_overprediction():
    result = analyze_residuals(
        y_true=[12, 8],
        y_pred=[10, 10],
    )

    assert result["residual_mean"] == pytest.approx(0)
    assert result["residual_mae"] == pytest.approx(2)


def test_constant_residuals_have_undefined_correlation():
    result = analyze_residuals(
        y_true=[3, 4, 5],
        y_pred=[2, 3, 4],
    )

    assert result["residual_std"] == pytest.approx(0)
    assert result["residual_prediction_correlation"] is None
    assert result["large_error_count"] == 0


def test_constant_predictions_have_undefined_correlation():
    result = analyze_residuals(
        y_true=[1, 3, 2],
        y_pred=[2, 2, 2],
    )

    assert result["residual_prediction_correlation"] is None


def test_large_residual_is_flagged():
    actual = np.zeros(20)
    predicted = np.zeros(20)
    actual[-1] = 100

    result = analyze_residuals(actual, predicted)

    assert result["large_error_count"] == 1
    assert result["large_error_indices"] == [19]


def test_output_is_json_serializable():
    result = analyze_residuals([1, 3, 2], [1.5, 2.5, 2])
    encoded = json.dumps(result, allow_nan=False)

    assert json.loads(encoded)["n_samples"] == 3


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [
        ([], []),
        ([1, 2], [1]),
        ([[1, 2]], [[1, 2]]),
        ([1, np.nan], [1, 2]),
        ([1, 2], [1, np.inf]),
        (["invalid", "data"], [1, 2]),
    ],
)
def test_invalid_inputs_are_rejected(actual, predicted):
    with pytest.raises(ValueError):
        analyze_residuals(actual, predicted)


def test_complex_inputs_are_rejected():
    with pytest.raises(ValueError, match="real-valued"):
        analyze_residuals(
            np.array([1 + 2j, 3 + 4j]),
            np.array([1 + 0j, 2 + 0j]),
        )


def test_complex_object_inputs_are_not_silently_truncated():
    actual = np.array([1 + 2j, 3 + 4j], dtype=object)
    with pytest.raises(ValueError, match="real-valued"):
        analyze_residuals(actual, [1.0, 2.0])
