# Scientific ML Reliability Workbench

A reproducible Python workbench for auditing scientific machine learning datasets, evaluating regression models, analyzing residuals, and detecting changes in numerical feature distributions.

## Overview

Scientific machine learning requires more than predictive performance. Data quality, evaluation protocols, statistical diagnostics, and reproducibility are also important when assessing model reliability.

The Scientific ML Reliability Workbench provides a modular foundation for these tasks. The current version uses a synthetic regression dataset to demonstrate its workflow without depending on external datasets.

## Current Features

- **Synthetic data generation:** Generate reproducible nonlinear regression data with configurable sample size, feature count, and random seed.
- **Dataset auditing:** Inspect missing values, duplicate rows, non-finite values, constant features, and basic schema issues.
- **Regression evaluation:** Compare a `DummyRegressor` baseline with a `RandomForestRegressor`.
- **Leakage-aware preprocessing:** Fit median imputation using training data only.
- **Residual diagnostics:** Summarize prediction errors and identify observations with unusually large residuals.
- **Data drift detection:** Compare numerical feature distributions using the two-sample Kolmogorov–Smirnov test and Holm multiple-testing correction.
- **Automated reporting:** Export evaluation metrics, audit findings, drift results, run metadata, and a dashboard image.
- **Automated testing:** Validate core functions and integrated reporting behavior.

## Project Structure

```text
scientific-ml-reliability-workbench/
├── pyproject.toml
├── src/
│   └── reliability_workbench/
│       ├── __init__.py
│       ├── data.py
│       ├── audit.py
│       ├── evaluation.py
│       ├── diagnostics.py
│       ├── drift.py
│       └── reporting.py
├── tests/
│   ├── test_data.py
│   ├── test_audit.py
│   ├── test_evaluation.py
│   ├── test_diagnostics.py
│   ├── test_drift.py
│   └── test_reporting.py
└── reports/
    ├── metrics.json
    ├── audit_report.json
    ├── drift_report.json
    ├── run_metadata.json
    └── reliability_dashboard.png
```

## Requirements

- Python 3.11 or later
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- pytest for testing

## Installation

Clone or download the project, then open a terminal in its root directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and its test dependencies:

```bash
python -m pip install -e ".[test]"
```

## Running the Tests

Run the complete test suite from the project root:

```bash
python -m pytest -q
```

Compile-check the source and test modules:

```bash
python -m compileall -q src/reliability_workbench tests
```

At the latest verified checkpoint, the full test suite passes **76 tests**. Re-run the test suite after changes to verify the current state.

## Generated Reports

The reporting module produces five outputs in the `reports/` directory.

| File | Purpose |
|---|---|
| `metrics.json` | Regression evaluation metrics |
| `audit_report.json` | Dataset quality and schema audit |
| `drift_report.json` | Feature-level statistical drift results |
| `run_metadata.json` | Runtime versions and evaluation protocol |
| `reliability_dashboard.png` | Visual summary of evaluation and drift results |

The dashboard contains regression metric comparisons, observed-versus-predicted values, residual analysis, and feature-level drift statistics.

## Current Synthetic Experiment

The current integration experiment uses:

- 1,200 synthetic observations
- 10 numerical features
- A reproducible random seed of 42
- An 80/20 random train/test split
- A Random Forest regressor with 200 trees

The recorded Random Forest results are:

| Metric | Value |
|---|---:|
| MAE | 0.8791 |
| RMSE | 1.1362 |
| R² | 0.8471 |

The drift check flagged 0 of 10 features in the particular comparison performed during the integration run.

These results demonstrate the software workflow on synthetic data. They are not evidence of performance on real scientific datasets or battery degradation data.

## Evaluation Protocol and Limitations

The current implementation uses a random train/test split. This is suitable for demonstrating the workflow, but it is not automatically appropriate for time-series, grouped, subject-specific, or cross-domain evaluation.

The workbench includes statistical feature-distribution checks, but a detected distribution shift does not necessarily imply degraded predictive performance. Conversely, an absence of detected drift does not prove that distributions are identical or that a model is reliable.

Dataset auditing identifies selected quality issues; it does not guarantee the absence of every form of data leakage or experimental bias.

The current benchmark uses synthetic regression data. Real-world validation, domain-specific validation protocols, and additional reliability analyses remain future development tasks.

## Reproducibility

The project records relevant runtime versions and evaluation protocol information in `run_metadata.json`. Synthetic data generation and evaluation use explicit random seeds to support reproducible experiments.

Exact numerical results may vary across software versions or execution environments.

## Development Status

**Version:** 0.1.0

The current implementation includes data generation, dataset auditing, regression evaluation, residual diagnostics, statistical drift detection, automated report generation, and unit/integration tests.

Further development can extend the framework with additional evaluation protocols, domain-specific checks, and more comprehensive reliability assessments.

## License

No license has been specified yet. A license should be selected and added before the project is distributed for reuse.
