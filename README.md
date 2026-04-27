# Kapelbok Energy Forecasting

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Build](https://img.shields.io/badge/build-manually%20verified-success)
![Python](https://img.shields.io/badge/python-3.10%2B-informational)

Kapelbok Energy Forecasting is a Python project for profiling, training, evaluating, and serving hourly energy demand forecasts from Excel-based meter data. It is designed for analysts, engineers, and project stakeholders who need an interpretable forecasting workflow rather than a black-box model.

The project currently supports three forecasting approaches:

- `SeasonalNaiveForecaster` for simple baseline comparison
- `LinearFeatureForecaster` for feature-based regression
- `StructuralTimeSeriesForecaster` for interpretable trend and seasonality modeling

In addition to the command-line workflow, the repository includes a Streamlit user interface for manually uploading an `.xlsx` workbook and running the training pipeline interactively.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration Options](#configuration-options)
- [Contributing](#contributing)
- [License](#license)
- [Troubleshooting](#troubleshooting)

## Project Overview

This repository processes energy meter data stored in Excel workbooks, cleans and aggregates the data to an hourly level, engineers forecasting features, compares multiple model families, and writes artifacts such as metrics, forecasts, and plots to disk.

Typical use cases include:

- forecasting system-wide hourly electricity consumption
- comparing interpretable time-series models against baseline methods
- testing alternative forecast horizons and train/test splits
- giving non-technical users a basic upload-and-run interface through Streamlit

The default project configuration targets the Kapelbok dataset and expects an Excel sheet with these columns:

- `serial`
- `localtime_hour`
- `last_energy`
- `kilowatt_hours`
- `voltage_min`
- `voltage_avg`
- `voltage_max`
- `current_max`
- `system`

The default forecast target is `kilowatt_hours`.

## Repository Structure

```text
energy_forecasting_kapelbok/
├── app.py
├── config/
│   └── default.yaml
├── data/
│   ├── January to April Kapelbok.xlsx
│   └── README.md
├── notebooks/
│   └── 01_quick_start.py
├── scripts/
│   └── run_training.py
├── src/
│   └── energy_modeling/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── models/
│           ├── baseline.py
│           ├── linear.py
│           └── structural_time_series.py
├── tests/
│   └── test_features.py
├── pyproject.toml
└── requirements.txt
```

## Installation

### Prerequisites

- Python `3.10` or later
- `pip`
- Access to an Excel workbook compatible with the configured schema

### Standard Installation

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install project dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Verify the Installation

The editable install exposes the CLI entry point:

```bash
energy-forecast --help
```

Expected output includes the available subcommands:

```text
profile
train
forecast
```

### Dataset Path Setup

The bundled workbook currently lives at `data/January to April Kapelbok.xlsx`, while the default config uses:

```yaml
data:
  path: "January to April Kapelbok.xlsx"
```

Before running the CLI workflow, either:

1. update `config/default.yaml` to point to `data/January to April Kapelbok.xlsx`, or
2. place the workbook at the project root using the configured filename

## Usage

### 1. Command-Line Workflow

Create a cleaned hourly data profile:

```bash
energy-forecast profile --config config/default.yaml
```

Train the forecasting models and write output artifacts:

```bash
energy-forecast train --config config/default.yaml
```

Generate a future forecast from a previously saved model:

```bash
energy-forecast forecast --config config/default.yaml --horizon 168
```

### 2. Python Usage Example

The project also exposes callable pipeline functions for notebooks or custom scripts.

```python
from energy_modeling.config import load_settings
from energy_modeling.pipeline import load_and_prepare, run_training

settings = load_settings("config/default.yaml")
frame = load_and_prepare(settings)
outputs = run_training(settings)

print(frame.head())
print(outputs)
```

### 3. Streamlit User Interface

Launch the local UI:

```bash
python -m streamlit run app.py
```

The Streamlit app allows you to:

- upload an `.xlsx` file manually
- preview the uploaded data
- run training without moving the file into the project structure
- review metrics, plots, and generated forecast rows
- download the generated forecast as CSV

### 4. Expected Output Files

Training writes artifacts to the configured output directory, which defaults to `outputs/`:

```text
outputs/
├── data_profile.json
├── metrics.csv
├── forecast.csv
├── forecast_plot.png
└── trained_model.joblib
```

## API Documentation

This project does not expose a REST or web API. The primary public interfaces are the CLI and a small Python API intended for scripts, notebooks, and internal integrations.

### CLI Commands

#### `energy-forecast profile`

Creates a JSON summary of the cleaned and aggregated hourly dataset.

```bash
energy-forecast profile --config config/default.yaml
```

#### `energy-forecast train`

Runs the training and evaluation workflow for the baseline, linear, and structural time series models.

```bash
energy-forecast train --config config/default.yaml
```

#### `energy-forecast forecast`

Loads the saved structural time series model and generates future forecasts.

```bash
energy-forecast forecast --config config/default.yaml --model outputs/trained_model.joblib --horizon 72
```

### Python Functions

#### `load_settings(config_path)`

Loads a YAML configuration file into a `Settings` object.

```python
from energy_modeling.config import load_settings

settings = load_settings("config/default.yaml")
```

#### `load_and_prepare(settings, df=None)`

Loads the configured Excel source, or an already-loaded pandas DataFrame, then cleans and aggregates it into an hourly modeling frame.

```python
prepared = load_and_prepare(settings)
```

#### `run_profile(settings, df=None)`

Profiles the cleaned hourly dataset and writes a JSON report.

```python
from energy_modeling.pipeline import run_profile

profile = run_profile(settings)
print(profile["rows"])
```

#### `run_training(settings, df=None)`

Runs the end-to-end training workflow and returns paths to the generated artifacts.

```python
from energy_modeling.pipeline import run_training

outputs = run_training(settings)
print(outputs["metrics"])
print(outputs["forecast"])
```

#### `run_future_forecast(settings, model_path=None, horizon_hours=None)`

Loads a saved model bundle and generates future predictions for the configured forecast horizon.

```python
from energy_modeling.pipeline import run_future_forecast

future = run_future_forecast(settings, horizon_hours=48)
print(future.head())
```

## Configuration Options

The main runtime settings are defined in `config/default.yaml`.

### `project`

| Key | Description | Default |
| --- | --- | --- |
| `name` | Project identifier used for organization | `kapelbok_energy_forecast` |
| `random_seed` | Seed for reproducibility-related workflows | `42` |

### `data`

| Key | Description | Default |
| --- | --- | --- |
| `path` | Path to the source Excel workbook | `January to April Kapelbok.xlsx` |
| `sheet_name` | Excel sheet to read | `Foglio1` |
| `datetime_col` | Timestamp column | `localtime_hour` |
| `target_col` | Forecast target column | `kilowatt_hours` |
| `id_col` | Meter or asset identifier | `serial` |
| `system_col` | System grouping column | `system` |
| `sample_serials` | Optional cap on number of serials loaded for faster experiments | `null` |
| `required_columns` | Columns that must exist in the workbook | See config file |

### `aggregation`

| Key | Description | Default |
| --- | --- | --- |
| `level` | Aggregation strategy label | `system_total` |
| `frequency` | Output time frequency | `h` |
| `target_agg` | Aggregation method for the target | `sum` |
| `numeric_agg` | Aggregation method for numeric exogenous variables | `mean` |
| `fill_target_method` | Missing target handling | `zero` |
| `fill_exog_method` | Missing exogenous feature handling | `interpolate` |

### `features`

| Key | Description | Default |
| --- | --- | --- |
| `calendar.include_hour` | Include hour-of-day features | `true` |
| `calendar.include_day_of_week` | Include weekday features | `true` |
| `calendar.include_month` | Include month features | `true` |
| `calendar.include_weekend` | Include weekend indicator | `true` |
| `calendar.include_cyclical` | Include cyclical encodings | `true` |
| `lags` | Lag offsets used in supervised modeling | `[1, 24, 48, 168]` |
| `rolling_windows` | Rolling window sizes for engineered features | `[24, 168]` |
| `exogenous_columns` | Numeric external predictors | `last_energy`, `voltage_min`, `voltage_avg`, `voltage_max`, `current_max` |

### `forecast`

| Key | Description | Default |
| --- | --- | --- |
| `horizon_hours` | Number of hours to forecast into the future | `168` |
| `test_days` | Holdout window used for evaluation | `14` |
| `confidence_alpha` | Confidence interval significance level | `0.05` |

### `models`

| Key | Description | Default |
| --- | --- | --- |
| `models.baseline.seasonal_period` | Repetition period for the naive baseline | `24` |
| `models.linear.alpha` | Ridge regression regularization strength | `1.0` |
| `models.structural_time_series.level` | State-space level component | `local level` |
| `models.structural_time_series.daily_period` | Daily seasonal period | `24` |
| `models.structural_time_series.weekly_period` | Weekly seasonal period | `168` |
| `models.structural_time_series.weekly_harmonics` | Weekly Fourier harmonics | `3` |
| `models.structural_time_series.maxiter` | Maximum optimizer iterations | `250` |
| `models.structural_time_series.use_exogenous` | Whether to include external regressors in STS | `false` |

### `outputs`

| Key | Description | Default |
| --- | --- | --- |
| `directory` | Output folder | `outputs` |
| `profile_file` | Dataset profile filename | `data_profile.json` |
| `metrics_file` | Model metrics filename | `metrics.csv` |
| `forecast_file` | Evaluation forecast filename | `forecast.csv` |
| `plot_file` | Forecast plot filename | `forecast_plot.png` |
| `model_file` | Serialized model bundle filename | `trained_model.joblib` |

### Example Configuration Snippet

```yaml
data:
  path: "data/January to April Kapelbok.xlsx"
  sheet_name: "Foglio1"
  datetime_col: "localtime_hour"
  target_col: "kilowatt_hours"
  sample_serials: 10

forecast:
  horizon_hours: 72
  test_days: 7

models:
  structural_time_series:
    use_exogenous: true
```

## Contributing

Contributions are welcome, especially improvements to model quality, data validation, user experience, and documentation.

### Recommended Contribution Workflow

1. Fork the repository.
2. Create a feature branch.
3. Set up a virtual environment and install the project in editable mode.
4. Make focused changes with clear commit messages.
5. Run the available checks before opening a pull request.
6. Submit a pull request with a concise description of the change and its impact.

### Development Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest
```

### Local Validation

Run the unit tests:

```bash
python -m pytest -q
```

Check the CLI:

```bash
energy-forecast --help
```

Launch the Streamlit app:

```bash
python -m streamlit run app.py
```

### Contribution Guidelines

- keep changes scoped and reviewable
- update documentation when behavior changes
- preserve compatibility with the existing YAML-driven workflow
- prefer reproducible examples and explicit configuration updates
- add or update tests when modifying data preparation or model behavior

## License

No license file is currently included in this repository.

Until a license is added, treat the codebase as proprietary or internal-use only and obtain approval before redistributing, modifying for external use, or incorporating it into another project.

For open-source distribution, adding a dedicated `LICENSE` file is strongly recommended.

## Troubleshooting

### Import errors such as `cannot import name 'run_training'`

Possible causes:

- dependencies are not fully installed
- the project was not installed in editable mode
- Python is importing an older package copy instead of the local `src/` tree

Recommended fix:

```bash
pip install -r requirements.txt
pip install -e .
```

### `No module named 'statsmodels'`

Install the project dependencies again:

```bash
pip install -r requirements.txt
```

### `No module named 'pytest'`

`pytest` is not currently listed in `requirements.txt`. Install it separately for development:

```bash
pip install pytest
```

### The app cannot find the Excel workbook

Confirm that the configured path in `config/default.yaml` matches the real workbook location. If you are using the bundled file in this repository, set:

```yaml
data:
  path: "data/January to April Kapelbok.xlsx"
```

### Streamlit starts but training fails after upload

Check the uploaded workbook against the required schema. The default configuration expects:

- `serial`
- `localtime_hour`
- `last_energy`
- `kilowatt_hours`
- `voltage_min`
- `voltage_avg`
- `voltage_max`
- `current_max`
- `system`

### Training is slow

This is expected for large workbooks. To speed up experimentation:

- reduce the dataset using `data.sample_serials`
- shorten the holdout period with `forecast.test_days`
- lower model complexity while iterating on the workflow

## Support

For implementation questions, start by reviewing the configuration file, CLI help output, and the Streamlit upload workflow. When reporting issues, include:

- the command you ran
- the config file used
- the full error message
- whether you ran the CLI or the Streamlit app
