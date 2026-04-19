# Time Series Forecasting — Applied Forecasting Methods Project

## Overview

This repository studies the Kaggle **Hedge fund — Time series forecasting** dataset as a panel forecasting problem. The cleaned workflow is now script-based and reproducible: exploratory notebooks can still be used for visual analysis, but the saved comparison artifacts should be generated from the Python scripts in `scripts/`.

The main methodological changes in this repo are:

- exact dataset naming based on the real schema (`y_target`, `weight`)
- explicit separation between **full-validation ML**, **sampled classical**, **sampled deep-learning**, and **ensemble holdout** studies
- leak-resistant ensemble evaluation using an early-validation **meta split** and a later-validation **holdout split**
- reproducible shared utilities for metrics, feature engineering, and time-aware splits

## Dataset

The dataset is a panel of integer-indexed time series with the following key columns:

| Column | Description |
| --- | --- |
| `id` | Unique row identifier |
| `code` | Primary series identifier |
| `sub_code` | Secondary series identifier |
| `sub_category` | Category grouping |
| `horizon` | Forecast horizon |
| `ts_index` | Integer time index |
| `y_target` | Continuous target value |
| `weight` | Per-row competition weight |

Raw data is expected at:

```text
data/ts-forecasting/train.parquet
data/ts-forecasting/test.parquet
```

The repository ignores `data/` because the Kaggle files are too large to version.

## Implemented Studies

### Classical sampled study

Implemented in `scripts/run_classical_models.py`.

- `Zero`
- `Naive`
- `Expanding Mean`
- `Drift`
- `ETS`
- `ARIMA`

Notes:

- classical models are run on a stratified sampled subset of eligible series
- causal walk-forward baselines are used for `Naive`, `Expanding Mean`, and `Drift`

### Full-validation ML study

Implemented in `scripts/run_ml_models.py`.

- `LightGBM`
- `XGBoost`

Notes:

- trained globally across all series
- uses the 86 anonymized `feature_*` columns plus causal lag and rolling features
- writes full-validation predictions for downstream evaluation

### Sampled deep-learning study

Implemented in `scripts/run_deep_learning_sampled.py`.

- `LSTM`
- `NBEATS`
- `NHITS`

Notes:

- this is intentionally labeled as a **sampled** study
- these models currently use target history only
- they should not be ranked directly against the full-feature ML models without matching the evaluation scope

### Ensemble holdout study

Implemented in `scripts/run_ensemble_holdout.py`.

- `Simple Average`
- `Inverse RMSE Weighted`
- `Optimal Weighted`
- `Stacking (Ridge)`
- `Per-Series Selection`

Notes:

- ensemble weights/selectors are fit on an early-validation **meta** slice
- final ensemble scores are reported only on a later-validation **holdout** slice
- the old notebook-style “cascade” has been removed from the cleaned pipeline because it was not a true residual-correction setup

## Metric

Primary metric: **Weighted Skill Score**.

```text
score = sqrt(1 - clip_0_1(sum(w * (y - y_hat)^2) / sum(w * y^2)))
```

Interpretation:

- `1.0` is perfect
- `0.0` means no gain over the clipped baseline threshold
- values are clipped into `[0, 1]` only at the final ratio stage

The shared implementation lives in [src/afm_project/metrics.py](/Users/jeetraj/Desktop/Everything/DAIICT/Sem_6/Applied_Forecasting_Methods/Project/src/afm_project/metrics.py:1).

## Repository Layout

```text
Project/
├── scripts/
│   ├── run_classical_models.py
│   ├── run_ml_models.py
│   ├── run_deep_learning_sampled.py
│   ├── run_ensemble_holdout.py
│   ├── build_comparison_report.py
│   └── run_pipeline.py
├── src/afm_project/
│   ├── config.py
│   ├── features.py
│   ├── io.py
│   ├── metrics.py
│   ├── splits.py
│   └── stats.py
├── tests/
├── notebooks/                      # exploratory notebooks, optional
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Setup

The cleaned pipeline was developed against Python `3.14.3`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

Run the cleaned pipeline without deep learning:

```bash
.venv/bin/python scripts/run_pipeline.py
```

Include the sampled deep-learning study:

```bash
.venv/bin/python scripts/run_pipeline.py --include-deep-learning
```

Run stages individually:

```bash
.venv/bin/python scripts/run_classical_models.py
.venv/bin/python scripts/run_ml_models.py
.venv/bin/python scripts/run_deep_learning_sampled.py
.venv/bin/python scripts/run_ensemble_holdout.py
.venv/bin/python scripts/build_comparison_report.py
```

## Outputs

By default, scripts write artifacts under `data/processed/`.

Key files:

- `classical_sample_results.parquet`
- `classical_sample_summary.parquet`
- `ml_validation_predictions.parquet`
- `ml_model_summary.parquet`
- `deep_learning_sampled_predictions.parquet`
- `deep_learning_sampled_summary.parquet`
- `ensemble_holdout_predictions.parquet`
- `ensemble_holdout_results.parquet`
- `comparison_report.md`

## Tests

Run the lightweight unit tests:

```bash
.venv/bin/python -m unittest discover -s tests
```

## Notes on Notebooks

The notebooks remain useful for:

- EDA visuals
- qualitative inspection of forecasts
- course-report plots

They are not the authoritative source for saved result artifacts anymore. The scripts are.

## Course

**Applied Forecasting Methods** — DA-IICT, Semester 6 (2026)
