# Hedge Fund Panel Forecasting

Course final project for Applied Forecasting Methods (DA-IICT, Sem 6, 2026).

This is a clean rebuild of the project after the midway presentation. The midway covered EDA + per-series classical baselines + target-only deep models. This rebuild adds:

- a corrected understanding of the data (multi-horizon cumulative structure, H3-dominant weighted metric)
- a global LGBM-per-horizon model with weighted training
- a fit-H1-and-aggregate experiment exploiting the cumulation constraint
- quantile and conformal prediction intervals
- weight-aware ensembles on a leak-resistant meta/holdout split

## Layout

```
claude_forecasting/
├── data/
│   ├── raw/               train.parquet, test.parquet
│   └── processed/         canonical model outputs (parquet)
├── src/cf/                shared utilities (metrics, splits, features, io, viz)
├── notebooks/             one notebook per study, in numbered order
├── report/                IEEE LaTeX source + figures
└── slides/                final presentation deck
```

## Notebook order

1. `00_data_structure.ipynb` — schema, missingness, target/weight distributions, stationarity diagnostics, cumulation hypothesis verification, H3-weight-dominance finding
2. `01_classical_baselines.ipynb` — 4-series ARIMA / SES / Holt baselines (ported from midway, kept for context)
3. `02_global_lgbm_per_horizon.ipynb` — global LightGBM trained per horizon with sample weights and bootstrap CIs
4. `03_h1_aggregation_experiment.ipynb` — fit H1 only, aggregate to H3/H10/H25 via cumulation, compare to independent per-horizon fits
5. `04_quantile_conformal.ipynb` — LGBM quantile regression + split conformal calibration on the four representative series
6. `05_ensembles.ipynb` — weighted average, ridge stacking on meta slice, per-series/per-horizon model selection
7. `06_final_comparison.ipynb` — unified leaderboard with bootstrap CIs, decision framework, headline findings

Each notebook ends with a "Viva cheat sheet" cell listing the 5–10 things to know for the final presentation.

## Canonical evaluation slice

All models score on the same chronological split:

- train: `ts_index <= 2880`
- meta (for ensembles only): `2880 < ts_index <= 3240`
- holdout: `3240 < ts_index <= 3601`

The Kaggle test set (`ts_index >= 3602`) has no labels and is only used for submission-style sanity checks, not for model selection.

## Setup

Python 3.14, dependencies in `pyproject.toml`. Polars for heavy data work, pandas for compatibility, LightGBM for global models, statsmodels for classical baselines, MAPIE for conformal, optuna for hyperparameter tuning, PyTorch (MPS) for any deep model.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Reproducing results

Each notebook is self-contained and writes one parquet to `data/processed/`. To regenerate everything:

```bash
for nb in notebooks/*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

Total runtime on M4 is approximately 30–60 minutes depending on hyperparameter sweep depth.

## Building the report

The report lives under `report/`. It uses single-column LaTeX (`article` class, 11pt, 1in margins) and pulls figures from `report/figures/`. To build:

```bash
cd report
tectonic main.tex          # self-contained, no LaTeX install required
# or
latexmk -pdf main.tex      # if you have a full LaTeX install
```

The current draft covers:

- Section 1: Problem statement
- Section 2: Forecast targets and series identity
- Section 3: Weighted skill score (formula, worked examples, comparison vs RMSE/MAE/R^2, design rationale)
- Section 4: Dataset and exploratory analysis (full diagnostic chain plus the three originality findings)

Modelling sections (5+) will be added as their notebooks land. Bibliography is deferred to the end of the project.
