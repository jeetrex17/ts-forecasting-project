# Time Series Forecasting — Applied Forecasting Methods Project

## Overview

A comprehensive study of time series forecasting methods applied to a hedge fund dataset from the Kaggle competition [*Hedge fund — Time series forecasting*](https://www.kaggle.com/competitions/ts-forecasting). This project systematically compares classical statistical models, gradient-boosted tree models, deep learning sequence models, and ensemble/hybrid approaches across thousands of related financial time series.

## Dataset

The dataset is a **panel of integer-indexed time series**. Each series is identified by a combination of categorical keys, and the target is a continuous numerical value.

| Column | Description |
|---|---|
| `code` | Primary series identifier |
| `sub_code` | Secondary series identifier |
| `sub_category` | Category grouping |
| `horizon` | Forecast horizon |
| `ts_index` | Integer time index (the time axis) |
| `target` | Continuous value to predict |

**Key characteristics:**
- Multiple related series sharing structure across codes and sub-categories
- Low signal-to-noise ratio
- Potentially non-stationary processes
- Data provided as Parquet files (`train.parquet` ~740MB, `test.parquet` ~139MB)

## Methods

### Baselines
- Zero prediction, last value, expanding mean, seasonal naive

### Classical Statistical Models
- **Exponential Smoothing (ETS)** — automatic model selection with trend/seasonality decomposition
- **ARIMA / SARIMA** — autoregressive integrated moving average via `pmdarima` auto order selection
- Full residual diagnostics (Ljung-Box, Shapiro-Wilk, heteroskedasticity tests)

### Machine Learning (Global Models)
- **LightGBM** and **XGBoost** — gradient boosted trees trained globally across all series
- Causal feature engineering: lag features, rolling/expanding statistics, group-level target encodings
- SHAP-based feature importance analysis
- Time-series-aware hyperparameter tuning

### Deep Learning
- **LSTM / GRU** — recurrent neural network baselines
- **N-BEATS** — neural basis expansion with interpretable trend/seasonality decomposition
- **NHITS** — hierarchical interpolation for multi-scale forecasting
- **Temporal Fusion Transformer (TFT)** — attention-based model with built-in interpretability
- All trained globally across the panel via NeuralForecast

### Ensemble & Hybrid Approaches
- Simple average and inverse-error weighted ensembles
- Stacking with a meta-learner trained on base model predictions
- Per-series model selection (best model per group)
- Sequential cascade: classical model residuals fed into ML/DL models
- Ablation study on ensemble combinations

## Evaluation

Models are compared using multiple metrics:

**Primary — Weighted Skill Score** (higher is better, range [0, 1]):

$$\text{Score} = \sqrt{1 - \min\!\left(\max\!\left(\frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i y_i^2},\, 0\right),\, 1\right)}$$

**Secondary:** RMSE, MAE, MAPE, MASE

**Statistical rigor:** Diebold-Mariano tests for pairwise model comparison, performance breakdown by code/sub_category/horizon.

## Project Structure

```
Project/
├── data/ts-forecasting/              # train.parquet, test.parquet (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_classical_models.ipynb     # Baselines, ETS, ARIMA
│   ├── 03_ml_models.ipynb            # LightGBM, XGBoost (global)
│   ├── 04_deep_learning.ipynb        # LSTM, N-BEATS, NHITS, TFT
│   ├── 05_ensemble_hybrid.ipynb      # Stacking, weighted, cascaded ensembles
│   └── 06_comparison_report.ipynb    # Statistical tests, final comparison
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Download Data

Download the dataset from [Kaggle](https://www.kaggle.com/competitions/ts-forecasting/data) and place the files in `data/ts-forecasting/`:

```
data/
└── ts-forecasting/
    ├── train.parquet   (~740 MB)
    └── test.parquet    (~139 MB)
```

Data files are excluded from git due to size.

## Usage

Run the notebooks in order:

1. `01_eda.ipynb` — understand the data, panel structure, stationarity, distributions
2. `02_classical_models.ipynb` — fit baselines, ETS, ARIMA with residual diagnostics
3. `03_ml_models.ipynb` — feature engineering, LightGBM/XGBoost, SHAP analysis
4. `04_deep_learning.ipynb` — train LSTM, N-BEATS, NHITS, TFT
5. `05_ensemble_hybrid.ipynb` — combine models (parallel, sequential, stacking)
6. `06_comparison_report.ipynb` — full comparison with statistical significance tests

## Course

**Applied Forecasting Methods** — DA-IICT, Semester 6 (2026)
