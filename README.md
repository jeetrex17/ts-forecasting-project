# Time Series Forecasting — Applied Forecasting Methods Project

## Overview

This project explores time series forecasting techniques on a hedge fund dataset sourced from the Kaggle competition [*Hedge fund — Time series forecasting*](https://www.kaggle.com/competitions/ts-forecasting). The goal is to compare classical statistical methods against modern deep learning approaches for forecasting continuous numerical values across multiple related time series.

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
- Data provided as Parquet files (`train.parquet`, `test.parquet`)

## Methods

### Classical Statistical Models
- **Exponential Smoothing (ETS)** — trend and seasonality decomposition
- **ARIMA / SARIMA** — autoregressive integrated moving average with automatic order selection

### Deep Learning Models
- **LSTM** — long short-term memory recurrent networks
- **N-BEATS** — neural basis expansion analysis for time series

## Evaluation

Models are compared using a **weighted skill score** (higher is better, range [0, 1]):

$$\text{Score} = \sqrt{1 - \min\!\left(\max\!\left(\frac{\sum_i w_i (y_i - \hat{y}_i)^2}{\sum_i w_i y_i^2},\, 0\right),\, 1\right)}$$

Additionally, standard metrics (RMSE, MAE) are reported for interpretability.

## Project Structure

```
Project/
├── data/                        # Dataset files (train.parquet, test.parquet)
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_classical.ipynb       # ARIMA, ETS models
│   └── 03_deep_learning.ipynb   # LSTM, N-BEATS models
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
1. `01_eda.ipynb` — understand the data, visualize trends, check stationarity
2. `02_classical.ipynb` — fit and evaluate ARIMA/ETS models
3. `03_deep_learning.ipynb` — train and evaluate LSTM/N-BEATS, compare with classical results

## Course

**Applied Forecasting Methods** — DA-IICT, Semester 6 (2026)
