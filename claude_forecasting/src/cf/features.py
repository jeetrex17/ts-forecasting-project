"""Causal feature engineering for the panel.

All lag and rolling features are computed strictly from the past so the
features at time t depend only on values at t-1 and earlier. Per-series
groupings use (code, sub_code, sub_category, horizon).
"""
from __future__ import annotations
import polars as pl

SERIES_KEYS = ["code", "sub_code", "sub_category", "horizon"]


def add_lag_features(lf: pl.LazyFrame, target_col: str = "y_target",
                     lags=(1, 2, 3, 5, 10)) -> pl.LazyFrame:
    """Add lagged target columns. Lazy-friendly."""
    exprs = [pl.col(target_col).shift(k).over(SERIES_KEYS).alias(f"lag_{k}")
             for k in lags]
    return lf.sort(SERIES_KEYS + ["ts_index"]).with_columns(exprs)


def add_rolling_features(lf: pl.LazyFrame, target_col: str = "y_target",
                         windows=(5, 10, 25)) -> pl.LazyFrame:
    """Add causal rolling mean and std. Uses shift(1) to avoid leakage."""
    exprs = []
    for w in windows:
        exprs.append(
            pl.col(target_col).shift(1).rolling_mean(w).over(SERIES_KEYS).alias(f"roll_mean_{w}")
        )
        exprs.append(
            pl.col(target_col).shift(1).rolling_std(w).over(SERIES_KEYS).alias(f"roll_std_{w}")
        )
    return lf.sort(SERIES_KEYS + ["ts_index"]).with_columns(exprs)
