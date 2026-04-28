"""Data loading helpers.

The raw parquet files live in data/raw/. Notebooks should import these
helpers rather than hard-coding paths so a single change here updates
every downstream notebook.
"""
from pathlib import Path
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_train_pandas(columns=None):
    """Load train.parquet as pandas. Pass `columns=` to skip features."""
    return pd.read_parquet(RAW_DIR / "train.parquet", columns=columns)


def load_test_pandas(columns=None):
    return pd.read_parquet(RAW_DIR / "test.parquet", columns=columns)


def load_train_polars():
    """Load train.parquet as polars LazyFrame for heavy panel operations."""
    return pl.scan_parquet(RAW_DIR / "train.parquet")


def load_test_polars():
    return pl.scan_parquet(RAW_DIR / "test.parquet")


def save_processed(df, name):
    """Save a DataFrame (pandas or polars) to data/processed/<name>.parquet."""
    path = PROCESSED_DIR / f"{name}.parquet"
    if isinstance(df, pl.DataFrame):
        df.write_parquet(path)
    else:
        df.to_parquet(path, index=False)
    return path


def load_processed(name):
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
