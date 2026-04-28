"""Canonical chronological splits used across all notebooks.

Train ends at ts_index 2880 (matches the midway cutoff so EDA carries over).
A meta slice 2881..3240 is reserved for ensemble fitting; the holdout
slice 3241..3601 is the final scored window. The Kaggle test set
(ts_index >= 3602) has no labels and is never used for model selection.
"""
TRAIN_END = 2880
META_END = 3240
HOLDOUT_END = 3601


def assign_split(ts_index):
    """Return one of 'train', 'meta', 'holdout', 'test' for an integer ts_index."""
    if ts_index <= TRAIN_END:
        return "train"
    if ts_index <= META_END:
        return "meta"
    if ts_index <= HOLDOUT_END:
        return "holdout"
    return "test"


def add_split_column(df, ts_col="ts_index"):
    """Add a 'split' column to a pandas DataFrame in-place-friendly fashion."""
    df = df.copy()
    df["split"] = df[ts_col].apply(assign_split)
    return df
