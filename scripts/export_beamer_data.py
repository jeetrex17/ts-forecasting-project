#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "xdg-cache"))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import presentation_deck_v2 as deck

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=Warning, module="statsmodels")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ts-forecasting"
DEFAULT_OUT = ROOT / "presentation" / "beamer" / "data"
SERIES_KEYS = ["code", "sub_code", "sub_category", "horizon"]
REASON_ORDER = ["longest history", "highest total weight", "most volatile", "most stable"]
REASON_LABELS = {
    "longest history": "Longest History",
    "highest total weight": "Highest Total Weight",
    "most volatile": "Most Volatile",
    "most stable": "Most Stable",
}
VAL_CUTOFF = 2880
SEED = 42

LOOKBACK = 24
TRAIN_FRACTION = 0.80
HIDDEN_SIZE = 32
NUM_LAYERS = 1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 12
ARCHITECTURES = ["RNN", "GRU", "LSTM"]
DEVICE = torch.device("cpu")
FAMILY_ORDER = ["Smoothing", "AR", "MA", "ARMA", "ARIMA"]
FOCUS_MODELS = [
    "ARIMA(1,1,1)",
    "ARIMA(1,1,0)",
    "ARIMA(0,1,1)",
    "AR(1)",
    "AR(2)",
    "AR(3)",
    "ARMA(1,1)",
    "ARMA(1,2)",
    "ARMA(1,3)",
]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = df.copy()
    for column in clean_df.select_dtypes(include=["object", "string"]).columns:
        clean_df[column] = clean_df[column].map(
            lambda value: value.replace(";", " - ").replace("\n", " ").replace("\r", " ") if isinstance(value, str) else value
        )
    clean_df.to_csv(path, index=False, sep=";")
    safe_path = path.with_name(path.name.replace("_", "-"))
    if safe_path != path:
        clean_df.to_csv(safe_path, index=False, sep=";")


def fmt_sig(value: float | int | str | None, decimals: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, str):
        return value
    value = float(value)
    abs_value = abs(value)
    if abs_value == 0:
        return "0.000"
    if abs_value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs_value >= 1e3:
        return f"{value:,.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.3g}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    if abs_value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


def fmt_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{int(round(float(value))):,}"


def fmt_pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{decimals}f}"


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    safe_path = path.with_name(path.name.replace("_", "-"))
    if safe_path != path:
        safe_path.write_text(json.dumps(payload, indent=2))


def write_panel_csvs(df: pd.DataFrame, path: Path, panel_col: str = "panel_id") -> None:
    if panel_col not in df.columns:
        return
    for panel_value, group in df.groupby(panel_col, sort=True):
        try:
            panel_label = int(panel_value)
        except Exception:
            panel_label = panel_value
        panel_path = path.with_name(f"{path.stem}_panel_{panel_label}{path.suffix}")
        write_csv(group.reset_index(drop=True), panel_path)


def title_case_reason(reason: str) -> str:
    return REASON_LABELS[reason]


def fmt_series_key(row: pd.Series) -> str:
    return f"{row['code']} / {row['sub_code']} / {row['sub_category']} / H{int(row['horizon'])}"


def deep_series_id(row: pd.Series) -> str:
    return f"{row['code']}_{row['sub_code']}_{row['sub_category']}_H{int(row['horizon'])}"


def weighted_skill(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    denom = np.sum(weight * (y_true**2))
    if denom == 0:
        return np.nan
    ratio = np.sum(weight * ((y_true - y_pred) ** 2)) / denom
    ratio = min(max(float(ratio), 0.0), 1.0)
    return float(np.sqrt(1.0 - ratio))


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight_sum = np.sum(weight)
    if weight_sum == 0:
        return np.nan
    return float(np.sqrt(np.sum(weight * ((y_true - y_pred) ** 2)) / weight_sum))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight_sum = np.sum(weight)
    if weight_sum == 0:
        return np.nan
    return float(np.sum(weight * np.abs(y_true - y_pred)) / weight_sum)


def mase(y_true: np.ndarray, y_pred: np.ndarray, train_scale: float) -> float:
    if not np.isfinite(train_scale) or train_scale == 0:
        return np.nan
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)) / train_scale)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, train_scale: float) -> dict[str, float]:
    return {
        "skill_score": weighted_skill(y_true, y_pred, weight),
        "weighted_rmse": weighted_rmse(y_true, y_pred, weight),
        "weighted_mae": weighted_mae(y_true, y_pred, weight),
        "mase": mase(y_true, y_pred, train_scale),
    }


def exact_unique(values: pd.Series) -> list:
    return sorted(values.dropna().unique().tolist())


def approx_sample_parquet(path: Path, columns: list[str], sample_size: int, seed: int = SEED, batch_size: int = 50_000) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    frac = min(1.0, sample_size / max(total_rows, 1) * 1.35)
    rng = np.random.default_rng(seed)
    batches: list[pd.DataFrame] = []
    for batch_idx, batch in enumerate(parquet.iter_batches(batch_size=batch_size, columns=columns), start=1):
        df = batch.to_pandas()
        if df.empty:
            continue
        local_seed = int(rng.integers(0, 2**31 - 1))
        sampled = df.sample(frac=frac, random_state=local_seed)
        if not sampled.empty:
            batches.append(sampled)
    sample = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=columns)
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    return sample.reset_index(drop=True)


def stream_missingness(path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    names = parquet.schema.names
    counts = {name: 0 for name in names}
    total = 0
    for batch in parquet.iter_batches(batch_size=50_000):
        total += batch.num_rows
        for name, arr in zip(names, batch.columns):
            counts[name] += arr.null_count
    return (
        pd.DataFrame(
            {
                "column": names,
                "missing_count": [counts[name] for name in names],
                "missing_rate": [counts[name] / total for name in names],
            }
        )
        .sort_values("missing_rate", ascending=False)
        .reset_index(drop=True)
    )


def histogram_df(values: np.ndarray, bins: int | np.ndarray, *, label: str, panel_id: int = 1) -> pd.DataFrame:
    counts, edges = np.histogram(values, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"panel": label, "panel_id": panel_id, "x": mids, "count": counts})


def longform_matrix(df: pd.DataFrame, *, x_name: str = "x", y_name: str = "y", value_name: str = "value") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for y_idx, y_key in enumerate(df.index):
        for x_idx, x_key in enumerate(df.columns):
            rows.append(
                {
                    x_name: str(x_key),
                    y_name: str(y_key),
                    "x_idx": x_idx + 1,
                    "y_idx": y_idx + 1,
                    value_name: float(df.loc[y_key, x_key]),
                }
            )
    return pd.DataFrame(rows)


def safe_adf(values: pd.Series) -> tuple[float, float]:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(arr) < 20 or np.std(arr) == 0:
        return np.nan, np.nan
    try:
        stat, pvalue, *_ = adfuller(arr, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def safe_kpss(values: pd.Series) -> float:
    arr = pd.Series(values).dropna().to_numpy(dtype=float)
    if len(arr) < 20 or np.std(arr) == 0:
        return np.nan
    try:
        stat, pvalue, *_ = kpss(arr, regression="c", nlags="auto")
        return float(pvalue)
    except Exception:
        return np.nan


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_windows(values: np.ndarray, lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= lookback:
        return np.empty((0, lookback, 1), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xs, ys = [], []
    for start in range(len(values) - lookback):
        end = start + lookback
        xs.append(values[start:end])
        ys.append(values[end])
    return np.asarray(xs, dtype=np.float32).reshape(-1, lookback, 1), np.asarray(ys, dtype=np.float32)


def make_validation_windows(train_scaled: np.ndarray, val_scaled: np.ndarray, lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    train_scaled = np.asarray(train_scaled, dtype=np.float32)
    val_scaled = np.asarray(val_scaled, dtype=np.float32)
    context = np.concatenate([train_scaled[-lookback:], val_scaled])
    xs, ys = [], []
    for idx in range(len(val_scaled)):
        xs.append(context[idx : idx + lookback])
        ys.append(val_scaled[idx])
    return np.asarray(xs, dtype=np.float32).reshape(-1, lookback, 1), np.asarray(ys, dtype=np.float32)


class SequenceRegressor(nn.Module):
    def __init__(self, model_kind: str, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        kind = model_kind.upper()
        if kind == "RNN":
            self.recurrent = nn.RNN(1, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity="tanh")
        elif kind == "GRU":
            self.recurrent = nn.GRU(1, hidden_size, num_layers=num_layers, batch_first=True)
        elif kind == "LSTM":
            self.recurrent = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError(f"Unknown model kind: {model_kind}")
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.recurrent(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def recursive_forecast(model: SequenceRegressor, train_scaled: np.ndarray, steps: int, mean: float, std: float) -> np.ndarray:
    model.eval()
    context = list(np.asarray(train_scaled[-LOOKBACK:], dtype=np.float32))
    preds_scaled: list[float] = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(context[-LOOKBACK:], dtype=torch.float32, device=DEVICE).view(1, LOOKBACK, 1)
            pred_scaled = float(model(x).detach().cpu().numpy()[0])
            preds_scaled.append(pred_scaled)
            context.append(pred_scaled)
    preds_scaled = np.asarray(preds_scaled, dtype=float)
    return preds_scaled * std + mean


def train_sequence_model(y_train: np.ndarray, y_val: np.ndarray, model_kind: str, seed: int = SEED) -> tuple[np.ndarray, pd.DataFrame]:
    set_all_seeds(seed)
    y_train = np.asarray(y_train, dtype=float)
    y_val = np.asarray(y_val, dtype=float)
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not np.isfinite(std) or std == 0:
        std = 1.0

    train_scaled = (y_train - mean) / std
    val_scaled = (y_val - mean) / std
    x_train, target_train = make_windows(train_scaled, LOOKBACK)
    x_val, target_val = make_validation_windows(train_scaled, val_scaled, LOOKBACK)
    if len(x_train) == 0:
        raise ValueError(f"not enough training points for LOOKBACK={LOOKBACK}")

    train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(target_train))
    train_loader = DataLoader(
        train_ds,
        batch_size=min(BATCH_SIZE, len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    target_val_t = torch.tensor(target_val, dtype=torch.float32, device=DEVICE)

    model = SequenceRegressor(model_kind).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = np.inf
    bad_epochs = 0
    history_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(batch_losses))
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), target_val_t).detach().cpu().item())

        history_rows.append({"epoch": epoch, "model": model_kind, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred = recursive_forecast(model, train_scaled, len(y_val), mean, std)
    return pred, pd.DataFrame(history_rows)


def load_small_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [*SERIES_KEYS, "ts_index", "y_target", "weight"]
    train = pd.read_parquet(DATA_DIR / "train.parquet", columns=cols)
    test = pd.read_parquet(DATA_DIR / "test.parquet", columns=[*SERIES_KEYS, "ts_index"])
    train = train.sort_values(SERIES_KEYS + ["ts_index"]).reset_index(drop=True)
    test = test.sort_values(SERIES_KEYS + ["ts_index"]).reset_index(drop=True)
    return train, test


def build_slide_manifest(out_root: Path) -> None:
    text = (ROOT / "scripts" / "presentation_deck_v2.py").read_text()
    pattern = re.compile(r'fig = builder\.new_slide\(\n\s+"([^"]+)",\n\s+"([^"]+)",\n\s+"([^"]+)"', re.M)
    rows = []
    for idx, (title, section, takeaway) in enumerate(pattern.findall(text), start=1):
        rows.append({"slide_no": idx, "section": section, "title": title, "takeaway": takeaway})
    write_csv(pd.DataFrame(rows), out_root / "tables" / "slide_manifest.csv")


def export_overview_and_eda(out_root: Path, train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = DATA_DIR / "train.parquet"
    test_path = DATA_DIR / "test.parquet"
    feature_cols = [col for col in pq.ParquetFile(train_path).schema.names if col.startswith("feature_")]

    write_csv(deck.SCHEMA_DF.copy(), out_root / "tables" / "overview_schema.csv")

    comparability = []
    for col in ["code", "sub_code", "sub_category", "horizon"]:
        comparability.append(
            {
                "Column": col,
                "Train unique": int(train[col].nunique()),
                "Test unique": int(test[col].nunique()),
                "Test subset of train": "Yes" if set(test[col].unique()).issubset(set(train[col].unique())) else "No",
            }
        )
    write_csv(pd.DataFrame(comparability), out_root / "tables" / "overview_comparability.csv")

    counts = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": len(feature_cols),
        "codes": int(train["code"].nunique()),
        "sub_codes_train": int(train["sub_code"].nunique()),
        "sub_codes_test": int(test["sub_code"].nunique()),
        "sub_categories": int(train["sub_category"].nunique()),
        "horizons": exact_unique(train["horizon"]),
        "train_range": [int(train["ts_index"].min()), int(train["ts_index"].max())],
        "test_range": [int(test["ts_index"].min()), int(test["ts_index"].max())],
    }
    write_json(counts, out_root / "tables" / "overview_counts.json")

    missingness = stream_missingness(train_path)
    missing_top = missingness[missingness["missing_rate"] > 0].head(10).copy()
    missing_top["Missing rate"] = (missing_top["missing_rate"] * 100).map(lambda value: f"{value:.2f}%")
    missing_top = missing_top.rename(columns={"column": "Most missing columns"})[["Most missing columns", "Missing rate"]]
    write_csv(missing_top, out_root / "tables" / "eda_missing_top.csv")
    write_csv(
        missingness[missingness["missing_rate"] > 0].head(10)[["column", "missing_rate"]].rename(
            columns={"column": "label", "missing_rate": "value"}
        ),
        out_root / "curves" / "eda_missing_rates.csv",
    )

    y = train["y_target"].to_numpy(dtype=float)
    q01, q99 = np.quantile(y, [0.01, 0.99])
    target_hist = pd.concat(
        [
            histogram_df(y, bins=70, label="full", panel_id=1),
            histogram_df(y[(y >= q01) & (y <= q99)], bins=70, label="central_1_99", panel_id=2),
        ],
        ignore_index=True,
    )
    write_csv(target_hist, out_root / "curves" / "eda_target_histograms.csv")
    write_json(
        {
            "median": float(np.quantile(y, 0.5)),
            "iqr_low": float(np.quantile(y, 0.25)),
            "iqr_high": float(np.quantile(y, 0.75)),
            "p1": float(q01),
            "p99": float(q99),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
        },
        out_root / "tables" / "eda_target_stats.json",
    )

    w = train["weight"].to_numpy(dtype=float)
    w_sorted = np.sort(w)[::-1]
    cum_share = np.cumsum(w_sorted) / np.sum(w_sorted)
    pct = np.arange(1, len(w_sorted) + 1) / len(w_sorted) * 100
    dense_idx = np.linspace(0, len(w_sorted) - 1, num=min(400, len(w_sorted)), dtype=int)
    head_idx = np.geomspace(1, max(len(w_sorted), 1), num=min(250, len(w_sorted))).astype(int) - 1
    sample_idx = np.unique(np.clip(np.concatenate([dense_idx, head_idx, [len(w_sorted) - 1]]), 0, len(w_sorted) - 1))
    weight_curve = pd.DataFrame({"pct_rows": pct[sample_idx], "cumulative_share": cum_share[sample_idx]})
    weight_hist = histogram_df(np.log10(np.clip(w, 1e-12, None)), bins=70, label="log10_weight", panel_id=1)
    write_csv(weight_curve, out_root / "curves" / "eda_weight_cumulative_share.csv")
    write_csv(weight_hist, out_root / "curves" / "eda_weight_histogram.csv")
    write_json(
        {
            "top_1pct_share": float(cum_share[int(len(w_sorted) * 0.01) - 1]),
            "top_5pct_share": float(cum_share[int(len(w_sorted) * 0.05) - 1]),
            "top_10pct_share": float(cum_share[int(len(w_sorted) * 0.10) - 1]),
            "median_weight": float(np.median(w)),
            "max_weight": float(np.max(w)),
        },
        out_root / "tables" / "eda_weight_stats.json",
    )

    series_stats = (
        train.groupby(SERIES_KEYS)
        .agg(
            length=("ts_index", "size"),
            start=("ts_index", "min"),
            end=("ts_index", "max"),
            total_weight=("weight", "sum"),
            target_std=("y_target", "std"),
        )
        .reset_index()
    )
    series_stats["crosses_cutoff"] = (series_stats["start"] <= VAL_CUTOFF) & (series_stats["end"] > VAL_CUTOFF)
    eligible = series_stats[(series_stats["crosses_cutoff"]) & (series_stats["length"] >= 120)].copy()
    eligible["target_std"] = eligible["target_std"].fillna(0.0)
    stable_pool = eligible[eligible["target_std"] > 0].copy()
    chosen = (
        pd.concat(
            [
                eligible.nlargest(1, "length").assign(reason="longest history"),
                eligible.nlargest(1, "total_weight").assign(reason="highest total weight"),
                eligible.nlargest(1, "target_std").assign(reason="most volatile"),
                stable_pool.nsmallest(1, "target_std").assign(reason="most stable"),
            ],
            ignore_index=True,
        )
        .drop_duplicates(subset=SERIES_KEYS)
        .copy()
    )
    chosen["reason"] = pd.Categorical(chosen["reason"], categories=REASON_ORDER, ordered=True)
    chosen = chosen.sort_values("reason").reset_index(drop=True)
    chosen["series_label"] = chosen["reason"].map(title_case_reason)
    chosen["series_key"] = chosen.apply(fmt_series_key, axis=1)
    chosen["deep_series"] = chosen.apply(deep_series_id, axis=1)

    chosen_table = chosen[["series_label", "series_key", "length", "total_weight", "target_std"]].rename(
        columns={
            "series_label": "Reason",
            "series_key": "Series key",
            "length": "Length",
            "total_weight": "Total weight",
            "target_std": "Target std",
        }
    )
    write_csv(chosen_table, out_root / "tables" / "eda_chosen_series.csv")
    chosen_display = pd.DataFrame(
        {
            "Role": chosen["series_label"],
            "Series key": chosen["series_key"],
            "Len": chosen["length"].map(fmt_int),
            "Total wt": chosen["total_weight"].map(fmt_sig),
            "Std": chosen["target_std"].map(fmt_sig),
        }
    )
    write_csv(chosen_display, out_root / "tables" / "eda_chosen_series_display.csv")

    write_csv(
        pd.DataFrame(
            [
                {"metric": "Total series", "value": int(len(series_stats))},
                {"metric": "Cross cutoff", "value": int(series_stats["crosses_cutoff"].sum())},
                {"metric": "Eligible length>=120", "value": int(len(eligible))},
            ]
        ),
        out_root / "tables" / "eda_series_coverage_stats.csv",
    )
    coverage_hist = pd.concat(
        [
            histogram_df(series_stats["length"].to_numpy(dtype=float), bins=np.arange(0, 420, 10), label="all", panel_id=1),
            histogram_df(eligible["length"].to_numpy(dtype=float), bins=np.arange(0, 420, 10), label="eligible", panel_id=2),
        ],
        ignore_index=True,
    )
    write_csv(coverage_hist, out_root / "curves" / "eda_series_coverage_histogram.csv")

    feature_sample = approx_sample_parquet(train_path, [*feature_cols, "y_target"], sample_size=200_000, seed=SEED)
    corr_with_y = feature_sample[feature_cols].corrwith(feature_sample["y_target"]).sort_values(key=np.abs, ascending=False)
    top_corr = corr_with_y.head(20)
    write_csv(
        pd.DataFrame({"feature": top_corr.head(8).index, "correlation": top_corr.head(8).values}),
        out_root / "curves" / "eda_feature_correlations.csv",
    )
    feature_corr_table = (
        pd.DataFrame({"Top feature": top_corr.head(8).index, "Correlation": top_corr.head(8).values})
        .assign(Correlation=lambda df: df["Correlation"].map(lambda value: f"{value:.4f}"))
    )
    write_csv(feature_corr_table, out_root / "tables" / "eda_feature_corr_top.csv")

    top_features = top_corr.index.tolist()
    feature_corr_heatmap = feature_sample[top_features + ["y_target"]].corr()
    write_csv(longform_matrix(feature_corr_heatmap), out_root / "heatmaps" / "eda_feature_corr_heatmap.csv")

    series_grouped = train.groupby(SERIES_KEYS, sort=False)

    def get_series(row: pd.Series) -> pd.DataFrame:
        key = (row["code"], row["sub_code"], row["sub_category"], row["horizon"])
        return series_grouped.get_group(key)[["ts_index", "y_target", "weight"]].reset_index(drop=True)

    raw_rows: list[pd.DataFrame] = []
    rolling_rows: list[pd.DataFrame] = []
    diff_rolling_rows: list[pd.DataFrame] = []
    for panel_id, (_, row) in enumerate(chosen.iterrows(), start=1):
        series_df = get_series(row).copy()
        panel = row["series_label"]
        series_df["panel_id"] = panel_id
        series_df["panel"] = panel
        raw_rows.append(series_df[["panel_id", "panel", "ts_index", "y_target", "weight"]])

        rolling = pd.DataFrame(
            {
                "panel_id": panel_id,
                "panel": panel,
                "ts_index": series_df["ts_index"],
                "y": series_df["y_target"],
                "rolling_mean": series_df["y_target"].rolling(24).mean(),
                "rolling_std": series_df["y_target"].rolling(24).std(),
            }
        )
        rolling = rolling.dropna(subset=["rolling_mean", "rolling_std"]).reset_index(drop=True)
        rolling_rows.append(rolling)

        diff_y = series_df["y_target"].diff()
        diff_rolling = pd.DataFrame(
            {
                "panel_id": panel_id,
                "panel": panel,
                "ts_index": series_df["ts_index"],
                "diff_y": diff_y,
                "rolling_mean": diff_y.rolling(24).mean(),
                "rolling_std": diff_y.rolling(24).std(),
            }
        )
        diff_rolling = diff_rolling.dropna(subset=["rolling_mean", "rolling_std"]).reset_index(drop=True)
        diff_rolling_rows.append(diff_rolling)

    write_csv(pd.concat(raw_rows, ignore_index=True), out_root / "series" / "eda_chosen_series_raw.csv")
    write_csv(pd.concat(rolling_rows, ignore_index=True), out_root / "series" / "eda_chosen_series_rolling.csv")
    write_csv(pd.concat(diff_rolling_rows, ignore_index=True), out_root / "series" / "eda_chosen_series_diff_rolling.csv")

    sample_ids = eligible.sample(n=min(200, len(eligible)), random_state=SEED).reset_index(drop=True)
    adf_rows = []
    for _, row in sample_ids.iterrows():
        series_df = get_series(row)
        stat, pv = safe_adf(series_df["y_target"])
        adf_rows.append(
            {
                "code": row["code"],
                "sub_code": row["sub_code"],
                "sub_category": row["sub_category"],
                "horizon": int(row["horizon"]),
                "length": int(row["length"]),
                "adf_stat": stat,
                "adf_p": pv,
            }
        )
    adf_df = pd.DataFrame(adf_rows)
    adf_df["stationary_5pct"] = adf_df["adf_p"] < 0.05
    adf_df["kpss_p"] = [safe_kpss(get_series(row)["y_target"]) for _, row in sample_ids.iterrows()]
    adf_df["kpss_stationary_5pct"] = adf_df["kpss_p"] >= 0.05
    both = adf_df.dropna(subset=["adf_p", "kpss_p"]).copy()
    both["verdict"] = np.where(
        both["stationary_5pct"] & both["kpss_stationary_5pct"],
        "Stationary",
        np.where(~both["stationary_5pct"] & ~both["kpss_stationary_5pct"], "Unit root", "Inconclusive"),
    )
    verdict = (
        both["verdict"].value_counts(normalize=True).rename_axis("Verdict").reset_index(name="Share")
        .assign(Share=lambda df: df["Share"] * 100)
    )
    verdict["Interpretation"] = verdict["Verdict"].map(
        {
            "Stationary": "ADF rejects unit root and KPSS does not reject stationarity.",
            "Inconclusive": "Conflicting evidence; often trend-stationary or noisy.",
            "Unit root": "Fails ADF and rejects KPSS, so differencing is needed.",
        }
    )
    write_csv(verdict, out_root / "tables" / "eda_kpss_verdict.csv")

    diff_rows = []
    for _, row in sample_ids.iterrows():
        diff_series = get_series(row)["y_target"].diff().dropna()
        stat, pv = safe_adf(diff_series)
        diff_rows.append(pv)
    adf_df["adf_p_diff"] = diff_rows
    adf_df["stationary_after_diff"] = adf_df["adf_p_diff"] < 0.05

    write_csv(
        histogram_df(adf_df["adf_p"].dropna().to_numpy(dtype=float), bins=np.linspace(0, 1, 31), label="adf_p", panel_id=1),
        out_root / "curves" / "eda_adf_hist.csv",
    )
    write_csv(
        adf_df.groupby("horizon", as_index=False)["stationary_5pct"].mean().rename(columns={"stationary_5pct": "fraction_stationary"}),
        out_root / "curves" / "eda_adf_by_horizon.csv",
    )
    write_csv(
        adf_df.groupby("code", as_index=False)["stationary_5pct"].mean()
        .sort_values("stationary_5pct")
        .rename(columns={"stationary_5pct": "fraction_stationary"}),
        out_root / "curves" / "eda_adf_by_code.csv",
    )
    write_csv(
        pd.DataFrame(
            [
                {"Representation": "Levels (d=0)", "Stationary share": float(adf_df["stationary_5pct"].mean() * 100)},
                {"Representation": "First difference (d=1)", "Stationary share": float(adf_df["stationary_after_diff"].mean() * 100)},
            ]
        ),
        out_root / "curves" / "eda_stationarity_after_diff.csv",
    )
    write_json(
        {
            "sampled": int(len(adf_df)),
            "stationary_share": float(adf_df["stationary_5pct"].mean()),
            "after_diff_share": float(adf_df["stationary_after_diff"].mean()),
        },
        out_root / "tables" / "eda_adf_summary.json",
    )

    anchor_row = chosen[chosen["reason"] == "longest history"].iloc[0]
    anchor_series = get_series(anchor_row)["y_target"].reset_index(drop=True)
    max_lag = min(40, len(anchor_series) // 2 - 1)
    anchor_acf = acf(anchor_series, nlags=max_lag, fft=True)
    anchor_pacf = pacf(anchor_series, nlags=max_lag, method="ywm")
    conf = 1.96 / math.sqrt(len(anchor_series))
    write_csv(
        pd.concat(
            [
                pd.DataFrame({"kind": "ACF", "lag": np.arange(len(anchor_acf)), "value": anchor_acf, "conf": conf}),
                pd.DataFrame({"kind": "PACF", "lag": np.arange(len(anchor_pacf)), "value": anchor_pacf, "conf": conf}),
            ],
            ignore_index=True,
        ).assign(kind_id=lambda df: np.where(df["kind"].eq("ACF"), 1, 2)),
        out_root / "series" / "eda_anchor_acf_pacf.csv",
    )

    k = 30
    acf_mat, pacf_mat = [], []
    for _, row in sample_ids.iterrows():
        arr = get_series(row)["y_target"].dropna().to_numpy(dtype=float)
        if len(arr) < 2 * k + 5 or np.std(arr) == 0:
            continue
        try:
            acf_mat.append(acf(arr, nlags=k, fft=True))
            pacf_mat.append(pacf(arr, nlags=k, method="ywm"))
        except Exception:
            continue
    acf_mat = np.asarray(acf_mat)
    pacf_mat = np.asarray(pacf_mat)
    write_csv(
        pd.DataFrame(
            {
                "lag": np.arange(1, k + 1),
                "mean_abs_acf": np.nanmean(np.abs(acf_mat), axis=0)[1:],
                "mean_abs_pacf": np.nanmean(np.abs(pacf_mat), axis=0)[1:],
            }
        ),
        out_root / "curves" / "eda_panel_acf_pacf.csv",
    )

    period = max(2, min(24, len(anchor_series) // 8))
    stl = STL(anchor_series, period=period, robust=True).fit()
    stl_df = pd.DataFrame(
        {
            "position": np.arange(len(anchor_series)),
            "Observed": anchor_series,
            "Trend": stl.trend,
            "Seasonal": stl.seasonal,
            "Residual": stl.resid,
        }
    )
    write_csv(stl_df, out_root / "decompositions" / "eda_stl.csv")

    pair_candidates = []
    for (code_value, horizon_value), grp in train.groupby(["code", "horizon"]):
        if grp["sub_code"].nunique() < 2:
            continue
        pivot = grp.pivot_table(index="ts_index", columns="sub_code", values="y_target", aggfunc="mean")
        cols = list(pivot.columns)
        best = (0, None, None)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                overlap = pivot[[cols[i], cols[j]]].dropna().shape[0]
                if overlap > best[0]:
                    best = (overlap, cols[i], cols[j])
        if best[0] > 50:
            pair_candidates.append((code_value, int(horizon_value), pivot, best[1], best[2]))
        if len(pair_candidates) == 3:
            break
    lagcorr_rows = []
    for panel_id, (code_value, horizon_value, pivot, c1, c2) in enumerate(pair_candidates, start=1):
        panel = f"{code_value} / H{horizon_value}: {c1} vs {c2}"
        for lag in range(-10, 11):
            aligned = pd.concat([pivot[c1], pivot[c2].shift(lag)], axis=1).dropna()
            if len(aligned) > 3 and aligned.iloc[:, 0].nunique() > 1 and aligned.iloc[:, 1].nunique() > 1:
                corr = aligned.corr().iloc[0, 1]
            else:
                corr = np.nan
            lagcorr_rows.append({"panel_id": panel_id, "panel": panel, "lag": lag, "correlation": corr})
    write_csv(pd.DataFrame(lagcorr_rows), out_root / "curves" / "eda_cross_series_lagcorr.csv")

    split_check = pd.DataFrame(
        {
            "Partition": ["train_pre_cutoff", "train_post_cutoff", "test"],
            "Rows": [int((train["ts_index"] <= VAL_CUTOFF).sum()), int((train["ts_index"] > VAL_CUTOFF).sum()), int(len(test))],
        }
    )
    write_csv(split_check, out_root / "tables" / "eda_leakage_partition.csv")
    write_csv(adf_df, out_root / "tables" / "eda_adf_raw.csv")

    return series_grouped, chosen, series_stats


def export_classical(out_root: Path, series_grouped: pd.core.groupby.DataFrameGroupBy, chosen: pd.DataFrame) -> None:
    def get_series(row: pd.Series) -> pd.DataFrame:
        key = (row["code"], row["sub_code"], row["sub_category"], row["horizon"])
        return series_grouped.get_group(key)[["ts_index", "y_target", "weight"]].reset_index(drop=True)

    def split_series(series_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return series_df[series_df["ts_index"] <= VAL_CUTOFF].copy(), series_df[series_df["ts_index"] > VAL_CUTOFF].copy()

    def forecast_rolling_mean(values: pd.Series, steps: int, window: int = 10) -> np.ndarray:
        width = min(window, len(values))
        return np.repeat(float(values.iloc[-width:].mean()), steps)

    def forecast_ses(values: pd.Series, steps: int) -> np.ndarray:
        fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
        return np.asarray(fit.forecast(steps), dtype=float)

    def forecast_holt(values: pd.Series, steps: int) -> np.ndarray:
        fit = Holt(values, initialization_method="estimated").fit(optimized=True)
        return np.asarray(fit.forecast(steps), dtype=float)

    def make_arima_forecaster(order: tuple[int, int, int]):
        def _forecast(values: pd.Series, steps: int) -> np.ndarray:
            fit = ARIMA(values, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
            pred = np.asarray(fit.forecast(steps), dtype=float)
            if not np.all(np.isfinite(pred)):
                raise ValueError("non_finite_forecast")
            return pred

        return _forecast

    model_specs: dict[str, dict[str, object]] = {}

    def register_model(name: str, family: str, min_train: int, func, p=np.nan, d=np.nan, q=np.nan) -> None:
        model_specs[name] = {"family": family, "min_train": min_train, "func": func, "p": p, "d": d, "q": q}

    register_model("Rolling Mean (w=10)", "Smoothing", 2, lambda values, steps: forecast_rolling_mean(values, steps, 10))
    register_model("SES", "Smoothing", 3, forecast_ses)
    register_model("Holt", "Smoothing", 5, forecast_holt)
    for p in range(1, 4):
        register_model(f"AR({p})", "AR", 20, make_arima_forecaster((p, 0, 0)), p=p, d=0, q=0)
    for q in range(1, 4):
        register_model(f"MA({q})", "MA", 20, make_arima_forecaster((0, 0, q)), p=0, d=0, q=q)
    for p in range(1, 4):
        for q in range(1, 4):
            register_model(f"ARMA({p},{q})", "ARMA", 20, make_arima_forecaster((p, 0, q)), p=p, d=0, q=q)
    for name, order in {"ARIMA(0,1,1)": (0, 1, 1), "ARIMA(1,1,0)": (1, 1, 0), "ARIMA(1,1,1)": (1, 1, 1)}.items():
        register_model(name, "ARIMA", 20, make_arima_forecaster(order), p=order[0], d=order[1], q=order[2])

    results_rows = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    model_meta = pd.DataFrame(
        [
            {"model": name, "family": spec["family"], "p": spec["p"], "d": spec["d"], "q": spec["q"], "min_train": spec["min_train"]}
            for name, spec in model_specs.items()
        ]
    )

    setup_full_rows = []
    setup_zoom_rows = []
    for panel_id, (_, row) in enumerate(chosen.iterrows(), start=1):
        series_df = get_series(row)
        train_part, val_part = split_series(series_df)
        y_train = train_part["y_target"].reset_index(drop=True)
        y_val = val_part["y_target"].to_numpy(dtype=float)
        w_val = val_part["weight"].to_numpy(dtype=float)
        steps = len(val_part)
        train_scale = float(np.mean(np.abs(np.diff(y_train)))) if len(y_train) > 1 else np.nan
        series_label = row["series_label"]

        full_df = series_df.assign(
            panel_id=panel_id,
            panel=series_label,
            partition=np.where(series_df["ts_index"] <= VAL_CUTOFF, "train", "validation"),
        )
        full_df["partition_id"] = np.where(full_df["partition"].eq("train"), 1, 2)
        setup_full_rows.append(full_df[["panel_id", "panel", "ts_index", "y_target", "partition", "partition_id"]])
        zoom_start = max(int(series_df["ts_index"].min()), VAL_CUTOFF - 40)
        zoom_end = min(int(series_df["ts_index"].max()), VAL_CUTOFF + 40)
        zoom_df = full_df[(full_df["ts_index"] >= zoom_start) & (full_df["ts_index"] <= zoom_end)].copy()
        setup_zoom_rows.append(zoom_df[["panel_id", "panel", "ts_index", "y_target", "partition", "partition_id"]])

        for model_name, spec in model_specs.items():
            result = {
                "series_label": series_label,
                "reason": row["reason"],
                "model": model_name,
                "family": spec["family"],
                "train_len": len(y_train),
                "val_len": steps,
                "status": "ok",
                "skip_reason": None,
                "skill_score": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "mase": np.nan,
                "p": spec["p"],
                "d": spec["d"],
                "q": spec["q"],
            }
            if len(y_train) < spec["min_train"]:
                result["status"] = "skipped"
                result["skip_reason"] = f"train_len<{spec['min_train']}"
                results_rows.append(result)
                continue
            try:
                pred = np.asarray(spec["func"](y_train, steps), dtype=float)
                if len(pred) != steps:
                    raise ValueError("forecast_length_mismatch")
                if not np.all(np.isfinite(pred)):
                    raise ValueError("non_finite_forecast")
                result["skill_score"] = weighted_skill(y_val, pred, w_val)
                result["rmse"] = weighted_rmse(y_val, pred, w_val)
                result["mae"] = weighted_mae(y_val, pred, w_val)
                result["mase"] = mase(y_val, pred, train_scale)
                predictions[(series_label, model_name)] = pred
            except Exception as exc:
                result["status"] = "failed"
                result["skip_reason"] = type(exc).__name__
            results_rows.append(result)

    results_df = pd.DataFrame(results_rows)
    ok_results = results_df[results_df["status"] == "ok"].copy()
    ok_results["reason"] = pd.Categorical(ok_results["reason"], categories=REASON_ORDER, ordered=True)
    ok_results = ok_results.sort_values(["reason", "skill_score", "rmse"], ascending=[True, False, True])

    per_series_leaderboard = ok_results[["series_label", "family", "model", "skill_score", "rmse", "mae", "mase"]].sort_values(
        ["series_label", "skill_score", "rmse"], ascending=[True, False, True]
    )
    overall_summary = (
        ok_results.groupby(["family", "model"], as_index=False)
        .agg(mean_skill=("skill_score", "mean"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_mase=("mase", "mean"), n_ok=("model", "size"))
        .sort_values(["mean_rmse", "mean_mae", "mean_skill"], ascending=[True, True, False])
    )
    top6_by_skill = overall_summary.sort_values(["mean_skill", "mean_rmse"], ascending=[False, True]).head(6).reset_index(drop=True)
    overall_summary["n_attempted"] = overall_summary["model"].map(results_df.groupby("model").size())
    overall_summary["ok_rate"] = overall_summary["n_ok"] / overall_summary["n_attempted"]

    best_by_series = (
        ok_results.sort_values(["reason", "skill_score", "rmse"], ascending=[True, False, True])
        .groupby("series_label", as_index=False)
        .first()[["series_label", "family", "model", "skill_score", "rmse", "mae", "mase", "train_len", "val_len"]]
    )

    write_csv(model_meta, out_root / "tables" / "classical_model_meta.csv")
    write_csv(results_df, out_root / "tables" / "classical_results_raw.csv")
    write_csv(per_series_leaderboard, out_root / "tables" / "classical_per_series_leaderboard.csv")
    write_csv(overall_summary, out_root / "tables" / "classical_summary.csv")
    write_csv(top6_by_skill, out_root / "tables" / "classical_summary_top6.csv")
    write_csv(best_by_series, out_root / "tables" / "classical_best_by_series.csv")
    classical_best_display = pd.DataFrame(
        {
            "Series": best_by_series["series_label"],
            "Family": best_by_series["family"],
            "Best model": best_by_series["model"],
            "Skill": best_by_series["skill_score"].map(fmt_sig),
            "RMSE": best_by_series["rmse"].map(fmt_sig),
            "Train": best_by_series["train_len"].map(fmt_int),
            "Val": best_by_series["val_len"].map(fmt_int),
        }
    )
    write_csv(classical_best_display, out_root / "tables" / "classical_best_by_series_display.csv")
    classical_lineup_display = deck.CLASSICAL_LINEUP_DF.rename(
        columns={"Models": "Examples", "Min train": "Min train", "Intuition": "Role"}
    )[["Family", "Examples", "Min train", "Role"]]
    write_csv(classical_lineup_display, out_root / "tables" / "classical_lineup_display.csv")
    setup_full_df = pd.concat(setup_full_rows, ignore_index=True)
    setup_zoom_df = pd.concat(setup_zoom_rows, ignore_index=True)
    write_csv(setup_full_df, out_root / "series" / "classical_setup_full.csv")
    write_csv(setup_zoom_df, out_root / "series" / "classical_setup_zoom.csv")
    write_panel_csvs(setup_full_df, out_root / "series" / "classical_setup_full.csv")
    write_panel_csvs(setup_zoom_df, out_root / "series" / "classical_setup_zoom.csv")
    write_csv(deck.CLASSICAL_LINEUP_DF.copy(), out_root / "tables" / "classical_lineup.csv")

    rank_df = ok_results.copy()
    rank_df["rmse_rank"] = rank_df.groupby("series_label")["rmse"].rank(method="dense", ascending=True)
    model_order = overall_summary["model"].tolist()
    rank_pivot = rank_df.pivot(index="series_label", columns="model", values="rmse_rank").reindex(chosen["series_label"]).reindex(columns=model_order)
    write_csv(longform_matrix(rank_pivot.fillna(22.0)), out_root / "heatmaps" / "classical_rank_heatmap.csv")
    rank_focus = (
        rank_df.pivot(index="series_label", columns="model", values="rmse_rank")
        .reindex(chosen["series_label"])
        .reindex(columns=FOCUS_MODELS)
        .fillna(22.0)
    )
    write_csv(longform_matrix(rank_focus), out_root / "heatmaps" / "classical_rank_heatmap_focus.csv")

    status_summary = (
        results_df.groupby(["model", "status"]).size().unstack(fill_value=0).reindex(model_meta["model"], fill_value=0).reset_index().rename(columns={"index": "model"})
    )
    write_csv(status_summary, out_root / "curves" / "classical_fit_status.csv")
    family_status = (
        results_df.groupby(["family", "status"]).size().unstack(fill_value=0).reindex(FAMILY_ORDER, fill_value=0).reset_index()
    )
    write_csv(family_status, out_root / "curves" / "classical_fit_status_family.csv")

    ar_summary = (
        ok_results[ok_results["family"] == "AR"]
        .groupby("p", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_skill=("skill_score", "mean"))
    )
    ma_summary = (
        ok_results[ok_results["family"] == "MA"]
        .groupby("q", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_skill=("skill_score", "mean"))
    )
    arma_rmse = ok_results[ok_results["family"] == "ARMA"].groupby(["p", "q"])["rmse"].mean().unstack("q")
    write_csv(ar_summary, out_root / "curves" / "classical_ar_order_rmse.csv")
    write_csv(ma_summary, out_root / "curves" / "classical_ma_order_rmse.csv")
    write_csv(longform_matrix(arma_rmse), out_root / "heatmaps" / "classical_arma_rmse.csv")

    def select_unique_forecasts(series_label: str, top_k: int = 3, atol: float = 1e-8, rtol: float = 1e-6) -> list[tuple[pd.Series, np.ndarray]]:
        ranked = ok_results[ok_results["series_label"] == series_label].sort_values(["skill_score", "rmse"], ascending=[False, True])
        selected: list[tuple[pd.Series, np.ndarray]] = []
        seen_preds: list[np.ndarray] = []
        for _, model_row in ranked.iterrows():
            pred = predictions.get((series_label, model_row["model"]))
            if pred is None:
                continue
            if any(np.allclose(pred, prev, atol=atol, rtol=rtol) for prev in seen_preds):
                continue
            selected.append((model_row, pred))
            seen_preds.append(pred)
            if len(selected) == top_k:
                break
        return selected

    overlay_full_rows = []
    overlay_zoom_rows = []
    error_rows = []
    for panel_id, (_, row) in enumerate(chosen.iterrows(), start=1):
        series_df = get_series(row)
        train_part, val_part = split_series(series_df)
        series_label = row["series_label"]
        selected = select_unique_forecasts(series_label, top_k=3)

        for store, context_points in [
            (overlay_full_rows, None),
            (overlay_zoom_rows, 12),
        ]:
            train_plot = train_part if context_points is None else train_part.tail(context_points)
            line_id = 1
            for _, train_row in train_plot.iterrows():
                store.append(
                    {
                        "panel_id": panel_id,
                        "panel": series_label,
                        "ts_index": train_row["ts_index"],
                        "line_id": line_id,
                        "line_name": "Train",
                        "kind": "train",
                        "value": train_row["y_target"],
                    }
                )
            line_id += 1
            for _, val_row in val_part.iterrows():
                store.append(
                    {
                        "panel_id": panel_id,
                        "panel": series_label,
                        "ts_index": val_row["ts_index"],
                        "line_id": line_id,
                        "line_name": "Validation truth",
                        "kind": "validation",
                        "value": val_row["y_target"],
                    }
                )
            line_id += 1
            for model_row, pred in selected:
                for ts_index, value in zip(val_part["ts_index"], pred):
                    store.append(
                        {
                            "panel_id": panel_id,
                            "panel": series_label,
                            "ts_index": ts_index,
                            "line_id": line_id,
                            "line_name": f"{model_row['model']} (skill={model_row['skill_score']:.3f})",
                            "kind": "forecast",
                            "value": value,
                        }
                    )
                line_id += 1

        best_row = best_by_series[best_by_series["series_label"] == series_label].iloc[0]
        best_pred = predictions[(series_label, best_row["model"])]
        abs_error = np.abs(val_part["y_target"].to_numpy(dtype=float) - best_pred)
        for ts_index, value in zip(val_part["ts_index"], abs_error):
            error_rows.append(
                {
                    "panel_id": panel_id,
                    "panel": series_label,
                    "ts_index": ts_index,
                    "abs_error": value,
                    "mean_abs_error": float(abs_error.mean()),
                    "best_model": best_row["model"],
                }
            )

    overlay_full_df = pd.DataFrame(overlay_full_rows)
    overlay_zoom_df = pd.DataFrame(overlay_zoom_rows)
    error_df = pd.DataFrame(error_rows)
    write_csv(overlay_full_df, out_root / "series" / "classical_forecast_overlay_full.csv")
    write_csv(overlay_zoom_df, out_root / "series" / "classical_forecast_overlay_zoom.csv")
    write_csv(error_df, out_root / "series" / "classical_abs_error.csv")
    write_panel_csvs(overlay_full_df, out_root / "series" / "classical_forecast_overlay_full.csv")
    write_panel_csvs(overlay_zoom_df, out_root / "series" / "classical_forecast_overlay_zoom.csv")
    write_panel_csvs(error_df, out_root / "series" / "classical_abs_error.csv")
    write_csv(best_by_series[["series_label", "rmse"]].rename(columns={"series_label": "Series", "rmse": "RMSE"}), out_root / "curves" / "classical_best_rmse.csv")


def export_deep(out_root: Path, train: pd.DataFrame, chosen: pd.DataFrame) -> None:
    set_all_seeds(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    def get_series_df(meta_row: pd.Series) -> pd.DataFrame:
        mask = np.ones(len(train), dtype=bool)
        for key in SERIES_KEYS:
            mask &= train[key].eq(meta_row[key]).to_numpy()
        return train.loc[mask, ["ts_index", "y_target", "weight"]].sort_values("ts_index").reset_index(drop=True)

    def adaptive_split(series_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(np.floor(len(series_df) * TRAIN_FRACTION))
        split_idx = max(split_idx, LOOKBACK + 1)
        split_idx = min(split_idx, len(series_df) - 1)
        return series_df.iloc[:split_idx].copy(), series_df.iloc[split_idx:].copy()

    versions = {}
    for module_name in ["pandas", "pyarrow", "numpy", "matplotlib", "sklearn", "torch"]:
        module = __import__(module_name)
        versions[module_name] = getattr(module, "__version__", "ok")
    write_csv(pd.DataFrame({"module": list(versions), "version": list(versions.values())}), out_root / "tables" / "deep_environment.csv")
    write_csv(
        pd.DataFrame(
            {
                "Module": ["pandas", "pyarrow", "numpy", "matplotlib", "sklearn", "torch"],
                "Version": [versions[name] for name in ["pandas", "pyarrow", "numpy", "matplotlib", "sklearn", "torch"]],
            }
        ),
        out_root / "tables" / "deep_environment_display.csv",
    )

    split_rows = []
    results_rows = []
    forecast_rows = []
    history_rows = []
    forecast_store: dict[tuple[str, str], np.ndarray] = {}

    for panel_id, (_, meta_row) in enumerate(chosen.iterrows(), start=1):
        series_df = get_series_df(meta_row)
        train_part, val_part = adaptive_split(series_df)
        series_label = meta_row["series_label"]
        series_id = meta_row["deep_series"]
        split_rows.append(
            {
                "Series": series_label,
                "Train len": len(train_part),
                "Val len": len(val_part),
                "Train range": f"{int(train_part['ts_index'].min())}–{int(train_part['ts_index'].max())}",
                "Validation range": f"{int(val_part['ts_index'].min())}–{int(val_part['ts_index'].max())}",
            }
        )

        y_train = train_part["y_target"].to_numpy(dtype=float)
        y_val = val_part["y_target"].to_numpy(dtype=float)
        train_scale = float(np.mean(np.abs(np.diff(y_train)))) if len(y_train) > 1 else np.nan
        naive_pred = np.repeat(float(y_train[-1]), len(y_val))
        forecast_store[(series_label, "Naive")] = naive_pred
        row = {
            "series": series_label,
            "reason": meta_row["reason"],
            "model": "Naive",
            "train_len": len(train_part),
            "val_len": len(val_part),
            "status": "ok",
            "error": None,
            **metric_dict(y_val, naive_pred, val_part["weight"].to_numpy(dtype=float), train_scale),
        }
        results_rows.append(row)

        for model_kind in ARCHITECTURES:
            try:
                pred, history = train_sequence_model(y_train, y_val, model_kind, seed=SEED)
                forecast_store[(series_label, model_kind)] = pred
                metrics = metric_dict(y_val, pred, val_part["weight"].to_numpy(dtype=float), train_scale)
                results_rows.append(
                    {
                        "series": series_label,
                        "reason": meta_row["reason"],
                        "model": model_kind,
                        "train_len": len(train_part),
                        "val_len": len(val_part),
                        "status": "ok",
                        "error": None,
                        **metrics,
                    }
                )
                history = history.copy()
                history["panel_id"] = panel_id
                history["series"] = series_label
                history["series_id"] = series_id
                history["reason"] = meta_row["reason"]
                history["curve_id"] = np.where(history["model"].eq(model_kind), 0, 0)
                history_rows.append(history)
            except Exception as exc:
                results_rows.append(
                    {
                        "series": series_label,
                        "reason": meta_row["reason"],
                        "model": model_kind,
                        "train_len": len(train_part),
                        "val_len": len(val_part),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "skill_score": np.nan,
                        "weighted_rmse": np.nan,
                        "weighted_mae": np.nan,
                        "mase": np.nan,
                    }
                )

        for _, point in train_part.iterrows():
            forecast_rows.append(
                {
                    "panel_id": panel_id,
                    "panel": series_label,
                    "ts_index": point["ts_index"],
                    "line_id": 1,
                    "line_name": "Train",
                    "kind": "train",
                    "value": point["y_target"],
                }
            )
        for _, point in val_part.iterrows():
            forecast_rows.append(
                {
                    "panel_id": panel_id,
                    "panel": series_label,
                    "ts_index": point["ts_index"],
                    "line_id": 2,
                    "line_name": "Validation truth",
                    "kind": "validation",
                    "value": point["y_target"],
                }
            )
        for line_id, model_name in enumerate(["Naive", *ARCHITECTURES], start=3):
            pred = forecast_store.get((series_label, model_name))
            if pred is None:
                continue
            for ts_index, value in zip(val_part["ts_index"], pred):
                forecast_rows.append(
                    {
                        "panel_id": panel_id,
                        "panel": series_label,
                        "ts_index": ts_index,
                        "line_id": line_id,
                        "line_name": f"{model_name} forecast",
                        "kind": "forecast",
                        "value": value,
                    }
                )

    results = pd.DataFrame(results_rows)
    ok = results[results["status"] == "ok"].copy()
    deep_only_ok = ok[ok["model"].isin(ARCHITECTURES)].copy()
    leaderboard = (
        ok.groupby("model", as_index=False)
        .agg(
            n_series=("series", "nunique"),
            mean_skill=("skill_score", "mean"),
            median_skill=("skill_score", "median"),
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_mae=("weighted_mae", "mean"),
            mean_mase=("mase", "mean"),
        )
        .sort_values(["mean_skill", "median_skill"], ascending=False)
    )
    winners = (
        deep_only_ok.loc[
            deep_only_ok.groupby("series")["skill_score"].idxmax(),
            ["series", "reason", "model", "skill_score", "weighted_rmse", "weighted_mae", "mase"],
        ]
        .sort_values("reason")
        .reset_index(drop=True)
    )

    write_csv(pd.DataFrame(split_rows), out_root / "tables" / "deep_split_summary.csv")
    write_csv(results, out_root / "tables" / "deep_results_raw.csv")
    write_csv(leaderboard, out_root / "tables" / "deep_leaderboard.csv")
    write_csv(winners, out_root / "tables" / "deep_winners.csv")
    write_csv(
        pd.DataFrame(split_rows).rename(columns={"Train len": "Train", "Val len": "Val", "Validation range": "Val range"}),
        out_root / "tables" / "deep_split_summary_display.csv",
    )
    deep_leaderboard_display = pd.DataFrame(
        {
            "Model": leaderboard["model"],
            "Mean skill": leaderboard["mean_skill"].map(fmt_sig),
            "Mean RMSE": leaderboard["mean_weighted_rmse"].map(fmt_sig),
            "Series": leaderboard["n_series"].map(fmt_int),
        }
    )
    write_csv(deep_leaderboard_display, out_root / "tables" / "deep_leaderboard_display.csv")
    deep_winners_display = pd.DataFrame(
        {
            "Series": winners["series"],
            "Winner": winners["model"],
            "Skill": winners["skill_score"].map(fmt_sig),
        }
    )
    write_csv(deep_winners_display, out_root / "tables" / "deep_winners_display.csv")
    forecast_df = pd.DataFrame(forecast_rows)
    write_csv(forecast_df, out_root / "series" / "deep_forecasts.csv")
    write_panel_csvs(forecast_df, out_root / "series" / "deep_forecasts.csv")
    best_deep_rows = []
    for panel_id, (_, meta_row) in enumerate(chosen.iterrows(), start=1):
        series_label = meta_row["series_label"]
        panel_forecast = forecast_df[forecast_df["panel"] == series_label].copy()
        winner_row = winners[winners["series"] == series_label].iloc[0]
        best_model = winner_row["model"]
        best_map = {"Naive": 3, "RNN": 4, "GRU": 5, "LSTM": 6}
        keep_ids = [1, 2, 3, best_map[best_model]]
        temp = panel_forecast[panel_forecast["line_id"].isin(keep_ids)].copy()
        line_map = {
            1: (1, "Train"),
            2: (2, "Validation truth"),
            3: (3, "Naive forecast"),
            best_map[best_model]: (4, f"{best_model} forecast"),
        }
        temp["line_id"] = temp["line_id"].map(lambda value: line_map[value][0])
        temp["line_name"] = temp["line_name"].map(
            lambda value: "Train" if value == "Train" else (
                "Validation truth" if value == "Validation truth" else (
                    "Naive forecast" if value == "Naive forecast" else f"{best_model} forecast"
                )
            )
        )
        temp["panel_id"] = panel_id
        best_deep_rows.append(temp)
    best_deep_df = pd.concat(best_deep_rows, ignore_index=True)
    write_csv(best_deep_df, out_root / "series" / "deep_best_forecasts.csv")
    write_panel_csvs(best_deep_df, out_root / "series" / "deep_best_forecasts.csv")
    if history_rows:
        history_df = pd.concat(history_rows, ignore_index=True)
        history_df["curve_label"] = history_df["model"] + " " + np.where(history_df["train_loss"].notna(), "train", "train")
        curve_rows = []
        for (panel_id, series, model_name), grp in history_df.groupby(["panel_id", "series", "model"], sort=False):
            for line_id, (metric_name, label_suffix) in enumerate([("train_loss", "train"), ("val_loss", "val")], start=1):
                temp = grp[["panel_id", "series", "series_id", "reason", "epoch", metric_name]].copy()
                temp = temp.rename(columns={metric_name: "value"})
                temp["line_id"] = (ARCHITECTURES.index(model_name) + 1) * 10 + line_id
                temp["line_name"] = f"{model_name} {label_suffix}"
                temp["metric"] = metric_name
                curve_rows.append(temp)
        history_export = pd.concat(curve_rows, ignore_index=True)
    else:
        history_export = pd.DataFrame(columns=["panel_id", "series", "series_id", "reason", "epoch", "value", "line_id", "line_name", "metric"])
    write_csv(history_export, out_root / "series" / "deep_training_curves.csv")
    write_panel_csvs(history_export, out_root / "series" / "deep_training_curves.csv")

    final_compare = (
        pd.DataFrame(
            {
                "Series": chosen["series_label"],
                "Best classical": np.nan,
                "Best deep": np.nan,
            }
        )
        .set_index("Series")
        .reset_index()
    )
    classical_best = pd.read_csv(out_root / "tables" / "classical_best_by_series.csv", sep=";")
    for _, row in classical_best.iterrows():
        final_compare.loc[final_compare["Series"] == row["series_label"], "Best classical"] = row["skill_score"]
    for _, row in winners.iterrows():
        final_compare.loc[final_compare["Series"] == row["series"], "Best deep"] = row["skill_score"]
    write_csv(final_compare, out_root / "tables" / "final_compare.csv")
    final_compare_display = final_compare.copy()
    final_compare_display["Classical"] = final_compare_display["Best classical"].map(fmt_sig)
    final_compare_display["Deep"] = final_compare_display["Best deep"].map(fmt_sig)
    final_compare_display["Edge"] = np.where(
        final_compare["Best deep"].fillna(-np.inf) > final_compare["Best classical"].fillna(-np.inf),
        "Deep",
        np.where(
            final_compare["Best deep"].fillna(-np.inf) < final_compare["Best classical"].fillna(-np.inf),
            "Classical",
            "Tie",
        ),
    )
    write_csv(final_compare_display[["Series", "Classical", "Deep", "Edge"]], out_root / "tables" / "final_compare_display.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structured numeric data for the Beamer deck.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Destination directory for exported CSV/JSON files.")
    args = parser.parse_args()

    out_root: Path = args.out
    for name in ["tables", "curves", "series", "heatmaps", "decompositions"]:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    build_slide_manifest(out_root)
    train, test = load_small_frames()
    series_grouped, chosen, _ = export_overview_and_eda(out_root, train, test)
    export_classical(out_root, series_grouped, chosen)
    export_deep(out_root, train, chosen)
    write_json({"status": "ok"}, out_root / "tables" / "export_manifest.json")
    print(out_root)


if __name__ == "__main__":
    sys.exit(main())
