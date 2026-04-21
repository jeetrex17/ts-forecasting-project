from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


SERIES_KEYS = ["code", "sub_code", "sub_category", "horizon"]
LEGACY_VAL_CUTOFF = 2880
SEED = 42
MIN_SERIES_LENGTH = 120
TRAIN_FRAC = 0.80
MIN_TRAIN_POINTS = 96
MIN_VAL_POINTS = 24
BENCHMARK_PER_HORIZON = 2
STATIONARITY_SAMPLE_SIZE = 200
LAG_SAMPLE_SIZE = 120
MAX_LAG = 20
LOOKBACK = 24

REQUIRED_ARTIFACTS = [
    "overview.json",
    "series_manifest.parquet",
    "chosen_manifest.parquet",
    "benchmark_manifest.parquet",
    "study_series.parquet",
    "target_hist_full.parquet",
    "target_hist_clip.parquet",
    "weight_hist_clip.parquet",
    "weight_cumshare.parquet",
    "stationarity_sample.parquet",
    "lag_summary.parquet",
]


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists():
        return cwd
    if (cwd.parent / "data").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate project root containing data/.")


ROOT = project_root()
DATA_DIR = ROOT / "data" / "ts-forecasting"
PROCESSED_DIR = ROOT / "data" / "processed"
CODEX_DIR = PROCESSED_DIR / "codex"
FIG_DIR = ROOT / "presentation" / "beamer" / "figures" / "codex"
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"


@dataclass(frozen=True)
class SplitInfo:
    train_end: int
    train_len: int
    val_len: int


def ensure_dirs() -> None:
    CODEX_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def series_id_from_frame(df: pd.DataFrame) -> pd.Series:
    return (
        df["code"].astype(str)
        + "__"
        + df["sub_code"].astype(str)
        + "__"
        + df["sub_category"].astype(str)
        + "__H"
        + df["horizon"].astype(int).astype(str)
    )


def make_series_id(row: pd.Series | dict) -> str:
    return (
        f"{row['code']}__{row['sub_code']}__{row['sub_category']}"
        f"__H{int(row['horizon'])}"
    )


def split_info(length: int) -> SplitInfo:
    train_end = max(int(np.floor(length * TRAIN_FRAC)), MIN_TRAIN_POINTS)
    train_end = min(train_end, length - MIN_VAL_POINTS)
    if train_end < MIN_TRAIN_POINTS or length - train_end < MIN_VAL_POINTS:
        raise ValueError(f"Series of length {length} does not satisfy codex split rules.")
    return SplitInfo(train_end=train_end, train_len=train_end, val_len=length - train_end)


def chronological_split_df(series_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    ordered = series_df.sort_values("ts_index").reset_index(drop=True)
    info = split_info(len(ordered))
    train_part = ordered.iloc[: info.train_end].copy()
    val_part = ordered.iloc[info.train_end :].copy()
    return train_part, val_part, info


def weighted_skill(y_true: Iterable[float], y_pred: Iterable[float], weight: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    weight = np.asarray(list(weight), dtype=float)
    denom = np.sum(weight * np.square(y_true))
    if denom <= 0:
        return 0.0
    ratio = np.sum(weight * np.square(y_true - y_pred)) / denom
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return float(np.sqrt(1.0 - ratio))


def weighted_rmse(y_true: Iterable[float], y_pred: Iterable[float], weight: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    weight = np.asarray(list(weight), dtype=float)
    return float(np.sqrt(np.average(np.square(y_true - y_pred), weights=weight)))


def weighted_mae(y_true: Iterable[float], y_pred: Iterable[float], weight: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    weight = np.asarray(list(weight), dtype=float)
    return float(np.average(np.abs(y_true - y_pred), weights=weight))


def mase(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    insample: Iterable[float],
) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    insample = np.asarray(list(insample), dtype=float)
    if len(insample) < 2:
        return float("nan")
    scale = np.mean(np.abs(np.diff(insample)))
    if scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def export_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_dirs()
    out = FIG_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    return out


def load_json_artifact(name: str) -> dict:
    return json.loads((CODEX_DIR / name).read_text())


def save_json_artifact(name: str, payload: dict) -> Path:
    ensure_dirs()
    out = CODEX_DIR / name
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def load_artifact(name: str) -> pd.DataFrame:
    return pd.read_parquet(CODEX_DIR / name)


def artifacts_ready() -> bool:
    return all((CODEX_DIR / name).exists() for name in REQUIRED_ARTIFACTS)


def load_study_series() -> pd.DataFrame:
    df = load_artifact("study_series.parquet")
    return df.sort_values(["series_id", "ts_index"]).reset_index(drop=True)


def safe_adf(values: Iterable[float]) -> tuple[float, float]:
    values = pd.Series(values).dropna().astype(float).to_numpy()
    if len(values) < 20 or np.nanstd(values) == 0:
        return np.nan, np.nan
    try:
        stat, pvalue, *_ = adfuller(values, autolag="AIC")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def safe_kpss(values: Iterable[float]) -> tuple[float, float]:
    values = pd.Series(values).dropna().astype(float).to_numpy()
    if len(values) < 20 or np.nanstd(values) == 0:
        return np.nan, np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            stat, pvalue, *_ = kpss(values, regression="c", nlags="auto")
        return float(stat), float(pvalue)
    except Exception:
        return np.nan, np.nan


def stationarity_verdict(adf_p: float, kpss_p: float) -> str:
    if np.isnan(adf_p) or np.isnan(kpss_p):
        return "failed"
    if adf_p < 0.05 and kpss_p > 0.05:
        return "stationary"
    if adf_p >= 0.05 and kpss_p < 0.05:
        return "unit_root"
    return "mixed"


def _overview_from_frames(train: pd.DataFrame, test: pd.DataFrame, feature_count: int) -> dict:
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_count": int(feature_count),
        "codes": int(train["code"].nunique()),
        "sub_codes_train": int(train["sub_code"].nunique()),
        "sub_codes_test": int(test["sub_code"].nunique()),
        "sub_categories": int(train["sub_category"].nunique()),
        "horizons": sorted(int(v) for v in train["horizon"].dropna().unique()),
        "train_ts_min": int(train["ts_index"].min()),
        "train_ts_max": int(train["ts_index"].max()),
        "test_ts_min": int(test["ts_index"].min()),
        "test_ts_max": int(test["ts_index"].max()),
        "legacy_val_cutoff": LEGACY_VAL_CUTOFF,
        "split_policy": {
            "type": "per_series_chronological_fraction",
            "train_fraction": TRAIN_FRAC,
            "min_series_length": MIN_SERIES_LENGTH,
            "min_train_points": MIN_TRAIN_POINTS,
            "min_val_points": MIN_VAL_POINTS,
            "lookback": LOOKBACK,
        },
    }


def _target_and_weight_artifacts(train: pd.DataFrame) -> dict:
    target = train["y_target"].astype(float)
    weight = train["weight"].astype(float)
    target_summary = {
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "q01": float(target.quantile(0.01)),
        "q05": float(target.quantile(0.05)),
        "q95": float(target.quantile(0.95)),
        "q99": float(target.quantile(0.99)),
        "min": float(target.min()),
        "max": float(target.max()),
    }

    full_counts, full_edges = np.histogram(target.to_numpy(), bins=150)
    clip_low, clip_high = target.quantile(0.01), target.quantile(0.99)
    clip_counts, clip_edges = np.histogram(
        np.clip(target.to_numpy(), clip_low, clip_high),
        bins=120,
    )
    weight_clip = float(weight.quantile(0.99))
    weight_counts, weight_edges = np.histogram(
        np.clip(weight.to_numpy(), 0, weight_clip),
        bins=120,
    )
    sorted_weight = weight.sort_values(ascending=False).reset_index(drop=True)
    cdf = sorted_weight.cumsum() / sorted_weight.sum()
    stride = max(1, len(sorted_weight) // 1500)
    cdf_table = pd.DataFrame(
        {
            "rank_pct": np.arange(len(sorted_weight))[::stride] / max(len(sorted_weight) - 1, 1),
            "cum_weight_share": cdf.iloc[::stride].to_numpy(),
        }
    )

    share_summary = {
        "top_1pct_share": float(sorted_weight.iloc[: max(1, int(len(sorted_weight) * 0.01))].sum() / sorted_weight.sum()),
        "top_5pct_share": float(sorted_weight.iloc[: max(1, int(len(sorted_weight) * 0.05))].sum() / sorted_weight.sum()),
        "top_10pct_share": float(sorted_weight.iloc[: max(1, int(len(sorted_weight) * 0.10))].sum() / sorted_weight.sum()),
    }

    pd.DataFrame(
        {
            "bin_left": full_edges[:-1],
            "bin_right": full_edges[1:],
            "count": full_counts,
        }
    ).to_parquet(CODEX_DIR / "target_hist_full.parquet", index=False)
    pd.DataFrame(
        {
            "bin_left": clip_edges[:-1],
            "bin_right": clip_edges[1:],
            "count": clip_counts,
        }
    ).to_parquet(CODEX_DIR / "target_hist_clip.parquet", index=False)
    pd.DataFrame(
        {
            "bin_left": weight_edges[:-1],
            "bin_right": weight_edges[1:],
            "count": weight_counts,
        }
    ).to_parquet(CODEX_DIR / "weight_hist_clip.parquet", index=False)
    cdf_table.to_parquet(CODEX_DIR / "weight_cumshare.parquet", index=False)

    return {
        "target_summary": target_summary,
        "weight_summary": {
            "clip_99pct": weight_clip,
            **share_summary,
        },
    }


def _build_series_manifest(train: pd.DataFrame) -> pd.DataFrame:
    manifest = (
        train.groupby(SERIES_KEYS)
        .agg(
            length=("ts_index", "size"),
            start=("ts_index", "min"),
            end=("ts_index", "max"),
            total_weight=("weight", "sum"),
            target_std=("y_target", "std"),
            target_mean_abs=("y_target", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    manifest["series_id"] = series_id_from_frame(manifest)
    manifest["target_std"] = manifest["target_std"].fillna(0.0)
    manifest["crosses_legacy_cutoff"] = (
        (manifest["start"] <= LEGACY_VAL_CUTOFF) & (manifest["end"] > LEGACY_VAL_CUTOFF)
    )
    manifest["eligible_codex"] = (
        (manifest["length"] >= MIN_SERIES_LENGTH) & (manifest["target_std"] > 0)
    )
    manifest["legacy_eligible"] = manifest["eligible_codex"] & manifest["crosses_legacy_cutoff"]
    manifest["split_train_len"] = manifest["length"].map(lambda n: split_info(int(n)).train_len if n >= MIN_SERIES_LENGTH else np.nan)
    manifest["split_val_len"] = manifest["length"].map(lambda n: split_info(int(n)).val_len if n >= MIN_SERIES_LENGTH else np.nan)
    manifest = manifest.sort_values(["code", "sub_code", "sub_category", "horizon"]).reset_index(drop=True)
    manifest.to_parquet(CODEX_DIR / "series_manifest.parquet", index=False)
    return manifest


def _pick_representatives(manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest[manifest["legacy_eligible"]].copy()
    stable_pool = eligible[eligible["target_std"] > 0].copy()
    chosen = pd.concat(
        [
            eligible.nlargest(1, "length").assign(reason="longest history"),
            eligible.nlargest(1, "total_weight").assign(reason="highest total weight"),
            eligible.nlargest(1, "target_std").assign(reason="most volatile"),
            stable_pool.nsmallest(1, "target_std").assign(reason="most stable"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["series_id"])
    chosen["role"] = "representative"
    chosen = chosen.sort_values("reason").reset_index(drop=True)
    chosen.to_parquet(CODEX_DIR / "chosen_manifest.parquet", index=False)
    return chosen


def _pick_benchmark(manifest: pd.DataFrame, chosen: pd.DataFrame) -> pd.DataFrame:
    pool = manifest[manifest["eligible_codex"] & ~manifest["series_id"].isin(chosen["series_id"])].copy()
    picks: list[pd.DataFrame] = []
    quantiles = [0.25, 0.75][:BENCHMARK_PER_HORIZON]
    for horizon, group in pool.groupby("horizon", sort=True):
        group = group.sort_values(["total_weight", "series_id"]).reset_index(drop=True)
        used: set[int] = set()
        rows = []
        for q in quantiles:
            idx = int(round((len(group) - 1) * q))
            while idx in used and idx + 1 < len(group):
                idx += 1
            used.add(idx)
            rows.append(group.iloc[[idx]].assign(benchmark_band=f"H{int(horizon)}_q{int(q * 100)}"))
        picks.append(pd.concat(rows, ignore_index=True))
    benchmark = pd.concat(picks, ignore_index=True)
    benchmark["role"] = "benchmark"
    benchmark = benchmark.sort_values(["horizon", "benchmark_band", "series_id"]).reset_index(drop=True)
    benchmark.to_parquet(CODEX_DIR / "benchmark_manifest.parquet", index=False)
    return benchmark


def _save_selected_series(train: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    keep_ids = selected["series_id"].unique().tolist()
    frame = train.loc[series_id_from_frame(train).isin(keep_ids), SERIES_KEYS + ["ts_index", "y_target", "weight"]].copy()
    frame["series_id"] = series_id_from_frame(frame)
    meta_cols = ["series_id", "role", "reason", "benchmark_band", "length", "total_weight", "target_std"]
    meta = selected.reindex(columns=[c for c in meta_cols if c in selected.columns]).drop_duplicates("series_id")
    merged = frame.merge(meta, on="series_id", how="left")
    merged = merged.sort_values(["role", "series_id", "ts_index"]).reset_index(drop=True)
    merged.to_parquet(CODEX_DIR / "study_series.parquet", index=False)
    return merged


def _build_stationarity(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest[manifest["eligible_codex"]].copy()
    sample = eligible.sample(n=min(STATIONARITY_SAMPLE_SIZE, len(eligible)), random_state=SEED)
    rows = []
    needed = set(sample["series_id"])
    study = train.loc[series_id_from_frame(train).isin(needed), SERIES_KEYS + ["ts_index", "y_target"]].copy()
    study["series_id"] = series_id_from_frame(study)
    for series_id, grp in study.groupby("series_id", sort=False):
        grp = grp.sort_values("ts_index")
        values = grp["y_target"].to_numpy()
        adf_stat, adf_p = safe_adf(values)
        kpss_stat, kpss_p = safe_kpss(values)
        diff_values = pd.Series(values).diff().dropna().to_numpy()
        adf_stat_d1, adf_p_d1 = safe_adf(diff_values)
        kpss_stat_d1, kpss_p_d1 = safe_kpss(diff_values)
        first = grp.iloc[0]
        rows.append(
            {
                "series_id": series_id,
                "code": first["code"],
                "sub_code": first["sub_code"],
                "sub_category": first["sub_category"],
                "horizon": int(first["horizon"]),
                "adf_stat": adf_stat,
                "adf_p": adf_p,
                "kpss_stat": kpss_stat,
                "kpss_p": kpss_p,
                "verdict": stationarity_verdict(adf_p, kpss_p),
                "adf_stat_d1": adf_stat_d1,
                "adf_p_d1": adf_p_d1,
                "kpss_stat_d1": kpss_stat_d1,
                "kpss_p_d1": kpss_p_d1,
                "verdict_d1": stationarity_verdict(adf_p_d1, kpss_p_d1),
            }
        )
    result = pd.DataFrame(rows).sort_values("series_id").reset_index(drop=True)
    result.to_parquet(CODEX_DIR / "stationarity_sample.parquet", index=False)
    return result


def _build_lag_summary(train: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    eligible = manifest[manifest["eligible_codex"]].copy()
    sample = eligible.sample(n=min(LAG_SAMPLE_SIZE, len(eligible)), random_state=SEED + 1)
    needed = set(sample["series_id"])
    frame = train.loc[series_id_from_frame(train).isin(needed), SERIES_KEYS + ["ts_index", "y_target"]].copy()
    frame["series_id"] = series_id_from_frame(frame)
    acf_rows: list[dict] = []
    for series_id, grp in frame.groupby("series_id", sort=False):
        values = grp.sort_values("ts_index")["y_target"].astype(float).to_numpy()
        if len(values) < MAX_LAG + 5 or np.std(values) == 0:
            continue
        values = (values - values.mean()) / (values.std() + 1e-8)
        try:
            acf_vals = acf(values, nlags=MAX_LAG, fft=True)
            pacf_vals = pacf(values, nlags=MAX_LAG, method="ywm")
        except Exception:
            continue
        for lag in range(1, MAX_LAG + 1):
            acf_rows.append(
                {
                    "series_id": series_id,
                    "lag": lag,
                    "acf_abs": float(abs(acf_vals[lag])),
                    "pacf_abs": float(abs(pacf_vals[lag])),
                }
            )
    lag_df = pd.DataFrame(acf_rows)
    summary = (
        lag_df.groupby("lag")
        .agg(mean_abs_acf=("acf_abs", "mean"), mean_abs_pacf=("pacf_abs", "mean"))
        .reset_index()
    )
    summary.to_parquet(CODEX_DIR / "lag_summary.parquet", index=False)
    return summary


def build_codex_artifacts(force: bool = False, verbose: bool = True) -> None:
    ensure_dirs()
    if artifacts_ready() and not force:
        if verbose:
            print(f"Codex artifacts already available in {CODEX_DIR}")
        return

    if verbose:
        print("Reading raw Kaggle parquet files...")
    schema_cols = pq.ParquetFile(TRAIN_PATH).schema.names
    feature_count = sum(col.startswith("feature_") for col in schema_cols)
    cols = SERIES_KEYS + ["ts_index", "y_target", "weight"]
    train = pd.read_parquet(TRAIN_PATH, columns=cols)
    test = pd.read_parquet(TEST_PATH, columns=SERIES_KEYS + ["ts_index"])

    overview = _overview_from_frames(train, test, feature_count=feature_count)
    extra = _target_and_weight_artifacts(train)
    overview.update(extra)

    if verbose:
        print("Building series manifest...")
    manifest = _build_series_manifest(train)
    chosen = _pick_representatives(manifest)
    benchmark = _pick_benchmark(manifest, chosen)
    selected = pd.concat([chosen, benchmark], ignore_index=True, sort=False)

    if verbose:
        print("Saving selected study series...")
    _save_selected_series(train, selected)

    if verbose:
        print("Running stationarity sample...")
    _build_stationarity(train, manifest)
    if verbose:
        print("Building lag summary...")
    _build_lag_summary(train, manifest)

    overview["legacy_cutoff_counts"] = {
        "total_series": int(len(manifest)),
        "crosses_legacy_cutoff": int(manifest["crosses_legacy_cutoff"].sum()),
        "legacy_eligible": int(manifest["legacy_eligible"].sum()),
    }
    overview["codex_counts"] = {
        "eligible_codex": int(manifest["eligible_codex"].sum()),
        "representatives": int(len(chosen)),
        "benchmark_series": int(len(benchmark)),
    }
    save_json_artifact("overview.json", overview)
    if verbose:
        print(f"Codex artifacts built in {CODEX_DIR}")
