from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "ts-forecasting" / "train.parquet"
OVERVIEW_PATH = ROOT / "data" / "processed" / "codex" / "overview.json"
FIG_DIR = ROOT / "presentation" / "beamer" / "figures" / "codex"

ACCENT = "#425B74"
INK = "#1F2933"
MUTED = "#667482"
SOFT = "#F4F7FA"
SOFT_ALT = "#E8EEF3"
GRID = "#DCE3EA"
WARM = "#EEE6DB"


def apply_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "DejaVu Sans", "Arial"],
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": GRID,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def load_overview() -> dict:
    return json.loads(OVERVIEW_PATH.read_text())


def compute_missingness() -> pd.DataFrame:
    parquet = pq.ParquetFile(TRAIN_PATH)
    counts = {name: 0 for name in parquet.schema_arrow.names}
    total_rows = 0

    for batch in parquet.iter_batches(batch_size=200_000):
        frame = batch.to_pandas()
        total_rows += len(frame)
        nulls = frame.isna().sum()
        for column, value in nulls.items():
            counts[column] += int(value)

    rows = [
        {
            "column": column,
            "missing_count": count,
            "missing_rate": count / total_rows,
        }
        for column, count in counts.items()
        if count > 0
    ]
    return pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def plot_missing_values(missingness: pd.DataFrame) -> Path:
    top = missingness.head(10).iloc[::-1].copy()
    top["missing_pct"] = top["missing_rate"] * 100

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(top["column"], top["missing_pct"], color=ACCENT, alpha=0.88, height=0.68)
    ax.set_xlabel("Missing rate (%)")
    ax.set_ylabel("")
    ax.set_xlim(0, max(13.5, float(top["missing_pct"].max()) + 0.8))
    ax.grid(axis="y", visible=False)
    ax.set_title("Top missing feature columns", loc="left", fontsize=15, fontweight="bold")

    for idx, value in enumerate(top["missing_pct"]):
        ax.text(value + 0.15, idx, f"{value:.2f}%", va="center", ha="left", fontsize=9.5, color=INK)

    fig.tight_layout()
    out = FIG_DIR / "01_missing_values_chart.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target_center(overview: dict) -> Path:
    target = pd.read_parquet(TRAIN_PATH, columns=["y_target"])["y_target"].astype(float)
    q05 = float(overview["target_summary"]["q05"])
    q95 = float(overview["target_summary"]["q95"])
    median = float(overview["target_summary"]["median"])
    center = target[(target >= q05) & (target <= q95)]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.histplot(center, bins=72, stat="count", color=ACCENT, edgecolor="white", linewidth=0.4, ax=ax)
    ax.axvline(median, color=MUTED, linestyle="--", linewidth=1.5)
    ax.set_xlabel("y_target")
    ax.set_ylabel("Count")
    ax.set_title("Central 90% window of the target distribution", loc="left", fontsize=15, fontweight="bold")
    ax.text(
        0.99,
        0.95,
        f"Median = {median:.4f}\nWindow = [{q05:.2f}, {q95:.2f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.3", "fc": SOFT, "ec": GRID},
    )

    fig.tight_layout()
    out = FIG_DIR / "01_target_distribution_center.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summaries(missingness: pd.DataFrame) -> None:
    top = missingness.head(10).copy()
    top["missing_pct"] = (top["missing_rate"] * 100).round(4)
    top.to_csv(FIG_DIR / "01_missing_values_top.csv", index=False)

    summary = {
        "missing_columns": int(len(missingness)),
        "top_missing": top[["column", "missing_pct"]].to_dict(orient="records"),
    }
    (FIG_DIR / "01_missing_values_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()
    overview = load_overview()
    missingness = compute_missingness()
    write_summaries(missingness)
    plot_missing_values(missingness)
    plot_target_center(overview)


if __name__ == "__main__":
    main()
