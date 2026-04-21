#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import presentation_deck_v2 as deck


ROOT = Path(__file__).resolve().parents[1]
MAIN_PDF = ROOT / "Finial_notebooks" / "main.pdf"
QUANTILE_GRID = ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_chosen_series_grid.png"
SUMMARY_CSV = ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_metrics_summary.csv"
BACKUP_PDF = Path("/tmp/main_before_quantile_regression_replace.pdf")

LGBM_PARAM_BULLETS = [
    "Same LightGBM quantile setup on every series and every rolling fold.",
    "objective='quantile'; alphas q02, q05, q10, q25, q50, q75, q90, q95, q98.",
    "n_estimators=400, learning_rate=0.04, max_depth=5, num_leaves=31, min_child_samples=10, subsample=0.85, colsample_bytree=0.85, random_state=42; folds use 70/20/20.",
]


def selected_recipe_bullets(summary: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{row['Series']}: {row['dominant_method']} with {row['dominant_interval_pair']} "
            f"(wPICP={row['wPICP_80']:.3f}, wMPIW={row['wMPIW_80']:.4g})."
        )
    return rows


def build_slide(slide_pdf: Path) -> None:
    missing = [path for path in (QUANTILE_GRID, SUMMARY_CSV) if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing quantile artifact(s): {missing_text}")

    summary = pd.read_csv(SUMMARY_CSV)

    deck.TOTAL_SLIDES = 51

    with PdfPages(slide_pdf) as pdf:
        builder = deck.DeckBuilder(pdf)
        builder.slide_no = 50
        fig = builder.new_slide(
            "Quantile Regression Forecasts",
            "Classical",
            "The rolling backtest now uses the stitched quantile grid directly: LightGBM provides the raw quantile curves, then the selected interval recipe balances coverage and width differently for each representative series.",
        )

        template = builder.template_full_width_figure(figure_fraction=0.64)
        asset = deck.AssetSpec(key="quantile_grid", path=QUANTILE_GRID, kind="multi_panel_chart", fit_mode="contain")
        builder.image_box(fig, asset, template.slots["figure"])

        params_rect, series_rect = template.slots["note"].split_cols([0.60, 0.40])
        builder.bullet_box(
            fig,
            params_rect,
            title="LightGBM Quantile Setup",
            bullets=LGBM_PARAM_BULLETS,
            body_size=7.8,
            body_min_size=7.3,
        )
        builder.bullet_box(
            fig,
            series_rect,
            title="Selected Interval Recipe",
            bullets=selected_recipe_bullets(summary),
            body_size=7.7,
            body_min_size=7.2,
        )

        builder.save(fig)


def page_count(pdf_path: Path) -> int:
    result = subprocess.run(["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not determine page count for {pdf_path}")


def replace_last_slide() -> None:
    if not MAIN_PDF.exists():
        raise FileNotFoundError(f"Main PDF not found: {MAIN_PDF}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="quantile_regression_slide_"))
    pages_dir = tmp_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    slide_pdf = tmp_dir / "quantile_regression_slide.pdf"
    rebuilt_pdf = tmp_dir / "main_with_replaced_quantile_slide.pdf"

    shutil.copy2(MAIN_PDF, BACKUP_PDF)
    build_slide(slide_pdf)

    total_pages = page_count(MAIN_PDF)
    if total_pages <= 1:
        shutil.move(slide_pdf, MAIN_PDF)
        print(MAIN_PDF)
        print(BACKUP_PDF)
        return

    page_pattern = pages_dir / "page-%03d.pdf"
    subprocess.run(["pdfseparate", str(MAIN_PDF), str(page_pattern)], check=True)

    kept_pages = [pages_dir / f"page-{idx:03d}.pdf" for idx in range(1, total_pages)]
    subprocess.run(["pdfunite", *[str(path) for path in kept_pages], str(slide_pdf), str(rebuilt_pdf)], check=True)
    shutil.move(rebuilt_pdf, MAIN_PDF)

    print(MAIN_PDF)
    print(BACKUP_PDF)


if __name__ == "__main__":
    replace_last_slide()
