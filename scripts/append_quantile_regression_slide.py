#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

import presentation_deck_v2 as deck


ROOT = Path(__file__).resolve().parents[1]
MAIN_PDF = ROOT / "Finial_notebooks" / "main.pdf"
BACKUP_PDF = Path("/tmp/main_before_quantile_regression_replace.pdf")
SWIFT_REPLACER = ROOT / "scripts" / "replace_last_pdf_page.swift"
QUANTILE_IMAGES = [
    ("Longest History", ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_longest_history_plot.png"),
    ("Highest Total Weight", ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_highest_total_weight_plot.png"),
    ("Most Volatile", ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_most_volatile_plot.png"),
    ("Most Stable", ROOT / "Finial_notebooks" / "quantile_outputs" / "quantile_most_stable_plot.png"),
]


def build_slide(slide_pdf: Path) -> None:
    missing = [path for _, path in QUANTILE_IMAGES if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing quantile artifact(s): {missing_text}")

    deck.TOTAL_SLIDES = 51

    with PdfPages(slide_pdf) as pdf:
        builder = deck.DeckBuilder(pdf)
        builder.slide_no = 50
        fig = builder.new_slide(
            "Quantile Regression Forecasts",
            "Classical",
            "Shared LightGBM setup: 400 trees, lr=0.04, depth=5, 31 leaves, min child=10, subsample=0.85, colsample=0.85, seed=42; rolling folds 70/20/20, with the interval selected separately per series and fold.",
        )

        template = builder.template_dashboard_2x2()
        for slot, (title, image_path) in zip(("tl", "tr", "bl", "br"), QUANTILE_IMAGES):
            asset = deck.AssetSpec(key=title.lower().replace(" ", "_"), path=image_path, kind="chart", fit_mode="contain")
            builder.image_box(fig, asset, template.slots[slot])

        builder.save(fig)
def replace_last_slide() -> None:
    if not MAIN_PDF.exists():
        raise FileNotFoundError(f"Main PDF not found: {MAIN_PDF}")
    if not SWIFT_REPLACER.exists():
        raise FileNotFoundError(f"Swift replacer not found: {SWIFT_REPLACER}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="quantile_regression_slide_"))
    slide_pdf = tmp_dir / "quantile_regression_slide.pdf"
    rebuilt_pdf = tmp_dir / "main_with_replaced_quantile_slide.pdf"

    shutil.copy2(MAIN_PDF, BACKUP_PDF)
    build_slide(slide_pdf)
    swift_env = os.environ.copy()
    swift_env["TMPDIR"] = "/tmp"
    subprocess.run(
        ["swift", str(SWIFT_REPLACER), str(MAIN_PDF), str(slide_pdf), str(rebuilt_pdf)],
        check=True,
        env=swift_env,
    )
    shutil.move(rebuilt_pdf, MAIN_PDF)

    print(MAIN_PDF)
    print(BACKUP_PDF)


if __name__ == "__main__":
    replace_last_slide()
