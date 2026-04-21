#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

import presentation_deck_v2 as deck


ROOT = Path(__file__).resolve().parents[1]
MAIN_PDF = ROOT / "Finial_notebooks" / "main.pdf"
QUANTILE_DIR = ROOT / "Finial_notebooks" / "Quantile"
BACKUP_PDF = Path("/tmp/main_before_quantile_regression_append.pdf")

QUANTILE_IMAGES = [
    ("Longest History", QUANTILE_DIR / "WhatsApp Image 2026-04-22 at 03.44.20.jpeg"),
    ("Highest Total Weight", QUANTILE_DIR / "WhatsApp Image 2026-04-22 at 03.43.56.jpeg"),
    ("Most Volatile", QUANTILE_DIR / "WhatsApp Image 2026-04-22 at 03.43.21.jpeg"),
    ("Most Stable", QUANTILE_DIR / "WhatsApp Image 2026-04-22 at 03.42.48.jpeg"),
]


def build_slide(slide_pdf: Path) -> None:
    missing = [path for _, path in QUANTILE_IMAGES if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing quantile image(s): {missing_text}")

    deck.TOTAL_SLIDES = 51

    with PdfPages(slide_pdf) as pdf:
        builder = deck.DeckBuilder(pdf)
        builder.slide_no = 50
        fig = builder.new_slide(
            "Quantile Regression Forecasts",
            "Deep",
            "Quantile regression adds a median path and a 10%-90% interval band: uncertainty stays narrow for the stable and weight cases, but expands materially on the longest-history and volatile series where forecast risk is higher.",
        )

        template = builder.template_dashboard_2x2()
        for slot, (title, image_path) in zip(("tl", "tr", "bl", "br"), QUANTILE_IMAGES):
            asset = deck.AssetSpec(key=title.lower().replace(" ", "_"), path=image_path)
            builder.image_box(fig, asset, template.slots[slot], title=title)

        builder.save(fig)


def append_slide() -> None:
    if not MAIN_PDF.exists():
        raise FileNotFoundError(f"Main PDF not found: {MAIN_PDF}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="quantile_regression_slide_"))
    slide_pdf = tmp_dir / "quantile_regression_slide.pdf"
    merged_pdf = tmp_dir / "main_with_quantile_regression.pdf"

    shutil.copy2(MAIN_PDF, BACKUP_PDF)
    build_slide(slide_pdf)

    subprocess.run(["pdfunite", str(MAIN_PDF), str(slide_pdf), str(merged_pdf)], check=True)
    shutil.move(merged_pdf, MAIN_PDF)

    print(MAIN_PDF)
    print(BACKUP_PDF)


if __name__ == "__main__":
    append_slide()
