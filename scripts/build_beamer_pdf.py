#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEAMER_DIR = ROOT / "presentation" / "beamer"
DATA_DIR = BEAMER_DIR / "data"
BUILD_DIR = BEAMER_DIR / "build"
TEXBIN = Path.home() / "Library" / "TinyTeX" / "bin" / "universal-darwin"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Beamer data and build the LaTeX deck.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse existing exported CSV data.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PATH"] = f"{TEXBIN}:{env.get('PATH', '')}"

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sections_data_link = BEAMER_DIR / "sections" / "data"
    build_data_link = BUILD_DIR / "data"

    if not args.skip_export:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_beamer_data.py"), "--out", str(DATA_DIR)],
            cwd=ROOT,
            env=env,
            check=True,
        )

    for link_path in (sections_data_link, build_data_link):
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(DATA_DIR, target_is_directory=True)

    subprocess.run(
        [
            str(TEXBIN / "latexmk"),
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-jobname=Hedge_Fund_Time_Series_Forecasting_Beamer",
            f"-outdir={BUILD_DIR}",
            "main.tex",
        ],
        cwd=BEAMER_DIR,
        env=env,
        check=True,
    )

    pdf_path = BUILD_DIR / "Hedge_Fund_Time_Series_Forecasting_Beamer.pdf"
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
