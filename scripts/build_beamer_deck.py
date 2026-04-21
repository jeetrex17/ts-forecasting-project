#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BEAMER_DIR = ROOT / "presentation" / "beamer"
BUILD_DIR = BEAMER_DIR / "build"
FIGURE_DIR = BEAMER_DIR / "figures"
TEXBIN = Path.home() / "Library" / "TinyTeX" / "bin" / "universal-darwin"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract figures and build the new Beamer presentation.")
    parser.add_argument("--skip-extract", action="store_true", help="Reuse existing extracted figures.")
    args = parser.parse_args()

    env = os.environ.copy()
    if TEXBIN.exists():
        env["PATH"] = f"{TEXBIN}:{env.get('PATH', '')}"

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "extract_beamer_figures.py"), "--out", str(FIGURE_DIR)],
            cwd=ROOT,
            env=env,
            check=True,
        )

    latexmk = TEXBIN / "latexmk" if (TEXBIN / "latexmk").exists() else Path("latexmk")
    subprocess.run(
        [
            str(latexmk),
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-outdir={BUILD_DIR}",
            "main.tex",
        ],
        cwd=BEAMER_DIR,
        env=env,
        check=True,
    )

    pdf_path = BUILD_DIR / "main.pdf"
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
