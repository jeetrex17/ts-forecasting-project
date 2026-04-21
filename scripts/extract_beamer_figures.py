#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "Finial_notebooks"
DEFAULT_OUT = ROOT / "presentation" / "beamer" / "figures"

# Notebook cell indexes are zero-based.
FIGURE_MAP: dict[str, dict[tuple[int, int], str]] = {
    "01_EDA.ipynb": {
        (15, 1): "eda_target_distribution.png",
        (19, 1): "eda_weight_distribution.png",
        (31, 1): "eda_representative_series.png",
        (35, 1): "eda_adf_summary.png",
        (53, 1): "eda_panel_lags.png",
    },
    "02_classical_codex.ipynb": {
        (5, 2): "classical_cutoff_zoom.png",
        (13, 2): "classical_forecast_zoom.png",
    },
    "03_deep_sequence_models_chosen.ipynb": {
        (21, 1): "deep_forecast_panels.png",
    },
}


def extract_png(data: str | list[str], target: Path) -> None:
    payload = "".join(data) if isinstance(data, list) else data
    target.write_bytes(base64.b64decode(payload))


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract curated PNG outputs from final notebooks for the Beamer deck.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for extracted figures.")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, str] = {}
    for notebook_name, targets in FIGURE_MAP.items():
        notebook_path = NOTEBOOK_DIR / notebook_name
        notebook = json.loads(notebook_path.read_text())

        for (cell_idx, image_idx), filename in targets.items():
            cell = notebook["cells"][cell_idx]
            seen = 0
            found = False
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" not in data:
                    continue
                seen += 1
                if seen != image_idx:
                    continue
                target = out_dir / filename
                extract_png(data["image/png"], target)
                extracted[filename] = str(target)
                found = True
                break

            if not found:
                raise RuntimeError(
                    f"Could not find image {image_idx} in cell {cell_idx} of {notebook_name}."
                )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(extracted, indent=2))
    print(f"Extracted {len(extracted)} figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
