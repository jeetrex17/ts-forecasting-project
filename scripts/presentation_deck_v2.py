#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.table import Table
from matplotlib.transforms import Bbox
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "Finial_notebooks"
OUTPUT_DIR = ROOT / "presentation" / "build"
ASSET_DIR = OUTPUT_DIR / "extracted_assets"
PDF_PATH = OUTPUT_DIR / "Hedge_Fund_Time_Series_Forecasting_Main_Deck.pdf"

SLIDE_W = 13.333
SLIDE_H = 7.5
TOTAL_SLIDES = 50

BG = "#F5F1EA"
CARD = "#FFFFFF"
INK = "#1F2933"
MUTED = "#58646E"
BORDER = "#D8D0C5"
GRID = "#E9E2D8"
SUCCESS = "#6B7A5E"

SECTION_COLORS = {
    "Overview": "#5B6E87",
    "EDA": "#5F7D8E",
    "Classical": "#B56A3A",
    "Deep": "#2F6E91",
    "Summary": "#6B7A5E",
}

HEADER_H = 0.07
SAFE_MARGIN_X = 0.045
SAFE_MARGIN_BOTTOM = 0.05
SAFE_TOP = 0.915
TITLE_H = 0.075
TITLE_GAP = 0.015
TAKEAWAY_H = 0.085
SECTION_GAP = 0.02
GUTTER = 0.018
CARD_PAD = 0.014
CARD_RADIUS = 0.02
MIN_BODY_FONT = 8.7
MIN_CHART_LABEL = 8.2

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    def inset(self, pad_x: float, pad_y: float | None = None) -> "Rect":
        if pad_y is None:
            pad_y = pad_x
        return Rect(self.x + pad_x, self.y + pad_y, self.w - 2 * pad_x, self.h - 2 * pad_y)

    def split_cols(self, fractions: list[float], gutter: float = GUTTER) -> list["Rect"]:
        total = sum(fractions)
        usable = self.w - gutter * (len(fractions) - 1)
        x = self.x
        rects: list[Rect] = []
        for idx, fraction in enumerate(fractions):
            width = usable * (fraction / total)
            rects.append(Rect(x, self.y, width, self.h))
            x += width
            if idx < len(fractions) - 1:
                x += gutter
        return rects

    def split_rows(self, fractions: list[float], gutter: float = GUTTER, top_first: bool = True) -> list["Rect"]:
        total = sum(fractions)
        usable = self.h - gutter * (len(fractions) - 1)
        heights = [usable * (fraction / total) for fraction in fractions]
        rects: list[Rect] = []
        if top_first:
            cursor_top = self.top
            for idx, height in enumerate(heights):
                y = cursor_top - height
                rects.append(Rect(self.x, y, self.w, height))
                cursor_top = y - (gutter if idx < len(heights) - 1 else 0)
        else:
            y = self.y
            for idx, height in enumerate(heights):
                rects.append(Rect(self.x, y, self.w, height))
                y += height + (gutter if idx < len(heights) - 1 else 0)
        return rects

    def union(self, other: "Rect") -> "Rect":
        x0 = min(self.x, other.x)
        y0 = min(self.y, other.y)
        x1 = max(self.right, other.right)
        y1 = max(self.top, other.top)
        return Rect(x0, y0, x1 - x0, y1 - y0)


@dataclass
class SlideTemplate:
    name: str
    content: Rect
    slots: dict[str, Rect]


@dataclass
class ValidationTarget:
    label: str
    artist: object
    bounds: Rect


@dataclass(frozen=True)
class AssetSpec:
    key: str
    path: Path
    kind: str = "chart"
    fit_mode: str = "contain"
    trim_mode: str = "none"
    manual_crop: tuple[float, float, float, float] | None = None
    focus: tuple[float, float] = (0.5, 0.5)
    allow_cover: bool = False


class LayoutValidationError(RuntimeError):
    pass


def blend(color: str, target: str = "#FFFFFF", alpha: float = 0.2) -> tuple[float, float, float]:
    def hex_to_rgb(value: str) -> np.ndarray:
        value = value.lstrip("#")
        return np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])

    src = hex_to_rgb(color)
    tgt = hex_to_rgb(target)
    mixed = (1 - alpha) * src + alpha * tgt
    return tuple(mixed)


def fmt_int(value: int) -> str:
    return f"{value:,}"


def bullet_text(items: list[str]) -> str:
    return "\n\n".join(f"• {item}" for item in items)


def wrap_paragraphs(text: str, width: int) -> str:
    wrapped_blocks: list[str] = []
    for block in text.split("\n"):
        stripped = block.strip()
        if not stripped:
            wrapped_blocks.append("")
            continue
        if stripped.startswith("• "):
            wrapped_blocks.append(
                textwrap.fill(
                    stripped[2:],
                    width=width,
                    initial_indent="• ",
                    subsequent_indent="  ",
                    break_long_words=False,
                )
            )
        else:
            wrapped_blocks.append(textwrap.fill(stripped, width=width, break_long_words=False))
    return "\n".join(wrapped_blocks)


def extract_png_assets() -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Path] = {}

    for notebook_path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = json.loads(notebook_path.read_text())
        stem = notebook_path.stem

        for idx, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            png_idx = 0
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "image/png" not in data:
                    continue
                png_idx += 1
                key = f"{stem}_cell{idx:02d}_img{png_idx}"
                png_data = data["image/png"]
                if isinstance(png_data, list):
                    png_data = "".join(png_data)
                target = ASSET_DIR / f"{key}.png"
                target.write_bytes(base64.b64decode(png_data))
                manifest[key] = target

    manifest_path = OUTPUT_DIR / "asset_manifest.json"
    manifest_path.write_text(json.dumps({k: str(v) for k, v in manifest.items()}, indent=2))
    return manifest


def build_asset_specs(manifest: dict[str, Path]) -> dict[str, AssetSpec]:
    specs = {key: AssetSpec(key=key, path=path) for key, path in manifest.items()}

    safe_trim_keys = {
        "01_EDA_cell15_img1",
        "01_EDA_cell19_img1",
        "01_EDA_cell23_img1",
        "01_EDA_cell35_img1",
        "02_classical_codex_cell05_img1",
        "02_classical_codex_cell05_img2",
        "02_classical_codex_cell11_img1",
        "02_classical_codex_cell11_img2",
        "02_classical_codex_cell11_img3",
        "02_classical_codex_cell15_img1",
        "02_classical_codex_cell17_img1",
        "03_deep_sequence_models_chosen_cell21_img1",
        "03_deep_sequence_models_chosen_cell24_img1",
    }
    multi_panel_keys = {
        "01_EDA_cell31_img1",
        "01_EDA_cell44_img1",
        "01_EDA_cell47_img1",
        "01_EDA_cell50_img1",
        "01_EDA_cell53_img1",
        "01_EDA_cell56_img1",
        "01_EDA_cell59_img1",
        "01_EDA_cell62_img1",
        "02_classical_codex_cell13_img1",
        "02_classical_codex_cell13_img2",
        "03_deep_sequence_models_chosen_cell21_img1",
        "03_deep_sequence_models_chosen_cell24_img1",
    }

    for key in safe_trim_keys:
        if key in specs:
            specs[key] = replace(specs[key], trim_mode="safe_chart")
    for key in multi_panel_keys:
        if key in specs:
            specs[key] = replace(specs[key], kind="multi_panel_chart")

    crop_variants = {
        "01_EDA_cell31_img1_top": ("01_EDA_cell31_img1", (0.0, 0.0, 1.0, 0.5)),
        "01_EDA_cell31_img1_bottom": ("01_EDA_cell31_img1", (0.0, 0.5, 1.0, 1.0)),
        "02_classical_codex_cell13_img1_top": ("02_classical_codex_cell13_img1", (0.0, 0.0, 1.0, 0.5)),
        "02_classical_codex_cell13_img1_bottom": ("02_classical_codex_cell13_img1", (0.0, 0.5, 1.0, 1.0)),
        "02_classical_codex_cell13_img2_top": ("02_classical_codex_cell13_img2", (0.0, 0.0, 1.0, 0.5)),
        "02_classical_codex_cell13_img2_bottom": ("02_classical_codex_cell13_img2", (0.0, 0.5, 1.0, 1.0)),
    }
    for key, (base_key, crop) in crop_variants.items():
        base = specs[base_key]
        specs[key] = replace(base, key=key, trim_mode="manual", manual_crop=crop)

    spec_manifest_path = OUTPUT_DIR / "asset_spec_manifest.json"
    spec_manifest_path.write_text(
        json.dumps(
            {
                key: {
                    "path": str(spec.path),
                    "kind": spec.kind,
                    "fit_mode": spec.fit_mode,
                    "trim_mode": spec.trim_mode,
                    "manual_crop": spec.manual_crop,
                }
                for key, spec in specs.items()
            },
            indent=2,
        )
    )
    return specs


def _trim_bbox(image: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    bg = Image.new("RGB", image.size, "white")
    diff = ImageChops.difference(image, bg).convert("L")
    diff = diff.point(lambda p: 255 if p > threshold else 0)
    return diff.getbbox()


def image_array(
    path: Path,
    *,
    trim_mode: str = "none",
    manual_crop: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if manual_crop is not None:
        left, top, right, bottom = manual_crop
        width, height = image.size
        image = image.crop(
            (
                int(width * left),
                int(height * top),
                int(width * right),
                int(height * bottom),
            )
        )

    if trim_mode in {"none", "manual"}:
        return np.asarray(image)

    if trim_mode == "safe_chart":
        bbox = _trim_bbox(image, threshold=14)
        if bbox:
            left, top, right, bottom = bbox
            pad_x = max(int(image.size[0] * 0.03), 20)
            pad_y = max(int(image.size[1] * 0.05), 24)
            image = image.crop(
                (
                    max(left - pad_x, 0),
                    max(top - pad_y, 0),
                    min(right + pad_x, image.size[0]),
                    min(bottom + pad_y, image.size[1]),
                )
            )
        return np.asarray(image)

    raise ValueError(f"Unsupported trim mode: {trim_mode}")
    return np.asarray(image)


def crop_to_aspect(arr: np.ndarray, target_ratio: float, focus: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    height, width = arr.shape[:2]
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 0.02:
        return arr

    fx, fy = focus
    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        center_x = int(width * fx)
        left = max(0, min(width - new_width, center_x - new_width // 2))
        return arr[:, left : left + new_width]

    new_height = int(width / target_ratio)
    center_y = int(height * fy)
    top = max(0, min(height - new_height, center_y - new_height // 2))
    return arr[top : top + new_height, :]


DATASET_COUNTS = {
    "train_rows": 5_337_414,
    "test_rows": 1_447_107,
    "feature_count": 86,
    "codes": 23,
    "sub_codes_train": 180,
    "sub_codes_test": 47,
    "sub_categories": 5,
    "horizons": "1, 3, 10, 25",
    "train_range": "1 to 3601",
    "test_range": "3602 to 4376",
}

COMPARABILITY_DF = pd.DataFrame(
    [
        ["code", 23, 23, "Yes"],
        ["sub_code", 180, 47, "No (narrower support)"],
        ["sub_category", 5, 5, "Yes"],
        ["horizon", 4, 4, "Yes"],
    ],
    columns=["Column", "Train unique", "Test unique", "Test subset of train"],
)

SCHEMA_DF = pd.DataFrame(
    [
        ["code", "Primary series identifier"],
        ["sub_code", "Secondary series identifier"],
        ["sub_category", "Category grouping"],
        ["horizon", "Forecast horizon"],
        ["ts_index", "Integer time index"],
        ["y_target", "Continuous target value"],
        ["weight", "Per-row competition weight"],
    ],
    columns=["Field", "Meaning"],
)

MISSING_TOP_DF = pd.DataFrame(
    [
        ["feature_at", "12.47%"],
        ["feature_by", "11.02%"],
        ["feature_ay", "8.54%"],
        ["feature_cd", "7.50%"],
        ["feature_ce", "5.17%"],
    ],
    columns=["Most missing columns", "Missing rate"],
)

FEATURE_CORR_DF = pd.DataFrame(
    [
        ["feature_bz", "0.0758"],
        ["feature_cd", "0.0730"],
        ["feature_af", "-0.0697"],
        ["feature_u", "-0.0660"],
        ["feature_s", "-0.0531"],
        ["feature_bo", "-0.0525"],
        ["feature_bm", "-0.0492"],
        ["feature_t", "-0.0372"],
    ],
    columns=["Top |corr(y, feature)|", "Correlation"],
)

CHOSEN_SERIES_DF = pd.DataFrame(
    [
        ["Longest History", "X9BZ68VQ / OYJGNSQK / DPPUO5X2 / H1", 212, "9.06e2", "2.6416"],
        ["Highest Total Weight", "SJZP0OVU / OYJGNSQK / NQ58FVQM / H25", 156, "4.35e10", "0.0004"],
        ["Most Volatile", "W4S29LF4 / KL66VIS3 / PHHHVYZI / H25", 162, "4.56e-1", "296.7600"],
        ["Most Stable", "SJZP0OVU / OYJGNSQK / NQ58FVQM / H1", 170, "3.87e10", "0.0001"],
    ],
    columns=["Reason", "Series key", "Length", "Total weight", "Target std"],
)

KPSS_VERDICT_DF = pd.DataFrame(
    [
        ["Stationary", "64.0%", "ADF rejects unit root and KPSS does not reject stationarity."],
        ["Inconclusive", "25.5%", "Conflicting evidence; often trend-stationary or noisy."],
        ["Unit root", "10.5%", "Fails ADF and rejects KPSS, so differencing is needed."],
    ],
    columns=["Verdict", "Share", "Interpretation"],
)

LEAKAGE_PARTITION_DF = pd.DataFrame(
    [
        ["train_pre_cutoff", fmt_int(4_121_749)],
        ["train_post_cutoff", fmt_int(1_215_665)],
        ["test", fmt_int(1_447_107)],
    ],
    columns=["Partition", "Rows"],
)

CLASSICAL_LINEUP_DF = pd.DataFrame(
    [
        ["Smoothing", "Rolling Mean (w=10)", 2, "Constant average of latest window"],
        ["Smoothing", "SES", 3, "Single smoothed level, flat multi-step forecast"],
        ["Smoothing", "Holt", 5, "Level + trend extrapolation"],
        ["AR", "AR(1), AR(2), AR(3)", 20, "Lagged target dependence only"],
        ["MA", "MA(1), MA(2), MA(3)", 20, "Uses serially correlated shocks"],
        ["ARMA", "ARMA(1,1) to ARMA(3,3)", 20, "AR and MA combined in levels"],
        ["ARIMA", "ARIMA(0,1,1), ARIMA(1,1,0), ARIMA(1,1,1)", 20, "Difference first, then fit ARMA"],
    ],
    columns=["Family", "Models", "Min train", "Intuition"],
)

CLASSICAL_BEST_DF = pd.DataFrame(
    [
        ["Highest Total Weight", "SES", "Smoothing", "0.0000", "0.0006", "0.0005", 7, 149],
        ["Longest History", "ARMA(1,3)", "ARMA", "0.0000", "3.0400", "1.6685", 55, 157],
        ["Most Stable", "SES", "Smoothing", "0.0000", "0.0001", "0.0001", 7, 163],
        ["Most Volatile", "ARIMA(1,1,1)", "ARIMA", "0.8412", "129.8104", "108.8193", 131, 31],
    ],
    columns=["Series", "Best model", "Family", "Skill", "RMSE", "MAE", "Train len", "Val len"],
)

CLASSICAL_SUMMARY_DF = pd.DataFrame(
    [
        ["SES", 0.2052, 35.0584, 1.000],
        ["Rolling Mean (w=10)", 0.1691, 44.9619, 1.000],
        ["Holt", 0.1221, 53.1606, 1.000],
        ["ARIMA(1,1,1)", 0.4206, 66.4288, 0.500],
        ["ARIMA(1,1,0)", 0.4131, 69.1758, 0.500],
        ["ARIMA(0,1,1)", 0.4120, 69.5437, 0.500],
        ["AR(1)", 0.3678, 82.8428, 0.500],
        ["ARMA(1,1)", 0.3500, 87.2554, 0.500],
    ],
    columns=["Model", "Mean skill", "Mean RMSE", "OK rate"],
)

DEEP_SPLIT_DF = pd.DataFrame(
    [
        ["Longest History", 169, 43, "2826–2994", "2995–3037"],
        ["Highest Total Weight", 124, 32, "2874–3000", "3001–3032"],
        ["Most Volatile", 129, 33, "2747–2878", "2879–2911"],
        ["Most Stable", 136, 34, "2874–3012", "3013–3052"],
    ],
    columns=["Series", "Train len", "Val len", "Train range", "Validation range"],
)

DEEP_LEADERBOARD_DF = pd.DataFrame(
    [
        ["GRU", 0.339034, 36.590587, 1.603287],
        ["LSTM", 0.326238, 38.938620, 1.657656],
        ["RNN", 0.322859, 35.809019, 1.588534],
        ["Naive", 0.277758, 46.019387, 1.863608],
    ],
    columns=["Model", "Mean skill", "Mean weighted RMSE", "Mean MASE"],
)

DEEP_WINNERS_DF = pd.DataFrame(
    [
        ["Highest Total Weight", "GRU", "0.511704"],
        ["Longest History", "LSTM", "0.034827"],
        ["Most Stable", "Naive", "0.000000"],
        ["Most Volatile", "RNN", "0.819422"],
    ],
    columns=["Series", "Best deep model", "Skill"],
)

FINAL_COMPARE_DF = pd.DataFrame(
    [
        ["Highest Total Weight", 0.0000, 0.511704],
        ["Longest History", 0.0000, 0.034827],
        ["Most Stable", 0.0000, 0.000000],
        ["Most Volatile", 0.8412, 0.819422],
    ],
    columns=["Series", "Best classical", "Best deep"],
)

FINAL_COMPARE_CHART_DF = pd.DataFrame(
    [
        ["High weight", 0.0000, 0.511704],
        ["Long history", 0.0000, 0.034827],
        ["Stable", 0.0000, 0.000000],
        ["Volatile", 0.8412, 0.819422],
    ],
    columns=["Series", "Best classical", "Best deep"],
)

FIXED_VS_ADAPTIVE_DF = pd.DataFrame(
    [
        ["Longest History", 55, 169],
        ["Highest Total Weight", 7, 124],
        ["Most Volatile", 131, 129],
        ["Most Stable", 7, 136],
    ],
    columns=["Series", "Fixed-cutoff train", "Adaptive train"],
)


class DeckBuilder:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.slide_no = 0
        self.safe_rect = Rect(SAFE_MARGIN_X, SAFE_MARGIN_BOTTOM, 1 - 2 * SAFE_MARGIN_X, SAFE_TOP - SAFE_MARGIN_BOTTOM)
        self._content_rect: Rect | None = None
        self._validation_targets: list[ValidationTarget] = []
        self._fig: plt.Figure | None = None

    def _reset_validation(self) -> None:
        self._validation_targets = []

    def _register(self, label: str, artist: object, bounds: Rect) -> None:
        self._validation_targets.append(ValidationTarget(label=label, artist=artist, bounds=bounds))

    def _artist_bbox(self, fig: plt.Figure, artist: object) -> Bbox:
        renderer = fig.canvas.get_renderer()
        if hasattr(artist, "get_window_extent"):
            window_bbox = artist.get_window_extent(renderer)
            return fig.transFigure.inverted().transform_bbox(window_bbox)
        raise TypeError(f"Artist {artist!r} does not expose get_window_extent")

    def _bbox_inside(self, inner: Bbox, outer: Rect, tol: float = 0.003) -> bool:
        return (
            inner.x0 >= outer.x - tol
            and inner.x1 <= outer.right + tol
            and inner.y0 >= outer.y - tol
            and inner.y1 <= outer.top + tol
        )

    def _wrap_width(self, rect: Rect, font_size: float) -> int:
        chars_per_in = 10.8 * (12.0 / max(font_size, 1.0))
        return max(12, int(rect.w * SLIDE_W * chars_per_in))

    def fit_text_block(
        self,
        fig: plt.Figure,
        rect: Rect,
        text: str,
        *,
        max_size: float = 12.0,
        min_size: float = MIN_BODY_FONT,
        weight: str = "normal",
        color: str = INK,
        ha: str = "left",
        va: str = "top",
        linespacing: float = 1.25,
        wrap: bool = True,
        max_lines: int | None = None,
    ):
        if rect.w <= 0 or rect.h <= 0:
            raise LayoutValidationError(f"Invalid text rect: {rect}")

        size = max_size
        while size >= min_size - 1e-9:
            wrapped = wrap_paragraphs(text, self._wrap_width(rect, size)) if wrap else text
            line_count = wrapped.count("\n") + 1
            if max_lines is not None and line_count > max_lines:
                size -= 0.5
                continue

            x = rect.x if ha == "left" else rect.center_x if ha == "center" else rect.right
            y = rect.top if va == "top" else rect.center_y if va == "center" else rect.y
            artist = fig.text(
                x,
                y,
                wrapped,
                fontsize=size,
                weight=weight,
                color=color,
                ha=ha,
                va=va,
                linespacing=linespacing,
            )
            fig.canvas.draw()
            bbox = self._artist_bbox(fig, artist)
            if self._bbox_inside(bbox, rect):
                self._register("text", artist, rect)
                return artist
            artist.remove()
            size -= 0.5

        raise LayoutValidationError(f"Text did not fit inside {rect}: {text[:80]!r}")

    def fit_title(self, fig: plt.Figure, rect: Rect, text: str, *, color: str = INK) -> None:
        self.fit_text_block(
            fig,
            rect,
            text,
            max_size=24,
            min_size=17,
            weight="bold",
            color=color,
            linespacing=1.05,
            max_lines=2,
        )

    def place_card(
        self,
        fig: plt.Figure,
        rect: Rect,
        *,
        facecolor: str | tuple[float, float, float] = CARD,
        edgecolor: str = BORDER,
        linewidth: float = 1.0,
        radius: float = CARD_RADIUS,
    ) -> None:
        patch = FancyBboxPatch(
            (rect.x, rect.y),
            rect.w,
            rect.h,
            transform=fig.transFigure,
            boxstyle=f"round,pad=0.006,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=0.2,
        )
        fig.add_artist(patch)

    def _card_text_regions(self, rect: Rect, title_lines: int = 1) -> tuple[Rect, Rect]:
        inner = rect.inset(CARD_PAD, CARD_PAD)
        title_h = 0.05 + 0.012 * max(title_lines - 1, 0)
        title_rect = Rect(inner.x, inner.top - title_h, inner.w, title_h)
        body_rect = Rect(inner.x, inner.y, inner.w, inner.h - title_h - 0.01)
        return title_rect, body_rect

    def new_slide(self, title: str, section: str, takeaway: str) -> plt.Figure:
        self.slide_no += 1
        self._reset_validation()
        color = SECTION_COLORS[section]

        fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=180)
        self._fig = fig
        fig.patch.set_facecolor(BG)

        fig.add_artist(
            Rectangle((0, 1 - HEADER_H), 1, HEADER_H, transform=fig.transFigure, facecolor=color, edgecolor="none")
        )
        section_text = fig.text(0.045, 0.963, section.upper(), fontsize=10, color="white", weight="bold", va="center")
        count_text = fig.text(0.955, 0.963, f"{self.slide_no:02d} / {TOTAL_SLIDES}", fontsize=10, color="white", ha="right", va="center")
        self._register("header-section", section_text, Rect(0.03, 0.94, 0.20, 0.05))
        self._register("header-count", count_text, Rect(0.85, 0.94, 0.12, 0.05))

        title_rect = Rect(self.safe_rect.x, SAFE_TOP - TITLE_H, self.safe_rect.w, TITLE_H)
        self.fit_title(fig, title_rect, title)

        takeaway_rect = Rect(self.safe_rect.x, SAFE_TOP - TITLE_H - TITLE_GAP - TAKEAWAY_H, self.safe_rect.w, TAKEAWAY_H)
        self.place_card(fig, takeaway_rect, facecolor=blend(color, alpha=0.84), edgecolor="none", radius=0.018)
        label_rect = Rect(takeaway_rect.x + 0.016, takeaway_rect.y + 0.02, 0.11, takeaway_rect.h - 0.04)
        body_rect = Rect(takeaway_rect.x + 0.11, takeaway_rect.y + 0.018, takeaway_rect.w - 0.13, takeaway_rect.h - 0.036)
        self.fit_text_block(fig, label_rect, "Takeaway", max_size=10.5, min_size=10, weight="bold", color=color, wrap=False, va="center")
        self.fit_text_block(fig, body_rect, takeaway, max_size=12, min_size=10.2, color=INK, va="center")

        content_top = takeaway_rect.y - SECTION_GAP
        self._content_rect = Rect(self.safe_rect.x, self.safe_rect.y, self.safe_rect.w, content_top - self.safe_rect.y)
        return fig

    def template_cover(self) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        left, right = self._content_rect.split_cols([0.63, 0.37])
        left_top, left_bottom = left.split_rows([0.56, 0.44])
        stats_row, narrative = left_bottom.split_rows([0.34, 0.66])
        s1, s2, s3, s4 = stats_row.split_cols([1, 1, 1, 1], gutter=0.012)
        r1, r2 = right.split_rows([0.54, 0.46])
        return SlideTemplate(
            name="cover",
            content=self._content_rect,
            slots={
                "hero": left_top,
                "stat1": s1,
                "stat2": s2,
                "stat3": s3,
                "stat4": s4,
                "narrative": narrative,
                "side_top": r1,
                "side_bottom": r2,
            },
        )

    def template_2_col_chart_sidebar(self, *, sidebar_side: str = "right", main_fraction: float = 0.68) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        fractions = [main_fraction, 1 - main_fraction]
        left, right = self._content_rect.split_cols(fractions)
        main = left if sidebar_side == "right" else right
        sidebar = right if sidebar_side == "right" else left
        side1, side2, side3 = sidebar.split_rows([0.36, 0.32, 0.32])
        return SlideTemplate(
            name="2_col_chart_sidebar",
            content=self._content_rect,
            slots={
                "main": main,
                "sidebar": sidebar,
                "side1": side1,
                "side2": side2,
                "side3": side3,
            },
        )

    def template_chart_dominant_sidebar(self, *, sidebar_side: str = "right", main_fraction: float = 0.72) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        fractions = [main_fraction, 1 - main_fraction]
        left, right = self._content_rect.split_cols(fractions)
        main = left if sidebar_side == "right" else right
        sidebar = right if sidebar_side == "right" else left
        side1, side2, side3 = sidebar.split_rows([0.28, 0.27, 0.45])
        return SlideTemplate(
            name="chart_dominant_sidebar",
            content=self._content_rect,
            slots={
                "main": main,
                "sidebar": sidebar,
                "side1": side1,
                "side2": side2,
                "side3": side3,
            },
        )

    def template_table_sidebar(self, *, main_fraction: float = 0.64) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        table_rect, sidebar = self._content_rect.split_cols([main_fraction, 1 - main_fraction])
        side1, side2, side3 = sidebar.split_rows([0.38, 0.30, 0.32])
        return SlideTemplate(
            name="table_sidebar",
            content=self._content_rect,
            slots={
                "main": table_rect,
                "sidebar": sidebar,
                "side1": side1,
                "side2": side2,
                "side3": side3,
            },
        )

    def template_full_width_figure(self, *, figure_fraction: float = 0.78) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        figure, note = self._content_rect.split_rows([figure_fraction, 1 - figure_fraction])
        return SlideTemplate(name="full_width_figure", content=self._content_rect, slots={"figure": figure, "note": note})

    def template_chart_full_width(self, *, figure_fraction: float = 0.84) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        figure, note = self._content_rect.split_rows([figure_fraction, 1 - figure_fraction])
        return SlideTemplate(name="chart_full_width", content=self._content_rect, slots={"figure": figure, "note": note})

    def template_dashboard_2x2(self) -> SlideTemplate:
        if self._content_rect is None:
            raise RuntimeError("No active slide")
        top, bottom = self._content_rect.split_rows([1, 1])
        tl, tr = top.split_cols([1, 1])
        bl, br = bottom.split_cols([1, 1])
        return SlideTemplate(
            name="dashboard_2x2",
            content=self._content_rect,
            slots={"tl": tl, "tr": tr, "bl": bl, "br": br},
        )

    def text_box(
        self,
        fig: plt.Figure,
        rect: Rect,
        *,
        title: str,
        body: str,
        title_size: float = 12,
        body_size: float = 10.6,
        body_wrap: bool = True,
        body_min_size: float = MIN_BODY_FONT,
    ) -> None:
        self.place_card(fig, rect)
        title_rect, body_rect = self._card_text_regions(rect, title_lines=2)
        self.fit_text_block(fig, title_rect, title, max_size=title_size, min_size=10.3, weight="bold", max_lines=2)
        self.fit_text_block(fig, body_rect, body, max_size=body_size, min_size=body_min_size, wrap=body_wrap)

    def bullet_box(
        self,
        fig: plt.Figure,
        rect: Rect,
        *,
        title: str,
        bullets: list[str],
        body_size: float = 10.2,
        body_min_size: float = MIN_BODY_FONT,
    ) -> None:
        self.place_card(fig, rect)
        title_rect, body_rect = self._card_text_regions(rect, title_lines=2)
        self.fit_text_block(fig, title_rect, title, max_size=12, min_size=10.3, weight="bold", max_lines=2)
        self.fit_text_block(fig, body_rect, bullet_text(bullets), max_size=body_size, min_size=body_min_size)

    def stat_card(self, fig: plt.Figure, rect: Rect, label: str, value: str, *, accent: str) -> None:
        self.place_card(fig, rect, facecolor=blend(accent, alpha=0.88), edgecolor="none")
        inner = rect.inset(CARD_PAD, CARD_PAD)
        label_h = min(0.026, max(0.018, inner.h * 0.34))
        label_rect = Rect(inner.x, inner.top - label_h, inner.w, label_h)
        value_rect = Rect(inner.x, inner.y + 0.002, inner.w, inner.h - label_h - 0.004)
        self.fit_text_block(fig, label_rect, label.upper(), max_size=8.9, min_size=8.0, weight="bold", color=accent, wrap=False)
        self.fit_text_block(fig, value_rect, value, max_size=17, min_size=10.5, weight="bold", color=INK, wrap=False, va="center")

    def image_box(
        self,
        fig: plt.Figure,
        asset: AssetSpec,
        rect: Rect,
        *,
        title: str | None = None,
        fit: str | None = None,
        focus: tuple[float, float] = (0.5, 0.5),
    ) -> None:
        resolved_fit = fit or asset.fit_mode
        if asset.kind in {"chart", "multi_panel_chart"} and resolved_fit == "cover" and not asset.allow_cover:
            raise LayoutValidationError(f"Chart asset {asset.key} may not use cover fit")

        self.place_card(fig, rect)
        inner_pad = 0.010 if asset.kind in {"chart", "multi_panel_chart"} else CARD_PAD
        inner = rect.inset(inner_pad, inner_pad)
        if title:
            title_rect = Rect(inner.x, inner.top - 0.045, inner.w, 0.04)
            self.fit_text_block(fig, title_rect, title, max_size=11.5, min_size=10, weight="bold", max_lines=2)
            image_rect = Rect(inner.x, inner.y, inner.w, inner.h - 0.050)
        else:
            image_rect = inner

        arr = image_array(asset.path, trim_mode=asset.trim_mode, manual_crop=asset.manual_crop)
        if resolved_fit == "cover":
            arr = crop_to_aspect(arr, image_rect.w / image_rect.h)
        elif resolved_fit == "crop_focus":
            arr = crop_to_aspect(arr, image_rect.w / image_rect.h, focus=asset.focus if asset.focus != (0.5, 0.5) else focus)

        ax = fig.add_axes([image_rect.x, image_rect.y, image_rect.w, image_rect.h])
        ax.set_zorder(3)
        ax.axis("off")
        ax.set_facecolor(CARD)
        if resolved_fit in {"cover", "crop_focus"}:
            ax.imshow(arr, aspect="auto")
        else:
            ax.imshow(arr)
        ax.set_anchor("C")

    def fit_table(
        self,
        fig: plt.Figure,
        rect: Rect,
        df: pd.DataFrame,
        *,
        title: str,
        max_font: float = 9.4,
        min_font: float = 7.5,
        header_font: float | None = None,
    ) -> None:
        self.place_card(fig, rect)
        inner = rect.inset(CARD_PAD, CARD_PAD)
        title_rect = Rect(inner.x, inner.top - 0.045, inner.w, 0.04)
        self.fit_text_block(fig, title_rect, title, max_size=12, min_size=10.3, weight="bold", max_lines=2)
        table_rect = Rect(inner.x, inner.y, inner.w, inner.h - 0.055)
        ax = fig.add_axes([table_rect.x, table_rect.y, table_rect.w, table_rect.h])
        ax.set_axis_off()
        ax.set_zorder(3)
        header_font = header_font or max_font

        col_lengths = []
        for col in df.columns:
            values = [str(col)] + [str(v) for v in df[col].tolist()]
            col_lengths.append(max(len(v) for v in values))
        total_len = sum(col_lengths)
        col_widths = [length / total_len for length in col_lengths]

        font = max_font
        scale_y = 1.26
        final_table: Table | None = None
        while font >= min_font - 1e-9:
            ax.clear()
            ax.set_axis_off()
            table = ax.table(
                cellText=df.values.tolist(),
                colLabels=df.columns.tolist(),
                cellLoc="left",
                colLoc="left",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font)
            table.scale(1, scale_y)
            cells = table.get_celld()
            for (r, c), cell in cells.items():
                cell.set_edgecolor(BORDER)
                cell.set_linewidth(0.7)
                cell.set_width(col_widths[c])
                if r == 0:
                    cell.set_facecolor(blend("#D3DEE8", alpha=0.15))
                    cell.set_text_props(weight="bold", color=INK, fontsize=header_font)
                else:
                    cell.set_facecolor(CARD)
                    cell.set_text_props(color=INK, fontsize=font)

            fig.canvas.draw()
            fits = True
            for cell in cells.values():
                bbox = self._artist_bbox(fig, cell.get_text())
                if not self._bbox_inside(bbox, table_rect, tol=0.001):
                    fits = False
                    break
            if fits:
                final_table = table
                break
            font -= 0.4
            scale_y -= 0.04

        if final_table is None:
            raise LayoutValidationError(f"Table {title!r} did not fit inside {rect}")

        for idx, cell in final_table.get_celld().items():
            self._register(f"table-cell-{idx}", cell.get_text(), table_rect)

    def bar_chart_box(
        self,
        fig: plt.Figure,
        rect: Rect,
        df: pd.DataFrame,
        *,
        title: str,
        label_col: str,
        value_col: str,
        color: str,
        horizontal: bool = True,
        value_fmt: str = "{:.3f}",
        xlabel: str | None = None,
    ) -> None:
        self.place_card(fig, rect)
        inner = rect.inset(CARD_PAD, CARD_PAD)
        title_rect = Rect(inner.x, inner.top - 0.045, inner.w, 0.04)
        self.fit_text_block(fig, title_rect, title, max_size=12, min_size=10.3, weight="bold", max_lines=2)
        plot_rect = Rect(inner.x + 0.01, inner.y + 0.005, inner.w - 0.02, inner.h - 0.065)
        ax = fig.add_axes([plot_rect.x, plot_rect.y, plot_rect.w, plot_rect.h])
        ax.set_zorder(3)
        ax.set_facecolor(CARD)
        ax.grid(axis="x" if horizontal else "y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=MIN_CHART_LABEL)

        if horizontal:
            order = df.iloc[::-1]
            ax.barh(order[label_col], order[value_col], color=color, alpha=0.92)
            for idx, (_, row) in enumerate(order.iterrows()):
                ax.text(row[value_col], idx, f" {value_fmt.format(row[value_col])}", va="center", fontsize=8.2, color=INK)
            if xlabel:
                ax.set_xlabel(xlabel, color=MUTED)
        else:
            ax.bar(df[label_col], df[value_col], color=color, alpha=0.92)
            ax.tick_params(axis="x", rotation=16)
            for idx, (_, row) in enumerate(df.iterrows()):
                ax.text(idx, row[value_col], value_fmt.format(row[value_col]), ha="center", va="bottom", fontsize=8.1, color=INK)
            if xlabel:
                ax.set_ylabel(xlabel, color=MUTED)

    def grouped_bar_box(
        self,
        fig: plt.Figure,
        rect: Rect,
        df: pd.DataFrame,
        *,
        title: str,
        label_col: str,
        series_cols: list[str],
        colors: list[str],
        legend_labels: list[str],
    ) -> None:
        self.place_card(fig, rect)
        inner = rect.inset(CARD_PAD, CARD_PAD)
        title_rect = Rect(inner.x, inner.top - 0.045, inner.w, 0.04)
        self.fit_text_block(fig, title_rect, title, max_size=12, min_size=10.3, weight="bold", max_lines=2)
        plot_rect = Rect(inner.x + 0.005, inner.y + 0.005, inner.w - 0.01, inner.h - 0.065)
        ax = fig.add_axes([plot_rect.x, plot_rect.y, plot_rect.w, plot_rect.h])
        ax.set_zorder(3)
        ax.set_facecolor(CARD)
        positions = np.arange(len(df))
        width = 0.34
        shifts = np.linspace(-width / 2, width / 2, num=len(series_cols))
        for idx, col in enumerate(series_cols):
            ax.bar(positions + shifts[idx], df[col], width=width, color=colors[idx], alpha=0.92, label=legend_labels[idx])
        ax.set_xticks(positions)
        ax.set_xticklabels(df[label_col], rotation=15, ha="right")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=8.3)
        ax.tick_params(colors=MUTED, labelsize=MIN_CHART_LABEL)
        for spine in ax.spines.values():
            spine.set_visible(False)

    def validate_slide_layout(self, fig: plt.Figure) -> None:
        fig.canvas.draw()
        for target in self._validation_targets:
            bbox = self._artist_bbox(fig, target.artist)
            if not self._bbox_inside(bbox, target.bounds):
                raise LayoutValidationError(f"{target.label} overflowed bounds {target.bounds}: {bbox}")
            if target.label.startswith("header-"):
                continue
            if not self._bbox_inside(bbox, self.safe_rect, tol=0.005):
                raise LayoutValidationError(f"{target.label} escaped safe area: {bbox}")

    def save(self, fig: plt.Figure) -> None:
        self.validate_slide_layout(fig)
        self.pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
        self._fig = None


def draw_overview_slides(builder: DeckBuilder, assets: dict[str, Path]) -> None:
    fig = builder.new_slide(
        "Kaggle Hedge Fund Panel Forecasting",
        "Overview",
        "This deck rebuilds the project as one professional, chart-first story: panel structure, diagnostics, classical baselines, deep models, and an honest comparison at the end.",
    )
    template = builder.template_cover()
    builder.place_card(fig, template.slots["hero"], facecolor=blend(SECTION_COLORS["Overview"], alpha=0.91), edgecolor="none")
    hero_inner = template.slots["hero"].inset(0.024)
    builder.fit_text_block(
        fig,
        Rect(hero_inner.x, hero_inner.top - 0.12, hero_inner.w, 0.10),
        "Forecasting anonymized hedge-fund series under a weighted multi-series evaluation metric.",
        max_size=24,
        min_size=17,
        weight="bold",
        max_lines=3,
    )
    builder.fit_text_block(
        fig,
        Rect(hero_inner.x, hero_inner.y + 0.02, hero_inner.w, hero_inner.h - 0.14),
        "The project combines exploratory analysis, stationarity testing, representative-series classical modeling, and adaptive-split recurrent networks. The goal is not just to score models, but to explain when each family succeeds, where the data limits matter, and why some comparisons are methodologically constrained.",
        max_size=12.4,
        min_size=10.0,
    )
    builder.stat_card(fig, template.slots["stat1"], "Train rows", fmt_int(DATASET_COUNTS["train_rows"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, template.slots["stat2"], "Test rows", fmt_int(DATASET_COUNTS["test_rows"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, template.slots["stat3"], "Features", str(DATASET_COUNTS["feature_count"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, template.slots["stat4"], "Horizons", "1 · 3 · 10 · 25", accent=SECTION_COLORS["Overview"])
    builder.text_box(
        fig,
        template.slots["narrative"],
        title="Project angle",
        body="The strongest part of the work is the diagnostic chain: weight concentration, cutoff feasibility, ADF/KPSS, differencing, and lag structure all feed directly into how the classical and deep studies are framed.",
        body_size=10.6,
    )
    builder.bullet_box(
        fig,
        template.slots["side_top"],
        title="Main questions",
        bullets=[
            "What does the hedge-fund panel look like before any modeling?",
            "Which stationarity and lag diagnostics materially change model choice?",
            "How do classical and deep models behave under realistic train-window constraints?",
        ],
        body_size=10.0,
    )
    builder.bullet_box(
        fig,
        template.slots["side_bottom"],
        title="Deck promises",
        bullets=[
            "No appendix: all major plot families stay in the main flow.",
            "Every slide has one explicit message, not just a pasted notebook figure.",
            "The final comparison keeps the methodological caveat that deep models use adaptive splits.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Agenda",
        "Overview",
        "The talk moves from the panel and metric, to diagnostics, to model families, and ends with a direct comparison plus project limitations.",
    )
    flow_card = builder._content_rect
    assert flow_card is not None
    upper, lower = flow_card.split_rows([0.58, 0.42])
    builder.place_card(fig, upper)
    steps = [
        "Problem",
        "Data",
        "EDA",
        "Classical",
        "Deep",
        "Compare",
        "Close",
    ]
    xs = np.linspace(upper.x + 0.075, upper.right - 0.075, len(steps))
    y = upper.center_y
    for idx, (x, label) in enumerate(zip(xs, steps), start=1):
        if idx < len(steps):
            fig.add_artist(
                FancyArrowPatch(
                    (x + 0.045, y),
                    (xs[idx] - 0.045, y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2.0,
                    color=blend(SECTION_COLORS["Overview"], alpha=0.20),
                )
            )
        fig.add_artist(Circle((x, y), 0.037, transform=fig.transFigure, facecolor=blend(SECTION_COLORS["Overview"], alpha=0.88), edgecolor="none"))
        num_text = fig.text(x, y + 0.006, str(idx), ha="center", va="center", fontsize=15, color=INK, weight="bold")
        label_text = fig.text(x, y - 0.057, label, ha="center", va="center", fontsize=10.5, color=INK, weight="bold")
        builder._register("agenda-number", num_text, Rect(x - 0.03, y - 0.02, 0.06, 0.04))
        builder._register("agenda-label", label_text, Rect(x - 0.05, y - 0.09, 0.10, 0.04))
    left_note, right_note = lower.split_cols([1, 1])
    builder.bullet_box(
        fig,
        left_note,
        title="Presentation logic",
        bullets=[
            "Use diagnostics to justify model choices rather than treating EDA as decoration.",
            "Keep all major plots visible, but anchor each one to a single explanatory message.",
        ],
        body_size=10.0,
    )
    builder.bullet_box(
        fig,
        right_note,
        title="What to watch for",
        bullets=[
            "Weight concentration changes how model quality should be read.",
            "The fixed cutoff is useful but not universally feasible across the panel.",
            "The final comparison is informative, not perfectly apples-to-apples.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Problem Statement",
        "Overview",
        "This is a weighted panel-forecasting problem, not a single-series curve-fitting exercise, so series identity, time order, and metric concentration all matter together.",
    )
    template = builder.template_dashboard_2x2()
    builder.text_box(
        fig,
        template.slots["tl"],
        title="What is forecast",
        body="The target y_target must be predicted over future ts_index values for many distinct series defined by code, sub_code, sub_category, and horizon.",
    )
    builder.text_box(
        fig,
        template.slots["tr"],
        title="Why it is hard",
        body="The panel is heterogeneous, heavy-tailed, and only a minority of series cleanly supports the fixed cutoff used for validation in the EDA and classical notebooks.",
    )
    builder.text_box(
        fig,
        template.slots["bl"],
        title="What counts as success",
        body="A useful model improves the weighted skill score on future data without breaking chronology, leaking information, or overclaiming from tiny train segments.",
    )
    builder.bullet_box(
        fig,
        template.slots["br"],
        title="Project structure",
        bullets=[
            "EDA explains the panel before any model comparison.",
            "Classical models are tested on four representative series.",
            "Deep models use adaptive splits only when fixed-cutoff training becomes absurdly short.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Dataset Overview",
        "Overview",
        "The forecasting target lives inside a large anonymized panel: 5.34M train rows, 1.45M test rows, 86 engineered features, and four horizons.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.46, 0.54])
    builder.fit_table(fig, left, SCHEMA_DF, title="Core schema", max_font=10.2, min_font=8.6)
    top_stats, bottom_note = right.split_rows([0.64, 0.36])
    s1, s2 = top_stats.split_rows([0.5, 0.5])
    row1a, row1b = s1.split_cols([1, 1])
    row2a, row2b = s2.split_cols([1, 1])
    builder.stat_card(fig, row1a, "Train rows", fmt_int(DATASET_COUNTS["train_rows"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, row1b, "Test rows", fmt_int(DATASET_COUNTS["test_rows"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, row2a, "Feature count", str(DATASET_COUNTS["feature_count"]), accent=SECTION_COLORS["Overview"])
    builder.stat_card(fig, row2b, "Codes", str(DATASET_COUNTS["codes"]), accent=SECTION_COLORS["Overview"])
    builder.text_box(
        fig,
        bottom_note,
        title="Panel scale",
        body="Train covers ts_index 1–3601 and test continues from 3602–4376. The panel spans 180 sub-codes in train, 47 in test, 5 sub-categories, and 4 horizons.",
        body_size=10.7,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Train vs Test Comparability",
        "Overview",
        "Train and test share the same schema and category families, but test uses a narrower sub-code universe, so the main risk is coverage shift rather than schema mismatch.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], COMPARABILITY_DF, title="Categorical support check", max_font=10.1, min_font=8.8)
    subcode_df = pd.DataFrame([["Train", 180], ["Test", 47]], columns=["Split", "Sub-codes"])
    builder.bar_chart_box(
        fig,
        template.slots["side1"],
        subcode_df,
        title="Sub-code universe",
        label_col="Split",
        value_col="Sub-codes",
        color=SECTION_COLORS["Overview"],
        horizontal=False,
        value_fmt="{:.0f}",
        xlabel="Count",
    )
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="What this means",
        bullets=[
            "Train-only fields are exactly y_target and weight, which is expected.",
            "Every category level seen in test already exists in train, so there is no true cold-start schema issue.",
            "The narrower test support still matters because validation and generalization should respect that coverage shift.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Validation Strategy",
        "Overview",
        "Because test is a later suffix in time, validation must also be chronological; random cross-validation would leak future information.",
    )
    content = builder._content_rect
    assert content is not None
    timeline, note = content.split_rows([0.68, 0.32])
    builder.place_card(fig, timeline)
    title_rect = Rect(timeline.x + 0.02, timeline.top - 0.06, timeline.w - 0.04, 0.045)
    builder.fit_text_block(fig, title_rect, "Timeline view of the forecasting problem", max_size=12, min_size=10.5, weight="bold", max_lines=1)
    bar_rect = Rect(timeline.x + 0.05, timeline.y + 0.18, timeline.w - 0.10, 0.11)
    fig.add_artist(Rectangle((bar_rect.x, bar_rect.y), bar_rect.w * 0.64, bar_rect.h, transform=fig.transFigure, facecolor=blend("#6D8BA2", alpha=0.25), edgecolor="none"))
    fig.add_artist(Rectangle((bar_rect.x + bar_rect.w * 0.64, bar_rect.y), bar_rect.w * 0.22, bar_rect.h, transform=fig.transFigure, facecolor=blend("#D08C60", alpha=0.30), edgecolor="none"))
    fig.add_artist(Rectangle((bar_rect.x + bar_rect.w * 0.86, bar_rect.y), bar_rect.w * 0.14, bar_rect.h, transform=fig.transFigure, facecolor=blend("#8AA67A", alpha=0.30), edgecolor="none"))
    for text, x in [
        ("Train before cutoff\n(ts_index 1 to 2880)", bar_rect.x + bar_rect.w * 0.32),
        ("Post-cutoff validation\n(2881 to 3601)", bar_rect.x + bar_rect.w * 0.75),
        ("Held-out test\n(3602 to 4376)", bar_rect.x + bar_rect.w * 0.93),
    ]:
        artist = fig.text(x, bar_rect.y + bar_rect.h / 2, text, ha="center", va="center", fontsize=11, color=INK, weight="bold")
        builder._register("timeline-label", artist, Rect(x - 0.12, bar_rect.y - 0.02, 0.24, bar_rect.h + 0.04))
    left_note, right_note = note.split_cols([1, 1])
    builder.bullet_box(
        fig,
        left_note,
        title="Why the split matters",
        bullets=[
            "The fixed cutoff is the shared boundary used throughout EDA and the classical study.",
            "Shuffled folds would mix future information into earlier states and overstate performance.",
        ],
        body_size=10.0,
    )
    builder.bullet_box(
        fig,
        right_note,
        title="What changes later",
        bullets=[
            "The deep notebook keeps chronology but moves to adaptive per-series splits.",
            "That change is methodological, not cosmetic, because some chosen series have only seven pre-cutoff points.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Metric Math: Weighted Skill Score",
        "Overview",
        "All model discussion is anchored to the weighted skill score, because weight concentration makes unweighted RMSE fundamentally misleading.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.58, 0.42])
    builder.text_box(
        fig,
        left,
        title="Primary score used in the project",
        body=r"$\mathrm{score}=\sqrt{1-\mathrm{clip}_{0,1}\left(\frac{\sum_i w_i(y_i-\hat y_i)^2}{\sum_i w_i y_i^2}\right)}$",
        body_size=23,
        body_wrap=False,
        body_min_size=16,
    )
    upper, lower = right.split_rows([0.58, 0.42])
    builder.bullet_box(
        fig,
        upper,
        title="Interpretation",
        bullets=[
            "1.0 is perfect forecasting.",
            "0.0 means no gain beyond the clipped baseline threshold.",
            "Large weights can dominate the score even if most rows are easy.",
        ],
        body_size=10.2,
    )
    builder.text_box(
        fig,
        lower,
        title="Why it changes decisions",
        body="The score rewards getting high-weight rows right. That is why the EDA treats weight concentration as a modeling issue, not just a descriptive statistic.",
        body_size=10.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Libraries and Workflow",
        "Overview",
        "The project uses a compact Python stack: pandas and pyarrow for data access, matplotlib and seaborn for visuals, statsmodels for classical diagnostics, and torch for sequence models.",
    )
    content = builder._content_rect
    assert content is not None
    flow, details = content.split_rows([0.42, 0.58])
    builder.place_card(fig, flow)
    nodes = ["EDA notebook", "Classical study", "Deep study", "Comparison"]
    xs = np.linspace(flow.x + 0.13, flow.right - 0.13, len(nodes))
    for idx, (x, label) in enumerate(zip(xs, nodes)):
        node_rect = Rect(x - 0.08, flow.center_y - 0.04, 0.16, 0.08)
        builder.place_card(fig, node_rect, facecolor=blend(SECTION_COLORS["Overview"], alpha=0.86), edgecolor="none")
        builder.fit_text_block(fig, node_rect.inset(0.012), label, max_size=11, min_size=9.7, weight="bold", ha="center", va="center", max_lines=2)
        if idx < len(nodes) - 1:
            fig.add_artist(
                FancyArrowPatch(
                    (x + 0.09, flow.center_y),
                    (xs[idx + 1] - 0.09, flow.center_y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2,
                    color=SECTION_COLORS["Overview"],
                )
            )
    d1, d2, d3 = details.split_cols([1, 1, 1])
    builder.text_box(
        fig,
        d1,
        title="Data and IO",
        body="pandas, numpy, pyarrow, pathlib\n\nUsed for parquet loading, grouped summaries, window construction, and result tables.",
    )
    builder.text_box(
        fig,
        d2,
        title="Visualization and diagnostics",
        body="matplotlib, seaborn, statsmodels\n\nUsed for missingness views, ADF/KPSS, ACF/PACF, STL, and classical forecasting.",
    )
    builder.text_box(
        fig,
        d3,
        title="Deep learning",
        body="torch, torch.utils.data\n\nUsed for one-step window training, GRU/LSTM/RNN models, recursive forecasting, and training-curve tracking.",
    )
    builder.save(fig)


def draw_eda_slides(builder: DeckBuilder, assets: dict[str, Path]) -> None:
    fig = builder.new_slide(
        "Missing Values",
        "EDA",
        "Missingness exists in 48 columns, but the maximum missing rate is only about 12.47%, so the main issue is selective feature reliability rather than catastrophic data loss.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell11_img1"], template.slots["main"])
    builder.fit_table(fig, template.slots["side1"], MISSING_TOP_DF, title="Top missing columns", max_font=9.8, min_font=8.4)
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="Modeling implication",
        bullets=[
            "The missingness pattern should be monitored when choosing feature sets or imputations.",
            "The rates are non-trivial, but still low enough that a meaningful feature audit remains possible.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Target Distribution",
        "EDA",
        "The target is centered near zero but has extremely heavy tails, so outliers can dominate error-based metrics even when most observations are small.",
    )
    template = builder.template_chart_dominant_sidebar(main_fraction=0.72)
    builder.image_box(fig, assets["01_EDA_cell15_img1"], template.slots["main"])
    builder.text_box(
        fig,
        template.slots["side1"],
        title="Summary statistics",
        body="Median: -0.0006\nIQR: -0.1291 to 0.0511\n1% / 99%: -82.80 / 62.92\nRange: -2201.88 to 2314.41",
        body_wrap=False,
        body_size=10.7,
    )
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="What we learn",
        bullets=[
            "Robustness matters because tail events sit far outside the central mass.",
            "Simple means can be distorted by rare spikes.",
            "Visual evaluation still has to be tied back to weighted metrics.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Weight Distribution",
        "EDA",
        "A tiny fraction of rows carries most of the evaluation weight, so the competition metric and unweighted losses can tell very different stories.",
    )
    template = builder.template_chart_dominant_sidebar(main_fraction=0.72)
    builder.image_box(fig, assets["01_EDA_cell19_img1"], template.slots["main"])
    top_stats, upper_note, lower_note = template.slots["sidebar"].split_rows([0.24, 0.30, 0.46])
    s1, s2, s3 = top_stats.split_cols([1, 1, 1], gutter=0.012)
    builder.stat_card(fig, s1, "Top 1%", "64.19%", accent=SECTION_COLORS["EDA"])
    builder.stat_card(fig, s2, "Top 5%", "92.58%", accent=SECTION_COLORS["EDA"])
    builder.stat_card(fig, s3, "Top 10%", "98.33%", accent=SECTION_COLORS["EDA"])
    builder.text_box(
        fig,
        upper_note,
        title="Scale of concentration",
        body="Median weight is 1,699.38, while the maximum weight is 1.391e13. The upper tail is therefore far more extreme than the central tendency suggests.",
        body_size=10.4,
    )
    builder.bullet_box(
        fig,
        lower_note,
        title="Implication",
        bullets=[
            "High-weight rows dominate ranking outcomes.",
            "Any evaluation that ignores weights can recommend the wrong model family.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Series Coverage and Split Feasibility",
        "EDA",
        "Only about 5% of all series cleanly cross the validation cutoff, so cutoff-based evaluation is informative but not universally feasible across the panel.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell23_img1"], template.slots["main"])
    s_top, s_mid, s_bot = template.slots["side1"], template.slots["side2"], template.slots["side3"]
    stats_row = s_top
    stat1, stat2 = stats_row.split_cols([1, 1])
    builder.stat_card(fig, stat1, "Total series", fmt_int(36_923), accent=SECTION_COLORS["EDA"])
    builder.stat_card(fig, stat2, "Cross cutoff", fmt_int(1_930), accent=SECTION_COLORS["EDA"])
    builder.text_box(
        fig,
        s_mid,
        title="Key point",
        body="The notebook keeps only eligible series for later diagnostics by enforcing both minimum length and cutoff-crossing requirements.",
        body_size=10.4,
    )
    builder.bullet_box(
        fig,
        s_bot,
        title="Why it matters",
        bullets=[
            "Some model families simply do not have enough pre-cutoff data.",
            "Representative-series studies become a practical compromise.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Feature Audit",
        "EDA",
        "The 86 anonymized features still show measurable target relationships, but their anonymous naming means we can only reason statistically, not semantically.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell27_img1"], template.slots["main"])
    builder.fit_table(fig, template.slots["side1"].union(template.slots["side2"]), FEATURE_CORR_DF, title="Top correlations", max_font=9.2, min_font=8.0)
    builder.bullet_box(
        fig,
        template.slots["side3"],
        title="Interpretation",
        bullets=[
            "Useful signal exists, but it is modest and spread across many features.",
            "The audit guides later model choice more than any domain-level interpretation.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Representative-Series Selection Logic",
        "EDA",
        "The four chosen series were selected to span the panel's length, weight concentration, volatility, and stability extremes rather than to show only easy cases.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], CHOSEN_SERIES_DF, title="Chosen representative series", max_font=8.8, min_font=7.8)
    builder.text_box(
        fig,
        template.slots["side1"],
        title="Why these four",
        body="They capture four distinct conditions: plenty of history, huge weight, extreme variability, and near-flat behavior. That makes later visual comparisons interpretable and honest.",
        body_size=10.2,
    )
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="What each one stresses",
        bullets=[
            "Longest history tests persistent lag structure.",
            "Highest total weight tests metric-sensitive behavior.",
            "Most volatile tests spike handling.",
            "Most stable tests tiny-scale numerical accuracy.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Representative-Series Raw Plots I",
        "EDA",
        "The first two representative series already show the panel contrast: one has meaningful lag structure, while the high-weight series is tiny in scale and much harder to interpret visually.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(fig, assets["01_EDA_cell31_img1_top"], template.slots["figure"])
    builder.text_box(
        fig,
        template.slots["note"],
        title="Visual message",
        body="These two series show why train-window length and target scale matter before any model is fit.",
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Representative-Series Raw Plots II",
        "EDA",
        "The second pair completes the contrast: the volatile series has obvious structure and swings, while the most-stable series is close to a numerical-precision problem.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(fig, assets["01_EDA_cell31_img1_bottom"], template.slots["figure"])
    builder.text_box(
        fig,
        template.slots["note"],
        title="Visual message",
        body="This is why a single model story is hard to defend. The panel contains both visible structure and series that are almost numerically flat.",
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "ADF Test",
        "EDA",
        "ADF is the first stationarity gate: if many raw series fail it, classical models in levels should not be trusted without transformation.",
    )
    template = builder.template_dashboard_2x2()
    builder.text_box(
        fig,
        template.slots["tl"],
        title="Null and alternative",
        body=r"$H_0$: the series has a unit root (non-stationary)" "\n" r"$H_1$: the series is stationary" "\n\nReject $H_0$ when p-value < 0.05.",
        body_wrap=False,
        body_size=14.5,
        body_min_size=11.0,
    )
    builder.text_box(
        fig,
        template.slots["tr"],
        title="Why it matters here",
        body="ARIMA-style models assume a stable mean and autocorrelation structure after any differencing. Running ADF first avoids blindly treating trending series as stationary.",
        body_size=10.4,
    )
    builder.text_box(
        fig,
        template.slots["bl"],
        title="Notebook setup",
        body="One ADF test is run per eligible series, with all four series keys retained on each result row. The notebook samples 200 eligible series to keep panel-level conclusions readable and practical.",
        body_size=10.2,
    )
    builder.bullet_box(
        fig,
        template.slots["br"],
        title="Decision rule used later",
        bullets=[
            "If many raw series fail ADF, first differencing becomes the default repair step.",
            "If most pass, low-differencing classical models remain viable on the eligible subset.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "ADF Results",
        "EDA",
        "The sampled ADF results show mixed behavior rather than universal stationarity: about 69% pass at 5%, leaving a meaningful minority that still needs repair.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell35_img1"], template.slots["main"])
    stat_row = template.slots["side1"]
    st1, st2 = stat_row.split_cols([1, 1])
    builder.stat_card(fig, st1, "Sampled", "200", accent=SECTION_COLORS["EDA"])
    builder.stat_card(fig, st2, "Stationary", "69%", accent=SECTION_COLORS["EDA"])
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="Reading the figure",
        bullets=[
            "The p-value histogram has a strong left-of-threshold mass, but not a complete collapse toward zero.",
            "The by-horizon and by-code views show that stationarity is not uniform across the panel.",
            "That heterogeneity justifies a cautious, diagnostics-driven classical setup.",
        ],
        body_size=9.7,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "KPSS Test and Joint Verdict",
        "EDA",
        "KPSS complements ADF by flipping the null hypothesis; together the tests show that most sampled series are usable after mild transformation, but ambiguity remains.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], KPSS_VERDICT_DF, title="ADF + KPSS combined verdict", max_font=8.9, min_font=7.8)
    kpss_bar = pd.DataFrame([["Stationary", 64.0], ["Inconclusive", 25.5], ["Unit root", 10.5]], columns=["Verdict", "Share"])
    builder.bar_chart_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]),
        kpss_bar,
        title="Verdict shares",
        label_col="Verdict",
        value_col="Share",
        color=SECTION_COLORS["EDA"],
        horizontal=True,
        value_fmt="{:.1f}%",
        xlabel="Percent of sampled series",
    )
    builder.text_box(
        fig,
        template.slots["side3"],
        title="Interpretation",
        body="The stationary share dominates, but the inconclusive bucket is large enough that one-size-fits-all claims about raw levels would be unsafe.",
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "First-Differencing Analysis",
        "EDA",
        "First differencing is a strong repair step in this sample: the stationary share rises from 69% in levels to 100% after differencing.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.42, 0.58])
    diff_df = pd.DataFrame([["Levels (d=0)", 69.0], ["First difference (d=1)", 100.0]], columns=["Representation", "Stationary share"])
    builder.bar_chart_box(
        fig,
        left,
        diff_df,
        title="Before vs after differencing",
        label_col="Representation",
        value_col="Stationary share",
        color=SECTION_COLORS["EDA"],
        horizontal=False,
        value_fmt="{:.0f}%",
        xlabel="Percent stationary",
    )
    upper, lower = right.split_rows([0.56, 0.44])
    builder.text_box(
        fig,
        upper,
        title="Mathematical idea",
        body=r"First difference: $\Delta y_t = y_t - y_{t-1}$"
        "\n\nRemoves slow-moving level shifts."
        "\nStabilizes mean and variance for many series.",
        body_wrap=False,
        body_size=13.0,
        body_min_size=9.8,
    )
    builder.bullet_box(
        fig,
        lower,
        title="Why it matters later",
        bullets=[
            "ARIMA variants with d=1 become well motivated.",
            "Any level-based model on raw data should now be interpreted carefully when the series still trends.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Rolling Mean / Variance on Raw Series",
        "EDA",
        "The rolling statistics on raw levels visually confirm what the tests suggest: several chosen series do not maintain a stable mean or variance over time.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell44_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Pointwise reading",
        bullets=[
            "A drifting rolling mean indicates non-stationarity in level.",
            "A widening or narrowing rolling standard deviation indicates non-stationarity in scale.",
            "The volatile series is especially useful here because its instability is visible even without a formal test.",
        ],
        body_size=9.9,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Rolling Mean / Variance After First Differencing",
        "EDA",
        "After differencing, the rolling mean and variance flatten noticeably, which visually confirms the statistical improvement seen in the ADF results.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell47_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Pointwise reading",
        bullets=[
            "The mean now oscillates closer to zero instead of drifting.",
            "Variance becomes more stable across time windows.",
            "This is the visual behavior we want before fitting differenced classical models.",
        ],
        body_size=9.9,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "ACF / PACF for an Anchor Series",
        "EDA",
        "The anchor-series correlograms expose which lags matter directly and which reflect shorter-lag mediation, giving concrete hints for AR and MA order selection.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell50_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="How to read it",
        bullets=[
            "ACF shows correlation with lagged copies of the series.",
            "PACF isolates the direct contribution of each lag after controlling for shorter lags.",
            "A sharp PACF cutoff hints at AR order; a sharp ACF cutoff hints at MA order.",
        ],
        body_size=9.6,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Panel-Level ACF / PACF",
        "EDA",
        "Aggregating correlograms across sampled series shows what lag structure is typical for the panel, not just for one convenient example.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell53_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Modeling message",
        bullets=[
            "The panel-wide view prevents overfitting model orders to one series.",
            "Persistent early lags support low-order autoregressive structure.",
            "The lack of clean seasonal bumps keeps the focus on short-memory dynamics rather than heavy seasonality.",
        ],
        body_size=9.6,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "STL Decomposition",
        "EDA",
        "STL makes the time-series structure more interpretable by separating observed behavior into trend, seasonal, and residual components.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell56_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Interpretation",
        bullets=[
            "A sloped trend component matches the non-stationarity seen in earlier tests.",
            "A weak seasonal amplitude argues against building the whole solution around seasonality.",
            "Residual structure shows whether the decomposition captured enough of the series behavior.",
        ],
        body_size=9.6,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Cross-Series Relationships",
        "EDA",
        "The panel is not a bag of isolated series: shared behavior within codes suggests that pooled or global models can potentially borrow signal across related sub-series.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell59_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Why this matters",
        bullets=[
            "If related sub-series co-move, a global model can use pooled structure more effectively than isolated per-series fitting.",
            "This is especially relevant when individual series are short or noisy.",
            "It explains why the project later compares local classical models against pooled deep-learning behavior.",
        ],
        body_size=9.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Top-Feature Correlation Heatmap",
        "EDA",
        "The top features are not independent; several high-ranking variables are correlated with one another, so importance should be read with collinearity in mind.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["01_EDA_cell62_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Reading the heatmap",
        bullets=[
            "The first row and column anchor feature-to-target relationships.",
            "Strong feature-feature blocks suggest redundancy rather than independent signal sources.",
            "That limits naive marginal interpretation and supports careful model choice.",
        ],
        body_size=9.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Leakage and Split Sanity",
        "EDA",
        "The raw schema does not expose an obvious leakage path: the only train-only fields are the target and the weight, which is exactly what we expect.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], LEAKAGE_PARTITION_DF, title="Row counts by partition", max_font=10.1, min_font=8.8)
    builder.text_box(
        fig,
        template.slots["side1"],
        title="Safety checks performed",
        body="The notebook verifies train-only and test-only columns, confirms that y_target and weight are absent from test, and separates pre-cutoff from later rows to preserve temporal causality.",
        body_size=10.0,
    )
    builder.bullet_box(
        fig,
        template.slots["side2"].union(template.slots["side3"]),
        title="Practical conclusion",
        bullets=[
            "No schema-level leakage is obvious.",
            "The bigger challenge is respecting time order and cutoff feasibility, not hidden future columns.",
        ],
        body_size=10.0,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "EDA Synthesis",
        "EDA",
        "The EDA narrows the modeling space: respect time order, evaluate with weights, difference when needed, and be honest about series-length constraints.",
    )
    template = builder.template_dashboard_2x2()
    builder.text_box(fig, template.slots["tl"], title="Data split", body="Use chronological validation only; test is a true temporal suffix.")
    builder.text_box(fig, template.slots["tr"], title="Metric", body="Weighted skill score is non-negotiable because the weight distribution is extremely concentrated.")
    builder.text_box(fig, template.slots["bl"], title="Stationarity", body="Differencing repairs most non-stationarity; classical level models need caution.")
    builder.bullet_box(
        fig,
        template.slots["br"],
        title="Modeling consequences",
        bullets=[
            "Short-memory dynamics matter more than strong seasonality.",
            "Cross-series dependence suggests value in global models.",
            "Many series are too short for a strict cutoff study, which motivates adaptive deep splits.",
        ],
        body_size=9.6,
    )
    builder.save(fig)


def draw_classical_slides(builder: DeckBuilder, assets: dict[str, Path]) -> None:
    fig = builder.new_slide(
        "Classical Study Setup",
        "Classical",
        "The classical notebook stays intentionally small and interpretable by focusing on four representative series rather than pretending every series supports the same level-based experiment.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["02_classical_codex_cell05_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Why this design",
        bullets=[
            "The four chosen series already span the practical edge cases discovered in EDA.",
            "Showing raw history first makes later forecast behavior easier to interpret.",
            "A small honest study is better than hiding weak train windows inside aggregate tables.",
        ],
        body_size=9.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Model Lineup",
        "Classical",
        "The lineup mixes smoothing, autoregressive, moving-average, and differenced ARIMA families so we can see which level of structure each representative series supports.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], CLASSICAL_LINEUP_DF, title="Model families included", max_font=8.6, min_font=7.5)
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Reading the lineup",
        bullets=[
            "Smoothing models are cheap and robust on tiny histories.",
            "AR, MA, and ARMA need at least 20 training points, so some series are skipped by design.",
            "The ARIMA block tests whether first differencing helps in the volatile case.",
        ],
        body_size=9.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Model Math",
        "Classical",
        "The family formulas matter because the later forecast plots make sense only if we remember which models naturally flatten, trend, or react to shocks.",
    )
    content = builder._content_rect
    assert content is not None
    top, bottom = content.split_rows([0.62, 0.38])
    c1, c2, c3 = top.split_cols([1, 1, 1])
    builder.text_box(
        fig,
        c1,
        title="Rolling / SES / Holt",
        body=r"Rolling mean: $\hat y_{t+1}=\frac{1}{w}\sum_{i=0}^{w-1}y_{t-i}$" "\n\n" r"SES: $\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}$" "\n\nHolt adds a trend term on top of the level.",
        body_wrap=False,
        body_size=13.7,
        body_min_size=10.6,
    )
    builder.text_box(
        fig,
        c2,
        title="AR and MA",
        body=r"AR(p): $y_t=c+\sum_{i=1}^{p}\phi_i y_{t-i}+\varepsilon_t$" "\n\n" r"MA(q): $y_t=\mu+\varepsilon_t+\sum_{i=1}^{q}\theta_i \varepsilon_{t-i}$",
        body_wrap=False,
        body_size=13.7,
        body_min_size=10.6,
    )
    builder.text_box(
        fig,
        c3,
        title="ARMA / ARIMA",
        body="ARMA combines lagged values\nand shocks.\n\nARIMA(d=1) differences first,\nthen fits ARMA on the\ntransformed series.",
        body_wrap=False,
        body_size=11.0,
        body_min_size=8.8,
    )
    builder.bullet_box(
        fig,
        bottom,
        title="Visual consequence in later plots",
        bullets=[
            "Rolling mean and SES often become almost flat multi-step forecasts.",
            "Holt can extrapolate a straight trend line even when the true series bends.",
            "AR, MA, ARMA, and ARIMA still look simple when the train segment is short.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Evaluation Protocol",
        "Classical",
        "The cutoff zoom shows why evaluation is hard here: two of the four representative series offer only seven pre-cutoff training points, which immediately constrains model complexity.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["02_classical_codex_cell05_img2"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Protocol summary",
        bullets=[
            "Fit each model on ts_index <= 2880 for the chosen series.",
            "Forecast the entire post-cutoff segment in one multi-step pass.",
            "Report skill, RMSE, MAE, and MASE, and explicitly mark skipped fits.",
            "Keep skipped models in the study because train-length failure is itself a result.",
        ],
        body_size=9.2,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Per-Series Leaderboard",
        "Classical",
        "The best classical model depends heavily on series type; the volatile series is the only one where a differenced ARIMA variant achieves clearly positive skill.",
    )
    template = builder.template_table_sidebar()
    builder.fit_table(fig, template.slots["main"], CLASSICAL_BEST_DF, title="Best classical model by representative series", max_font=8.7, min_font=7.6)
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Reading the table",
        bullets=[
            "SES wins the weight-heavy and stable series because there is almost no shape to learn from seven points.",
            "ARMA(1,3) only edges out others on the longest-history series, and even there skill stays at zero.",
            "ARIMA(1,1,1) is the clear winner on the volatile series, matching the differencing story from EDA.",
        ],
        body_size=9.1,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Overall Model Comparison",
        "Classical",
        "Overall averages should be read cautiously, but they still show two themes: smoothing models always fit, and differenced ARIMA is the only complex family with strong upside on the volatile case.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.50, 0.50])
    builder.bar_chart_box(
        fig,
        left,
        CLASSICAL_SUMMARY_DF.head(6),
        title="Top models by mean skill",
        label_col="Model",
        value_col="Mean skill",
        color=SECTION_COLORS["Classical"],
        horizontal=True,
        value_fmt="{:.3f}",
        xlabel="Mean skill",
    )
    upper, lower = right.split_rows([0.58, 0.42])
    builder.image_box(fig, assets["02_classical_codex_cell11_img2"], upper, title="Fit status by model")
    builder.text_box(
        fig,
        lower,
        title="What the status chart adds",
        body="Smoothing models are fully feasible on all four series, while many AR, MA, ARMA, and ARIMA variants are skipped on the short-history cases because train_len < 20.",
        body_size=10.2,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Order Sensitivity and Parameter Patterns",
        "Classical",
        "Order selection matters, but the visual summaries confirm that tuning only helps when the train window is long enough to support it.",
    )
    template = builder.template_dashboard_2x2()
    top = template.slots["tl"].union(template.slots["tr"])
    builder.image_box(fig, assets["02_classical_codex_cell11_img1"], top, title="Per-series model-rank heatmap")
    builder.image_box(fig, assets["02_classical_codex_cell11_img3"], template.slots["bl"].union(template.slots["br"]), title="AR / MA order plots and ARMA heatmap")
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Forecast Overlays I",
        "Classical",
        "The top-half forecast overlays show two extremes: one moderately dynamic series where many models collapse toward similar shapes, and one nearly flat high-weight series where level models dominate by default.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(
        fig,
        assets["02_classical_codex_cell13_img1_top"],
        template.slots["figure"],
        title="Longest History and Highest Total Weight",
    )
    builder.text_box(
        fig,
        template.slots["note"],
        title="Interpretation",
        body="The first pair shows why train-window length and target scale matter. More history helps, but even the longest-history case does not automatically become easy.",
        body_size=10.4,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Forecast Overlays II",
        "Classical",
        "The bottom-half overlays make the key contrast obvious: the volatile series benefits from differencing, while the stable series offers so little movement that sophisticated models have almost nothing to exploit.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(
        fig,
        assets["02_classical_codex_cell13_img1_bottom"],
        template.slots["figure"],
        title="Most Volatile and Most Stable",
    )
    builder.text_box(
        fig,
        template.slots["note"],
        title="Interpretation",
        body="This pair is where ARIMA earns its keep. The volatile series has real structure to capture, while the stable series mostly becomes a precision problem.",
        body_size=10.4,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Validation Zoom I",
        "Classical",
        "The validation zoom removes long-history context and shows the actual forecasting challenge around the cutoff, where short-train problems become most visible.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(
        fig,
        assets["02_classical_codex_cell13_img2_top"],
        template.slots["figure"],
        title="Cutoff zoom: Longest History and Highest Total Weight",
    )
    builder.text_box(
        fig,
        template.slots["note"],
        title="Interpretation",
        body="The zoom emphasizes what the models actually saw and predicted near the evaluation window, not just how they look over the whole history.",
        body_size=10.4,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Validation Zoom II",
        "Classical",
        "The zoomed volatile and stable series show the real behavior: ARIMA tracks the volatile case better, while the stable case remains almost a numerical-precision exercise.",
    )
    template = builder.template_chart_full_width(figure_fraction=0.80)
    builder.image_box(
        fig,
        assets["02_classical_codex_cell13_img2_bottom"],
        template.slots["figure"],
        title="Cutoff zoom: Most Volatile and Most Stable",
    )
    builder.text_box(
        fig,
        template.slots["note"],
        title="Interpretation",
        body="The stable case can make many models look similar. The volatile case is where model family choice has a visibly meaningful effect.",
        body_size=10.4,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Best-Model Error Plots",
        "Classical",
        "Looking at error over time is more informative than a single scalar score, because it shows whether a model is consistently a bit wrong or badly wrong at a few decisive spikes.",
    )
    template = builder.template_2_col_chart_sidebar()
    builder.image_box(fig, assets["02_classical_codex_cell15_img1"], template.slots["main"])
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Pointwise reading",
        bullets=[
            "The volatile series has large concentrated error swings, which is why model choice matters there.",
            "The stable and high-weight series are numerically tiny, so even visually small deviations can matter under weighting.",
            "The longest-history series remains challenging despite having more pre-cutoff points.",
        ],
        body_size=9.4,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Issues Faced",
        "Classical",
        "The classical study is useful precisely because it exposes its limits: tiny train windows flatten forecasts, reduce feasible models, and cap the value of order tuning on some series.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.58, 0.42])
    builder.fit_table(fig, left, CLASSICAL_BEST_DF, title="Winning model and train/validation lengths", max_font=8.5, min_font=7.5)
    upper, lower = right.split_rows([0.48, 0.52])
    builder.image_box(fig, assets["02_classical_codex_cell17_img1"], upper, title="Best-model RMSE by series", fit="contain")
    builder.bullet_box(
        fig,
        lower,
        title="Main issues",
        bullets=[
            "Two representative series have only 7 pre-cutoff points, forcing most complex models to skip.",
            "Flat or straight forecasts are often a genuine model consequence, not a plotting bug.",
            "Fit feasibility itself becomes part of the result, not just an implementation nuisance.",
        ],
        body_size=9.6,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Classical Takeaways",
        "Classical",
        "There is no universal classical winner: smoothing is safest on short or near-flat series, while differenced ARIMA is the only family with strong upside on the volatile case.",
    )
    template = builder.template_dashboard_2x2()
    builder.text_box(fig, template.slots["tl"], title="Safe models", body="SES and rolling averages remain useful because they still behave sensibly when the training window is tiny.")
    builder.text_box(fig, template.slots["tr"], title="Where complexity helps", body="ARIMA(1,1,1) stands out on the volatile series because differencing actually addresses the underlying non-stationarity.")
    builder.text_box(fig, template.slots["bl"], title="What not to overclaim", body="More parameters do not rescue the high-weight and stable series when there is almost no pre-cutoff structure to estimate.")
    builder.bullet_box(
        fig,
        template.slots["br"],
        title="Transition to deep models",
        bullets=[
            "The cutoff problem motivates adaptive splits for recurrent models.",
            "That change aims to make training feasible without abandoning chronology.",
        ],
        body_size=9.8,
    )
    builder.save(fig)


def draw_deep_slides(builder: DeckBuilder, assets: dict[str, Path]) -> None:
    fig = builder.new_slide(
        "Deep-Study Motivation",
        "Deep",
        "The deep notebook changes the validation rule for a practical reason: the fixed cutoff leaves two chosen series with only seven training points, which is not enough for sequence modeling.",
    )
    template = builder.template_2_col_chart_sidebar(main_fraction=0.60)
    builder.grouped_bar_box(
        fig,
        template.slots["main"],
        FIXED_VS_ADAPTIVE_DF,
        title="Train length: fixed cutoff vs adaptive split",
        label_col="Series",
        series_cols=["Fixed-cutoff train", "Adaptive train"],
        colors=[blend("#D08C60", alpha=0.10), SECTION_COLORS["Deep"]],
        legend_labels=["Fixed cutoff", "Adaptive split"],
    )
    builder.bullet_box(
        fig,
        template.slots["side1"].union(template.slots["side2"]).union(template.slots["side3"]),
        title="Why the change is justified",
        bullets=[
            "Target-only recurrent models need enough sequential context to learn more than a constant baseline.",
            "Adaptive chronological splits keep train/validation order intact while avoiding absurd 7-point training windows.",
            "The final comparison therefore has to acknowledge a methodological mismatch.",
        ],
        body_size=9.6,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Deep Setup and Libraries",
        "Deep",
        "The deep study is intentionally modest: CPU-only PyTorch, target-only windows, adaptive chronological splits, and explicit environment checks.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.42, 0.58])
    builder.text_box(
        fig,
        left,
        title="Environment check",
        body="Python: 3.14.3\npandas: 3.0.2\npyarrow: 23.0.1\nnumpy: 2.4.4\nmatplotlib: 3.10.8\nsklearn: 1.8.0\ntorch: 2.11.0",
        body_wrap=False,
        body_size=10.8,
    )
    upper, lower = right.split_rows([0.62, 0.38])
    builder.fit_table(fig, upper, DEEP_SPLIT_DF, title="Adaptive splits for chosen series", max_font=8.8, min_font=7.8)
    builder.bullet_box(
        fig,
        lower,
        title="Stack summary",
        bullets=[
            "Only the columns required for target-only sequence modeling are loaded from parquet.",
            "The training stack stays small and reproducible on CPU.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Adaptive Splits and Training Constraints",
        "Deep",
        "Adaptive splits solve a feasibility problem, but they also change what the deep results can legitimately claim against the fixed-cutoff classical study.",
    )
    template = builder.template_dashboard_2x2()
    builder.text_box(fig, template.slots["tl"], title="Chronology preserved", body="Each adaptive split still keeps training strictly before validation. The change is in the cutoff location, not in the causal direction.")
    builder.text_box(fig, template.slots["tr"], title="Why fixed cutoff fails", body="The highest-weight and most-stable series offer only seven training points under the global cutoff, which makes recurrent learning effectively impossible.")
    builder.text_box(fig, template.slots["bl"], title="Training constraints", body="Training uses CPU and deterministic seeds; forecasts are recursive and open-loop, which is harder than teacher-forced one-step validation.")
    builder.bullet_box(
        fig,
        template.slots["br"],
        title="Interpretation rule",
        bullets=[
            "The deep study is fair within its own setup.",
            "The final classical-vs-deep comparison must keep the split mismatch explicit.",
        ],
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Deep Model Design and Math",
        "Deep",
        "The deep study uses simple recurrent architectures on lag windows; the real risk is not architectural complexity but whether recursive forecasting stays stable over the validation horizon.",
    )
    content = builder._content_rect
    assert content is not None
    top, bottom = content.split_rows([0.62, 0.38])
    c1, c2, c3 = top.split_cols([1, 1, 1])
    builder.text_box(
        fig,
        c1,
        title="Window input",
        body="Use the last L targets\nas input context.\n\nPredict the next value.",
        body_wrap=False,
        body_size=10.8,
        body_min_size=8.4,
    )
    builder.text_box(
        fig,
        c2,
        title="Architectures",
        body="RNN: simplest recurrent baseline\n\nGRU: gated recurrence\nwith fewer parameters\n\nLSTM: gated memory cell\nfor longer dependencies",
        body_wrap=False,
        body_size=10.3,
    )
    builder.text_box(
        fig,
        c3,
        title="Inference rule",
        body="Recursive forecast:\npredict one step,\nappend it to the context,\nand repeat.\n\nHarder than one-step validation,\nbut closer to deployment.",
        body_wrap=False,
        body_size=10.0,
        body_min_size=8.2,
    )
    builder.bullet_box(
        fig,
        bottom,
        title="Evaluation setup",
        bullets=[
            "Naive persistence is kept as a baseline on every chosen series.",
            "Training curves use teacher-forced validation for diagnostics, but the reported scored forecasts are recursive.",
            "This keeps the deep study transparent about what the loss curves do and do not prove.",
        ],
        body_size=9.7,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Deep Results",
        "Deep",
        "Deep models help most on the highest-weight and longest-history cases, while the volatile series remains competitive with classical ARIMA and the most stable series is still hard to beat beyond a naive baseline.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.58, 0.42])
    builder.image_box(fig, assets["03_deep_sequence_models_chosen_cell21_img1"], left, title="Deep forecast plots")
    upper_right, lower_right = right.split_rows([0.44, 0.56])
    builder.bar_chart_box(
        fig,
        upper_right,
        DEEP_LEADERBOARD_DF[["Model", "Mean skill"]],
        title="Deep leaderboard",
        label_col="Model",
        value_col="Mean skill",
        color=SECTION_COLORS["Deep"],
        horizontal=True,
        value_fmt="{:.3f}",
        xlabel="Mean skill",
    )
    builder.text_box(
        fig,
        lower_right,
        title="Result message",
        body="GRU leads on mean skill overall, LSTM edges the longest-history case, RNN stays best on the volatile case, and naive persistence still ties the most-stable series at zero skill.",
        body_size=9.8,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Deep Training Curves",
        "Deep",
        "The training curves are diagnostic rather than decisive: they show optimization behavior, but the scored forecasts still need to be judged on recursive validation performance.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.62, 0.38])
    builder.image_box(fig, assets["03_deep_sequence_models_chosen_cell24_img1"], left, title="Training curves")
    upper_right, lower_right = right.split_rows([0.52, 0.48])
    builder.fit_table(fig, upper_right, DEEP_WINNERS_DF, title="Per-series winners", max_font=8.5, min_font=7.5)
    builder.bullet_box(
        fig,
        lower_right,
        title="How to read this",
        bullets=[
            "The curves help diagnose stability and overfitting trends.",
            "They do not replace recursive multi-step validation scores.",
            "The final comparison should still privilege forecast behavior over loss-curve aesthetics.",
        ],
        body_size=9.5,
    )
    builder.save(fig)

    fig = builder.new_slide(
        "Final Comparison and Close",
        "Summary",
        "The cleaned comparison is useful but not perfectly apples-to-apples: deep models gain from adaptive splits, while classical models stay on the harsher fixed-cutoff setup.",
    )
    content = builder._content_rect
    assert content is not None
    left, right = content.split_cols([0.54, 0.46])
    upper_left, lower_left = left.split_rows([0.54, 0.46])
    builder.grouped_bar_box(
        fig,
        upper_left,
        FINAL_COMPARE_CHART_DF,
        title="Best classical vs best deep",
        label_col="Series",
        series_cols=["Best classical", "Best deep"],
        colors=[SECTION_COLORS["Classical"], SECTION_COLORS["Deep"]],
        legend_labels=["Classical", "Deep"],
    )
    builder.fit_table(fig, lower_left, FINAL_COMPARE_DF, title="Final per-series comparison", max_font=8.7, min_font=7.6)
    upper_right, lower_right = right.split_rows([0.52, 0.48])
    builder.bullet_box(
        fig,
        upper_right,
        title="Honest conclusion",
        bullets=[
            "Deep models clearly help on the highest-weight case and slightly help on the longest-history case.",
            "Classical ARIMA remains highly competitive on the volatile series.",
            "The most-stable series remains difficult to improve beyond a naive forecast.",
        ],
        body_size=9.6,
    )
    builder.bullet_box(
        fig,
        lower_right,
        title="Limits and future work",
        bullets=[
            "The split mismatch means the final ranking should be read as directional, not definitive.",
            "Future work should bring pooled feature-based ML and ensembles into the same cleaned evaluation pipeline.",
            "The main project lesson is that diagnostics and evaluation design matter as much as model family choice.",
        ],
        body_size=9.6,
    )
    builder.save(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_manifest = extract_png_assets()
    assets = build_asset_specs(asset_manifest)

    with PdfPages(PDF_PATH) as pdf:
        builder = DeckBuilder(pdf)
        draw_overview_slides(builder, assets)
        draw_eda_slides(builder, assets)
        draw_classical_slides(builder, assets)
        draw_deep_slides(builder, assets)

        pdf.infodict()["Title"] = "Hedge Fund Time-Series Forecasting Main Deck"
        pdf.infodict()["Author"] = "OpenAI Codex"
        pdf.infodict()["Subject"] = "Applied Forecasting Methods project presentation"
        pdf.infodict()["Keywords"] = "time series, forecasting, presentation, classical models, deep learning"

    if builder.slide_no != TOTAL_SLIDES:
        raise RuntimeError(f"Expected {TOTAL_SLIDES} slides, built {builder.slide_no}")

    print(f"Built {builder.slide_no} slides")
    print(PDF_PATH)


if __name__ == "__main__":
    main()
