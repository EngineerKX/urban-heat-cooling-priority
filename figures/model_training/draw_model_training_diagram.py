#!/usr/bin/env python
"""Report figures: how each model in the pipeline is trained.

Model-training companions to the system flow chart: same plain box-and-arrow
style and the same stage names (Downscaling, Land Cover, Heat Typology, Cooling
Priority Score, Scenario Modeling), in two levels of detail:

  detailed  what each model learns from, where it trains, how it is checked.
            Every number that is a project setting (trees, patch size, splits,
            season months, ...) is read from config/settings.py, so the figure
            follows the code instead of drifting from it.
  simple    just the machine-learning models and their approach (model name,
            supervised / unsupervised, one line on what it is for). No numbers.

Both share one layout and arrow set. Writes model_training_pipeline[_simple].png
(2x resolution, for the report) and .svg into ./out/. Skips any PNG that already
exists; pass --force to rebuild.

Usage:
  python figures/model_training/draw_model_training_diagram.py
  python figures/model_training/draw_model_training_diagram.py --only simple --force
"""

import argparse
import calendar
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config.settings import (
    BUCKET_NAMES,
    CNN_TRAIN_VAL_SPLIT,
    DRY_SEASON_MONTHS,
    INDEX_BANDS,
    RF_NUM_TREES,
    S2_FEATURE_BANDS,
    TRAINING_POINTS_PER_CLASS,
    UNET_PATCH_SIZE,
    UNET_TRAIN_VAL_SPLIT,
    VALIDATION_EXCLUSION_BUFFER_M,
    WET_SEASON_MONTHS,
    YEARS,
)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"
FILE_STEMS = {"detailed": "model_training_pipeline", "simple": "model_training_pipeline_simple"}

# Presentation-only layout constants, kept local on purpose (1 data unit = 1 px).
WIDTH = 1600
DPI = 100
PNG_SCALE = 2
MARGIN = 30
BOX_PAD_X, BOX_PAD_Y = 16, 14
WRAP_SLACK = 10  # px kept free inside each box so wrapped text never touches the border
TITLE_PT, BODY_PT = 13.5, 11
TITLE_LH, TITLE_GAP, TAG_LH, BODY_LH = 26, 4, 24, 21
LINE_W, ARROW_W, RADIUS = 1.6, 1.5, 14
LANE_W = 50  # px of the left column kept clear below Downscaling for the dashed benchmark arrow
DASH = (0, (5, 4))

_FONT_CANDIDATES = ["Calibri", "Segoe UI", "Arial", "DejaVu Sans"]

# Values below that are NOT in config/settings.py:
#   300         scripts/generate_validation_sample.py::TOTAL_POINTS
#   80/20       src/heat_model/tabular.py::train_xgb_model(test_size=0.2)
#   2-8 groups  src/hotspots/cluster.py::sweep_kmeans / sweep_gmm (k_range)
N_VALIDATION_POINTS = 300
XGB_TEST_SPLIT = "80/20"

DRY_MONTHS = ", ".join(calendar.month_abbr[m] for m in DRY_SEASON_MONTHS)
WET_MONTHS = ", ".join(calendar.month_abbr[m] for m in WET_SEASON_MONTHS)
YEAR_SPAN = f"{YEARS[0]}–{YEARS[-1]}"
N_FEATURES = len(S2_FEATURE_BANDS) + len(INDEX_BANDS)
N_CLASSES = len(BUCKET_NAMES)
UNET_HELD_OUT = f"{1 - UNET_TRAIN_VAL_SPLIT:.0%}"
CNN_HELD_OUT = f"{1 - CNN_TRAIN_VAL_SPLIT:.0%}"


@dataclass
class Box:
    title: str
    body: list  # paragraphs; each is word-wrapped on its own
    x: float
    w: float
    tag: str = ""  # optional one-line approach label under the title (e.g. "Supervised")
    y: float = 0.0
    h: float = 0.0
    lines: list = field(default_factory=list)

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h


def box_texts(simple: bool) -> dict:
    """name -> (title, tag, body paragraphs). Geometry lives in build_layout."""
    if simple:
        return {
            "input": ("Input", "", ["Satellite images + neighbourhood boundaries + population data + land-cover labels"]),
            "prep": ("Data Preparation", "", ["Clean + blend matching-season satellite images into composites"]),
            "container": ("Machine Learning Models", "", []),
            "downscale": ("Linear Regression", "Supervised",
                          ["Downscaling: sharpens heat maps from 30 m to 10 m"]),
            "rf": ("Random Forest", "Supervised",
                   ["Land cover: classifies each pixel by surface type"]),
            "unet": ("U-Net", "Supervised · deep learning",
                     ["Land cover: classifies pixels using their neighbours"]),
            "typology": ("K-means", "Unsupervised · compared with GMM",
                         ["Heat typology: groups neighbourhoods by heat pattern"]),
            "hybrid": ("Hybrid (RF + U-Net)", "Combines two models",
                       ["Land cover: averages both models' predictions"]),
            "score": ("PCA", "Unsupervised",
                      ["Cooling priority score: learns weights for heat, population and greenery"]),
            "cnn": ("CNN", "Supervised · deep learning",
                    ["Scenario modeling: predicts heat from images to test added greenery"]),
            "xgb": ("XGBoost", "Supervised · gradient boosting",
                    ["Scenario modeling: predicts heat per neighbourhood to test added greenery"]),
            "validation": ("Validation", "", ["Held-out accuracy checks + confidence bands on the final ranking"]),
            "dashboard": ("Dashboard", "", ["Interactive map of ranked neighbourhoods + the greening what-if"]),
        }
    return {
        "input": ("Input", "", [
            "Satellite images (Sentinel-2 + Landsat) + neighbourhood boundaries + population data "
            f"+ land-cover labels (ESA WorldCover) + {N_VALIDATION_POINTS} hand-labelled check points",
        ]),
        "prep": ("Data Preparation", "", [
            f"Cloud-mask and blend matching-season images (dry season: {DRY_MONTHS}; {YEAR_SPAN}, plus "
            f"{WET_MONTHS} for the typology) into median composites: Sentinel-2 at 10 m "
            f"({len(S2_FEATURE_BANDS)} bands + {', '.join(INDEX_BANDS)}) and Landsat heat at 30 m. "
            f"Land-cover labels are collapsed to {N_CLASSES} classes, and a {VALIDATION_EXCLUSION_BUFFER_M} m buffer "
            "around each check point is left out of training.",
        ]),
        "container": ("Model Training", "", []),
        "downscale": ("Downscaling", "", [
            "Supervised (linear regression). Learns 30 m heat from NDVI + NDBI + NDWI, applies it at 10 m, "
            "then adds back the 30 m residual. Benchmarked against native 30 m and bicubic 10 m heat. "
            "Runs on Earth Engine.",
        ]),
        "rf": ("Land Cover: Random Forest", "", [
            f"Supervised. {RF_NUM_TREES} trees learn {N_CLASSES} land-cover classes from "
            f"{TRAINING_POINTS_PER_CLASS:,} sampled pixels per class. Runs on Earth Engine, no GPU.",
        ]),
        "unet": ("Land Cover: U-Net", "", [
            f"Supervised. Reads {UNET_PATCH_SIZE}×{UNET_PATCH_SIZE} image patches so neighbouring pixels "
            f"inform each pixel; early stopping on {UNET_HELD_OUT} held-out patches. Runs on Colab GPU.",
        ]),
        "typology": ("Heat Typology", "", [
            "Unsupervised (K-means, compared with GMM). Groups neighbourhoods by dry- and wet-season heat, greenery and "
            "built-up values; 2–8 groups tested, best silhouette score kept. Runs locally.",
        ]),
        "hybrid": ("Land Cover: Hybrid", "", [
            "No extra training: averages the RF and U-Net class probabilities for each pixel, then takes "
            "the most likely class → land-cover map.",
        ]),
        "score": ("Cooling Priority Score", "", [
            "Unsupervised (PCA). Weights fitted on standardised pillars: heat (native 30 m), population + "
            "elderly, and greenery deficit (from the hybrid map). Runs locally.",
        ]),
        "cnn": ("Scenario Modeling: CNN", "", [
            f"Supervised. Same backbone as the U-Net, but predicts 10 m heat from {N_FEATURES} image features + "
            f"{N_CLASSES} land-cover channels (hybrid map). Target is bicubic-interpolated heat, so it does not "
            f"simply repeat the downscaling regression; early stopping on {CNN_HELD_OUT} held-out patches. "
            "Runs on Colab GPU.",
        ]),
        "xgb": ("Scenario Modeling: XGBoost", "", [
            "Supervised. One row per neighbourhood: land-cover mix, seasonal NDVI / NDBI, population and "
            f"elderly share. Target is native 30 m heat; {XGB_TEST_SPLIT} split. Runs locally.",
        ]),
        "validation": ("Validation", "", [
            "Downscaling: cross-checked against NEA station and MODIS heat data",
            f"Land cover: {N_VALIDATION_POINTS} hand-labelled points, never used in training",
            f"Heat models: held-out patches ({CNN_HELD_OUT}) and neighbourhoods (20%); CNN and XGBoost cross-checked",
            "Typology: silhouette score + land-cover coherence check",
            "Score: bootstrap confidence bands + tests of which choices change the ranking",
        ]),
        "tracking": ("Save + Track", "", [
            "Every training and evaluation run is logged in MLflow. Models trained on Colab GPU (U-Net, CNN) "
            "are saved to Google Cloud Storage, then pulled to the laptop for the app.",
        ]),
        "dashboard": ("Dashboard", "", [
            "Trained models and their outputs power the interactive map, the confidence bands and the greening what-if.",
        ]),
    }


def pick_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in _FONT_CANDIDATES if name in installed), "DejaVu Sans")


def make_measurer():
    """Text-width function (layout px). Measured at the PNG's real render DPI
    and scaled back down: glyph advances are hinted per-DPI, so measuring at
    the 1x DPI under-reports the width of what the 2x PNG actually draws."""
    render_dpi = DPI * PNG_SCALE
    fig = plt.figure(figsize=(1, 1), dpi=render_dpi)
    renderer = fig.canvas.get_renderer()

    def width(text, size, **kwargs):
        t = fig.text(0, 0, text, fontsize=size, **kwargs)
        w = t.get_window_extent(renderer).width
        t.remove()
        return w / PNG_SCALE

    return width


def wrap(text, max_w, width_fn):
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and width_fn(trial, BODY_PT, fontstyle="italic") > max_w:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def place_row(boxes, y, width_fn):
    """Wraps each box's text, gives every box in the row the tallest height,
    and returns the row's bottom edge."""
    inner_of = lambda b: b.w - 2 * BOX_PAD_X - WRAP_SLACK
    for b in boxes:
        assert width_fn(b.title, TITLE_PT, fontweight="bold") <= inner_of(b), f"title too wide: {b.title}"
        assert not b.tag or width_fn(b.tag, BODY_PT, fontweight="bold") <= inner_of(b), f"tag too wide: {b.tag}"
        b.lines = [line for para in b.body for line in wrap(para, inner_of(b), width_fn)]
        b.h = 2 * BOX_PAD_Y + TITLE_LH + TITLE_GAP + (TAG_LH if b.tag else 0) + len(b.lines) * BODY_LH
    row_h = max(b.h for b in boxes)
    for b in boxes:
        b.y, b.h = y, row_h
    return y + row_h


def build_layout(width_fn, simple: bool):
    """Returns (boxes by name, container rect, bus y-levels, total height)."""
    texts = box_texts(simple)
    cont_x, cont_w, cont_pad, gap = MARGIN, WIDTH - 2 * MARGIN, 26, 28
    col_w = (cont_w - 2 * cont_pad - 3 * gap) / 4
    col_x = [cont_x + cont_pad + i * (col_w + gap) for i in range(4)]
    mid = WIDTH / 2

    # name -> (x, w). Text is short in the simple variant, so its wide boxes are narrower
    # and the middle heat model is column-width instead of spanning the gap.
    wide = 700 if simple else 800
    geometry = {
        "input": (mid - wide / 2, wide),
        "prep": (mid - (wide if simple else 1080) / 2, wide if simple else 1080),
        "downscale": (col_x[0], col_w),
        "rf": (col_x[1], col_w),
        "unet": (col_x[2], col_w),
        "typology": (col_x[3], col_w),
        "hybrid": (mid - (220 if simple else 270), 440 if simple else 540),
        # Narrower and pushed right: leaves a free lane down the left of the container for the
        # dashed "benchmarked" arrow from Downscaling to Validation (see render).
        "score": (col_x[0] + LANE_W, col_w - LANE_W),
        "cnn": (mid - col_w / 2, col_w) if simple else (mid - 290, 580),
        "xgb": (col_x[3], col_w),
        "dashboard": (mid - wide / 2, wide),
    }

    def make(name, x, w):
        title, tag, body = texts[name]
        return Box(title, body, x=x, w=w, tag=tag)

    boxes = {name: make(name, *xw) for name, xw in geometry.items()}

    y = MARGIN
    y = place_row([boxes["input"]], y, width_fn)
    y = place_row([boxes["prep"]], y + 40, width_fn)

    cont_top = y + 44
    y = place_row([boxes[k] for k in ("downscale", "rf", "unet", "typology")], cont_top + 46, width_fn)
    y = place_row([boxes["hybrid"]], y + 44, width_fn)
    bus_hybrid = y + 26
    y = place_row([boxes[k] for k in ("score", "cnn", "xgb")], y + 60, width_fn)
    cont_bottom = y + cont_pad
    container = (cont_x, cont_top, cont_w, cont_bottom - cont_top)

    if simple:
        # One evaluation box, straight down to the dashboard (no experiment-tracking box).
        boxes["validation"] = make("validation", mid - wide / 2, wide)
        y = place_row([boxes["validation"]], cont_bottom + 46, width_fn)
        bus_final = None
        dashboard_gap = 46
    else:
        boxes["validation"] = make("validation", MARGIN, 930)
        boxes["tracking"] = make("tracking", MARGIN + 970, WIDTH - 2 * MARGIN - 970)
        y = place_row([boxes["validation"], boxes["tracking"]], cont_bottom + 46, width_fn)
        bus_final = y + 26
        dashboard_gap = 62
    y = place_row([boxes["dashboard"]], y + dashboard_gap, width_fn)

    return boxes, container, texts["container"][0], (bus_hybrid, bus_final), y + MARGIN


def draw_box(ax, b):
    ax.add_patch(FancyBboxPatch(
        (b.x, b.y), b.w, b.h, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
        fc="white", ec="black", lw=LINE_W, zorder=2,
    ))
    y = b.y + BOX_PAD_Y  # titles top-aligned so boxes in one row share a title baseline
    ax.text(b.cx, y, b.title, ha="center", va="top", fontsize=TITLE_PT, fontweight="bold", zorder=3)
    y += TITLE_LH + TITLE_GAP
    if b.tag:
        ax.text(b.cx, y, b.tag, ha="center", va="top", fontsize=BODY_PT, fontweight="bold", zorder=3)
        y += TAG_LH
    for line in b.lines:
        ax.text(b.cx, y, line, ha="center", va="top", fontsize=BODY_PT, fontstyle="italic", zorder=3)
        y += BODY_LH


def draw_line(ax, points, dashed=False):
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        ax.plot([x0, x1], [y0, y1], color="black", lw=ARROW_W, zorder=1,
                linestyle=DASH if dashed else "-",
                solid_capstyle="projecting", dash_capstyle="butt")


def draw_arrow(ax, points, dashed=False):
    """Polyline whose last segment ends in an arrowhead. A dashed shaft stops just short of the
    tip and finishes with a short solid arrow, so the head itself is never drawn dashed."""
    if dashed:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        length = math.hypot(x1 - x0, y1 - y0)
        stub_start = (x1 - (x1 - x0) / length * 12, y1 - (y1 - y0) / length * 12)
        draw_line(ax, points[:-1] + [stub_start], dashed=True)
        points = [stub_start, points[-1]]
    elif len(points) > 2:
        draw_line(ax, points[:-1])
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=16, lw=ARROW_W,
        color="black", shrinkA=0, shrinkB=0, zorder=1,
    ))


def render(boxes, container, container_title, buses, height, font):
    plt.rcParams["font.family"] = font
    plt.rcParams["svg.fonttype"] = "none"  # keep text as text in the SVG so it stays editable
    fig = plt.figure(figsize=(WIDTH / DPI, height / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")

    cx0, cy0, cw, ch = container
    ax.add_patch(FancyBboxPatch(
        (cx0, cy0), cw, ch, boxstyle=f"round,pad=0,rounding_size={RADIUS}",
        fc="white", ec="black", lw=LINE_W, zorder=0,
    ))
    ax.text(WIDTH / 2, cy0 + 14, container_title, ha="center", va="top",
            fontsize=TITLE_PT, fontweight="bold", zorder=3)
    for b in boxes.values():
        draw_box(ax, b)

    b, mid = boxes, WIDTH / 2
    bus_hybrid, bus_final = buses

    draw_arrow(ax, [(mid, b["input"].bottom), (mid, b["prep"].top)])
    draw_arrow(ax, [(mid, b["prep"].bottom), (mid, cy0)])

    # Trained-model dependencies inside the container.
    draw_arrow(ax, [(b["rf"].cx, b["rf"].bottom), (b["rf"].cx, b["hybrid"].top)])
    draw_arrow(ax, [(b["unet"].cx, b["unet"].bottom), (b["unet"].cx, b["hybrid"].top)])
    draw_arrow(ax, [(mid, b["hybrid"].bottom), (mid, b["cnn"].top)])
    score_in, xgb_in = b["score"].cx + 60, b["xgb"].cx - 60
    draw_arrow(ax, [(mid, b["hybrid"].bottom), (mid, bus_hybrid), (score_in, bus_hybrid), (score_in, b["score"].top)])
    draw_arrow(ax, [(mid, b["hybrid"].bottom), (mid, bus_hybrid), (xgb_in, bus_hybrid), (xgb_in, b["xgb"].top)])
    # No Downscaling -> Score arrow: the score's heat pillar is the native 30 m layer
    # (config REFERENCE_VARIANT); the downscaled map is only benchmarked against it.
    # No Typology -> XGBoost arrow either: the cluster label is not an XGBoost feature (it was built
    # from lst_dry, which equals the target -- see src/heat_model/tabular.py::XGB_FEATURE_COLUMNS).

    # PCA's other two pillars (heat, population + elderly) come straight from the data, not from
    # another model; only the greenery pillar comes from the hybrid (the bus arrow above).
    score = b["score"]
    label_y, heat_x = score.top - 74, score.cx - 30
    ax.text(score.cx, label_y, "Landsat heat + population data", ha="center", va="top",
            fontsize=BODY_PT - 0.5, fontstyle="italic", zorder=3)
    draw_arrow(ax, [(heat_x, label_y + 22), (heat_x, score.top)])

    # Out of the container, then into the dashboard.
    cont_bottom = cy0 + ch

    # Downscaling's output is compared against native 30 m heat, not consumed by any later model
    # (the score, typology and XGBoost all use native 30 m; the CNN uses bicubic). Dashed = evaluated only.
    ds, val = b["downscale"], b["validation"]
    lane_x, y_elbow = ds.x + LANE_W / 2, cont_bottom + 23
    end_x = min(max(lane_x, val.x + 40), val.x + val.w - 40)  # straight down if Validation is under the lane
    path = [(lane_x, ds.bottom), (lane_x, val.top)] if end_x == lane_x else [
        (lane_x, ds.bottom), (lane_x, y_elbow), (end_x, y_elbow), (end_x, val.top)]
    draw_arrow(ax, path, dashed=True)
    ax.text(lane_x + 8, ds.bottom + 10, "benchmarked", ha="left", va="top",
            fontsize=BODY_PT - 0.5, fontstyle="italic", zorder=3)
    if "tracking" in b:
        for key in ("validation", "tracking"):
            draw_arrow(ax, [(b[key].cx, cont_bottom), (b[key].cx, b[key].top)])
        draw_line(ax, [(b["tracking"].cx, b["tracking"].bottom), (b["tracking"].cx, bus_final), (mid, bus_final)])
        draw_arrow(ax, [(b["validation"].cx, b["validation"].bottom), (b["validation"].cx, bus_final),
                        (mid, bus_final), (mid, b["dashboard"].top)])
    else:
        draw_arrow(ax, [(mid, cont_bottom), (mid, b["validation"].top)])
        draw_arrow(ax, [(mid, b["validation"].bottom), (mid, b["dashboard"].top)])
    return fig


def save_figure(fig, path, **kwargs):
    """savefig with a short retry. In this OneDrive-synced folder, open() intermittently fails with
    OSError 22 on a file that was just written or read (seen 3 times); a few seconds' wait clears it."""
    for attempt in range(4):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 3:
                raise
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", choices=list(FILE_STEMS), help="build just one variant (default: both)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the PNG already exists")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = pick_font()
    plt.rcParams["font.family"] = font
    width_fn = make_measurer()

    for variant, stem in FILE_STEMS.items():
        if args.only and args.only != variant:
            continue
        png_path, svg_path = args.out_dir / f"{stem}.png", args.out_dir / f"{stem}.svg"
        if png_path.exists() and not args.force:
            print(f"{png_path} already exists - pass --force to rebuild.")
            continue
        boxes, container, container_title, buses, height = build_layout(width_fn, simple=variant == "simple")
        fig = render(boxes, container, container_title, buses, height, font)
        save_figure(fig, png_path, dpi=DPI * PNG_SCALE, facecolor="white")
        save_figure(fig, svg_path, facecolor="white")
        plt.close(fig)
        print(f"[{variant}] font {font}; wrote {png_path.name} and {svg_path.name} ({WIDTH}x{height:.0f} layout units).")


if __name__ == "__main__":
    main()
