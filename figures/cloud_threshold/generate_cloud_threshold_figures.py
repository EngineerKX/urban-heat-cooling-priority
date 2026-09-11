#!/usr/bin/env python
"""Report figures: how the cloud threshold changes composite quality.

Renders, straight from Earth Engine, what a Sentinel-2 / Landsat composite
looks like when the cloud threshold is set too loose (cloud + haze leak
through), at the project's setting, and too strict (real data gets punched
out). Reuses the exact masking helpers the pipeline uses
(`src/ingest/gee.py`) so the figures reflect real S1 behaviour, not a
reimplementation.

Three figures, each written as a labelled montage PNG into ./out/:

  figure1_single_scene.png    one partly-cloudy S2 scene: raw / s2cloudless
                              probability / after masking -- shows why one
                              scene can't be "fixed" by a threshold alone.
  figure2_threshold_sweep.png dry-season median RGB at several s2cloudless
                              probability cutoffs (config: S2_CLOUD_PROB_MAX).
  figure3_landsat_lst.png     Landsat LST median with the QA cloud mask
                              off vs on -- unmasked, cloud tops read cold
                              and collapse the median.

Skips any figure whose montage already exists; pass --force to rebuild.

Usage:
  python figures/cloud_threshold/generate_cloud_threshold_figures.py
  python figures/cloud_threshold/generate_cloud_threshold_figures.py --thresholds 90 70 40 --year 2023 --force
  python figures/cloud_threshold/generate_cloud_threshold_figures.py --only sweep --out-dir data/processed/diagnostics/cloud_threshold
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ee
import requests
from PIL import Image, ImageDraw, ImageFont

from config.settings import DRY_SEASON_MONTHS
from src.ingest.gee import (
    apply_s2_cloud_mask,
    date_filter_for_years_months,
    init_ee,
    join_s2_with_cloud_prob,
    mask_landsat_c2l2,
    scale_landsat_st_celsius,
)

# Presentation crop over mainland Singapore. Deliberately NOT in
# config.settings -- this is a figure-framing choice, not a pipeline AOI
# (the real analysis boundary is the dissolved URA subzone polygon).
DEFAULT_AOI = [103.605, 1.205, 104.045, 1.475]  # minLon, minLat, maxLon, maxLat
DEFAULT_YEAR = 2024
DEFAULT_THRESHOLDS = [100, 80, 70, 40, 20]  # s2cloudless probability cutoffs (%)
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "out"

# Viz params are presentation-only (stretch + palette), kept local on purpose.
S2_RGB = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.15}
PROB_VIZ = {"bands": ["probability"], "min": 0, "max": 100,
            "palette": ["001133", "1e6091", "52b69a", "d9ed92", "ffba08", "d00000"]}
LST_VIZ = {"bands": ["LST_C"], "min": 24, "max": 40,
           "palette": ["040274", "0502a3", "235cb1", "30c8e2", "9ffff5",
                       "ffef00", "ff8b00", "d00000", "7a0403"]}

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _thumb(image, viz, region, dimensions):
    url = image.getThumbURL({**viz, "region": region, "dimensions": dimensions, "format": "png"})
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _caption(img, title, subtitle=""):
    bar_h = 58 if subtitle else 34
    out = Image.new("RGB", (img.size[0], img.size[1] + bar_h), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((10, img.size[1] + 6), title, fill="black", font=_font(22))
    if subtitle:
        draw.text((10, img.size[1] + 32), subtitle, fill="#555555", font=_font(16))
    return out


def _montage(tiles, cols, path, pad=6):
    w = max(t.size[0] for t in tiles)
    h = max(t.size[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), "white")
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet.paste(tile, (pad + c * (w + pad), pad + r * (h + pad)))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"  \u2705 {path}")


# --- Sentinel-2 collection builders (thin wrappers over src/ingest/gee.py) ---

def _s2_pair(aoi, year, months, scene_cloud_max):
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", scene_cloud_max))
    )
    s2_sr = date_filter_for_years_months(s2_sr, [year], months)
    prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(aoi)
    prob = date_filter_for_years_months(prob, [year], months)
    return s2_sr, prob


def _s2_median(prob_threshold, aoi, year, months, scene_cloud_max):
    """Median RGB composite. prob_threshold=None -> no per-pixel cloud mask."""
    s2_sr, prob = _s2_pair(aoi, year, months, scene_cloud_max)
    n_scenes = s2_sr.size().getInfo()
    if prob_threshold is None:
        return s2_sr.median(), n_scenes
    joined = join_s2_with_cloud_prob(s2_sr, prob)
    return apply_s2_cloud_mask(joined, prob_threshold).median(), n_scenes


# --- Figures -----------------------------------------------------------------

def figure_single_scene(aoi, region, dims, out_dir, force):
    out = out_dir / "figure1_single_scene.png"
    if out.exists() and not force:
        print(f"  \u2013 skip (exists): {out}")
        return
    print("[1] single partly-cloudy Sentinel-2 scene")
    s2_sr, prob = _s2_pair(aoi, DEFAULT_YEAR, [10, 11, 12], 100)
    s2_sr = s2_sr.filter(ee.Filter.rangeContains("CLOUDY_PIXEL_PERCENTAGE", 35, 65))
    joined = ee.ImageCollection(join_s2_with_cloud_prob(s2_sr, prob)).sort(
        "CLOUDY_PIXEL_PERCENTAGE", False
    )
    scene = ee.Image(joined.first())
    cloud_pct = scene.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    date = ee.Date(scene.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"    scene {date}, CLOUDY_PIXEL_PERCENTAGE={cloud_pct:.1f}")

    prob_img = ee.Image(scene.get("cloud_mask"))
    masked = scene.updateMask(prob_img.select("probability").lt(40))

    tiles = [
        _caption(_thumb(scene, S2_RGB, region, dims),
                 "Raw scene \u2014 no cloud handling",
                 f"{date}   scene CLOUDY_PIXEL_PERCENTAGE = {cloud_pct:.0f}%"),
        _caption(_thumb(prob_img, PROB_VIZ, region, dims),
                 "s2cloudless cloud probability",
                 "blue = clear ... red = near-certain cloud"),
        _caption(_thumb(masked, S2_RGB, region, dims),
                 "Same scene, prob < 40 masked out",
                 "black = removed pixels (one scene can't fill the holes)"),
    ]
    _montage(tiles, 3, out)


def figure_threshold_sweep(aoi, region, dims, out_dir, force, thresholds, year, months,
                           scene_cloud_max):
    out = out_dir / "figure2_threshold_sweep.png"
    if out.exists() and not force:
        print(f"  \u2013 skip (exists): {out}")
        return
    print(f"[2] {year} season median \u2014 s2cloudless probability threshold sweep")
    specs = [("no per-pixel mask", None)] + [(f"prob < {t}", t) for t in thresholds]
    tiles = []
    for name, thr in specs:
        comp, n_scenes = _s2_median(thr, aoi, year, months, scene_cloud_max)
        note = f"{n_scenes} scenes   (scene filter < {scene_cloud_max}%)"
        print(f"    {name:<22} {note}")
        tiles.append(_caption(_thumb(comp, S2_RGB, region, dims), name, note))
    cols = 3 if len(tiles) > 4 else len(tiles)
    _montage(tiles, cols, out)


def figure_landsat_lst(aoi, region, dims, out_dir, force, year, months, cloud_cover_max):
    out = out_dir / "figure3_landsat_lst.png"
    if out.exists() and not force:
        print(f"  \u2013 skip (exists): {out}")
        return
    print("[3] Landsat LST \u2014 QA cloud mask off vs on")
    l89 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi)
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(aoi))
    )
    l89 = date_filter_for_years_months(l89, [year - 1, year], months)
    loose_col = l89.filter(ee.Filter.lt("CLOUD_COVER", 100))
    strict_col = l89.filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))

    loose = loose_col.map(scale_landsat_st_celsius).select("LST_C").median()
    strict = (
        strict_col.map(mask_landsat_c2l2).map(scale_landsat_st_celsius).select("LST_C").median()
    )
    n_loose = loose_col.size().getInfo()
    n_strict = strict_col.size().getInfo()

    tiles = [
        _caption(_thumb(loose, LST_VIZ, region, dims),
                 f"LST median, NO QA cloud mask, CLOUD_COVER < 100",
                 f"{n_loose} scenes \u2014 clouds read cold: blue blotches, cool bias"),
        _caption(_thumb(strict, LST_VIZ, region, dims),
                 f"LST median, QA cloud/shadow masked, CLOUD_COVER < {cloud_cover_max}",
                 f"{n_strict} scenes \u2014 clean urban-heat gradient"),
    ]
    _montage(tiles, 2, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"where to write the montage PNGs (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--only", choices=["single", "sweep", "lst"], nargs="+",
                    help="render only these figures (default: all three)")
    ap.add_argument("--thresholds", type=int, nargs="+", default=DEFAULT_THRESHOLDS,
                    help=f"s2cloudless probability cutoffs for the sweep (default: {DEFAULT_THRESHOLDS})")
    ap.add_argument("--year", type=int, default=DEFAULT_YEAR,
                    help=f"composite year for the sweep + LST figures (default: {DEFAULT_YEAR}). "
                         "Kept to one year on purpose -- more years median clouds out and weaken the demo.")
    ap.add_argument("--months", type=int, nargs="+", default=list(DRY_SEASON_MONTHS),
                    help=f"season months (default: config DRY_SEASON_MONTHS = {list(DRY_SEASON_MONTHS)})")
    ap.add_argument("--scene-cloud-max", type=int, default=100,
                    help="CLOUDY_PIXEL_PERCENTAGE scene prefilter for the sweep (default: 100 = keep all, "
                         "so the per-pixel threshold is the only variable)")
    ap.add_argument("--landsat-cloud-cover-max", type=int, default=70,
                    help="CLOUD_COVER cutoff for the 'strict' Landsat panel (default: 70)")
    ap.add_argument("--aoi", type=float, nargs=4, metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
                    default=DEFAULT_AOI, help=f"bounding box to render (default: {DEFAULT_AOI})")
    ap.add_argument("--dimensions", type=int, default=900, help="thumbnail long-edge px (default: 900)")
    ap.add_argument("--force", action="store_true", help="rebuild figures even if the PNG already exists")
    args = ap.parse_args()

    init_ee()
    aoi = ee.Geometry.Rectangle(list(args.aoi))
    region = aoi  # getThumbURL accepts a Geometry directly
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(args.only) if args.only else {"single", "sweep", "lst"}

    if "single" in wanted:
        figure_single_scene(aoi, region, args.dimensions, args.out_dir, args.force)
    if "sweep" in wanted:
        figure_threshold_sweep(aoi, region, args.dimensions, args.out_dir, args.force,
                               args.thresholds, args.year, args.months, args.scene_cloud_max)
    if "lst" in wanted:
        figure_landsat_lst(aoi, region, args.dimensions, args.out_dir, args.force,
                           args.year, args.months, args.landsat_cloud_cover_max)

    print(f"\nDone. Figures in {args.out_dir}")


if __name__ == "__main__":
    main()
