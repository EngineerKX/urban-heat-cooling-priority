#!/usr/bin/env python
"""Downscaling skill test (coarsen-and-recover): aggregate the production
Landsat LST composite to a coarse scale, rebuild a 3x finer scale with each
method (interpolation baselines, the production TsHARP regression, and two
ablations), and score every reconstruction against the real Landsat LST at
that finer scale. See validation/input_validation/downscaling_skill.py for
the method and its limits.

Two scale pairs, both run locally from one download of the 30 m LST and 30 m
Sentinel-2 indices (cached in data/interim/):
  90 -> 30 m    the production method's own factor-of-3 step, one level coarser
  360 -> 120 m  above Landsat's ~100 m thermal resolution, so the reference
                holds detail the sensor really measured
--gee instead runs the 90 -> 30 m pair entirely in Earth Engine (the
cross-check on the local interpolation kernels).

Reads Earth Engine only (no exports, no assets). Writes, per pair, the skill
table, block-bootstrap RMSE differences and the scored pixels under
data/processed/diagnostics/. --force recomputes.

Usage: python scripts/evaluate_downscaling_skill.py [--force] [--gee]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import numpy as np
import pandas as pd

from config.settings import (
    DIAGNOSTICS_DIR, DRY_SEASON_MONTHS, INTERIM_DIR, LANDSAT_CLOUD_COVER_MAX, NATIVE_SCALE_M, RANDOM_SEED,
    S2_CLOUD_PROB_MAX, S2_UTM_CRS, SG_BBOX, YEARS,
)
from src.downscaling.variants import build_lst_30m, build_s2_indices_10m
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, as_geodataframe, dissolve_boundary, fetch_subzones_geojson
from validation.input_validation.downscaling_skill import (
    INDEX_BANDS, block_bootstrap_rmse_diff, block_mean, build_prediction_stack, download_grids,
    local_skill_frame, sample_pixels, skill_table,
)

GRIDS_CACHE = INTERIM_DIR / "downscaling_skill_grids_30m.npz"
REG_SAMPLE_N = 4000      # same as scripts/build_heat_variants.py
EVAL_PIXELS = 20000      # Earth Engine path only
BLOCK_M = 1000           # bootstrap block size: whole 1 km squares are resampled
N_BOOT = 2000
SCALE_PAIRS = [(30, 90), (120, 360)]
COMPARISONS = [("tsharp", "bicubic"), ("tsharp", "nearest"), ("bicubic", "nearest"),
               ("tsharp_no_residual", "tsharp"), ("tsharp_ndvi_only", "tsharp")]


def _composites():
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    lst_30m = build_lst_30m(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX, S2_UTM_CRS, NATIVE_SCALE_M)
    s2_indices_10m = build_s2_indices_10m(sg_bbox, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    return sg_bbox, lst_30m, s2_indices_10m


def _save(skill, boot, pixels, suffix):
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    skill.to_csv(DIAGNOSTICS_DIR / f"downscaling_skill{suffix}.csv", index=False)
    boot.to_csv(DIAGNOSTICS_DIR / f"downscaling_skill_bootstrap{suffix}.csv", index=False)
    pixels.to_csv(DIAGNOSTICS_DIR / f"downscaling_skill_pixels{suffix}.csv", index=False)
    fmt = lambda x: f"{x:.3f}"
    print("\n" + skill.to_string(index=False, float_format=fmt))
    print("\nRMSE differences, 95% spatial block bootstrap (negative = first method more accurate):")
    print(boot.to_string(index=False, float_format=fmt))


def _score(pixels):
    skill = skill_table(pixels)
    boot = pd.DataFrame([block_bootstrap_rmse_diff(pixels, a, b, BLOCK_M, N_BOOT, RANDOM_SEED) for a, b in COMPARISONS])
    return skill, boot


def run_gee(force):
    out = DIAGNOSTICS_DIR / "downscaling_skill_gee.csv"
    if out.exists() and not force:
        print(f"{out} already exists — skipping (pass --force to rerun).")
        return
    sg_bbox, lst_30m, s2 = _composites()
    land = dissolve_boundary(as_ee_feature_collection(fetch_subzones_geojson()))
    stack, coefs = build_prediction_stack(lst_30m, s2, sg_bbox, S2_UTM_CRS, sample_n=REG_SAMPLE_N,
                                          seed=RANDOM_SEED, fine_scale=30, coarse_scale=90)
    for name, c in coefs.items():
        print(f"{name} coefficients (fitted at 90 m): " + ", ".join(f"{k}={v:.3f}" for k, v in c.items()))
    pixels = sample_pixels(stack, land, 30, EVAL_PIXELS, RANDOM_SEED)
    print(f"Sampled {len(pixels)} pixels with every method valid.")
    _save(*_score(pixels), pixels, "_gee")


def _land_mask_30m(shape, x0, y1):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    subzones = as_geodataframe(fetch_subzones_geojson()).to_crs(S2_UTM_CRS)
    return rasterize(((g, 1) for g in subzones.geometry), out_shape=shape,
                     transform=from_origin(x0, y1, 30, 30), fill=0, dtype="uint8").astype(bool)


def run_local(force):
    if GRIDS_CACHE.exists() and not force:
        cached = np.load(GRIDS_CACHE)
        grids = {b: cached[b] for b in ["LST"] + INDEX_BANDS}
        x0, y1 = float(cached["x0"]), float(cached["y1"])
        print(f"Using cached 30 m grids: {GRIDS_CACHE}")
    else:
        from pyproj import Transformer

        _, lst_30m, s2 = _composites()
        to_utm = Transformer.from_crs("EPSG:4326", S2_UTM_CRS, always_xy=True)
        (xa, xb), (ya, yb) = to_utm.transform([SG_BBOX[0], SG_BBOX[2]], [SG_BBOX[1], SG_BBOX[3]])
        print("Downloading 30 m LST and Sentinel-2 indices ...")
        grids, x0, y1 = download_grids(lst_30m, s2, S2_UTM_CRS, (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)))
        GRIDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(GRIDS_CACHE, x0=x0, y1=y1, **grids)
    land30 = _land_mask_30m(grids["LST"].shape, x0, y1)
    print(f"Grid {grids['LST'].shape}, land pixels {land30.sum()}, LST valid on land "
          f"{np.isfinite(grids['LST'][land30]).mean():.1%}")

    for fine, coarse in SCALE_PAIRS:
        print(f"\n===== {coarse} m -> {fine} m =====")
        land_fine = block_mean(land30.astype(float), fine // NATIVE_SCALE_M) >= 0.5
        pixels, coefs = local_skill_frame(grids, land_fine, fine, coarse, REG_SAMPLE_N, RANDOM_SEED)
        for name, c in coefs.items():
            print(f"{name} coefficients (fitted at {coarse} m): " + ", ".join(f"{k}={v:.3f}" for k, v in c.items()))
        print(f"Scored {len(pixels)} land pixels at {fine} m.")
        _save(*_score(pixels), pixels, f"_{fine}m")

        # Exploratory: the production regression is fitted over the whole bounding box, sea
        # included. Repeat with every input masked to land, so sea pixels can't drive the fit.
        land_grids = {b: np.where(land30, a, np.nan) for b, a in grids.items()}
        land_pixels, land_coefs = local_skill_frame(land_grids, land_fine, fine, coarse, REG_SAMPLE_N, RANDOM_SEED)
        print(f"\n--- land-only fit, {coarse} m -> {fine} m --- coefficients: "
              + "; ".join(f"{n}: " + ", ".join(f"{k}={v:.2f}" for k, v in c.items()) for n, c in land_coefs.items()))
        land_skill = skill_table(land_pixels)
        land_skill.to_csv(DIAGNOSTICS_DIR / f"downscaling_skill_landfit_{fine}m.csv", index=False)
        print(land_skill.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main(force: bool = False, gee: bool = False):
    init_ee()
    run_gee(force) if gee else run_local(force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gee", action="store_true", help="Run the 90 -> 30 m pair in Earth Engine instead.")
    args = parser.parse_args()
    main(force=args.force, gee=args.gee)
