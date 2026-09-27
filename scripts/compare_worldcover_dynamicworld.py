#!/usr/bin/env python
"""Dynamic World vs WorldCover (checkpoint_DynamicWorld): how much land changed
between the WorldCover year (2021) and 2025-26, and how well WorldCover,
Dynamic World and the hybrid classifier agree with the hand labels.

Earth Engine results are cached under data/interim/dw_vs_worldcover/ (pass
--force to refetch); the analysis is recomputed every run.

Usage: python scripts/compare_worldcover_dynamicworld.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
    DW_BASELINE_END, DW_BASELINE_SPLIT, DW_BASELINE_START, DW_RECENT_END, DW_RECENT_START,
    INTERIM_DIR, LC_CHANGE_FRACTION_THRESHOLD, PROCESSED_DIR, S2_UTM_CRS, SG_BBOX,
    SUBZONE_ID_PROPERTY, TARGET_SCALE_M, TOP_N,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import BUCKET_NAMES, get_worldcover_bucket_image
from src.landcover.hybrid import HYBRID_RASTER_PATH
from src.utils.caching import load_or_fetch_csv
from validation.input_validation.dw_comparison import (
    area_weighted_agreement, dw_bucket_image, island_summary, sample_dw_at_points, zonal_change_table,
)
from validation.input_validation.labeling_sample import class_area_histogram
from validation.landcover_validation.classifier_evaluation import sample_raster_at_points

OUT_DIR = INTERIM_DIR / "dw_vs_worldcover"
VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
PRIORITY_CSV = PROCESSED_DIR / "priority_score.csv"


def main(force: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(VALIDATION_CSV)
    ee_state = {}

    def _ee():
        """Earth Engine images, built only if some cached table is missing."""
        if not ee_state:
            init_ee()
            bbox = ee.Geometry.Rectangle(list(SG_BBOX))
            ee_state["subzones_fc"] = as_ee_feature_collection(fetch_subzones_geojson())
            ee_state["images"] = (
                dw_bucket_image(bbox, DW_BASELINE_START, DW_BASELINE_END),
                dw_bucket_image(bbox, DW_BASELINE_START, DW_BASELINE_SPLIT),
                dw_bucket_image(bbox, DW_BASELINE_SPLIT, DW_BASELINE_END),
                dw_bucket_image(bbox, DW_RECENT_START, DW_RECENT_END),
            )
        return ee_state["subzones_fc"], ee_state["images"]

    def _zonal():
        fc, (base, half_1, half_2, recent) = _ee()
        return zonal_change_table(base, half_1, half_2, recent, fc, SUBZONE_ID_PROPERTY, TARGET_SCALE_M)

    def _points():
        _, (base, _h1, _h2, recent) = _ee()
        return sample_dw_at_points(base, recent, labels, TARGET_SCALE_M)

    def _shares():
        fc, _ = _ee()
        boundary = dissolve_boundary(fc)
        frame = get_worldcover_bucket_image(boundary, S2_UTM_CRS, TARGET_SCALE_M)
        hist = class_area_histogram(frame, boundary, TARGET_SCALE_M)
        return pd.DataFrame({"worldcover_class": list(hist), "pixels": list(hist.values())})

    zonal = load_or_fetch_csv(OUT_DIR / "land_change_zonal.csv", _zonal, force=force)
    dw_points = load_or_fetch_csv(OUT_DIR / "points_dw.csv", _points, force=force)
    shares_df = load_or_fetch_csv(OUT_DIR / "worldcover_area_shares.csv", _shares, force=force)

    # --- 1. Land change, baseline (~2021) -> recent (2025-26) ---------------
    island = island_summary(zonal)
    print("\n== Land change, ~2021 -> 2025/26 (Dynamic World, 4 classes) ==")
    print(f"Island-wide pixels that changed class: {island['chg'] * 100:.1f}%  "
          f"(noise floor, two halves of the baseline: {island['chg_noise'] * 100:.1f}%)")
    print(f"Vegetation share {island['veg_base'] * 100:.1f}% -> {island['veg_recent'] * 100:.1f}%  "
          f"(noise: {island['veg_half1'] * 100:.1f}% vs {island['veg_half2'] * 100:.1f}%)")
    print(f"Built-up share   {island['built_base'] * 100:.1f}% -> {island['built_recent'] * 100:.1f}%")
    z = zonal.dropna(subset=["chg"]).copy()
    print(f"Subzones changing more than {LC_CHANGE_FRACTION_THRESHOLD:.0%}: {(z['chg'] > LC_CHANGE_FRACTION_THRESHOLD).sum()} of {len(z)} "
          f"(noise alone: {(z['chg_noise'] > LC_CHANGE_FRACTION_THRESHOLD).sum()})")
    if PRIORITY_CSV.exists():
        ranked = pd.read_csv(PRIORITY_CSV).merge(z, on="subzone_id")
        top = ranked.sort_values("priority_score", ascending=False).head(TOP_N)
        print(f"Top {TOP_N}: median change {top['chg'].median():.3f}, max {top['chg'].max():.3f}, "
              f"above threshold: {(top['chg'] > LC_CHANGE_FRACTION_THRESHOLD).sum()}")

    # --- 2. Agreement with hand labels, re-weighted to real area shares ------
    to_id = {name: bucket for bucket, name in BUCKET_NAMES.items()}
    pts = labels.merge(dw_points, on="point_id")
    pts = pts[pts["agreed_label"].isin(to_id)].dropna(subset=["dw_baseline", "dw_recent"]).copy()
    pts["true_bucket"] = pts["agreed_label"].map(to_id)
    maps = {"WorldCover 2021": "worldcover_class", "Dynamic World baseline": "dw_baseline",
            "Dynamic World recent": "dw_recent"}
    if HYBRID_RASTER_PATH.exists():
        pts = sample_raster_at_points(HYBRID_RASTER_PATH, pts)
        pts = pts[pts["pred_bucket"].notna() & (pts["pred_bucket"] != 0)]
        maps["Hybrid model"] = "pred_bucket"
    shares = dict(zip(shares_df["worldcover_class"].astype(int), shares_df["pixels"]))
    summary, boot = area_weighted_agreement(pts, shares, maps)

    print(f"\n== Agreement with hand labels, area-weighted (n={len(pts)} points) ==")
    for row in summary.itertuples():
        print(f"  {row.map:24s} {row.agreement * 100:5.1f}%   ({row.ci_low * 100:.0f}-{row.ci_high * 100:.0f}%)")
    for other in [m for m in maps if m != "Dynamic World baseline"]:
        diff = boot["Dynamic World baseline"] - boot[other]
        print(f"  Dynamic World baseline minus {other}: {diff.mean() * 100:.1f} points "
              f"({diff.quantile(0.025) * 100:.0f} to {diff.quantile(0.975) * 100:.0f})")
    summary.to_csv(OUT_DIR / "agreement_summary.csv", index=False)
    print(f"\nSaved: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
