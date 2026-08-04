#!/usr/bin/env python
"""Build the adaptive-capacity (greenery) pillar. Replaces
adaptive_capacity_pillar.ipynb. Computes BOTH greenery-fraction sources —
the original NDVI-threshold proxy and the validated RF/U-Net land-cover
ensemble — and selects the canonical `greenery_fraction` column per
config.settings.ADAPTIVE_CAPACITY_SOURCE, printing their Spearman
correlation as a built-in comparison diagnostic rather than silently
dropping the older column.

Usage: python scripts/build_adaptive_capacity_pillar.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import (
    ADAPTIVE_CAPACITY_SOURCE,
    DRY_SEASON_MONTHS,
    INTERIM_DIR,
    NDVI_VEGETATION_THRESHOLD,
    S2_CLOUD_PROB_MAX,
    SG_BBOX,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, as_geodataframe, fetch_subzones_geojson
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH
from src.priority_score.pillars import build_adaptive_capacity_pillar, build_adaptive_capacity_pillar_landcover

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "adaptive_capacity_pillar.csv"


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    init_ee()
    geojson = fetch_subzones_geojson()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(geojson)
    heat_ids = pd.read_csv(HEAT_CSV_PATH)["subzone_id"]

    print("--- NDVI-threshold greenery fraction ---")
    ndvi_df = build_adaptive_capacity_pillar(
        sg_bbox, subzones_fc, SUBZONE_ID_PROPERTY, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX,
        TARGET_SCALE_M, heat_ids, ndvi_threshold=NDVI_VEGETATION_THRESHOLD,
    ).rename(columns={"greenery_fraction": "greenery_fraction_ndvi"})

    landcover_col = None
    if ENSEMBLE_RASTER_PATH.exists():
        print("\n--- Land-cover-ensemble greenery fraction ---")
        subzones_gdf = as_geodataframe(geojson)
        landcover_df = build_adaptive_capacity_pillar_landcover(
            subzones_gdf, SUBZONE_ID_PROPERTY, heat_ids,
        ).rename(columns={"greenery_fraction": "greenery_fraction_landcover"})
        landcover_col = "greenery_fraction_landcover"
    else:
        print(f"\n⚠️  {ENSEMBLE_RASTER_PATH} not found — run scripts/build_landcover_ensemble.py first. "
              f"Falling back to NDVI regardless of ADAPTIVE_CAPACITY_SOURCE.")
        landcover_df = None

    if landcover_df is not None:
        ac_df = ndvi_df.merge(landcover_df, on="subzone_id", how="outer")
        both_present = ac_df[["greenery_fraction_ndvi", landcover_col]].dropna()
        if len(both_present) >= 2:
            corr, _ = spearmanr(both_present["greenery_fraction_ndvi"], both_present[landcover_col])
            print(f"\nSpearman(NDVI proxy, land-cover ensemble) = {corr:.3f} "
                  f"over {len(both_present)} subzones with both values.")
        else:
            print("\n⚠️  Too few overlapping subzones to compute a comparison correlation.")
    else:
        ac_df = ndvi_df.copy()

    if ADAPTIVE_CAPACITY_SOURCE == "landcover" and landcover_col is not None:
        ac_df["greenery_fraction"] = ac_df[landcover_col].where(
            ac_df[landcover_col].notna(), ac_df["greenery_fraction_ndvi"]
        )
        n_fallback = int(ac_df[landcover_col].isna().sum())
        if n_fallback:
            print(f"⚠️  {n_fallback} subzones missing a land-cover value — fell back to NDVI for those rows.")
    elif ADAPTIVE_CAPACITY_SOURCE == "ndvi":
        ac_df["greenery_fraction"] = ac_df["greenery_fraction_ndvi"]
    else:
        raise ValueError(f"Unknown ADAPTIVE_CAPACITY_SOURCE '{ADAPTIVE_CAPACITY_SOURCE}', expected 'landcover' or 'ndvi'.")

    print(f"\nCanonical source: ADAPTIVE_CAPACITY_SOURCE = '{ADAPTIVE_CAPACITY_SOURCE}'")
    if ADAPTIVE_CAPACITY_SOURCE == "ndvi":
        print(f"⚠️  Reminder: NDVI threshold ({NDVI_VEGETATION_THRESHOLD}) is a placeholder proxy, "
              f"not the real S3 land cover.")

    ac_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(ac_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
