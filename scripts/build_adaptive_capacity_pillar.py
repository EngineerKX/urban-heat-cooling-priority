#!/usr/bin/env python
"""Build the interim NDVI-threshold adaptive-capacity (greenery) pillar.
Replaces adaptive_capacity_pillar.ipynb. Stand-in until Track B's real S3
land-cover output is validated and ready to use instead.

Usage: python scripts/build_adaptive_capacity_pillar.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
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
from src.ingest.subzones import as_ee_feature_collection, fetch_subzones_geojson
from src.priority_score.pillars import build_adaptive_capacity_pillar

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "adaptive_capacity_pillar.csv"


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    heat_ids = pd.read_csv(HEAT_CSV_PATH)["subzone_id"]

    ac_df = build_adaptive_capacity_pillar(
        sg_bbox, subzones_fc, SUBZONE_ID_PROPERTY, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX,
        TARGET_SCALE_M, heat_ids, ndvi_threshold=NDVI_VEGETATION_THRESHOLD,
    )
    print(f"\n⚠️  Reminder: NDVI threshold ({NDVI_VEGETATION_THRESHOLD}) is a placeholder proxy, "
          f"not Track B's real S3 land cover.")

    ac_df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(ac_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
