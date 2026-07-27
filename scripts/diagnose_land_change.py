#!/usr/bin/env python
"""Land-change stability diagnostic (Dynamic World, early vs. late period).
Replaces land_change_diagnostic.ipynb.

Usage: python scripts/diagnose_land_change.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
    INTERIM_DIR,
    LC_CHANGE_FRACTION_THRESHOLD,
    LC_EARLY_END,
    LC_EARLY_START,
    LC_LATE_END,
    LC_LATE_START,
    LC_MIN_VALID_PIXELS,
    SG_BBOX,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, fetch_subzones_geojson
from validation.input_validation.land_change import (
    build_dominant_class_composites,
    flag_unstable_subzones,
    overall_change_fraction,
    zonal_land_change,
)

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "land_change_flags.csv"


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

    early_class, late_class, _, _ = build_dominant_class_composites(
        sg_bbox, LC_EARLY_START, LC_EARLY_END, LC_LATE_START, LC_LATE_END,
    )
    overall_change_fraction(early_class, late_class, sg_bbox, TARGET_SCALE_M)

    lc_df = zonal_land_change(early_class, late_class, subzones_fc, SUBZONE_ID_PROPERTY, TARGET_SCALE_M)
    full = flag_unstable_subzones(lc_df, heat_ids, LC_CHANGE_FRACTION_THRESHOLD, LC_MIN_VALID_PIXELS)

    full.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(full)} subzones, all heat-CSV subzones covered)")
    print("Every subzone has a 'status': 'flagged', 'stable', or 'insufficient_data' — "
          "treat 'insufficient_data' as genuinely unknown, not as 'stable'.")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
