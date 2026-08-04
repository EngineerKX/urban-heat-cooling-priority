#!/usr/bin/env python
"""Build the MODIS LST secondary cross-check table -- in the proposal's
data plan table, never implemented until now. Mirrors
scripts/build_nea_heldout.py.

Usage: python scripts/build_modis_heldout.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import DRY_SEASON_MONTHS, INTERIM_DIR, MODIS_LST_SCALE_M, SG_BBOX, SUBZONE_ID_PROPERTY, YEARS
from src.ingest.gee import fetch_modis_lst_collection, init_ee
from src.ingest.subzones import as_ee_feature_collection, fetch_subzones_geojson
from validation.input_validation.modis_heldout import build_modis_heldout

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "modis_heldout_lst.csv"


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

    modis_collection = fetch_modis_lst_collection(sg_bbox, YEARS, DRY_SEASON_MONTHS, day_or_night="day")
    modis_composite = modis_collection.median().clip(sg_bbox)

    heldout_df = build_modis_heldout(modis_composite, subzones_fc, SUBZONE_ID_PROPERTY, heat_ids, scale=MODIS_LST_SCALE_M)
    heldout_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(heldout_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
