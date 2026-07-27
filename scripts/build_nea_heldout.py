#!/usr/bin/env python
"""Build the NEA held-out LST validation table. Replaces nea_heldout_lst.ipynb.

Usage: python scripts/build_nea_heldout.py [--force] [--api-key KEY]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import INTERIM_DIR, RANDOM_SEED, SUBZONE_ID_PROPERTY
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from validation.input_validation.nea_heldout import build_nea_heldout

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "nea_heldout_lst.csv"

N_SAMPLE_DAYS = 20
N_MONTHS_BACK = 6


def main(force: bool = False, api_key: str = ""):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    subzones_gdf = as_geodataframe(fetch_subzones_geojson())
    heat_ids = pd.read_csv(HEAT_CSV_PATH)["subzone_id"]

    heldout_df = build_nea_heldout(
        subzones_gdf, SUBZONE_ID_PROPERTY, heat_ids,
        n_sample_days=N_SAMPLE_DAYS, n_months_back=N_MONTHS_BACK, seed=RANDOM_SEED, api_key=api_key,
    )
    heldout_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(heldout_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--api-key", default="", help="Optional data.gov.sg API key (avoids HTTP 429 rate limiting).")
    args = parser.parse_args()
    main(force=args.force, api_key=args.api_key)
