#!/usr/bin/env python
"""Season-window candidate diagnostic. Replaces season_window_diagnostic.ipynb.
Locked result already lives in config.settings.DRY_SEASON_MONTHS — this
script exists to re-justify that choice if it's ever revisited.

Usage: python scripts/diagnose_season_window.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee

from config.settings import LANDSAT_CLOUD_COVER_MAX, SG_BBOX, YEARS
from src.ingest.gee import init_ee
from validation.input_validation.season_window import recommend, run_season_window_candidates

CANDIDATES = {
    "A: Apr-May (inter-monsoon 1)": [4, 5],
    "B: Oct-Nov (inter-monsoon 2)": [10, 11],
    "C: Apr/May + Oct/Nov (both inter-monsoon)": [4, 5, 10, 11],
    "D: Feb-Apr (earlier placeholder guess)": [2, 3, 4],
    "E: All 12 months (reference, no season restriction)": list(range(1, 13)),
}
CLOUD_THRESHOLDS = [70, 20]


def main():
    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    results_df = run_season_window_candidates(sg_bbox, YEARS, CANDIDATES, CLOUD_THRESHOLDS)
    recommend(results_df, cloud_cover_max_col_value=LANDSAT_CLOUD_COVER_MAX)


if __name__ == "__main__":
    main()
