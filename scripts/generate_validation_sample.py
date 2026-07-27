#!/usr/bin/env python
"""Generate the stratified 200-point validation sample for hand-labeling.
Replaces generate_validation_sample.ipynb (Track B / SB1).

Usage: python scripts/generate_validation_sample.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee

from config.settings import (
    DRY_SEASON_MONTHS,
    INTERIM_DIR,
    RANDOM_SEED,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    TARGET_SCALE_M,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import get_worldcover_bucket_image
from validation.input_validation.labeling_sample import (
    allocate_sample_sizes,
    build_labeling_table,
    build_valid_data_mask,
    class_area_histogram,
    draw_stratified_sample,
    export_labeling_table,
)

OUT_DIR = INTERIM_DIR / "validation_sample"
TOTAL_POINTS = 200
COVERAGE_THRESHOLD = 0.90


def main(force: bool = False):
    csv_path = OUT_DIR / "validation_sample_200.csv"
    if csv_path.exists() and not force:
        print(f"{csv_path} already exists — skipping recompute (pass --force to redraw the sample).")
        return csv_path

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    sg_boundary = dissolve_boundary(subzones_fc)

    valid_mask = build_valid_data_mask(sg_bbox, sg_boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX, S2_UTM_CRS, TARGET_SCALE_M)
    wc_sampling_frame = get_worldcover_bucket_image(sg_boundary, S2_UTM_CRS, TARGET_SCALE_M, valid_mask=valid_mask)

    coverage = wc_sampling_frame.select("wc_class").mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=sg_boundary, scale=TARGET_SCALE_M, maxPixels=1e10, bestEffort=True,
    ).getInfo()
    coverage_frac = list(coverage.values())[0]
    print(f"Sampling-frame valid-pixel coverage: {coverage_frac * 100:.1f}%")
    if coverage_frac < COVERAGE_THRESHOLD:
        print(f"⚠️  Below {COVERAGE_THRESHOLD * 100:.0f}% — widen the season window or cloud thresholds if this looks wrong.")

    class_hist = class_area_histogram(wc_sampling_frame, sg_boundary, TARGET_SCALE_M)
    class_alloc = allocate_sample_sizes(class_hist, TOTAL_POINTS)
    sample_records, drawn_counts, shortfalls = draw_stratified_sample(wc_sampling_frame, sg_boundary, TARGET_SCALE_M, class_alloc, RANDOM_SEED)

    if len(sample_records) != TOTAL_POINTS:
        print(f"⚠️  Drew {len(sample_records)} points, expected {TOTAL_POINTS}.")

    labeling_df = build_labeling_table(sample_records)
    export_labeling_table(labeling_df, OUT_DIR, prefix="validation_sample_200")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
