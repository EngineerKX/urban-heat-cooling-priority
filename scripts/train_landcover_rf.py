#!/usr/bin/env python
"""Train the Random Forest land-cover baseline and classify all of
Singapore. Replaces train_rf_baseline.ipynb (Track B / RF1).

NOT included here: the formal RF-vs-U-Net-vs-ensemble evaluation (confusion
matrix, per-class F1) — that runs once all three exist, scored identically.

Usage: python scripts/train_landcover_rf.py [--force-retrain]
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
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    TARGET_SCALE_M,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import get_worldcover_bucket_image
from src.landcover.rf_baseline import (
    RF_RASTER_PATH,
    build_feature_image,
    build_training_region,
    classify,
    export_classified_raster,
    extract_training_samples,
    informal_accuracy_check,
    train_rf_classifier,
)

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"


def main(use_asset_cache: bool = True):
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"{VALIDATION_CSV} not found — generate + label the validation sample first "
            f"(scripts/generate_validation_sample.py, then app/pages/1_Label_Validation_Points.py)."
        )

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    boundary = dissolve_boundary(subzones_fc)

    feature_image, valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    wc_bucket_image = get_worldcover_bucket_image(boundary, S2_UTM_CRS, TARGET_SCALE_M, valid_mask=valid_mask)

    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(validation_df)} validation points from {VALIDATION_CSV}")

    training_region, _ = build_training_region(boundary, validation_df)
    training_fc, _ = extract_training_samples(feature_image, wc_bucket_image, training_region)

    classifier = train_rf_classifier(training_fc, use_asset_cache=use_asset_cache)
    classified = classify(feature_image, classifier, boundary)

    informal_accuracy_check(classified, validation_df)
    export_classified_raster(classified, boundary)
    print(f"\nDone. Classified raster: {RF_RASTER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-asset-cache", action="store_true", help="Always retrain instead of reusing a cached GEE asset classifier.")
    args = parser.parse_args()
    main(use_asset_cache=not args.no_asset_cache)
