#!/usr/bin/env python
"""Clean WorldCover RF baseline (checkpoint_DynamicWorld branch): retrain the
WorldCover-labelled RF with the 15 m exclusion buffer around the CURRENT 300
validation points, and rebuild the WorldCover hybrid from it.

Why this exists: the production WorldCover RF (data/processed/landcover/
rf_landcover.tif, 6 Aug) was trained when the validation sample had 200
points, so its exclusion buffer does not cover the current 300 (a different
draw). Some current points may sit next to its training pixels, which would
inflate its accuracy. Every other model in the comparison was trained against
the current 300, so this baseline has to be too for the comparison to be fair.

Same features, labels, trees and sampling as scripts/train_landcover_rf.py;
the only change is the validation set the buffer is built from. Writes its own
files and never touches the shared GEE classifier asset (fresh train, no asset
cache), so the production RF, hybrid and rf_landcover_classifier asset are
untouched. The hybrid reuses the WorldCover U-Net's probabilities
(unet_landcover_prob.tif, 18 Sep), which were already trained against the 300.

Usage: python scripts/train_landcover_rf_worldcover_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
    DRY_SEASON_MONTHS, INTERIM_DIR, PROCESSED_DIR, S2_CLOUD_PROB_MAX, S2_UTM_CRS, SG_BBOX,
    TARGET_SCALE_M, UNET_PROB_RASTER_PATH, YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import get_worldcover_bucket_image
from src.landcover.hybrid import build_hybrid
from src.landcover.rf_baseline import (
    build_feature_image, build_training_region, classify, classify_probability,
    export_classified_raster, extract_training_samples, train_rf_classifier,
)
from validation.landcover_validation.classifier_evaluation import evaluate_classifier

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
LC = PROCESSED_DIR / "landcover"
RF_RASTER = LC / "rf_landcover_wc300.tif"
RF_PROB_RASTER = LC / "rf_landcover_wc300_prob.tif"
HYBRID_RASTER = LC / "hybrid_landcover_wc300.tif"
HYBRID_PROB_RASTER = LC / "hybrid_landcover_wc300_prob.tif"


def main():
    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Exclusion buffer built from {len(validation_df)} validation points ({VALIDATION_CSV.name}).")

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    boundary = dissolve_boundary(as_ee_feature_collection(fetch_subzones_geojson()))

    feature_image, valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    wc_bucket_image = get_worldcover_bucket_image(boundary, S2_UTM_CRS, TARGET_SCALE_M, valid_mask=valid_mask)

    training_region, _ = build_training_region(boundary, validation_df)
    training_fc, _ = extract_training_samples(feature_image, wc_bucket_image, training_region)

    # Fresh train, no asset cache: a cached asset would be the old 200-point
    # classifier, and MULTIPROBABILITY output needs a same-session classifier.
    classifier = train_rf_classifier(training_fc, use_asset_cache=False)
    export_classified_raster(classify(feature_image, classifier, boundary), boundary, out_path=RF_RASTER)
    export_classified_raster(
        classify_probability(feature_image, classifier, boundary, valid_mask), boundary, out_path=RF_PROB_RASTER,
    )

    build_hybrid(RF_PROB_RASTER, UNET_PROB_RASTER_PATH, label_out_path=HYBRID_RASTER, prob_out_path=HYBRID_PROB_RASTER)

    for name, path in (("rf_worldcover_300", RF_RASTER), ("hybrid_worldcover_300", HYBRID_RASTER)):
        evaluate_classifier(path, validation_df, name)


if __name__ == "__main__":
    main()
