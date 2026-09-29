#!/usr/bin/env python
"""RF trial, texture variant (checkpoint_DynamicWorld branch): same as
scripts/train_landcover_rf_dynamicworld.py (Dynamic World labels, same 15m
exclusion, same evaluation), but RF's feature image gets 6 extra bands --
the local mean and standard deviation of NDVI/NDBI/NDWI over a fixed 3x3
(30m) neighborhood (src/landcover/rf_baseline.py::add_neighborhood_texture_bands).

Why: a diagnostic (majority-voting the plain RF's OUTPUT over a spatial
window, then re-scored against the 300 hand labels) showed most of RF's
accuracy gap vs U-Net closes with a bit of spatial context -- but choosing
that window's SIZE by testing it against the validation labels would be
circular. This instead gives the classifier texture as an INPUT feature, at
a modest neighborhood size fixed in advance (see
add_neighborhood_texture_bands's docstring for why 3x3), then trains and
scores it exactly once, the same procedure as every other model here.

Writes to its own raster path, its own GEE classifier asset, and its own
output folder -- nothing from scripts/train_landcover_rf_dynamicworld.py or
the production pipeline is touched.

Usage: python scripts/train_landcover_rf_dynamicworld_texture.py [--use-asset-cache] [--with-probabilities]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import (
    ALL_FEATURE_BANDS,
    DRY_SEASON_MONTHS,
    DW_TRAIN_END,
    DW_TRAIN_START,
    GEE_PROJECT_ID,
    INDEX_BANDS,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    TARGET_SCALE_M,
    YEARS,
)
from src.ingest.dynamic_world import get_dynamicworld_bucket_image
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.landcover.rf_baseline import (
    add_neighborhood_texture_bands,
    build_feature_image,
    build_training_region,
    classify,
    classify_probability,
    export_classified_raster,
    extract_training_samples,
    train_rf_classifier,
)
from validation.landcover_validation.classifier_evaluation import (
    compare_classifiers,
    evaluate_classifier,
    save_evaluation_outputs,
)

TEXTURE_BANDS = [f"{b}_mean" for b in INDEX_BANDS] + [f"{b}_stdDev" for b in INDEX_BANDS]
FEATURE_BANDS_TEXTURE = ALL_FEATURE_BANDS + TEXTURE_BANDS

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
RF_RASTER_PATH_DW = PROCESSED_DIR / "landcover" / "rf_landcover_texture.tif"
RF_PROB_RASTER_PATH_DW = PROCESSED_DIR / "landcover" / "rf_landcover_texture_prob.tif"
EVAL_DIR_DW = PROCESSED_DIR / "landcover" / "evaluation" / "texture"
CLASSIFIER_ASSET_ID_DW = f"projects/{GEE_PROJECT_ID}/assets/rf_landcover_classifier_texture"
PRODUCTION_COMPARISON_CSV = PROCESSED_DIR / "landcover" / "evaluation" / "comparison_table.csv"
TRIAL_COMPARISON_CSV = EVAL_DIR_DW / "comparison_table.csv"


def main(use_asset_cache: bool = False, with_probabilities: bool = False):
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(f"{VALIDATION_CSV} not found — label the validation sample first.")

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    boundary = dissolve_boundary(subzones_fc)

    feature_image, valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    feature_image = add_neighborhood_texture_bands(feature_image)
    print(f"Feature bands (with texture): {feature_image.bandNames().getInfo()}")

    dw_bucket_image = get_dynamicworld_bucket_image(
        boundary, S2_UTM_CRS, TARGET_SCALE_M, DW_TRAIN_START, DW_TRAIN_END, valid_mask=valid_mask,
    )

    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(validation_df)} validation points from {VALIDATION_CSV}")

    training_region, _ = build_training_region(boundary, validation_df)
    training_fc, _ = extract_training_samples(
        feature_image, dw_bucket_image, training_region, class_band="dw_class",
    )

    effective_asset_cache = use_asset_cache and not with_probabilities
    classifier = train_rf_classifier(
        training_fc, feature_bands=FEATURE_BANDS_TEXTURE, use_asset_cache=effective_asset_cache,
        class_band="dw_class", asset_id=CLASSIFIER_ASSET_ID_DW,
    )
    classified = classify(feature_image, classifier, boundary)
    export_classified_raster(classified, boundary, out_path=RF_RASTER_PATH_DW)
    print(f"\nDynamic-World-trained, texture-augmented RF raster: {RF_RASTER_PATH_DW}")

    if with_probabilities:
        prob_image = classify_probability(feature_image, classifier, boundary, valid_mask)
        export_classified_raster(prob_image, boundary, out_path=RF_PROB_RASTER_PATH_DW)
        print(f"Probability raster: {RF_PROB_RASTER_PATH_DW}")

    results = {"rf_dynamicworld_texture": evaluate_classifier(RF_RASTER_PATH_DW, validation_df, "rf_dynamicworld_texture")}
    comparison_df = compare_classifiers(results)

    existing = [p for p in (TRIAL_COMPARISON_CSV, PRODUCTION_COMPARISON_CSV) if p.exists()]
    if existing:
        prior = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        prior = prior[prior["model"] != "rf_dynamicworld_texture"]
        comparison_df = pd.concat([prior, comparison_df], ignore_index=True).drop_duplicates("model", keep="last")

    print("\n" + comparison_df.to_string(index=False))
    save_evaluation_outputs(results, comparison_df, out_dir=EVAL_DIR_DW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-asset-cache", action="store_true",
        help="Cache the trained classifier to its own GEE asset and reuse it on later runs. "
             "Off by default, so a re-run always retrains fresh.",
    )
    parser.add_argument(
        "--with-probabilities", action="store_true",
        help="Also export a per-class probability raster (needed for a hybrid rebuild).",
    )
    args = parser.parse_args()
    main(use_asset_cache=args.use_asset_cache, with_probabilities=args.with_probabilities)
