#!/usr/bin/env python
"""Train the Random Forest land-cover baseline and classify all of
Singapore. Replaces train_rf_baseline.ipynb (Track B / RF1).

Training labels follow config.settings.LANDCOVER_LABEL_SOURCE (Dynamic World
since 2026-09-29; WorldCover is the comparison). The optional GEE classifier
cache is keyed on the label source AND the validation points the training
region excludes (src/landcover/labels.py::rf_classifier_asset_name), so a
classifier trained on different labels or a different validation sample is
never reused.

NOT included here: the formal RF-vs-U-Net-vs-hybrid evaluation (confusion
matrix, per-class F1) — that runs once all three exist, scored identically.

Usage: python scripts/train_landcover_rf.py [--force-retrain]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import mlflow
import pandas as pd

from config.settings import (
    DRY_SEASON_MONTHS,
    GEE_PROJECT_ID,
    INTERIM_DIR,
    RF_BAG_FRACTION,
    RF_MIN_LEAF_POPULATION,
    RF_NUM_TREES,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    TARGET_SCALE_M,
    TRAINING_POINTS_PER_CLASS,
    VALIDATION_EXCLUSION_BUFFER_M,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.landcover.labels import get_training_label_image, label_source_tag, rf_classifier_asset_name
from src.landcover.rf_baseline import (
    RF_PROB_RASTER_PATH,
    RF_RASTER_PATH,
    build_feature_image,
    build_training_region,
    classify,
    classify_probability,
    export_classified_raster,
    extract_training_samples,
    train_rf_classifier,
)
from src.utils.experiment_tracking import log_artifact_safe, start_run
from validation.landcover_validation.classifier_evaluation import evaluate_classifier

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"


def main(use_asset_cache: bool = True, with_probabilities: bool = False):
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
    label_image, class_band = get_training_label_image(boundary, S2_UTM_CRS, TARGET_SCALE_M, valid_mask=valid_mask)

    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(validation_df)} validation points from {VALIDATION_CSV}")

    training_region, _ = build_training_region(boundary, validation_df)
    training_fc, _ = extract_training_samples(feature_image, label_image, training_region, class_band=class_band)
    asset_id = f"projects/{GEE_PROJECT_ID}/assets/{rf_classifier_asset_name(VALIDATION_CSV)}"

    with start_run("rf"):
        mlflow.log_params({
            "rf_num_trees": RF_NUM_TREES,
            "rf_min_leaf_population": RF_MIN_LEAF_POPULATION,
            "rf_bag_fraction": RF_BAG_FRACTION,
            "training_points_per_class": TRAINING_POINTS_PER_CLASS,
            "validation_exclusion_buffer_m": VALIDATION_EXCLUSION_BUFFER_M,
            "n_validation_points": len(validation_df),
            "use_asset_cache": use_asset_cache,
            "with_probabilities": with_probabilities,
            "label_source": label_source_tag(),
            "classifier_asset": asset_id,
        })

        # GEE constraint discovered empirically (not documented anywhere): a
        # classifier round-tripped through Export.classifier.toAsset /
        # ee.Classifier.load() only supports CLASSIFICATION output mode --
        # calling setOutputMode("MULTIPROBABILITY") on a loaded asset classifier
        # fails with "Trainer only supports the mode 'CLASSIFICATION'". Only a
        # classifier trained fresh in the current session supports it. So when
        # probabilities are requested, force a fresh train regardless of
        # --no-asset-cache, and use that SAME classifier for both outputs below
        # -- using a different (cached) classifier for the hard-label raster
        # would let it disagree with the probability raster's own argmax.
        effective_asset_cache = use_asset_cache and not with_probabilities
        if with_probabilities and use_asset_cache:
            print("--with-probabilities forces a fresh classifier train (asset-cached "
                  "classifiers don't support MULTIPROBABILITY output mode).")
        classifier = train_rf_classifier(
            training_fc, use_asset_cache=effective_asset_cache, class_band=class_band, asset_id=asset_id,
        )
        classified = classify(feature_image, classifier, boundary)

        # Export first, then score the downloaded raster locally -- the same code
        # (and the same point sampling) as the formal evaluation. Scoring the
        # un-exported image with a synchronous GEE sampleRegions call timed out
        # (EE's 5-minute limit) on 2026-09-29 before anything had been saved.
        export_classified_raster(classified, boundary)
        print(f"\nDone. Classified raster: {RF_RASTER_PATH}")

        result = evaluate_classifier(RF_RASTER_PATH, validation_df, "rf")
        mlflow.log_metrics({
            "accuracy": result["accuracy"], "macro_f1": result["macro_f1"], "n_scored": result["n_scored"],
        })
        log_artifact_safe(RF_RASTER_PATH)

        if with_probabilities:
            prob_image = classify_probability(feature_image, classifier, boundary, valid_mask)
            export_classified_raster(prob_image, boundary, out_path=RF_PROB_RASTER_PATH)
            print(f"Probability raster: {RF_PROB_RASTER_PATH}")
            log_artifact_safe(RF_PROB_RASTER_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-asset-cache", action="store_true", help="Always retrain instead of reusing a cached GEE asset classifier.")
    parser.add_argument("--with-probabilities", action="store_true",
                         help="Also export a per-class probability raster (needed for the soft-voting hybrid).")
    args = parser.parse_args()
    main(use_asset_cache=not args.no_asset_cache, with_probabilities=args.with_probabilities)
