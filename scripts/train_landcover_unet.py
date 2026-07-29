#!/usr/bin/env python
"""Train the U-Net land-cover classifier and classify all of Singapore.
Replaces train_unet.ipynb (Track B / UN1). Same feature bands, season
window and validation-point exclusion as train_landcover_rf.py, kept
identical deliberately so the RF-vs-U-Net comparison isolates model choice.

Compute note: the compute-heavy step (UN1.9 equivalent) is TensorFlow
training, which runs locally/on your own GPU, NOT against GEE quota — only
the patch exports count against EECU-hours.

Usage: python scripts/train_landcover_unet.py [--force-retrain]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import pandas as pd

from config.settings import DRY_SEASON_MONTHS, INTERIM_DIR, S2_CLOUD_PROB_MAX, S2_UTM_CRS, SG_BBOX, TARGET_SCALE_M, YEARS
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import get_worldcover_bucket_image
from src.landcover.rf_baseline import build_feature_image, build_training_region
from src.landcover.unet import (
    CLASSIFIED_RASTER_PATH,
    build_training_stack,
    export_inference_patches,
    export_training_patches,
    informal_accuracy_check,
    parse_training_patches,
    run_inference_and_reconstruct,
    train_unet,
)
from src.utils.seed import set_all_seeds

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"


def main(force_retrain: bool = False):
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(
            f"{VALIDATION_CSV} not found — generate + label the validation sample first "
            f"(scripts/generate_validation_sample.py, then app/pages/1_Label_Validation_Points.py)."
        )

    set_all_seeds()
    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    boundary = dissolve_boundary(subzones_fc)

    feature_image, valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    wc_bucket_image = get_worldcover_bucket_image(boundary, S2_UTM_CRS, TARGET_SCALE_M, valid_mask=valid_mask)

    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(validation_df)} validation points from {VALIDATION_CSV}")

    training_region, _ = build_training_region(boundary, validation_df)
    training_stack = build_training_stack(feature_image, wc_bucket_image, training_region)

    train_patch_dir = export_training_patches(training_stack, training_region)
    train_ds, val_ds, _ = parse_training_patches(train_patch_dir)

    model, _ = train_unet(train_ds, val_ds, force_retrain=force_retrain)

    inference_patch_dir = export_inference_patches(feature_image, boundary)
    raster_path, crs_str = run_inference_and_reconstruct(model, inference_patch_dir, boundary)

    informal_accuracy_check(raster_path, crs_str, validation_df)
    print(f"\nDone. Classified raster: {CLASSIFIED_RASTER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a saved model already exists.")
    args = parser.parse_args()
    main(force_retrain=args.force_retrain)
