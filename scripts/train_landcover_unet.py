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
import mlflow
import pandas as pd

from config import settings
from config.settings import DRY_SEASON_MONTHS, INTERIM_DIR, S2_CLOUD_PROB_MAX, S2_UTM_CRS, SG_BBOX, TARGET_SCALE_M, YEARS
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.ingest.worldcover import get_worldcover_bucket_image
from src.landcover.rf_baseline import build_feature_image, build_training_region
from src.landcover.unet import (
    CLASSIFIED_RASTER_PATH,
    MODEL_SAVE_PATH,
    build_training_stack,
    export_inference_patches,
    export_training_patches,
    informal_accuracy_check,
    parse_training_patches,
    run_inference_and_reconstruct,
    train_unet,
    training_fingerprint,
)
from src.utils.experiment_tracking import start_run
from src.utils.seed import set_all_seeds

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"


def main(force_retrain: bool = False, with_probabilities: bool = False):
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

    with start_run("unet"):
        mlflow.log_params({
            "unet_patch_size": settings.UNET_PATCH_SIZE,
            "unet_batch_size": settings.UNET_BATCH_SIZE,
            "unet_epochs": settings.UNET_EPOCHS,
            "unet_learning_rate": settings.UNET_LEARNING_RATE,
            "unet_base_filters": settings.UNET_BASE_FILTERS,
            "unet_early_stop_patience": settings.UNET_EARLY_STOP_PATIENCE,
            "unet_train_val_split": settings.UNET_TRAIN_VAL_SPLIT,
            "n_validation_points": len(validation_df),
            "force_retrain": force_retrain,
            "with_probabilities": with_probabilities,
        })

        # export_training_patches fingerprints VALIDATION_CSV and compares it to
        # the fingerprint saved alongside the cached patches -- so it re-exports
        # automatically when the validation sample gets relabeled (the exclusion
        # zone moved, old patches would be wrong), and reuses the cache for free
        # when you just want to retrain with different hyperparameters against
        # the same labels. No need to route force_retrain through here at all.
        train_patch_dir = export_training_patches(training_stack, training_region, VALIDATION_CSV)
        train_ds, val_ds, _ = parse_training_patches(train_patch_dir)

        # Same fingerprint used to decide the patch cache also guards the model
        # cache one level up -- otherwise patches can correctly auto-refresh on
        # a relabel while train_unet still silently reloads the OLD model that
        # was actually fit on the since-replaced patches.
        data_fingerprint = training_fingerprint(VALIDATION_CSV)
        model, history = train_unet(train_ds, val_ds, data_fingerprint=data_fingerprint, force_retrain=force_retrain)

        if history is not None:
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
                history.history["loss"], history.history["accuracy"],
                history.history["val_loss"], history.history["val_accuracy"],
            )):
                mlflow.log_metrics(
                    {"train_loss": loss, "train_accuracy": acc, "val_loss": val_loss, "val_accuracy": val_acc},
                    step=epoch,
                )
            mlflow.log_metric("best_val_loss", min(history.history["val_loss"]))

        inference_patch_dir = export_inference_patches(feature_image, boundary)
        raster_path, crs_str, prob_path = run_inference_and_reconstruct(
            model, inference_patch_dir, boundary, also_write_probabilities=with_probabilities,
        )

        accuracy, crosstab = informal_accuracy_check(raster_path, crs_str, validation_df)
        print(f"\nDone. Classified raster: {CLASSIFIED_RASTER_PATH}")

        mlflow.log_metric("informal_accuracy", accuracy)
        mlflow.log_metric("n_scored", int(crosstab.values.sum()))
        mlflow.log_artifact(str(raster_path))
        mlflow.log_artifact(str(MODEL_SAVE_PATH))

        if prob_path:
            print(f"Probability raster: {prob_path}")
            mlflow.log_artifact(str(prob_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a saved model already exists.")
    parser.add_argument("--with-probabilities", action="store_true",
                         help="Also export a per-class probability raster (needed for the soft-voting ensemble).")
    args = parser.parse_args()
    main(force_retrain=args.force_retrain, with_probabilities=args.with_probabilities)
