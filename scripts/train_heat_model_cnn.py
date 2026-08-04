#!/usr/bin/env python
"""Train S5's CNN patch-level half (C2) -- predicts lst_bicubic10 from
Sentinel-2 + spectral-index + land-cover 10m patches, reusing U-Net's
already-downloaded inference patches (see src/heat_model/cnn_patches.py's
module docstring). Needs WSL2's GPU TensorFlow, same environment as
scripts/train_landcover_unet.py -- native Windows TensorFlow is confirmed
broken in this repo's .venv.

Usage (from WSL2): python scripts/train_heat_model_cnn.py [--force-retrain] [--force-export]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import mlflow

from config import settings
from config.settings import (
    CNN_MODEL_SAVE_PATH,
    DRY_SEASON_MONTHS,
    GEE_EXPORT_BUCKET,
    INTERIM_DIR,
    LANDSAT_CLOUD_COVER_MAX,
    NATIVE_SCALE_M,
    S2_UTM_CRS,
    SG_BBOX,
    TARGET_SCALE_M,
    YEARS,
)
from src.downscaling.variants import build_lst_30m, variant_bicubic10
from src.heat_model.cnn_patches import build_local_feature_target_patches, train_cnn_regressor
from src.ingest.gee import export_geotiff_to_gcs, init_ee
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH
from src.utils.experiment_tracking import HEAT_MODEL_EXPERIMENT_NAME, start_run
from src.utils.seed import set_all_seeds

INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
MIXER_JSON_PATH = INFERENCE_PATCH_DIR / "unet_inference.json"
LST_BICUBIC10_RASTER_PATH = INTERIM_DIR / "lst_bicubic10_full.tif"


def _export_lst_bicubic10():
    """Only new GEE export S5's CNN half needs -- variant_bicubic10 is
    reused completely unmodified, over the same season window as the
    production heat variants."""
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    lst_30m = build_lst_30m(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX, S2_UTM_CRS, NATIVE_SCALE_M)
    bicubic_image = variant_bicubic10(lst_30m, S2_UTM_CRS, TARGET_SCALE_M)
    return export_geotiff_to_gcs(
        bicubic_image, description="lst_bicubic10_full", bucket=GEE_EXPORT_BUCKET,
        prefix="heat_model/lst_bicubic10_full", region=sg_bbox, scale=TARGET_SCALE_M, crs=S2_UTM_CRS,
        out_path=LST_BICUBIC10_RASTER_PATH,
    )


def main(force_retrain: bool = False, force_export: bool = False):
    if CNN_MODEL_SAVE_PATH.exists() and not force_retrain:
        print(f"{CNN_MODEL_SAVE_PATH} already exists — skipping retrain (pass --force-retrain to redo).")
        return CNN_MODEL_SAVE_PATH

    if not MIXER_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{MIXER_JSON_PATH} not found — run scripts/train_landcover_unet.py first "
            f"(this script reuses its already-exported inference patches)."
        )
    if not ENSEMBLE_RASTER_PATH.exists():
        raise FileNotFoundError(f"{ENSEMBLE_RASTER_PATH} not found — run scripts/build_landcover_ensemble.py first.")

    set_all_seeds()

    if LST_BICUBIC10_RASTER_PATH.exists() and not force_export:
        print(f"{LST_BICUBIC10_RASTER_PATH} already exists — skipping export (pass --force-export to redo).")
    else:
        init_ee()
        _export_lst_bicubic10()

    X, y, valid_mask = build_local_feature_target_patches(
        INFERENCE_PATCH_DIR, MIXER_JSON_PATH, ENSEMBLE_RASTER_PATH, LST_BICUBIC10_RASTER_PATH,
    )

    with start_run("cnn", experiment_name=HEAT_MODEL_EXPERIMENT_NAME):
        mlflow.log_params({
            "n_patches": int(X.shape[0]), "patch_size": int(X.shape[1]), "n_channels": int(X.shape[3]),
            "cnn_base_filters": settings.UNET_BASE_FILTERS, "cnn_epochs": settings.UNET_EPOCHS,
            "cnn_learning_rate": settings.UNET_LEARNING_RATE, "cnn_batch_size": settings.UNET_BATCH_SIZE,
            "cnn_early_stop_patience": settings.UNET_EARLY_STOP_PATIENCE,
            "force_retrain": force_retrain,
        })

        model, history = train_cnn_regressor(X, y, valid_mask, force_retrain=force_retrain)

        if history is not None:
            for epoch, (loss, rmse, val_loss, val_rmse) in enumerate(zip(
                history.history["loss"], history.history["rmse"],
                history.history["val_loss"], history.history["val_rmse"],
            )):
                mlflow.log_metrics({"train_loss": loss, "train_rmse": rmse, "val_loss": val_loss, "val_rmse": val_rmse}, step=epoch)
            mlflow.log_metric("best_val_loss", min(history.history["val_loss"]))
            mlflow.log_metric("best_val_rmse", min(history.history["val_rmse"]))

        mlflow.log_artifact(str(CNN_MODEL_SAVE_PATH))

    print(f"\nSaved: {CNN_MODEL_SAVE_PATH}")
    return CNN_MODEL_SAVE_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if a saved model already exists.")
    parser.add_argument("--force-export", action="store_true", help="Re-export lst_bicubic10 even if already cached.")
    args = parser.parse_args()
    main(force_retrain=args.force_retrain, force_export=args.force_export)
