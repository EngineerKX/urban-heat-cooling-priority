#!/usr/bin/env python
"""Run the trained U-Net over all of Singapore and reconstruct the
classified (+ optional probability) raster. Local-only, CPU -- U-Net
inference was never the GPU bottleneck, only training was (which now
happens exclusively in Colab, see notebooks/colab_training/train_unet.ipynb).

Replaces the inference half of the old (deleted) scripts/train_landcover_unet.py.

Usage:
  python scripts/run_landcover_unet_inference.py [--force-export] [--with-probabilities]
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
    SG_BBOX,
    UNET_CLASSIFIED_RASTER_PATH,
    UNET_MODEL_SAVE_PATH,
    YEARS,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, dissolve_boundary, fetch_subzones_geojson
from src.landcover.rf_baseline import build_feature_image
from src.landcover.unet_data import export_inference_patches
from src.landcover.unet_infer import informal_accuracy_check, load_unet, run_inference_and_reconstruct
from src.utils.seed import set_all_seeds

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"


def main(force_export: bool = False, with_probabilities: bool = False):
    if not UNET_MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(
            f"{UNET_MODEL_SAVE_PATH} not found — train it via "
            f"notebooks/colab_training/train_unet.ipynb, then `python scripts/pull_models.py --model unet`."
        )

    set_all_seeds()
    init_ee()
    subzones_fc = as_ee_feature_collection(fetch_subzones_geojson())
    boundary = dissolve_boundary(subzones_fc)
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))

    feature_image, _valid_mask = build_feature_image(sg_bbox, boundary, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)

    inference_patch_dir = export_inference_patches(feature_image, boundary, force_export=force_export)

    model = load_unet(UNET_MODEL_SAVE_PATH)
    raster_path, crs_str, prob_path = run_inference_and_reconstruct(
        model, inference_patch_dir, boundary, also_write_probabilities=with_probabilities,
    )

    if VALIDATION_CSV.exists():
        validation_df = pd.read_csv(VALIDATION_CSV)
        informal_accuracy_check(raster_path, crs_str, validation_df)

    print(f"\nDone. Classified raster: {UNET_CLASSIFIED_RASTER_PATH}")
    if prob_path:
        print(f"Probability raster: {prob_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-export", action="store_true", help="Re-export inference patches even if cached locally/in GCS.")
    parser.add_argument("--with-probabilities", action="store_true",
                         help="Also write a per-class probability raster (needed for the soft-voting ensemble).")
    args = parser.parse_args()
    main(force_export=args.force_export, with_probabilities=args.with_probabilities)
