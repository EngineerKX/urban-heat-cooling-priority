#!/usr/bin/env python
"""RF trial, smoothing variant (checkpoint_DynamicWorld branch): applies a
fixed, modest 3x3 (30m) spatial majority-vote smoothing to the plain
Dynamic-World-trained RF's output (scripts/train_landcover_rf_dynamicworld.py)
-- see src/landcover/spatial_smoothing.py for why this window and not a
bigger, validation-tuned one.

This is the "keep it simple" choice: plain RF (no retraining, no texture
features -- see scripts/train_landcover_rf_dynamicworld_texture.py, which
tested that route and found it didn't stack with smoothing) plus a cheap,
principled post-processing step. Writes its own raster(s) and its own
output-folder row; nothing produced by the plain/texture RF trials, U-Net
trial, or hybrid trial is touched.

Usage: python scripts/smooth_landcover_rf_dynamicworld.py [--window N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import INTERIM_DIR, PROCESSED_DIR
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.spatial_smoothing import DEFAULT_WINDOW, write_smoothed_outputs
from validation.landcover_validation.classifier_evaluation import (
    compare_classifiers,
    evaluate_classifier,
    save_evaluation_outputs,
)

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
LANDCOVER_DIR = PROCESSED_DIR / "landcover"
RF_RASTER_PATH_DW = LANDCOVER_DIR / "rf_landcover_dw_trial.tif"
RF_SMOOTH_RASTER_PATH = LANDCOVER_DIR / "rf_landcover_dw_trial_smoothed.tif"
RF_SMOOTH_PROB_RASTER_PATH = LANDCOVER_DIR / "rf_landcover_dw_trial_smoothed_prob.tif"
EVAL_DIR_DW = LANDCOVER_DIR / "evaluation" / "dw_trial"
TRIAL_COMPARISON_CSV = EVAL_DIR_DW / "comparison_table.csv"
PRODUCTION_COMPARISON_CSV = LANDCOVER_DIR / "evaluation" / "comparison_table.csv"


def main(window: int = DEFAULT_WINDOW):
    if not RF_RASTER_PATH_DW.exists():
        raise FileNotFoundError(f"{RF_RASTER_PATH_DW} not found — run scripts/train_landcover_rf_dynamicworld.py first.")
    if not VALIDATION_CSV.exists():
        raise FileNotFoundError(f"{VALIDATION_CSV} not found — label the validation sample first.")

    class_ids = sorted(BUCKET_NAMES.keys())
    label_path, prob_path = write_smoothed_outputs(
        RF_RASTER_PATH_DW, class_ids, BUCKET_NAMES, window=window,
        label_out_path=RF_SMOOTH_RASTER_PATH, prob_out_path=RF_SMOOTH_PROB_RASTER_PATH,
    )

    validation_df = pd.read_csv(VALIDATION_CSV)
    model_name = f"rf_dynamicworld_smooth{window}x{window}"
    results = {model_name: evaluate_classifier(label_path, validation_df, model_name)}
    comparison_df = compare_classifiers(results)

    existing = [p for p in (TRIAL_COMPARISON_CSV, PRODUCTION_COMPARISON_CSV) if p.exists()]
    if existing:
        prior = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        prior = prior[prior["model"] != model_name]
        comparison_df = pd.concat([prior, comparison_df], ignore_index=True).drop_duplicates("model", keep="last")

    print("\n" + comparison_df.to_string(index=False))
    save_evaluation_outputs(results, comparison_df, out_dir=EVAL_DIR_DW)
    print(f"\nSmoothed RF probability raster (for a hybrid rebuild, if wanted): {prob_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                         help=f"Smoothing window in pixels (default {DEFAULT_WINDOW} = 30m, fixed in advance -- "
                              "see src/landcover/spatial_smoothing.py's docstring before changing this.")
    args = parser.parse_args()
    main(window=args.window)
