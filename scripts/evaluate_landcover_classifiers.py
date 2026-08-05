#!/usr/bin/env python
"""Formal RF-vs-U-Net-vs-ensemble land-cover evaluation: confusion matrix,
per-class precision/recall/F1, macro/weighted F1, scored identically for
all three classifiers against the same hand-labeled validation points.
Replaces the ad hoc informal_accuracy_check() sanity checks each training
script runs on its own -- this is the formal version those explicitly defer.

Usage: python scripts/evaluate_landcover_classifiers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import INTERIM_DIR
from config.settings import UNET_CLASSIFIED_RASTER_PATH as UNET_RASTER_PATH
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH
from src.landcover.rf_baseline import RF_RASTER_PATH
from validation.landcover_validation.classifier_evaluation import (
    compare_classifiers,
    evaluate_classifier,
    save_evaluation_outputs,
)

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"

RASTERS = {
    "rf": RF_RASTER_PATH,
    "unet": UNET_RASTER_PATH,
    "ensemble": ENSEMBLE_RASTER_PATH,
}


def main():
    missing = [str(p) for p in RASTERS.values() if not p.exists()]
    if not VALIDATION_CSV.exists():
        missing.append(str(VALIDATION_CSV))
    if missing:
        raise FileNotFoundError(
            f"Missing prerequisite file(s): {', '.join(missing)} — run "
            "train_landcover_rf.py --with-probabilities, "
            "the U-Net Colab notebook + pull_models.py + "
            "run_landcover_unet_inference.py --with-probabilities, and "
            "build_landcover_ensemble.py first."
        )

    validation_df = pd.read_csv(VALIDATION_CSV)
    print(f"Loaded {len(validation_df)} validation points from {VALIDATION_CSV}")

    results = {
        model_name: evaluate_classifier(raster_path, validation_df, model_name)
        for model_name, raster_path in RASTERS.items()
    }
    comparison_df = compare_classifiers(results)
    print("\n" + comparison_df.to_string(index=False))

    save_evaluation_outputs(results, comparison_df)
    return comparison_df


if __name__ == "__main__":
    main()
