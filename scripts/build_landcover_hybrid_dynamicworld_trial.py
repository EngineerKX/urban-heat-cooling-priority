#!/usr/bin/env python
"""Hybrid trial (checkpoint_DynamicWorld branch): soft-vote the Dynamic-
World-labeled RF and U-Net probability rasters into a single hybrid raster,
then run the SAME formal evaluation used for RF/U-Net/hybrid, so the final
accuracy/F1 numbers this branch's Dynamic World trial is judged on are
directly comparable to comparison_table.csv. This is the actual candidate
for the merge decision -- U-Net alone already beat the plain-RF hybrid
(83.1% vs 81.8%), so whether a BETTER-combined hybrid can beat U-Net alone
is the open question this script answers.

`--rf-source raw` (default) uses scripts/train_landcover_rf_dynamicworld.py
--with-probabilities's output. `--rf-source smoothed` uses the "keep it
simple" improved RF -- plain RF + a fixed 3x3 spatial majority-vote smooth,
scripts/smooth_landcover_rf_dynamicworld.py -- the best RF variant tested so
far, so this is the one worth checking against U-Net alone.

Writes its own raster(s), one per --rf-source, and its own output folder;
the production hybrid (`data/processed/landcover/hybrid_landcover.tif`,
`comparison_table.csv`) is never touched.

Usage: python scripts/build_landcover_hybrid_dynamicworld_trial.py [--rf-source raw|smoothed] [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import INTERIM_DIR, PROCESSED_DIR
from src.landcover.hybrid import build_hybrid
from validation.landcover_validation.classifier_evaluation import (
    compare_classifiers,
    evaluate_classifier,
    save_evaluation_outputs,
)

VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"
LANDCOVER_DIR = PROCESSED_DIR / "landcover"
UNET_PROB_RASTER_PATH_DW = LANDCOVER_DIR / "unet_landcover_dw_trial_prob.tif"
EVAL_DIR_DW = LANDCOVER_DIR / "evaluation" / "dw_trial"
TRIAL_COMPARISON_CSV = EVAL_DIR_DW / "comparison_table.csv"
PRODUCTION_COMPARISON_CSV = LANDCOVER_DIR / "evaluation" / "comparison_table.csv"

RF_SOURCES = {
    "raw": (LANDCOVER_DIR / "rf_landcover_dw_trial_prob.tif", "hybrid_dynamicworld"),
    "smoothed": (LANDCOVER_DIR / "rf_landcover_dw_trial_smoothed_prob.tif", "hybrid_dynamicworld_rfsmoothed"),
}


def main(rf_source: str = "raw", force: bool = False):
    rf_prob_path, model_name = RF_SOURCES[rf_source]
    hybrid_raster_path = LANDCOVER_DIR / f"{model_name}.tif"
    hybrid_prob_raster_path = LANDCOVER_DIR / f"{model_name}_prob.tif"

    if hybrid_raster_path.exists() and not force:
        print(f"{hybrid_raster_path} already exists — skipping recompute (pass --force to rebuild).")
    else:
        missing = [p for p in (rf_prob_path, UNET_PROB_RASTER_PATH_DW) if not p.exists()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Missing probability raster(s): {names} — run "
                "'python scripts/train_landcover_rf_dynamicworld.py --with-probabilities' "
                "(and/or 'python scripts/smooth_landcover_rf_dynamicworld.py' for --rf-source smoothed) and "
                "the U-Net Dynamic World trial notebook + "
                "'python scripts/run_landcover_unet_inference_dynamicworld_trial.py --with-probabilities' first."
            )
        label_path, _prob_path = build_hybrid(
            rf_prob_path, UNET_PROB_RASTER_PATH_DW,
            label_out_path=hybrid_raster_path, prob_out_path=hybrid_prob_raster_path,
        )
        print(f"\nDynamic-World-trained hybrid raster (rf_source={rf_source}): {label_path}")

    if not VALIDATION_CSV.exists():
        return

    validation_df = pd.read_csv(VALIDATION_CSV)
    results = {model_name: evaluate_classifier(hybrid_raster_path, validation_df, model_name)}
    comparison_df = compare_classifiers(results)

    existing = [p for p in (TRIAL_COMPARISON_CSV, PRODUCTION_COMPARISON_CSV) if p.exists()]
    if existing:
        prior = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        prior = prior[prior["model"] != model_name]
        comparison_df = pd.concat([prior, comparison_df], ignore_index=True).drop_duplicates("model", keep="last")

    print("\n" + comparison_df.to_string(index=False))
    save_evaluation_outputs(results, comparison_df, out_dir=EVAL_DIR_DW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-source", choices=list(RF_SOURCES), default="raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(rf_source=args.rf_source, force=args.force)
