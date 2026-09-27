#!/usr/bin/env python
"""Hybrid trial (checkpoint_DynamicWorld branch): soft-vote the Dynamic-
World-labeled RF and U-Net probability rasters (from
scripts/train_landcover_rf_dynamicworld.py --with-probabilities and
scripts/run_landcover_unet_inference_dynamicworld_trial.py --with-probabilities)
into a single hybrid raster, then run the SAME formal evaluation used for
RF/U-Net/hybrid, so the final accuracy/F1 numbers this branch's Dynamic
World trial is judged on are directly comparable to comparison_table.csv.

Writes its own raster and its own output folder; the production hybrid
(`data/processed/landcover/hybrid_landcover.tif`, `comparison_table.csv`)
is never touched.

Usage: python scripts/build_landcover_hybrid_dynamicworld_trial.py [--force]
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
RF_PROB_RASTER_PATH_DW = LANDCOVER_DIR / "rf_landcover_dw_trial_prob.tif"
UNET_PROB_RASTER_PATH_DW = LANDCOVER_DIR / "unet_landcover_dw_trial_prob.tif"
HYBRID_RASTER_PATH_DW = LANDCOVER_DIR / "hybrid_landcover_dw_trial.tif"
HYBRID_PROB_RASTER_PATH_DW = LANDCOVER_DIR / "hybrid_landcover_dw_trial_prob.tif"
EVAL_DIR_DW = LANDCOVER_DIR / "evaluation" / "dw_trial"
TRIAL_COMPARISON_CSV = EVAL_DIR_DW / "comparison_table.csv"
PRODUCTION_COMPARISON_CSV = LANDCOVER_DIR / "evaluation" / "comparison_table.csv"


def main(force: bool = False):
    if HYBRID_RASTER_PATH_DW.exists() and not force:
        print(f"{HYBRID_RASTER_PATH_DW} already exists — skipping recompute (pass --force to rebuild).")
    else:
        missing = [p for p in (RF_PROB_RASTER_PATH_DW, UNET_PROB_RASTER_PATH_DW) if not p.exists()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"Missing probability raster(s): {names} — run "
                "'python scripts/train_landcover_rf_dynamicworld.py --with-probabilities' and "
                "the U-Net Dynamic World trial notebook + "
                "'python scripts/run_landcover_unet_inference_dynamicworld_trial.py --with-probabilities' first."
            )
        label_path, _prob_path = build_hybrid(
            RF_PROB_RASTER_PATH_DW, UNET_PROB_RASTER_PATH_DW,
            label_out_path=HYBRID_RASTER_PATH_DW, prob_out_path=HYBRID_PROB_RASTER_PATH_DW,
        )
        print(f"\nDynamic-World-trained hybrid raster: {label_path}")

    if not VALIDATION_CSV.exists():
        return

    validation_df = pd.read_csv(VALIDATION_CSV)
    results = {"hybrid_dynamicworld": evaluate_classifier(HYBRID_RASTER_PATH_DW, validation_df, "hybrid_dynamicworld")}
    comparison_df = compare_classifiers(results)

    existing = [p for p in (TRIAL_COMPARISON_CSV, PRODUCTION_COMPARISON_CSV) if p.exists()]
    if existing:
        prior = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        prior = prior[prior["model"] != "hybrid_dynamicworld"]
        comparison_df = pd.concat([prior, comparison_df], ignore_index=True).drop_duplicates("model", keep="last")

    print("\n" + comparison_df.to_string(index=False))
    save_evaluation_outputs(results, comparison_df, out_dir=EVAL_DIR_DW)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
