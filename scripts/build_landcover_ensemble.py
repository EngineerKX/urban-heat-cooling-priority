#!/usr/bin/env python
"""Build the soft-voting RF+U-Net land-cover ensemble raster.

Combines the per-class probability rasters from both classifiers (RF via
GEE MULTIPROBABILITY, U-Net via persisted softmax) onto a common grid,
averages them, and argmaxes to a single full-Singapore ensemble raster --
the "Hybrid ML/ensemble" deliverable named in the project proposal.

Usage: python scripts/build_landcover_ensemble.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import UNET_PROB_RASTER_PATH
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH, build_ensemble
from src.landcover.rf_baseline import RF_PROB_RASTER_PATH


def main(force: bool = False):
    if ENSEMBLE_RASTER_PATH.exists() and not force:
        print(f"{ENSEMBLE_RASTER_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return ENSEMBLE_RASTER_PATH

    missing = [p for p in (RF_PROB_RASTER_PATH, UNET_PROB_RASTER_PATH) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Missing probability raster(s): {names} — run "
            "'python scripts/train_landcover_rf.py --with-probabilities' and "
            "the U-Net Colab notebook + 'python scripts/pull_models.py --model unet' + "
            "'python scripts/run_landcover_unet_inference.py --with-probabilities' first."
        )

    label_path, prob_path = build_ensemble(RF_PROB_RASTER_PATH, UNET_PROB_RASTER_PATH)
    print(f"\nDone. Ensemble raster: {label_path}")
    return label_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
