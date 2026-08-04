#!/usr/bin/env python
"""Build S6 calibrated confidence bands on the production priority score.
Bootstraps using already-built validation error estimates (see
validation/score_validation/confidence_bands.py's module docstring for the
exact noise models) -- not ad hoc Monte Carlo over unvalidated input
ranges.

Usage: python scripts/build_priority_score_confidence_bands.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import PROCESSED_DIR, REFERENCE_VARIANT
from src.priority_score.io import load_and_join
from validation.score_validation.confidence_bands import (
    adaptive_capacity_noise_std,
    bootstrap_priority_score,
    exposure_noise_std,
)

CONFUSION_MATRIX_PATH = PROCESSED_DIR / "landcover" / "evaluation" / "confusion_matrix_ensemble.csv"
OUT_PATH = PROCESSED_DIR / "priority_score_confidence_bands.csv"


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH

    df, heldout = load_and_join(toy_mode=False)

    exp_std = exposure_noise_std(df, REFERENCE_VARIANT, heldout)
    print(f"Exposure noise std (NEA-heldout RMSE for {REFERENCE_VARIANT}): {exp_std}")
    print("⚠️  This RMSE mixes real random error with the systematic LST-vs-air-temperature offset — "
          "see validation/score_validation/confidence_bands.py's module docstring. Treat the resulting "
          "bands as a pessimistic upper bound on rank uncertainty, not a tight calibrated interval.")

    if CONFUSION_MATRIX_PATH.exists():
        confusion_df = pd.read_csv(CONFUSION_MATRIX_PATH)
        ac_std = adaptive_capacity_noise_std(confusion_df, class_name="vegetation")
    else:
        print(f"⚠️  {CONFUSION_MATRIX_PATH} not found — run scripts/evaluate_landcover_classifiers.py first. "
              f"Bands will reflect exposure uncertainty only.")
        ac_std = None

    result = bootstrap_priority_score(df, REFERENCE_VARIANT, "pca", exp_std, ac_std)

    ranked = result.sort_values("priority_score_point", ascending=False).reset_index(drop=True)
    top20 = ranked.head(20)
    overlaps = sum(
        top20.iloc[i]["priority_score_p05"] <= top20.iloc[i + 1]["priority_score_p95"]
        for i in range(len(top20) - 1)
    )
    print(f"\n{overlaps}/{len(top20) - 1} adjacent pairs among the top-20 ranked subzones have overlapping "
          f"[p05, p95] bands — i.e. their rank order isn't statistically distinguishable at this noise level.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
