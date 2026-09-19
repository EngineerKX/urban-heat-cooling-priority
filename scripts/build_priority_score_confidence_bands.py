#!/usr/bin/env python
"""Build S6 confidence bands on the production priority score.
Bootstraps using already-built validation error estimates (see
validation/score_validation/confidence_bands.py's module docstring for the
exact noise models, the fixed-scale scoring, and the stated limitations) --
not ad hoc Monte Carlo over unvalidated input ranges. Writes per-subzone
score quantiles plus `p_top<N>`, each subzone's probability of landing in
the top N.

Exposure noise comes from the held-out LST source named by
config.settings.EXPOSURE_NOISE_SOURCE ("modis" or "nea"), so that source's
build script must have been run first.

Usage: python scripts/build_priority_score_confidence_bands.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import EXPOSURE_NOISE_SOURCE, INTERIM_DIR, PROCESSED_DIR, REFERENCE_VARIANT, TOP_N
from src.priority_score.io import load_and_join
from validation.score_validation.confidence_bands import (
    adaptive_capacity_noise_std,
    bootstrap_priority_score,
    exposure_noise_std,
)

CONFUSION_MATRIX_PATH = PROCESSED_DIR / "landcover" / "evaluation" / "confusion_matrix_hybrid.csv"
OUT_PATH = PROCESSED_DIR / "priority_score_confidence_bands.csv"

# source name -> (held-out CSV, the script that builds it)
HELDOUT_SOURCES = {
    "modis": (INTERIM_DIR / "modis_heldout_lst.csv", "scripts/build_modis_heldout.py"),
    "nea": (INTERIM_DIR / "nea_heldout_lst.csv", "scripts/build_nea_heldout.py"),
}


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH

    if EXPOSURE_NOISE_SOURCE not in HELDOUT_SOURCES:
        raise ValueError(
            f"Unknown EXPOSURE_NOISE_SOURCE '{EXPOSURE_NOISE_SOURCE}', expected one of {sorted(HELDOUT_SOURCES)}."
        )
    heldout_path, build_script = HELDOUT_SOURCES[EXPOSURE_NOISE_SOURCE]
    if not heldout_path.exists():
        # Not a soft warning: running with no exposure noise would silently produce far-too-tight bands.
        raise FileNotFoundError(
            f"{heldout_path} not found (exposure noise source '{EXPOSURE_NOISE_SOURCE}') — run {build_script} first."
        )

    df, _ = load_and_join(toy_mode=False)
    heldout = pd.read_csv(heldout_path)

    print(f"Exposure noise source: {EXPOSURE_NOISE_SOURCE} ({heldout_path.name})")
    exp_std = exposure_noise_std(df, REFERENCE_VARIANT, heldout)
    print("⚠️  This is an UPPER bound on random exposure error: the residual spread also contains the mismatch "
          "between the held-out footprint and the subzone — see validation/score_validation/confidence_bands.py's "
          "module docstring.")

    if CONFUSION_MATRIX_PATH.exists():
        confusion_df = pd.read_csv(CONFUSION_MATRIX_PATH)
        ac_std = adaptive_capacity_noise_std(confusion_df, class_name="vegetation")
        print("⚠️  This counts vegetation RECALL only; the hybrid's larger built-up→vegetation error is not in the "
              "noise model, so adaptive-capacity uncertainty is probably understated.")
    else:
        print(f"⚠️  {CONFUSION_MATRIX_PATH} not found — run scripts/evaluate_landcover_classifiers.py first. "
              f"Bands will reflect exposure uncertainty only.")
        ac_std = None

    result = bootstrap_priority_score(df, REFERENCE_VARIANT, "pca", exp_std, ac_std)

    ranked = result.sort_values("priority_score_point", ascending=False).reset_index(drop=True)
    top = ranked.head(TOP_N)
    overlaps = sum(
        top.iloc[i]["priority_score_p05"] <= top.iloc[i + 1]["priority_score_p95"]
        for i in range(len(top) - 1)
    )
    print(f"\n{overlaps}/{len(top) - 1} adjacent pairs among the top-{TOP_N} ranked subzones have overlapping "
          f"[p05, p95] bands — i.e. their rank order isn't statistically distinguishable at this noise level.")

    p_col = f"p_top{TOP_N}"
    print(f"Of the {TOP_N} subzones ranked highest by point score: "
          f"{int((top[p_col] >= 0.9).sum())} have {p_col} >= 0.9, "
          f"{int(((top[p_col] >= 0.5) & (top[p_col] < 0.9)).sum())} are 0.5-0.9, "
          f"{int((top[p_col] < 0.5).sum())} are < 0.5.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
