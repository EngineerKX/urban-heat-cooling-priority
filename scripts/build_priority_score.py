#!/usr/bin/env python
"""Build the cooling-priority score + the C3 ablation table (rank-impact
test). Replaces rank_impact.ipynb.

Runs PCA (primary) and equal-weight (mandatory sensitivity check), per the
locked eval rules, and additionally writes a single production
`priority_score.csv` using the reference variant's PCA-weighted score —
packaging the already-computed result for the app to consume, not new
modeling. Also writes the PCA-vs-equal weighting comparison (the C1
benchmark): how far the two weightings disagree on the ranking, and which
top-N subzones each one adds or removes.

Usage: python scripts/build_priority_score.py [--toy]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import PROCESSED_DIR, REFERENCE_VARIANT, TOP_N
from src.priority_score.io import load_and_join
from src.priority_score.score import build_score
from validation.score_validation.rank_impact import run_rank_impact, run_weighting_comparison
from validation.score_validation.sensitivity_specs import (
    REQUIRED_COLUMNS as SPEC_REQUIRED_COLUMNS,
    run_sensitivity_spec_comparison,
)

PRIORITY_SCORE_OUT = PROCESSED_DIR / "priority_score.csv"
RANK_IMPACT_OUT_TEMPLATE = PROCESSED_DIR / "rank_impact_results_{weighting}.csv"
WEIGHTING_COMPARISON_OUT = PROCESSED_DIR / "weighting_comparison.csv"
WEIGHTING_MEMBERSHIP_OUT = PROCESSED_DIR / f"weighting_comparison_top{TOP_N}_membership.csv"
SENSITIVITY_SPEC_OUT = PROCESSED_DIR / "sensitivity_spec_comparison.csv"


def main(toy_mode: bool = False):
    df, heldout = load_and_join(toy_mode)

    for weighting in ("pca", "equal"):
        print(f"\n=== Rank-impact table — weighting: {weighting} ===")
        results_df, _ = run_rank_impact(df, heldout, weighting)
        out_path = Path(str(RANK_IMPACT_OUT_TEMPLATE).format(weighting=weighting))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("\n=== PCA vs. equal weighting (C1 benchmark) ===")
    weighting_df, membership_df = run_weighting_comparison(df)
    weighting_df.to_csv(WEIGHTING_COMPARISON_OUT, index=False)
    print(f"Saved: {WEIGHTING_COMPARISON_OUT}")
    if membership_df is not None:
        membership_df.to_csv(WEIGHTING_MEMBERSHIP_OUT, index=False)
        print(f"Saved: {WEIGHTING_MEMBERSHIP_OUT}")

    print("\n=== Sensitivity-pillar specification comparison (formula uncertainty, not noise) ===")
    if all(c in df.columns for c in SPEC_REQUIRED_COLUMNS):
        spec_df = run_sensitivity_spec_comparison(df)
        spec_df.to_csv(SENSITIVITY_SPEC_OUT, index=False)
        print(f"Saved: {SENSITIVITY_SPEC_OUT}")
    else:
        print("⚠️  Skipped — the pillar table lacks the population/area columns this needs "
              "(toy mode, or rebuild with scripts/build_sensitivity_pillar.py --force).")

    print(f"\n=== Production priority score (reference variant: {REFERENCE_VARIANT}, PCA-weighted) ===")
    score, weights = build_score(df, REFERENCE_VARIANT, "pca")
    priority_df = pd.DataFrame({
        "subzone_id": df["subzone_id"],
        "priority_score": score,
        "exposure_variant": REFERENCE_VARIANT,
        "weighting": "pca",
    }).sort_values("priority_score", ascending=False)
    print(f"Weights: {weights}")
    print(priority_df.head(10).to_string(index=False))

    PRIORITY_SCORE_OUT.parent.mkdir(parents=True, exist_ok=True)
    priority_df.to_csv(PRIORITY_SCORE_OUT, index=False)
    print(f"\nSaved: {PRIORITY_SCORE_OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Use seeded placeholder pillars to test wiring only.")
    args = parser.parse_args()
    main(toy_mode=args.toy)
