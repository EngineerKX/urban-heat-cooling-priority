#!/usr/bin/env python
"""Build the cooling-priority score + the C3 ablation table (rank-impact
test). Replaces rank_impact.ipynb.

Runs PCA (primary) and equal-weight (mandatory sensitivity check), per the
locked eval rules, and additionally writes a single production
`priority_score.csv` using the reference variant's PCA-weighted score —
packaging the already-computed result for the app to consume, not new
modeling.

Usage: python scripts/build_priority_score.py [--toy]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import PROCESSED_DIR, REFERENCE_VARIANT
from src.priority_score.io import load_and_join
from src.priority_score.score import build_score
from validation.score_validation.rank_impact import run_rank_impact

PRIORITY_SCORE_OUT = PROCESSED_DIR / "priority_score.csv"
RANK_IMPACT_OUT_TEMPLATE = PROCESSED_DIR / "rank_impact_results_{weighting}.csv"


def main(toy_mode: bool = False):
    df, heldout = load_and_join(toy_mode)

    for weighting in ("pca", "equal"):
        print(f"\n=== Rank-impact table — weighting: {weighting} ===")
        results_df, _ = run_rank_impact(df, heldout, weighting)
        out_path = Path(str(RANK_IMPACT_OUT_TEMPLATE).format(weighting=weighting))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

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
