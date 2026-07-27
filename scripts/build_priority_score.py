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

import numpy as np
import pandas as pd

from config.settings import INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED, REFERENCE_VARIANT, VARIANT_COLUMNS
from src.priority_score.score import build_score
from validation.score_validation.rank_impact import run_rank_impact

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
ADAPTIVE_CSV_PATH = INTERIM_DIR / "adaptive_capacity_pillar.csv"
HELDOUT_CSV_PATH = INTERIM_DIR / "nea_heldout_lst.csv"

PRIORITY_SCORE_OUT = PROCESSED_DIR / "priority_score.csv"
RANK_IMPACT_OUT_TEMPLATE = PROCESSED_DIR / "rank_impact_results_{weighting}.csv"


def _make_toy_pillars(subzone_ids: pd.Series, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    sensitivity = pd.DataFrame({"subzone_id": subzone_ids, "sensitivity_raw": rng.uniform(0.3, 1.0, size=len(subzone_ids))})
    adaptive = pd.DataFrame({"subzone_id": subzone_ids, "greenery_fraction": rng.uniform(0.05, 0.6, size=len(subzone_ids))})
    return sensitivity, adaptive


def load_and_join(toy_mode: bool):
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")
    heat = pd.read_csv(HEAT_CSV_PATH)
    missing = set(VARIANT_COLUMNS) - set(heat.columns)
    if missing:
        raise ValueError(f"{HEAT_CSV_PATH} is missing columns {missing}.")
    print(f"Loaded {len(heat)} subzones from heat-variants CSV.")

    if toy_mode:
        print("⚠️  TOY_MODE: seeded placeholder sensitivity/adaptive pillars. Not real data — do not report these numbers.")
        sensitivity, adaptive = _make_toy_pillars(heat["subzone_id"])
    else:
        for label, path in [("sensitivity", SENSITIVITY_CSV_PATH), ("adaptive capacity", ADAPTIVE_CSV_PATH)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found (needed for the {label} pillar). Run the corresponding build_*_pillar.py "
                    f"script first, or pass --toy to test wiring with placeholder data."
                )
        sensitivity = pd.read_csv(SENSITIVITY_CSV_PATH)
        adaptive = pd.read_csv(ADAPTIVE_CSV_PATH)

    df = heat.merge(sensitivity, on="subzone_id", how="inner").merge(adaptive, on="subzone_id", how="inner")
    n_dropped = len(heat) - len(df)
    print(f"Join summary: {len(heat)} -> {len(df)} subzones retained after pillar join.")
    if n_dropped:
        print(f"⚠️  {n_dropped} subzones dropped — check subzone_id spelling/casing across files.")

    heldout = pd.read_csv(HELDOUT_CSV_PATH) if HELDOUT_CSV_PATH.exists() else None
    if heldout is None:
        print(f"ℹ️  {HELDOUT_CSV_PATH} not found — lst_rmse_heldout will be NaN (run scripts/build_nea_heldout.py to fill it in).")

    return df, heldout


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
