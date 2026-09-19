#!/usr/bin/env python
"""Standalone verification for validation/score_validation/rank_impact.py's
run_weighting_comparison against synthetic data -- no GEE required. Same
runnable-script + printed pass/fail style as the rest of tests/.

Usage: python tests/test_rank_impact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validation.score_validation.rank_impact import run_weighting_comparison

TOP_N = 5
VARIANTS = ["lst_native30", "lst_bicubic10"]


def _make_table(seed=42, n=40):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        "lst_native30": rng.uniform(30, 45, size=n),
        "lst_bicubic10": rng.uniform(30, 45, size=n),
        "sensitivity_raw": rng.uniform(0.2, 0.9, size=n),
        "greenery_fraction": rng.uniform(0.05, 0.6, size=n),
    })


def test_identical_pillars_give_identical_rankings():
    """If all three normalized pillars are the same vector, PCA's first
    component loads equally on each, so PCA weights == equal weights and the
    two rankings must agree exactly."""
    x = np.random.default_rng(0).uniform(0.1, 0.9, size=30)
    df = pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(30)],
        "lst_native30": x, "sensitivity_raw": x, "greenery_fraction": 1 - x,
    })
    summary, membership = run_weighting_comparison(df, ["lst_native30"], "lst_native30", TOP_N)
    row = summary.iloc[0]
    assert np.isclose(row["spearman_pca_vs_equal"], 1.0), row["spearman_pca_vs_equal"]
    assert row[f"top{TOP_N}_overlap_pca_vs_equal"] == 1.0
    assert row[f"n_top{TOP_N}_changed"] == 0
    assert (membership["status"] == "both").all() and len(membership) == TOP_N
    print("PASS: identical pillars -> identical PCA and equal rankings")


def test_summary_shape_and_weights():
    summary, _ = run_weighting_comparison(_make_table(), VARIANTS, "lst_native30", TOP_N)
    assert list(summary["variant"]) == VARIANTS
    weight_cols = [c for c in summary.columns if c.startswith("pca_weight_")]
    assert len(weight_cols) == 3
    assert np.allclose(summary[weight_cols].sum(axis=1), 1.0)
    assert summary["spearman_pca_vs_equal"].between(-1, 1).all()
    print("PASS: one row per variant, PCA weights sum to 1")


def test_membership_consistent_with_summary():
    summary, membership = run_weighting_comparison(_make_table(), VARIANTS, "lst_native30", TOP_N)
    n_changed = int(summary.loc[summary["variant"] == "lst_native30", f"n_top{TOP_N}_changed"].iloc[0])
    counts = membership["status"].value_counts()
    assert counts.get("pca_only", 0) == n_changed == counts.get("equal_only", 0)
    assert len(membership) == TOP_N + n_changed
    assert (membership.loc[membership["status"] != "equal_only", "rank_pca"] <= TOP_N).all()
    assert (membership.loc[membership["status"] != "pca_only", "rank_equal"] <= TOP_N).all()
    print("PASS: membership table agrees with summary's changed-count and top-N rank bounds")


def test_missing_reference_variant_returns_no_membership():
    _, membership = run_weighting_comparison(_make_table(), ["lst_native30"], "not_a_variant", TOP_N)
    assert membership is None
    print("PASS: membership is None when reference variant isn't among the variants")


if __name__ == "__main__":
    test_identical_pillars_give_identical_rankings()
    test_summary_shape_and_weights()
    test_membership_consistent_with_summary()
    test_missing_reference_variant_returns_no_membership()
    print("\nAll rank_impact tests passed.")
