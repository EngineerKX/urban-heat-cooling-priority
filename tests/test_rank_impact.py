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

from validation.score_validation.rank_impact import heldout_agreement, run_rank_impact, run_weighting_comparison

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


def test_heldout_agreement_separates_offset_from_spread():
    """Regression for the misleading-RMSE finding: Landsat LST reads several
    degrees hotter than either held-out source, and RMSE mostly restated that
    systematic offset (11.6C against NEA) rather than the ~2C spread that can
    actually reorder subzones."""
    rng = np.random.default_rng(0)
    lst = rng.uniform(32, 46, size=200)
    noise = rng.normal(0, 2.0, size=200)
    df = pd.DataFrame({"subzone_id": range(200), "lst_native30": lst})
    heldout = pd.DataFrame({"subzone_id": range(200), "lst_heldout_c": lst - 7.0 + noise})  # residual = 7 - noise

    a = heldout_agreement(df, "lst_native30", heldout)
    assert a["n"] == 200
    assert np.isclose(a["mean_offset_c"], 7.0 - noise.mean())
    assert np.isclose(a["spread_c"], noise.std(ddof=1))
    assert a["spearman"] > 0.8, f"expected ~0.89 (4C spread of true LST vs 2C noise), got {a['spearman']:.3f}"
    assert np.isclose(a["rmse_c"] ** 2, a["mean_offset_c"] ** 2 + a["spread_c"] ** 2 * 199 / 200), "RMSE^2 = offset^2 + spread^2"
    assert a["rmse_c"] > 3 * a["spread_c"], "RMSE should be dominated by the offset on this fixture"

    shifted = heldout.assign(lst_heldout_c=heldout["lst_heldout_c"] + 3.0)
    b = heldout_agreement(df, "lst_native30", shifted)
    assert np.isclose(b["spread_c"], a["spread_c"]) and np.isclose(b["spearman"], a["spearman"]), (
        "a constant offset must not change the spread or the rank agreement"
    )
    assert np.isclose(b["mean_offset_c"], a["mean_offset_c"] - 3.0)
    print(f"PASS: heldout_agreement splits offset ({a['mean_offset_c']:.2f}) from spread ({a['spread_c']:.2f}); "
          f"RMSE ({a['rmse_c']:.2f}) mostly restates the offset")


def test_heldout_agreement_handles_missing_inputs():
    df = pd.DataFrame({"subzone_id": range(10), "lst_native30": np.arange(10.0)})

    none = heldout_agreement(df, "lst_native30", None)
    assert none["n"] == 0 and all(np.isnan(none[k]) for k in ("mean_offset_c", "spread_c", "spearman", "rmse_c"))

    tiny = heldout_agreement(df, "lst_native30", pd.DataFrame({"subzone_id": [0, 1], "lst_heldout_c": [1.0, 2.0]}))
    assert tiny["n"] == 2 and np.isnan(tiny["spread_c"]), "fewer than 3 overlapping subzones -> NaN, with n as counted"

    with_nan = pd.DataFrame({"subzone_id": range(10), "lst_heldout_c": [np.nan] * 7 + [1.0, 2.0, 3.0]})
    assert heldout_agreement(df, "lst_native30", with_nan)["n"] == 3, "NaN held-out rows must be dropped, not counted"
    print("PASS: heldout_agreement returns NaNs for a missing/too-small held-out table and drops NaN rows")


def test_run_rank_impact_reports_agreement_not_rmse():
    rng = np.random.default_rng(1)
    n = 60
    z = rng.normal(size=n)
    base = 38 + 2.5 * z + rng.normal(0, 1, n)
    df = pd.DataFrame({
        "subzone_id": range(n), "lst_native30": base,
        "lst_bicubic10": base + rng.normal(0, 0.05, n), "lst_regress10": base + rng.normal(0, 0.05, n),
        "sensitivity_raw": np.clip(0.5 + 0.15 * z + rng.normal(0, 0.1, n), 0, None),
        "greenery_fraction": np.clip(0.35 - 0.12 * z + rng.normal(0, 0.08, n), 0, 1),
    })
    heldout = pd.DataFrame({"subzone_id": range(n), "lst_heldout_c": base - 6 + rng.normal(0, 1.5, n)})

    results, _ = run_rank_impact(df, heldout, "equal", top_n=10)
    assert "lst_rmse_heldout" not in results.columns, "the offset-dominated RMSE column must be gone"
    for col in ("heldout_n", "heldout_offset_c", "heldout_spread_c", "heldout_spearman"):
        assert col in results.columns, f"missing column {col}"
    assert (results["heldout_offset_c"] > 4).all() and (results["heldout_n"] == n).all()
    assert results.loc[0, "spearman_vs_lst_native30"] == 1.0

    without, _ = run_rank_impact(df, None, "equal", top_n=10)
    assert without["heldout_offset_c"].isna().all() and (without["heldout_n"] == 0).all()
    print("PASS: run_rank_impact reports offset / spread / Spearman per variant (NaN without a held-out table), no RMSE")


if __name__ == "__main__":
    test_identical_pillars_give_identical_rankings()
    test_summary_shape_and_weights()
    test_membership_consistent_with_summary()
    test_missing_reference_variant_returns_no_membership()
    test_heldout_agreement_separates_offset_from_spread()
    test_heldout_agreement_handles_missing_inputs()
    test_run_rank_impact_reports_agreement_not_rmse()
    print("\nAll rank_impact tests passed.")
