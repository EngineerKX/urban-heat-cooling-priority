#!/usr/bin/env python
"""Standalone verification for the pure-pandas parts of
validation/input_validation/dw_comparison.py (no GEE needed). Same style as
tests/test_landcover_zonal.py: plain asserts, printed pass/fail.

Usage: python tests/test_dw_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validation.input_validation.dw_comparison import area_weighted_agreement, island_summary


def _points():
    """Stratum 1: 4 points, map A right on 3, map B right on 1. Stratum 2: 2 points, both maps right."""
    return pd.DataFrame({
        "worldcover_class": [1, 1, 1, 1, 2, 2],
        "true_bucket": [1, 1, 1, 1, 2, 2],
        "map_a": [1, 1, 1, 9, 2, 2],
        "map_b": [1, 9, 9, 9, 2, 2],
    })


def test_area_weighted_agreement_uses_area_shares_not_sample_shares():
    summary, boot = area_weighted_agreement(_points(), {1: 0.75, 2: 0.25}, {"A": "map_a", "B": "map_b"}, n_boot=200)
    est = dict(zip(summary["map"], summary["agreement"]))
    assert np.isclose(est["A"], 0.75 * 0.75 + 0.25 * 1.0)  # 0.8125, not raw 5/6
    assert np.isclose(est["B"], 0.75 * 0.25 + 0.25 * 1.0)
    assert len(boot) == 200 and (summary["ci_low"] <= summary["agreement"] + 1e-9).all()
    print("PASS: agreement is weighted by area share, with a bootstrap interval")


def test_missing_stratum_is_renormalised():
    points = _points()[lambda d: d["worldcover_class"] == 1]
    summary, _ = area_weighted_agreement(points, {1: 0.5, 2: 0.5}, {"A": "map_a"}, n_boot=50)
    assert np.isclose(summary.loc[0, "agreement"], 0.75)
    print("PASS: a stratum with no points drops out and shares are renormalised")


def test_bootstrap_is_reproducible():
    args = (_points(), {1: 0.75, 2: 0.25}, {"A": "map_a"})
    first, _ = area_weighted_agreement(*args, n_boot=100, seed=1)
    second, _ = area_weighted_agreement(*args, n_boot=100, seed=1)
    assert first.equals(second)
    print("PASS: same seed gives the same interval")


def test_island_summary_weights_by_valid_pixels():
    zonal = pd.DataFrame({
        "subzone_id": ["A", "B", "C"], "n_valid": [100, 300, 50], "chg": [0.10, 0.50, np.nan],
    })
    summary = island_summary(zonal)  # subzone C has no data and is ignored
    assert np.isclose(summary["chg"], (0.10 * 100 + 0.50 * 300) / 400)
    print("PASS: island-wide shares are pixel-weighted and skip empty subzones")


def main():
    test_area_weighted_agreement_uses_area_shares_not_sample_shares()
    test_missing_stratum_is_renormalised()
    test_bootstrap_is_reproducible()
    test_island_summary_weights_by_valid_pixels()
    print("\nAll dw_comparison.py checks passed.")


if __name__ == "__main__":
    main()
