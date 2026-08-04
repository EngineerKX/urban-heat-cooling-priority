#!/usr/bin/env python
"""Standalone verification for validation/score_validation/confidence_bands.py
against synthetic data -- no GEE required. Same runnable-script + printed
pass/fail style as the rest of tests/.

Usage: python tests/test_confidence_bands.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validation.score_validation.confidence_bands import adaptive_capacity_noise_std, bootstrap_priority_score


def _make_synthetic_pillar_table(seed=42, n=20):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        "lst_native30": rng.uniform(30, 45, size=n),
        "sensitivity_raw": rng.uniform(0.2, 0.9, size=n),
        "greenery_fraction": rng.uniform(0.05, 0.6, size=n),
    })


def test_zero_noise_collapses_quantiles_to_point_estimate():
    df = _make_synthetic_pillar_table()
    result = bootstrap_priority_score(
        df, "lst_native30", "pca", exposure_noise_std=0.0, adaptive_capacity_noise_std=0.0, n_iterations=50,
    )
    assert np.allclose(result["priority_score_p05"], result["priority_score_point"]), "zero noise should collapse p05 to the point estimate"
    assert np.allclose(result["priority_score_p50"], result["priority_score_point"]), "zero noise should collapse p50 to the point estimate"
    assert np.allclose(result["priority_score_p95"], result["priority_score_point"]), "zero noise should collapse p95 to the point estimate"
    assert np.allclose(result["band_width"], 0.0), "zero noise should give zero-width bands"
    print("PASS: zero noise std collapses all quantiles to the point estimate")


def test_nonzero_noise_gives_monotonic_quantiles():
    df = _make_synthetic_pillar_table()
    result = bootstrap_priority_score(
        df, "lst_native30", "pca", exposure_noise_std=2.0, adaptive_capacity_noise_std=0.05, n_iterations=300,
    )
    assert (result["priority_score_p05"] <= result["priority_score_p50"] + 1e-9).all(), "p05 must not exceed p50"
    assert (result["priority_score_p50"] <= result["priority_score_p95"] + 1e-9).all(), "p50 must not exceed p95"
    assert (result["band_width"] > 0).all(), "nonzero noise should produce nonzero-width bands"
    print("PASS: nonzero noise gives monotonically ordered quantiles with positive band width")


def test_larger_noise_gives_wider_bands():
    df = _make_synthetic_pillar_table()
    small = bootstrap_priority_score(df, "lst_native30", "pca", exposure_noise_std=0.5, adaptive_capacity_noise_std=0.01, n_iterations=300)
    large = bootstrap_priority_score(df, "lst_native30", "pca", exposure_noise_std=5.0, adaptive_capacity_noise_std=0.10, n_iterations=300)
    assert large["band_width"].mean() > small["band_width"].mean(), (
        f"expected larger injected noise to widen bands, got small={small['band_width'].mean():.4f} "
        f"vs large={large['band_width'].mean():.4f}"
    )
    print(f"PASS: larger noise widens bands (small={small['band_width'].mean():.4f}, large={large['band_width'].mean():.4f})")


def test_adaptive_capacity_noise_std_binomial_se():
    confusion = pd.DataFrame({
        "Unnamed: 0": ["true:vegetation", "true:built_up", "true:bare", "true:water"],
        "pred:vegetation": [90, 5, 1, 0],
        "pred:built_up": [10, 80, 3, 1],
        "pred:bare": [0, 5, 5, 0],
        "pred:water": [0, 0, 1, 19],
    })
    std = adaptive_capacity_noise_std(confusion, class_name="vegetation")
    expected = np.sqrt(0.9 * 0.1 / 100)
    assert np.isclose(std, expected, atol=1e-6), f"expected binomial SE {expected:.4f}, got {std:.4f}"
    print(f"PASS: adaptive_capacity_noise_std computes the correct binomial SE ({std:.4f})")


def main():
    test_zero_noise_collapses_quantiles_to_point_estimate()
    test_nonzero_noise_gives_monotonic_quantiles()
    test_larger_noise_gives_wider_bands()
    test_adaptive_capacity_noise_std_binomial_se()
    print("\nAll confidence_bands.py checks passed.")


if __name__ == "__main__":
    main()
