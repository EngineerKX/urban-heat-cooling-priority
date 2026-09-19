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

from src.priority_score.score import build_score
from validation.score_validation.confidence_bands import (
    adaptive_capacity_noise_std,
    bootstrap_priority_score,
    exposure_noise_std,
)


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


def test_bootstrap_is_not_biased_by_renormalisation():
    """Regression for the 2026-09-19 bug: re-min-maxing every noisy draw against
    its own range let the noise stretch the exposure range, so noisy scores
    drifted away from the point estimate (median shift 0.026-0.042 on this
    fixture; the point estimate landed outside its own band for 31/332 real
    subzones). Fixed-scale scoring must keep the median on the point estimate.
    Equal weights isolate the normalisation effect from PCA-loading instability
    on independent random pillars."""
    df = _make_synthetic_pillar_table(n=200)
    result = bootstrap_priority_score(
        df, "lst_native30", "equal", exposure_noise_std=3.0, adaptive_capacity_noise_std=0.03, n_iterations=300, top_n=10,
    )
    shift = (result["priority_score_p50"] - result["priority_score_point"]).abs().mean()
    outside = ((result["priority_score_point"] > result["priority_score_p95"])
               | (result["priority_score_point"] < result["priority_score_p05"])).mean()
    assert shift < 0.015, f"median should stay on the point estimate, got mean |p50 - point| = {shift:.4f}"
    assert outside < 0.15, f"point estimate should sit inside its own band, but {outside:.1%} of subzones fall outside"
    print(f"PASS: bootstrap is unbiased (mean |p50 - point| = {shift:.4f}, {outside:.1%} outside own band)")


def test_p_top_n_invariants():
    df = _make_synthetic_pillar_table()
    top_n = 5

    noisy = bootstrap_priority_score(
        df, "lst_native30", "pca", exposure_noise_std=2.0, adaptive_capacity_noise_std=0.05, n_iterations=200, top_n=top_n,
    )
    p = noisy[f"p_top{top_n}"]
    assert p.between(0, 1).all(), "probabilities must lie in [0, 1]"
    assert np.isclose(p.sum(), top_n), f"every draw contributes exactly {top_n} top-{top_n} slots, so p must sum to {top_n}, got {p.sum():.4f}"

    quiet = bootstrap_priority_score(
        df, "lst_native30", "pca", exposure_noise_std=0.0, adaptive_capacity_noise_std=0.0, n_iterations=50, top_n=top_n,
    )
    expected = np.zeros(len(df))
    expected[np.argsort(-quiet["priority_score_point"].values)[:top_n]] = 1.0
    assert np.allclose(quiet[f"p_top{top_n}"], expected), "zero noise: p_top must be 1 for the point top-N and 0 elsewhere"
    print(f"PASS: p_top{top_n} is in [0,1], sums to {top_n}, and is exactly 0/1 with zero noise")


def test_exposure_noise_std_removes_constant_offset():
    df = _make_synthetic_pillar_table(n=30)
    wobble = np.tile([-1.0, 1.0], 15)  # residual = lst - heldout = 7 + wobble
    heldout = pd.DataFrame({"subzone_id": df["subzone_id"], "lst_heldout_c": df["lst_native30"] - 7.0 - wobble})

    std = exposure_noise_std(df, "lst_native30", heldout)
    assert np.isclose(std, np.std(7.0 + wobble, ddof=1)), f"expected the residual std, got {std:.4f}"

    shifted = heldout.assign(lst_heldout_c=heldout["lst_heldout_c"] + 4.0)
    assert np.isclose(exposure_noise_std(df, "lst_native30", shifted), std), "a constant offset must not change the noise std"

    assert np.isnan(exposure_noise_std(df, "lst_native30", None)), "no held-out table -> NaN"
    assert np.isnan(exposure_noise_std(df, "lst_native30", heldout.head(2))), "fewer than 3 overlapping subzones -> NaN"
    print(f"PASS: exposure_noise_std is the offset-removed residual std ({std:.4f}), NaN when it can't be estimated")


def test_build_score_reference_df_pins_the_scale():
    df = _make_synthetic_pillar_table()
    assert np.allclose(
        build_score(df, "lst_native30", "equal")[0], build_score(df, "lst_native30", "equal", reference_df=df)[0]
    ), "reference_df=df must be identical to the default self-normalisation"

    shift = 3.0
    shifted = df.assign(lst_native30=df["lst_native30"] + shift)
    default_score = build_score(shifted, "lst_native30", "equal")[0]
    assert np.allclose(default_score, build_score(df, "lst_native30", "equal")[0]), (
        "self-normalising cancels a constant shift entirely (this is the behaviour that biased the bootstrap)"
    )
    span = df["lst_native30"].max() - df["lst_native30"].min()
    pinned_score = build_score(shifted, "lst_native30", "equal", reference_df=df)[0]
    assert np.allclose(pinned_score - build_score(df, "lst_native30", "equal")[0], shift / (3 * span)), (
        "against a pinned scale a +3C shift must raise every equal-weight score by shift / (3 * range)"
    )
    print("PASS: build_score(reference_df=...) scores against a fixed range; default behaviour unchanged")


def main():
    test_zero_noise_collapses_quantiles_to_point_estimate()
    test_nonzero_noise_gives_monotonic_quantiles()
    test_larger_noise_gives_wider_bands()
    test_adaptive_capacity_noise_std_binomial_se()
    test_bootstrap_is_not_biased_by_renormalisation()
    test_p_top_n_invariants()
    test_exposure_noise_std_removes_constant_offset()
    test_build_score_reference_df_pins_the_scale()
    print("\nAll confidence_bands.py checks passed.")


if __name__ == "__main__":
    main()
