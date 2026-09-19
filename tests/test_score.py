#!/usr/bin/env python
"""Standalone verification for src/priority_score/score.py's PCA weighting
against synthetic data -- no GEE required. Same runnable-script + printed
pass/fail style as the rest of tests/.

Usage: python tests/test_score.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.priority_score.score import build_score
from src.utils.geo import normalize


def _make_table(skew_k=0.0, seed=1, n=400):
    """Three positively co-varying pillars driven by one latent factor, so PCA
    has a real first component. `skew_k` > 0 exponentiates the sensitivity
    pillar (a monotone transform): same ranking of subzones, heavier tail."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    base = 0.5 + 0.15 * z + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        "lst_native30": 38 + 2.5 * z + rng.normal(0, 1.0, n),
        "sensitivity_raw": np.exp(skew_k * base) if skew_k else base,
        "greenery_fraction": np.clip(0.35 - 0.12 * z + rng.normal(0, 0.08, n), 0, 1),
    })


def _minmax_pca_sensitivity_weight(df):
    """The behaviour before 2026-09-19: PCA fitted on the min-max pillars."""
    pillars = pd.DataFrame({
        "e": normalize(df["lst_native30"]), "s": normalize(df["sensitivity_raw"]), "a": normalize(1 - df["greenery_fraction"]),
    })
    raw = PCA(n_components=1, random_state=42).fit(pillars.values).components_[0]
    raw = -raw if raw[0] < 0 else raw
    return (raw / raw.sum())[1]


def test_pca_weights_sum_to_one_and_score_stays_on_unit_scale():
    score, weights = build_score(_make_table(), "lst_native30", "pca")
    assert np.isclose(sum(weights.values()), 1.0), f"weights must sum to 1, got {sum(weights.values()):.4f}"
    assert all(w > 0 for w in weights.values()), f"co-varying pillars should all get positive weight: {weights}"
    assert score.between(0, 1).all(), "min-max pillars with positive weights summing to 1 must score in [0, 1]"
    print(f"PASS: PCA weights sum to 1 and are positive ({ {k: round(float(v), 3) for k, v in weights.items()} }); score in [0, 1]")


def test_identical_pillars_give_equal_weights():
    x = np.random.default_rng(0).uniform(0.1, 0.9, size=50)
    df = pd.DataFrame({"subzone_id": range(50), "lst_native30": x, "sensitivity_raw": x, "greenery_fraction": 1 - x})
    _, weights = build_score(df, "lst_native30", "pca")
    assert np.allclose(list(weights.values()), 1 / 3), f"identical pillars must weigh 1/3 each, got {weights}"
    print("PASS: identical pillars -> equal PCA weights")


def test_pca_weights_do_not_follow_pillar_skew():
    """Regression for the 2026-09-19 finding: PCA on min-max pillars let a
    skewed pillar's weight collapse (min-max squeezes it toward 0, shrinking
    its variance) even though its ranking and correlation barely changed --
    on this fixture 0.29 -> 0.17 as skew rises. Standardising first must keep
    the weight put."""
    plain, skewed = _make_table(skew_k=0), _make_table(skew_k=4)

    old_shift = _minmax_pca_sensitivity_weight(skewed) - _minmax_pca_sensitivity_weight(plain)
    assert old_shift < -0.08, f"fixture must actually discriminate: min-max PCA shift was only {old_shift:+.3f}"

    new_shift = build_score(skewed, "lst_native30", "pca")[1]["sensitivity"] - build_score(plain, "lst_native30", "pca")[1]["sensitivity"]
    assert abs(new_shift) < 0.03, f"standardised PCA weight should ignore skew, but moved {new_shift:+.3f}"
    print(f"PASS: sensitivity PCA weight moves {new_shift:+.3f} under heavy skew (min-max PCA would have moved {old_shift:+.3f})")


def test_equal_weighting_is_unchanged():
    df = _make_table()
    score, weights = build_score(df, "lst_native30", "equal")
    pillars = pd.DataFrame({
        "e": normalize(df["lst_native30"]), "s": normalize(df["sensitivity_raw"]), "a": normalize(1 - df["greenery_fraction"]),
    })
    assert np.allclose(score, pillars.mean(axis=1)) and np.allclose(list(weights.values()), 1 / 3)
    print("PASS: equal weighting is still the plain mean of the min-max pillars")


def main():
    test_pca_weights_sum_to_one_and_score_stays_on_unit_scale()
    test_identical_pillars_give_equal_weights()
    test_pca_weights_do_not_follow_pillar_skew()
    test_equal_weighting_is_unchanged()
    print("\nAll score.py checks passed.")


if __name__ == "__main__":
    main()
