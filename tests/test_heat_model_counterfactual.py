#!/usr/bin/env python
"""Standalone verification for src/heat_model/counterfactual.py against a
tiny synthetic patch -- no TensorFlow/GEE required. Same runnable-script +
printed pass/fail style as the rest of tests/ (no pytest in this repo).

Usage: python tests/test_heat_model_counterfactual.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.heat_model.counterfactual import (
    apply_patch_counterfactual,
    class_mean_feature_vectors,
    rescale_subzone_delta,
)

FEATURE_BANDS = ["B2", "B3"]
N_CLASSES = 4  # vegetation=1, built_up=2, bare=3, water=4


def _make_synthetic_patch_batch():
    """1 patch, 2x2 pixels, 2 spectral bands + 4 one-hot land-cover
    channels. (0,0) and (1,0) are vegetation; (0,1) and (1,1) are
    built_up. No bare/water pixels at all -- exercises the "no pixels for
    this class" branch too."""
    X = np.zeros((1, 2, 2, len(FEATURE_BANDS) + N_CLASSES), dtype=np.float32)
    n_bands = len(FEATURE_BANDS)

    X[0, 0, 0, :n_bands] = [10.0, 20.0]
    X[0, 0, 0, n_bands:] = [1, 0, 0, 0]  # vegetation

    X[0, 0, 1, :n_bands] = [30.0, 40.0]
    X[0, 0, 1, n_bands:] = [0, 1, 0, 0]  # built_up

    X[0, 1, 0, :n_bands] = [12.0, 22.0]
    X[0, 1, 0, n_bands:] = [1, 0, 0, 0]  # vegetation

    X[0, 1, 1, :n_bands] = [32.0, 42.0]
    X[0, 1, 1, n_bands:] = [0, 1, 0, 0]  # built_up

    return X


def test_class_mean_feature_vectors():
    X = _make_synthetic_patch_batch()
    means = class_mean_feature_vectors(X, feature_bands=FEATURE_BANDS, n_landcover_classes=N_CLASSES)

    assert np.allclose(means[1], [11.0, 21.0]), f"vegetation mean should be [11, 21], got {means[1]}"
    assert np.allclose(means[2], [31.0, 41.0]), f"built_up mean should be [31, 41], got {means[2]}"
    assert np.allclose(means[3], [0.0, 0.0]), "bare has no pixels -> zero vector"
    assert np.allclose(means[4], [0.0, 0.0]), "water has no pixels -> zero vector"
    print("PASS: class_mean_feature_vectors computes correct per-class means, zero for absent classes")


def test_apply_patch_counterfactual_edits_only_masked_pixels():
    X = _make_synthetic_patch_batch()
    means = class_mean_feature_vectors(X, feature_bands=FEATURE_BANDS, n_landcover_classes=N_CLASSES)
    patch_X = X[0]  # (2, 2, 6)

    edit_mask = np.zeros((2, 2), dtype=bool)
    edit_mask[1, 1] = True  # convert the bottom-right built_up pixel to vegetation

    edited = apply_patch_counterfactual(
        patch_X, edit_mask, target_class=1, class_mean_vectors=means, feature_bands=FEATURE_BANDS,
        n_landcover_classes=N_CLASSES,
    )

    n_bands = len(FEATURE_BANDS)
    assert np.allclose(edited[1, 1, :n_bands], means[1]), "edited pixel's spectral bands should be class 1's mean vector"
    assert np.allclose(edited[1, 1, n_bands:], [1, 0, 0, 0]), "edited pixel's one-hot should be a clean class-1 one-hot"

    for r, c in [(0, 0), (0, 1), (1, 0)]:
        assert np.allclose(edited[r, c], patch_X[r, c]), f"unedited pixel ({r},{c}) must be unchanged"

    assert np.allclose(patch_X[1, 1, n_bands:], [0, 1, 0, 0]), "original patch_X must not be mutated in place"
    print("PASS: apply_patch_counterfactual edits only masked pixels and leaves the original array untouched")


def test_rescale_subzone_delta():
    result = rescale_subzone_delta(patch_mean_delta=-2.0, edit_area_m2=1000.0, subzone_area_m2=10000.0)
    assert np.isclose(result, -0.2), f"expected -2.0 * (1000/10000) = -0.2, got {result}"
    print(f"PASS: rescale_subzone_delta area-weights correctly ({result})")

    try:
        rescale_subzone_delta(patch_mean_delta=-2.0, edit_area_m2=1000.0, subzone_area_m2=0.0)
        raise AssertionError("expected ValueError for non-positive subzone_area_m2")
    except ValueError:
        print("PASS: rescale_subzone_delta rejects non-positive subzone_area_m2")


def main():
    test_class_mean_feature_vectors()
    test_apply_patch_counterfactual_edits_only_masked_pixels()
    test_rescale_subzone_delta()
    print("\nAll heat_model/counterfactual.py checks passed.")


if __name__ == "__main__":
    main()
