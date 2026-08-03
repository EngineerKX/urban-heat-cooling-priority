#!/usr/bin/env python
"""Standalone verification for src/landcover/ensemble.py against small
synthetic in-memory GeoTIFFs -- no GEE/GPU required. The repo has no test
framework (no pytest, no tests/ convention yet), so this follows the
existing codebase style of a runnable script with printed pass/fail checks
(see e.g. build_training_region's excluded-area sanity check) rather than
introducing a new dependency.

Usage: python tests/test_landcover_ensemble.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from affine import Affine

from src.landcover.ensemble import (
    average_probabilities,
    build_ensemble,
    load_prob_raster,
    probabilities_to_hard_labels,
)

CRS = "EPSG:32648"
N_CLASSES = 4
BAND_NAMES = ["prob_vegetation", "prob_built_up", "prob_bare", "prob_water", "valid_mask"]


def _write_prob_raster(path, prob, valid_mask, transform):
    """prob: (4, H, W) float32-like, valid_mask: (H, W) float-like."""
    h, w = valid_mask.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=N_CLASSES + 1,
        dtype=np.float32, crs=CRS, transform=transform,
    ) as dst:
        for i in range(N_CLASSES):
            dst.write(prob[i].astype(np.float32), i + 1)
        dst.write(valid_mask.astype(np.float32), N_CLASSES + 1)
        dst.descriptions = tuple(BAND_NAMES)


def test_load_prob_raster_nan_masked_pixels(tmp_dir):
    """Regression test: GEE's export pipeline writes NaN (not 0.0) for
    masked/clipped-outside-boundary pixels in float32 bands -- confirmed
    empirically on a real RF probability export, where `.astype(bool)`
    treated those NaN pixels as valid (any nonzero bit pattern is truthy),
    silently inflating RF's apparent valid area by ~58%. load_prob_raster
    must treat NaN as invalid regardless of which convention (NaN vs clean
    0.0) the source raster used."""
    ref_transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    shape = (4, 4)
    prob = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    prob[0] = 0.7  # vegetation-leaning where valid
    valid_mask = np.ones(shape, dtype=np.float32)
    valid_mask[-1, :] = np.nan  # last row "masked out", GEE-style (NaN, not 0.0)
    prob[:, -1, :] = np.nan  # GEE masks every band together, not just valid_mask

    path = tmp_dir / "nan_masked.tif"
    _write_prob_raster(path, prob, valid_mask, ref_transform)

    loaded_prob, loaded_valid = load_prob_raster(path, ref_transform, CRS, shape)
    assert loaded_valid[-1, :].sum() == 0, "NaN-masked row must be treated as invalid, not valid"
    assert loaded_valid[:-1, :].all(), "non-NaN rows should still be valid"
    assert not np.isnan(loaded_prob).any(), "NaN must not leak into the returned probability array"
    print("PASS: load_prob_raster treats GEE-style NaN masking as invalid")


def test_load_prob_raster_same_grid(tmp_dir):
    ref_transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    shape = (4, 4)
    prob = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    prob[0] = 1.0  # all vegetation
    valid_mask = np.ones(shape, dtype=np.float32)
    valid_mask[-1, :] = 0.0  # last row invalid

    path = tmp_dir / "same_grid.tif"
    _write_prob_raster(path, prob, valid_mask, ref_transform)

    loaded_prob, loaded_valid = load_prob_raster(path, ref_transform, CRS, shape)
    assert loaded_prob.shape == (N_CLASSES, *shape), f"unexpected shape {loaded_prob.shape}"
    assert np.allclose(loaded_prob[0], 1.0), "same-grid read should pass values through unchanged"
    assert loaded_valid[-1, :].sum() == 0, "last row should be invalid"
    assert loaded_valid[:-1, :].all(), "all other rows should be valid"
    print("PASS: load_prob_raster same-grid passthrough")


def test_load_prob_raster_reproject(tmp_dir):
    ref_transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    ref_shape = (4, 4)

    # Coarser 2x2 grid covering the identical real-world extent, uniform
    # values everywhere -- bilinear resampling of a spatially constant field
    # should reproduce that same constant, so this doesn't depend on exact
    # resampling-weight arithmetic.
    src_transform = Affine(20.0, 0.0, 100000.0, 0.0, -20.0, 200000.0)
    src_shape = (2, 2)
    prob = np.zeros((N_CLASSES, *src_shape), dtype=np.float32)
    prob[1] = 0.7  # uniform built_up=0.7
    prob[3] = 0.3  # uniform water=0.3
    valid_mask = np.ones(src_shape, dtype=np.float32)

    path = tmp_dir / "coarse_grid.tif"
    _write_prob_raster(path, prob, valid_mask, src_transform)

    loaded_prob, loaded_valid = load_prob_raster(path, ref_transform, CRS, ref_shape)
    assert loaded_prob.shape == (N_CLASSES, *ref_shape)
    assert np.allclose(loaded_prob[1], 0.7, atol=1e-4), "reprojected uniform field should stay ~0.7"
    assert np.allclose(loaded_prob[3], 0.3, atol=1e-4), "reprojected uniform field should stay ~0.3"
    assert loaded_valid.all(), "uniform valid_mask=1 source should reproject to all-valid"
    print("PASS: load_prob_raster reprojection onto a finer reference grid")


def test_average_and_argmax():
    shape = (2, 2)
    prob_a = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    prob_a[0] = 0.8  # vegetation
    prob_b = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    prob_b[1] = 0.6  # built_up

    valid_a = np.array([[True, True], [True, False]])
    valid_b = np.array([[True, True], [False, True]])

    avg_prob, combined_valid = average_probabilities([prob_a, prob_b], [valid_a, valid_b])
    expected_valid = np.array([[True, True], [False, False]])
    assert (combined_valid == expected_valid).all(), "combined_valid must require validity in BOTH inputs"

    # Vegetation: 0.8 (A) + 0.0 (B), averaged = 0.4. Built_up: 0.0 + 0.6 = 0.3.
    assert np.allclose(avg_prob[0, 0, 0], 0.4), avg_prob[0, 0, 0]
    assert np.allclose(avg_prob[1, 0, 0], 0.3), avg_prob[1, 0, 0]

    labels = probabilities_to_hard_labels(avg_prob, combined_valid)
    assert labels[0, 0] == 1, "vegetation (0.4) should beat built_up (0.3) -> bucket 1"
    assert labels[1, 0] == 0, "invalid pixel must be labeled nodata (0)"
    assert labels[1, 1] == 0, "invalid pixel must be labeled nodata (0)"
    print("PASS: average_probabilities + probabilities_to_hard_labels")


def test_build_ensemble_end_to_end(tmp_dir):
    ref_transform = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
    shape = (3, 3)

    rf_prob = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    rf_prob[0] = 0.9  # vegetation everywhere
    rf_valid = np.ones(shape, dtype=np.float32)

    unet_prob = np.zeros((N_CLASSES, *shape), dtype=np.float32)
    unet_prob[0] = 0.7  # also vegetation-leaning, same grid this time
    unet_valid = np.ones(shape, dtype=np.float32)

    rf_path = tmp_dir / "rf_prob.tif"
    unet_path = tmp_dir / "unet_prob.tif"
    _write_prob_raster(rf_path, rf_prob, rf_valid, ref_transform)
    _write_prob_raster(unet_path, unet_prob, unet_valid, ref_transform)

    label_path, prob_path = build_ensemble(
        rf_path, unet_path,
        label_out_path=tmp_dir / "ensemble.tif", prob_out_path=tmp_dir / "ensemble_prob.tif",
    )

    with rasterio.open(label_path) as src:
        label_data = src.read(1)
    assert (label_data == 1).all(), "both models agree on vegetation -> ensemble should be all bucket 1"

    with rasterio.open(prob_path) as src:
        veg_band = src.read(1)
    assert np.allclose(veg_band, 0.8, atol=1e-4), f"expected avg(0.9, 0.7)=0.8, got {veg_band.mean()}"
    print("PASS: build_ensemble end-to-end")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        test_load_prob_raster_nan_masked_pixels(tmp_dir)
        test_load_prob_raster_same_grid(tmp_dir)
        test_load_prob_raster_reproject(tmp_dir)
        test_average_and_argmax()
        test_build_ensemble_end_to_end(tmp_dir)
    print("\nAll ensemble.py checks passed.")


if __name__ == "__main__":
    main()
