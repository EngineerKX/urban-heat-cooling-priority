"""Soft-voting ensemble combining the RF and U-Net land-cover classifiers
(Track B / EN1): averages each model's per-class probability raster onto a
common grid, then argmax -> a single full-Singapore ensemble raster. The
formal RF-vs-U-Net-vs-ensemble evaluation (validation/landcover_validation/)
scores all three identically once this exists.

Both prob rasters carry an explicit `valid_mask` band from their own
inference paths (see rf_baseline.py::classify_probability,
unet.py::run_inference_and_reconstruct) -- that band, not a nodata sentinel
on the probability values themselves, is the source of truth for validity.
A probability of exactly 0.0 for a class at a genuinely valid pixel is a
legitimate model output (e.g. 0 of 200 RF trees voted that class), not a
"missing" signal.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config.settings import PROCESSED_DIR
from src.ingest.worldcover import BUCKET_NAMES

ENSEMBLE_RASTER_PATH = PROCESSED_DIR / "landcover" / "ensemble_landcover.tif"
ENSEMBLE_PROB_RASTER_PATH = PROCESSED_DIR / "landcover" / "ensemble_landcover_prob.tif"

N_CLASSES = len(BUCKET_NAMES)


def _to_valid_bool(valid_mask_f: np.ndarray) -> np.ndarray:
    """valid_mask_f is meant to be a clean 0.0/1.0 float band, but GEE's
    export pipeline writes masked/clipped-outside-boundary pixels as NaN
    for float32 bands (confirmed empirically: RF's exported probability
    raster has NaN, not 0, at every pixel outside the Singapore boundary --
    U-Net's locally-written raster uses clean 0.0 instead, so this only
    bit RF). `.astype(bool)` treats NaN as truthy (any nonzero bit pattern
    is truthy), which silently turned "masked out" into "valid" for any
    NaN-writing source. Route through nan_to_num first so NaN is always
    treated as invalid regardless of which convention the source used."""
    return np.nan_to_num(valid_mask_f, nan=0.0) > 0.5


def load_prob_raster(path, ref_transform, ref_crs, ref_shape):
    """Read a 5-band probability raster (4 class-probability bands + a
    trailing valid_mask band). If its grid doesn't already match
    (ref_transform, ref_crs, ref_shape), reproject onto it: probability
    bands with bilinear resampling (continuous), valid_mask with nearest
    (categorical 0/1, so reprojection can't invent fractional "half-valid"
    pixels)."""
    with rasterio.open(path) as src:
        same_grid = (
            src.transform == ref_transform and src.crs == ref_crs and (src.height, src.width) == ref_shape
        )
        if same_grid:
            data = src.read()  # (bands, H, W)
            prob = np.nan_to_num(data[:N_CLASSES], nan=0.0).astype(np.float32)
            valid_mask = _to_valid_bool(data[N_CLASSES])
            return prob, valid_mask

        prob = np.zeros((N_CLASSES, *ref_shape), dtype=np.float32)
        for i in range(N_CLASSES):
            reproject(
                source=rasterio.band(src, i + 1), destination=prob[i],
                dst_transform=ref_transform, dst_crs=ref_crs, resampling=Resampling.bilinear,
            )
        prob = np.nan_to_num(prob, nan=0.0)
        valid_mask_f = np.zeros(ref_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, N_CLASSES + 1), destination=valid_mask_f,
            dst_transform=ref_transform, dst_crs=ref_crs, resampling=Resampling.nearest,
        )
        return prob, _to_valid_bool(valid_mask_f)


def average_probabilities(prob_arrays, valid_masks):
    """Mean per-class probability across all input models, valid only where
    EVERY input model is valid. List-based (not fixed to 2 args) so a third
    model could be added later without a signature change."""
    combined_valid = np.logical_and.reduce(valid_masks)
    avg_prob = np.mean(np.stack(prob_arrays, axis=0), axis=0)
    avg_prob[:, ~combined_valid] = 0.0
    return avg_prob, combined_valid


def probabilities_to_hard_labels(avg_prob, combined_valid):
    """argmax + 1 (bucket ids are 1-indexed, see config.settings.BUCKET_NAMES
    / BUCKET_VEGETATION..BUCKET_WATER), 0 (nodata) outside combined_valid."""
    label_array = (np.argmax(avg_prob, axis=0) + 1).astype(np.uint8)
    label_array[~combined_valid] = 0
    return label_array


def export_ensemble_rasters(avg_prob, combined_valid, label_array, ref_transform, ref_crs,
                             label_out_path=ENSEMBLE_RASTER_PATH, prob_out_path=ENSEMBLE_PROB_RASTER_PATH):
    label_out_path = Path(label_out_path)
    label_out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        label_out_path, "w", driver="GTiff", height=label_array.shape[0], width=label_array.shape[1],
        count=1, dtype=label_array.dtype, crs=ref_crs, transform=ref_transform,
    ) as dst:
        dst.write(label_array, 1)
    print(f"Ensemble raster written to {label_out_path}")

    class_values = sorted(BUCKET_NAMES.keys())
    band_names = [f"prob_{BUCKET_NAMES[v]}" for v in class_values] + ["valid_mask"]
    prob_out_path = Path(prob_out_path)
    prob_out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        prob_out_path, "w", driver="GTiff", height=avg_prob.shape[1], width=avg_prob.shape[2],
        count=N_CLASSES + 1, dtype=np.float32, crs=ref_crs, transform=ref_transform,
    ) as dst:
        for i in range(N_CLASSES):
            dst.write(avg_prob[i], i + 1)
        dst.write(combined_valid.astype(np.float32), N_CLASSES + 1)
        dst.descriptions = tuple(band_names)
    print(f"Ensemble probability raster written to {prob_out_path}")
    return label_out_path, prob_out_path


def build_ensemble(rf_prob_path, unet_prob_path, label_out_path=ENSEMBLE_RASTER_PATH,
                    prob_out_path=ENSEMBLE_PROB_RASTER_PATH):
    """Top-level orchestrator. RF's own raster grid is the reference: it's
    exported straight from the real boundary geometry (export_geotiff_to_gcs),
    while U-Net's grid is an artifact of the rectangular patch-export
    mechanism -- reprojecting U-Net onto RF's grid (not the other way)
    keeps the ensemble on the same footprint convention as the pre-existing
    RF raster already in the repo."""
    with rasterio.open(rf_prob_path) as ref:
        ref_transform, ref_crs, ref_shape = ref.transform, ref.crs, (ref.height, ref.width)

    rf_prob, rf_valid = load_prob_raster(rf_prob_path, ref_transform, ref_crs, ref_shape)
    unet_prob, unet_valid = load_prob_raster(unet_prob_path, ref_transform, ref_crs, ref_shape)

    both = rf_valid & unet_valid
    rf_only = rf_valid & ~unet_valid
    unet_only = unet_valid & ~rf_valid
    total = rf_valid.size
    print(f"Valid in both: {both.sum():,} ({both.sum() / total * 100:.1f}%)")
    print(f"Valid in RF only: {rf_only.sum():,} ({rf_only.sum() / total * 100:.1f}%)")
    print(f"Valid in U-Net only: {unet_only.sum():,} ({unet_only.sum() / total * 100:.1f}%)")

    avg_prob, combined_valid = average_probabilities([rf_prob, unet_prob], [rf_valid, unet_valid])
    prob_sums = avg_prob[:, combined_valid].sum(axis=0)
    if prob_sums.size:
        print(f"Averaged probability sum at valid pixels: mean={prob_sums.mean():.4f} (expect ~1.0)")

    label_array = probabilities_to_hard_labels(avg_prob, combined_valid)
    return export_ensemble_rasters(
        avg_prob, combined_valid, label_array, ref_transform, ref_crs, label_out_path, prob_out_path,
    )
