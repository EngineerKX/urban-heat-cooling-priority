"""S5 (C2) patch-level data I/O: building the CNN's (features, target,
valid_mask) patch stack -- Sentinel-2 + spectral-index + one-hot land-cover
channels as input, `lst_bicubic10` as target.

Reuses U-Net's ALREADY-DOWNLOADED inference patches
(data/interim/unet_patches/inference/) as the spatial feature source -- the
only new GEE work this needs is a single full-Singapore lst_bicubic10
GeoTIFF export (variant_bicubic10, unmodified, via the existing
export_geotiff_to_gcs). The ensemble land-cover raster and that LST raster
are each on their own grid, so per-patch windows are read via
rasterio.warp.reproject onto the patch's own transform (same technique
src/landcover/ensemble.py::load_prob_raster uses to reconcile RF's vs.
U-Net's differing grids) rather than assumed to already align pixel-for-
pixel with the U-Net TFRecord patch grid.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config.settings import ALL_FEATURE_BANDS, UNET_PATCH_SIZE
from src.heat_model.cnn_model import N_LANDCOVER_CLASSES
from src.landcover.unet_data import read_raw_patches


def _read_patch_window(src, dst_transform, dst_crs, patch_size, resampling, nodata):
    dest = np.full((patch_size, patch_size), nodata, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1), destination=dest,
        src_nodata=nodata, dst_nodata=nodata,
        dst_transform=dst_transform, dst_crs=dst_crs, resampling=resampling,
    )
    return dest


def build_local_feature_target_patches(
    unet_inference_patch_dir, mixer_json_path, ensemble_raster_path, lst_bicubic10_raster_path,
    patch_size=UNET_PATCH_SIZE, feature_bands=ALL_FEATURE_BANDS,
):
    """Returns (X, y, valid_mask):
    X: (n_patches, len(feature_bands) + N_LANDCOVER_CLASSES, patch, patch) float32,
       channels-first -- S2/index bands from the TFRecord patches + one-hot
       land-cover channels.
    y: (n_patches, patch, patch) float32 -- lst_bicubic10, 0.0 at invalid pixels.
    valid_mask: (n_patches, patch, patch) bool -- both land-cover AND LST valid.
    """
    with open(mixer_json_path) as f:
        mixer = json.load(f)
    patches_per_row = mixer["patchesPerRow"]
    total_patches = mixer["totalPatches"]
    proj = mixer["projection"]
    full_transform = Affine(*proj["affine"]["doubleMatrix"])
    crs = proj["crs"]

    s2_patches = read_raw_patches(unet_inference_patch_dir, feature_bands, patch_size)  # (n, n_bands, H, W)
    n_patches = s2_patches.shape[0]
    print(f"Loaded {n_patches} S2/index feature patches (mixer total: {total_patches}).")

    n_bands = len(feature_bands)
    X = np.zeros((n_patches, n_bands + N_LANDCOVER_CLASSES, patch_size, patch_size), dtype=np.float32)
    y = np.zeros((n_patches, patch_size, patch_size), dtype=np.float32)
    valid_mask = np.zeros((n_patches, patch_size, patch_size), dtype=bool)

    with rasterio.open(ensemble_raster_path) as lc_src, rasterio.open(lst_bicubic10_raster_path) as lst_src:
        for idx in range(n_patches):
            row, col = idx // patches_per_row, idx % patches_per_row
            window = rasterio.windows.Window(col * patch_size, row * patch_size, patch_size, patch_size)
            patch_transform = rasterio.windows.transform(window, full_transform)

            lc_window = _read_patch_window(lc_src, patch_transform, crs, patch_size, Resampling.nearest, nodata=0.0)
            lst_window = _read_patch_window(lst_src, patch_transform, crs, patch_size, Resampling.bilinear, nodata=np.nan)

            X[idx, :n_bands] = s2_patches[idx]
            for class_id in range(1, N_LANDCOVER_CLASSES + 1):
                X[idx, n_bands + class_id - 1] = (lc_window == class_id).astype(np.float32)

            y[idx] = np.nan_to_num(lst_window, nan=0.0)
            valid_mask[idx] = (lc_window != 0) & ~np.isnan(lst_window)

            if (idx + 1) % 200 == 0:
                print(f"  ...built {idx + 1}/{n_patches} patches")

    print(f"Valid pixels: {valid_mask.mean() * 100:.1f}% of {valid_mask.size:,} total.")
    return X, y, valid_mask
