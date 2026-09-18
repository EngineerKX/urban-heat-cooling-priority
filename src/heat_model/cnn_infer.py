"""S5 (C2) CNN heat-model inference — local-only (CPU; regression on a
handful of patches at a time is cheap, was never the GPU bottleneck).
"""

import json

import numpy as np
import tensorflow as tf

from config.settings import CNN_MODEL_SAVE_PATH, UNET_PATCH_SIZE


def load_cnn_regressor(path=CNN_MODEL_SAVE_PATH):
    """A `.keras` file embeds the full architecture alongside the weights,
    so no `in_channels`/`base_filters` reconstruction is needed here (unlike
    the old PyTorch `state_dict`-based load)."""
    return tf.keras.models.load_model(path)


def run_cnn_inference(model, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """X: (n_patches, H, W, C). Returns (n_patches, H, W) predicted LST."""
    return model.predict(X, batch_size=batch_size, verbose=0)[..., 0]


def predict_patch(model, patch_X: np.ndarray) -> np.ndarray:
    """`patch_X`: (H, W, C) float32, channels-last (matches
    `cnn_data.build_local_feature_target_patches`'s per-patch layout — the
    single-patch counterpart of `run_cnn_inference`, used by
    `src/heat_model/counterfactual.py` for the original-vs-edited patch
    pair). Returns (H, W) predicted LST."""
    return model.predict(patch_X[np.newaxis, ...], verbose=0)[0, ..., 0]


def locate_patch_and_pixel(lon: float, lat: float, mixer_json_path, patch_size=UNET_PATCH_SIZE):
    """Maps a WGS84 (lon, lat) onto (patch_idx, local_row, local_col) in
    the U-Net/CNN patch grid described by mixer_json_path -- shared by
    scripts/run_counterfactual.py and scripts/diagnose_heat_model.py so the
    same pixel-locating math isn't duplicated across scripts (this repo's
    scripts/ has no __init__.py; scripts import from src/, not each other)."""
    from affine import Affine
    from pyproj import Transformer

    with open(mixer_json_path) as f:
        mixer = json.load(f)
    full_transform = Affine(*mixer["projection"]["affine"]["doubleMatrix"])
    crs = mixer["projection"]["crs"]
    patches_per_row = mixer["patchesPerRow"]

    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    world_x, world_y = transformer.transform(lon, lat)
    col_full, row_full = ~full_transform * (world_x, world_y)
    col_full, row_full = int(col_full), int(row_full)

    patch_row, patch_col = row_full // patch_size, col_full // patch_size
    patch_idx = patch_row * patches_per_row + patch_col
    local_row, local_col = row_full % patch_size, col_full % patch_size
    return patch_idx, local_row, local_col
