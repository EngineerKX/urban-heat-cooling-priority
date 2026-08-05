"""S5 (C2) CNN heat-model inference — local-only (CPU; regression on a
handful of patches at a time is cheap, was never the GPU bottleneck).
"""

import json

import numpy as np
import torch

from config.settings import ALL_FEATURE_BANDS, CNN_MODEL_SAVE_PATH, UNET_BASE_FILTERS, UNET_PATCH_SIZE
from src.heat_model.cnn_model import N_LANDCOVER_CLASSES, CNNRegressor


def load_cnn_regressor(path=CNN_MODEL_SAVE_PATH, in_channels=len(ALL_FEATURE_BANDS) + N_LANDCOVER_CLASSES,
                        base_filters=UNET_BASE_FILTERS) -> CNNRegressor:
    """`map_location="cpu"` is mandatory here — weights are always saved
    from a Colab CUDA session, and loading without it raises on any
    machine (yours, your partner's) that has no GPU."""
    model = CNNRegressor(in_channels=in_channels, base_filters=base_filters)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_cnn_inference(model: CNNRegressor, X: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """X: (n_patches, C, H, W). Returns (n_patches, H, W) predicted LST,
    batched to avoid holding every patch through the model at once."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start:start + batch_size])
            pred = model(batch)  # (b, 1, H, W)
            preds.append(pred[:, 0].numpy())
    return np.concatenate(preds, axis=0)


def predict_patch(model: CNNRegressor, patch_X: np.ndarray) -> np.ndarray:
    """`patch_X`: (C, H, W) float32, channels-first (matches
    `cnn_data.build_local_feature_target_patches`'s per-patch layout — the
    single-patch counterpart of `run_cnn_inference`, used by
    `src/heat_model/counterfactual.py` for the original-vs-edited patch
    pair). Returns (H, W) predicted LST."""
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(patch_X).unsqueeze(0)  # (1, C, H, W)
        pred = model(tensor)  # (1, 1, H, W)
    return pred[0, 0].numpy()


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
