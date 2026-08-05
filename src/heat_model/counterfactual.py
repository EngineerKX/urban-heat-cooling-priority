"""S5 (C2) patch-level counterfactual mechanism: "what if this carpark
were a park?" Concretely: rasterize the edited region into a pixel mask on
the CNN's 10m patch grid, overwrite the spectral+index channels with the
target class's precomputed GLOBAL MEAN feature vector (keeps the edited
patch in-distribution for the CNN -- just flipping the categorical
land-cover flag while leaving built-up reflectance values in place would
hand the model an out-of-training-distribution input), overwrite the
one-hot land-cover channels to a clean one-hot for the target class, then
re-run inference on the original vs. edited patch.

XGBoost (src/heat_model/tabular.py) and this CNN answer at different
granularities (subzone-level tabular vs. 10m patch-level) and are
deliberately NOT soft-voted like the RF+U-Net land-cover ensemble --
rescale_subzone_delta lets their counterfactual deltas be compared
side-by-side as a cross-model sanity check instead.
"""

import numpy as np

from config.settings import ALL_FEATURE_BANDS
from src.heat_model.cnn_model import N_LANDCOVER_CLASSES


def class_mean_feature_vectors(X, valid_mask=None, feature_bands=ALL_FEATURE_BANDS,
                                n_landcover_classes=N_LANDCOVER_CLASSES) -> dict:
    """`X`: (n_patches, len(feature_bands)+n_landcover_classes, H, W),
    channels-first (matches `cnn_data.build_local_feature_target_patches`).
    Global (all-patches) mean spectral+index feature vector per land-cover
    bucket id (1..n_landcover_classes), computed only over pixels genuinely
    belonging to that class (and, if given, valid_mask).
    Returns {class_id: np.ndarray(len(feature_bands),)}."""
    n_bands = len(feature_bands)
    spectral = np.moveaxis(X[:, :n_bands], 1, -1)  # (n, H, W, n_bands) for the mask-indexing below
    onehot = X[:, n_bands:n_bands + n_landcover_classes]  # (n, n_landcover_classes, H, W)

    means = {}
    for class_idx in range(n_landcover_classes):
        class_id = class_idx + 1  # bucket ids are 1-indexed
        class_pixel_mask = onehot[:, class_idx] > 0.5  # (n, H, W)
        if valid_mask is not None:
            class_pixel_mask = class_pixel_mask & valid_mask
        n_pixels = int(class_pixel_mask.sum())
        if n_pixels == 0:
            means[class_id] = np.zeros(n_bands, dtype=np.float32)
            print(f"⚠️  No valid pixels found for class {class_id} — mean vector is all-zero.")
            continue
        means[class_id] = spectral[class_pixel_mask].mean(axis=0)
        print(f"Class {class_id} mean feature vector computed from {n_pixels:,} pixels.")
    return means


def apply_patch_counterfactual(patch_X, edit_mask, target_class: int, class_mean_vectors: dict,
                                feature_bands=ALL_FEATURE_BANDS, n_landcover_classes=N_LANDCOVER_CLASSES):
    """`patch_X`: single patch, (len(feature_bands)+n_landcover_classes, H, W),
    channels-first. `edit_mask`: (H, W) bool, pixels to convert to
    `target_class` (a bucket id, 1..n_landcover_classes). Returns a NEW
    array -- patch_X is not modified in place."""
    n_bands = len(feature_bands)
    edited = patch_X.copy()

    edited[:n_bands][:, edit_mask] = class_mean_vectors[target_class][:, np.newaxis]

    onehot = np.zeros(n_landcover_classes, dtype=np.float32)
    onehot[target_class - 1] = 1.0
    edited[n_bands:n_bands + n_landcover_classes][:, edit_mask] = onehot[:, np.newaxis]

    return edited


def run_patch_counterfactual(model, patch_X, edit_mask, target_class: int, class_mean_vectors: dict) -> dict:
    """Runs the CNN on the original and edited patch, returns full per-
    pixel delta plus the mean delta within the edited area."""
    from src.heat_model.cnn_infer import predict_patch

    edited_X = apply_patch_counterfactual(patch_X, edit_mask, target_class, class_mean_vectors)

    original_pred = predict_patch(model, patch_X)
    edited_pred = predict_patch(model, edited_X)
    delta = edited_pred - original_pred

    return {
        "original_lst_patch": original_pred,
        "counterfactual_lst_patch": edited_pred,
        "delta_lst_patch": delta,
        "mean_delta_lst_in_edit_area": float(delta[edit_mask].mean()) if edit_mask.any() else 0.0,
        "n_edited_pixels": int(edit_mask.sum()),
    }


def rescale_subzone_delta(patch_mean_delta: float, edit_area_m2: float, subzone_area_m2: float) -> float:
    """Area-weighted extrapolation: if editing edit_area_m2 produces
    patch_mean_delta of local cooling, the naive subzone-wide equivalent is
    that same delta scaled by the edited area's share of the whole
    subzone. Lets the CNN's local counterfactual and XGBoost's subzone-
    level counterfactual (src/heat_model/tabular.py::predict_counterfactual_subzone)
    be compared on the same footing (see module docstring)."""
    if subzone_area_m2 <= 0:
        raise ValueError("subzone_area_m2 must be positive.")
    return patch_mean_delta * (edit_area_m2 / subzone_area_m2)
