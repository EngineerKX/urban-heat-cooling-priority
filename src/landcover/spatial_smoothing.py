"""Local spatial majority-vote smoothing for a classified raster
(checkpoint_DynamicWorld): a post-processing fix for RF's biggest single
weakness relative to U-Net -- RF classifies every pixel independently, with
no idea what's next to it, so isolated pixels flip class in a way a
spatially-aware model doesn't ("salt-and-pepper" noise). A modest, fixed
neighborhood vote cleans up most of that, for free, no retraining.

`DEFAULT_WINDOW=3` (30m at the project's 10m grid) is fixed in advance, not
tuned against the 300 hand-labeled validation points -- picking a window
size by testing it against the same labels the final accuracy gets reported
on would be circular (a wider sweep showed accuracy keeps climbing well past
any defensible size, while macro F1 -- which weighs the rare classes equally
-- stalls, meaning big windows likely erase small real features rather than
denoise). 3x3 is the smallest neighborhood beyond the pixel itself, the same
modest, non-tuned scale used for src/landcover/rf_baseline.py's texture-
feature experiment (matching Landsat's own native 30m resolution, an
existing anchor already used elsewhere in this pipeline).
"""

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

DEFAULT_WINDOW = 3  # pixels; 30m at the project's 10m grid -- fixed, not tuned


def majority_smooth(label_array: np.ndarray, class_ids, window: int = DEFAULT_WINDOW, nodata: int = 0):
    """Per-pixel local vote fraction for each of `class_ids` (summing to ~1 at
    valid pixels) over a `window` x `window` neighborhood, plus the resulting
    argmax hard-label raster. `nodata` pixels are excluded from the vote and
    never overwritten with a real class.

    `class_ids` is taken explicitly (not inferred from what's present in
    `label_array`) so every class always gets a band, even if one happens to
    be entirely absent from a given raster -- matches this project's existing
    "never silently drop a class" discipline (see e.g.
    src/heat_model/counterfactual.py::class_mean_feature_vectors).

    Returns (smoothed_labels, vote_fractions): vote_fractions is
    (len(class_ids), H, W) float32 in `class_ids` order -- usable directly as
    a probability-style raster (src/landcover/hybrid.py's convention: N
    class bands + a trailing valid_mask) if a hybrid is ever rebuilt from
    this smoothed RF.
    """
    valid = label_array != nodata
    fractions = np.zeros((len(class_ids),) + label_array.shape, dtype=np.float32)
    for i, c in enumerate(class_ids):
        fractions[i] = ndimage.uniform_filter((label_array == c).astype(np.float32), size=window, mode="nearest")

    smoothed = np.array(class_ids, dtype=label_array.dtype)[fractions.argmax(axis=0)]
    smoothed = np.where(valid, smoothed, nodata).astype(label_array.dtype)
    fractions[:, ~valid] = 0.0
    return smoothed, fractions


def write_smoothed_outputs(raster_path, class_ids, class_names: dict, window: int = DEFAULT_WINDOW,
                            label_out_path=None, prob_out_path=None):
    """Reads a hard-label GeoTIFF, smooths it, and writes both a hard-label
    raster and a probability-style raster (matching
    src/landcover/hybrid.py::load_prob_raster's expected band convention:
    one band per class in `class_ids` order, plus a trailing `valid_mask`
    band) on the SAME grid as the input."""
    raster_path = Path(raster_path)
    with rasterio.open(raster_path) as src:
        label = src.read(1)
        profile = src.profile
        transform, crs = src.transform, src.crs

    smoothed, fractions = majority_smooth(label, class_ids, window=window)
    valid = smoothed != 0

    label_out_path = Path(label_out_path) if label_out_path else raster_path.with_stem(raster_path.stem + f"_smooth{window}x{window}")
    label_out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(label_out_path, "w", **profile) as dst:
        dst.write(smoothed, 1)
    print(f"Smoothed ({window}x{window}) raster written to {label_out_path}")

    n_classes = len(class_ids)
    band_names = [f"prob_{class_names[c]}" for c in class_ids] + ["valid_mask"]
    prob_out_path = Path(prob_out_path) if prob_out_path else label_out_path.with_stem(label_out_path.stem + "_prob")
    prob_out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        prob_out_path, "w", driver="GTiff", height=fractions.shape[1], width=fractions.shape[2],
        count=n_classes + 1, dtype=np.float32, crs=crs, transform=transform,
    ) as dst:
        for i in range(n_classes):
            dst.write(fractions[i], i + 1)
        dst.write(valid.astype(np.float32), n_classes + 1)
        dst.descriptions = tuple(band_names)
    print(f"Vote-fraction (probability-style) raster written to {prob_out_path}")

    return label_out_path, prob_out_path
