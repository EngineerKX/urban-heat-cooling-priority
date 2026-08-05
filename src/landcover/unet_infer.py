"""U-Net inference — local-only (CPU is fine; inference was never the GPU
bottleneck, only 30-epoch training was). Reconstructs a full-Singapore
classified/probability raster from per-patch predictions, reading
`mixer.json` (patch grid layout + georeferencing, written automatically
alongside the patches) to lay patches back into a single georeferenced
raster.

Known limitation carried over unchanged from the original Keras version:
non-overlapping patches can leave minor discontinuities at patch boundaries
("tile seams") — acceptable for a baseline comparison.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import settings
from config.settings import (
    ALL_FEATURE_BANDS,
    UNET_BASE_FILTERS,
    UNET_CLASSIFIED_RASTER_PATH,
    UNET_MODEL_SAVE_PATH,
    UNET_PROB_RASTER_PATH,
)
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.unet_data import N_CLASSES, read_raw_patches
from src.landcover.unet_model import UNet


def load_unet(path=UNET_MODEL_SAVE_PATH, in_channels=len(ALL_FEATURE_BANDS), n_classes=N_CLASSES,
              base_filters=UNET_BASE_FILTERS) -> UNet:
    """`map_location="cpu"` is mandatory here — weights are always saved
    from a Colab CUDA session, and loading without it raises on any
    machine (yours, your partner's) that has no GPU."""
    model = UNet(in_channels=in_channels, n_classes=n_classes, base_filters=base_filters)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def run_inference_and_reconstruct(model: UNet, inference_patch_dir, boundary, patch_size=settings.UNET_PATCH_SIZE,
                                   feature_bands=ALL_FEATURE_BANDS, out_path=UNET_CLASSIFIED_RASTER_PATH,
                                   also_write_probabilities: bool = False, prob_out_path=UNET_PROB_RASTER_PATH,
                                   batch_size: int = 16):
    """Unlike `ee.Classifier.classify()` (used by the RF baseline), which
    automatically carries the input mask through to the output, per-patch
    CNN inference has no notion of "outside the mask" — every patch gets a
    real class prediction, including the padding area between Singapore's
    actual coastline and the rectangular patch grid's bounding box (the
    model tends to predict "water" for that blank/near-zero input, which
    silently inflated the water class). `boundary` is used to mask those
    pixels back to nodata (0) after reconstruction, matching RF's behavior.

    `also_write_probabilities=True` persists the per-class softmax array
    (needed for the soft-voting ensemble) into a second 5-band GeoTIFF
    instead of discarding it.
    """
    import rasterio
    from rasterio.transform import Affine

    inference_dir = Path(inference_patch_dir)
    mixer_files = sorted(glob.glob(str(inference_dir / "*.json")))
    if not mixer_files:
        raise FileNotFoundError(f"Missing mixer JSON in {inference_dir}.")
    with open(mixer_files[0]) as f:
        mixer = json.load(f)

    patches_per_row = mixer["patchesPerRow"]
    patches_per_col = mixer["totalPatches"] // patches_per_row
    proj = mixer["projection"]

    patches = read_raw_patches(inference_dir, feature_bands, patch_size)  # (n, C, H, W)
    print(f"Loaded {len(patches)} inference patches for prediction.")

    all_pred_class = np.zeros(len(patches), dtype=object)
    all_pred_prob = np.zeros(len(patches), dtype=object)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch = torch.from_numpy(patches[start:start + batch_size])
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).numpy()  # (b, n_classes, H, W)
            pred_class = probs.argmax(axis=1).astype(np.uint8) + 1  # (b, H, W)
            for i in range(pred_class.shape[0]):
                all_pred_class[start + i] = pred_class[i]
                all_pred_prob[start + i] = np.transpose(probs[i], (1, 2, 0))  # (H, W, n_classes)
            if (start + batch_size) % 160 < batch_size:
                print(f"  ...predicted {min(start + batch_size, len(patches))}/{len(patches)} patches")

    full_raster = np.zeros((patches_per_col * patch_size, patches_per_row * patch_size), dtype=np.uint8)
    full_prob = (
        np.zeros((patches_per_col * patch_size, patches_per_row * patch_size, N_CLASSES), dtype=np.float32)
        if also_write_probabilities else None
    )
    for idx in range(len(patches)):
        row, col = idx // patches_per_row, idx % patches_per_row
        row_slice = slice(row * patch_size, (row + 1) * patch_size)
        col_slice = slice(col * patch_size, (col + 1) * patch_size)
        full_raster[row_slice, col_slice] = all_pred_class[idx]
        if full_prob is not None:
            full_prob[row_slice, col_slice, :] = all_pred_prob[idx]

    crs_str = proj["crs"]
    affine_params = proj.get("affine", {}).get("doubleMatrix")
    if affine_params:
        transform = Affine(*affine_params)
    else:
        raise RuntimeError("mixer.json has no affine transform — cannot georeference the reconstructed raster.")

    import geopandas as gpd
    from rasterio.features import geometry_mask
    from shapely.geometry import shape

    boundary_gdf = gpd.GeoSeries([shape(boundary.getInfo())], crs="EPSG:4326").to_crs(crs_str)
    outside_boundary = geometry_mask(boundary_gdf.geometry, out_shape=full_raster.shape, transform=transform, invert=False)
    full_raster[outside_boundary] = 0
    if full_prob is not None:
        full_prob[outside_boundary, :] = 0.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=full_raster.shape[0], width=full_raster.shape[1],
        count=1, dtype=full_raster.dtype, crs=crs_str, transform=transform,
    ) as dst:
        dst.write(full_raster, 1)
    print(f"✅ Classified raster written to {out_path}")

    written_prob_path = None
    if full_prob is not None:
        class_values = sorted(BUCKET_NAMES.keys())
        band_names = [f"prob_{BUCKET_NAMES[v]}" for v in class_values] + ["valid_mask"]
        valid_mask = (~outside_boundary).astype(np.float32)

        prob_out_path = Path(prob_out_path)
        prob_out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            prob_out_path, "w", driver="GTiff", height=full_prob.shape[0], width=full_prob.shape[1],
            count=N_CLASSES + 1, dtype=np.float32, crs=crs_str, transform=transform,
        ) as dst:
            for i in range(N_CLASSES):
                dst.write(full_prob[:, :, i], i + 1)
            dst.write(valid_mask, N_CLASSES + 1)
            dst.descriptions = tuple(band_names)
        print(f"✅ Probability raster written to {prob_out_path}")
        written_prob_path = prob_out_path

    return out_path, crs_str, written_prob_path


def informal_accuracy_check(raster_path, crs_str: str, validation_df: pd.DataFrame):
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        raster_transform = src.transform

    validation_df = validation_df.copy()
    validation_df["agreed_label"] = validation_df["agreed_label"].fillna("").astype(str)
    valid_rows = validation_df[validation_df["agreed_label"].isin(BUCKET_NAMES.values())].copy()

    transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)

    def _lonlat_to_pixel(lon, lat):
        x, y = transformer.transform(lon, lat)
        col, row = ~raster_transform * (x, y)
        return int(row), int(col)

    preds = []
    for row in valid_rows.itertuples():
        r, c = _lonlat_to_pixel(row.lon, row.lat)
        preds.append(int(raster_data[r, c]) if 0 <= r < raster_data.shape[0] and 0 <= c < raster_data.shape[1] else None)

    valid_rows["unet_pred_bucket"] = preds
    valid_rows["unet_pred_name"] = valid_rows["unet_pred_bucket"].map(BUCKET_NAMES)
    bucket_name_to_id = {v: k for k, v in BUCKET_NAMES.items()}
    valid_rows["true_bucket"] = valid_rows["agreed_label"].map(bucket_name_to_id)

    scored = valid_rows.dropna(subset=["unet_pred_bucket"])
    accuracy = (scored["unet_pred_bucket"] == scored["true_bucket"]).mean()
    crosstab = pd.crosstab(scored["agreed_label"], scored["unet_pred_name"])
    print(f"Informal U-Net accuracy on {len(scored)} validation points: {accuracy * 100:.1f}%")
    return accuracy, crosstab
