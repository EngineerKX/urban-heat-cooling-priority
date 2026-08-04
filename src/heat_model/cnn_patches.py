"""S5 (C2) patch-level half: a CNN predicting `lst_bicubic10` (pure
Landsat interpolation, no spectral info baked in -- avoids the circularity
that targeting `lst_regress10` would have, since that variant was itself
regressed on the same NDVI/NDBI/NDWI bands used as CNN input here) from
Sentinel-2 + spectral-index + land-cover patches at 10m.

Reuses U-Net's ALREADY-DOWNLOADED inference patches
(data/interim/unet_patches/inference/) as the spatial feature source --
the only new GEE work this needs is a single full-Singapore lst_bicubic10
GeoTIFF export (variant_bicubic10, unmodified, via the existing
export_geotiff_to_gcs). The ensemble land-cover raster and that LST raster
are each on their own grid, so per-patch windows are read via
rasterio.warp.reproject onto the patch's own transform (same technique
src/landcover/ensemble.py::load_prob_raster uses to reconcile RF's vs.
U-Net's differing grids) rather than assumed to already align pixel-for-
pixel with the U-Net TFRecord patch grid.
"""

import glob
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config.settings import (
    ALL_FEATURE_BANDS,
    CNN_MODEL_SAVE_PATH,
    RANDOM_SEED,
    UNET_BASE_FILTERS,
    UNET_BATCH_SIZE,
    UNET_EARLY_STOP_PATIENCE,
    UNET_EPOCHS,
    UNET_LEARNING_RATE,
    UNET_PATCH_SIZE,
    UNET_TRAIN_VAL_SPLIT,
)
from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.unet import build_unet_backbone

N_LANDCOVER_CLASSES = len(BUCKET_NAMES)


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
    X: (n_patches, patch, patch, len(feature_bands) + N_LANDCOVER_CLASSES) float32
       -- S2/index bands from the TFRecord patches + one-hot land-cover channels.
    y: (n_patches, patch, patch) float32 -- lst_bicubic10, 0.0 at invalid pixels.
    valid_mask: (n_patches, patch, patch) bool -- both land-cover AND LST valid.
    """
    import tensorflow as tf

    with open(mixer_json_path) as f:
        mixer = json.load(f)
    patches_per_row = mixer["patchesPerRow"]
    total_patches = mixer["totalPatches"]
    proj = mixer["projection"]
    from affine import Affine
    full_transform = Affine(*proj["affine"]["doubleMatrix"])
    crs = proj["crs"]

    tfrecord_files = sorted(glob.glob(str(Path(unet_inference_patch_dir) / "*.tfrecord.gz")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {unet_inference_patch_dir}.")

    feature_description = {
        band: tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for band in feature_bands
    }

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return tf.stack([parsed[b] for b in feature_bands], axis=-1)

    raw = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
    s2_patches = list(raw.map(_parse, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator())
    n_patches = len(s2_patches)
    print(f"Loaded {n_patches} S2/index feature patches (mixer total: {total_patches}).")

    n_bands = len(feature_bands)
    X = np.zeros((n_patches, patch_size, patch_size, n_bands + N_LANDCOVER_CLASSES), dtype=np.float32)
    y = np.zeros((n_patches, patch_size, patch_size), dtype=np.float32)
    valid_mask = np.zeros((n_patches, patch_size, patch_size), dtype=bool)

    with rasterio.open(ensemble_raster_path) as lc_src, rasterio.open(lst_bicubic10_raster_path) as lst_src:
        for idx in range(n_patches):
            row, col = idx // patches_per_row, idx % patches_per_row
            window = rasterio.windows.Window(col * patch_size, row * patch_size, patch_size, patch_size)
            patch_transform = rasterio.windows.transform(window, full_transform)

            lc_window = _read_patch_window(lc_src, patch_transform, crs, patch_size, Resampling.nearest, nodata=0.0)
            lst_window = _read_patch_window(lst_src, patch_transform, crs, patch_size, Resampling.bilinear, nodata=np.nan)

            X[idx, :, :, :n_bands] = s2_patches[idx]
            for class_id in range(1, N_LANDCOVER_CLASSES + 1):
                X[idx, :, :, n_bands + class_id - 1] = (lc_window == class_id).astype(np.float32)

            y[idx] = np.nan_to_num(lst_window, nan=0.0)
            valid_mask[idx] = (lc_window != 0) & ~np.isnan(lst_window)

            if (idx + 1) % 200 == 0:
                print(f"  ...built {idx + 1}/{n_patches} patches")

    print(f"Valid pixels: {valid_mask.mean() * 100:.1f}% of {valid_mask.size:,} total.")
    return X, y, valid_mask


def build_cnn_regressor(input_shape, base_filters=UNET_BASE_FILTERS):
    """Same encoder-decoder backbone as src/landcover/unet.py::build_unet,
    with a linear regression head instead of a softmax classification head."""
    from tensorflow.keras import layers, models

    inputs, d1 = build_unet_backbone(input_shape, base_filters)
    outputs = layers.Conv2D(1, 1, activation="linear")(d1)
    return models.Model(inputs, outputs, name="heat_cnn_regressor")


def train_cnn_regressor(
    X, y, valid_mask, model_save_path=CNN_MODEL_SAVE_PATH, epochs=UNET_EPOCHS,
    learning_rate=UNET_LEARNING_RATE, patience=UNET_EARLY_STOP_PATIENCE,
    train_val_split=UNET_TRAIN_VAL_SPLIT, batch_size=UNET_BATCH_SIZE, seed=RANDOM_SEED,
    force_retrain: bool = False,
):
    """Same "skip if a saved model already exists" caching convention as
    src/landcover/unet.py::train_unet. No content-fingerprint cache here --
    unlike U-Net's training labels, this model's target/features come from
    already-cached rasters rather than a validation CSV that gets relabeled,
    so there's no separate "did the underlying data change" signal to hash."""
    import tensorflow as tf

    if model_save_path.exists() and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        return tf.keras.models.load_model(model_save_path), None

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_idx = int(n * train_val_split)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    y_expanded = y[..., np.newaxis]
    weight = valid_mask.astype(np.float32)

    def _make_dataset(idx, shuffle):
        ds = tf.data.Dataset.from_tensor_slices((X[idx], y_expanded[idx], weight[idx]))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(idx), seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_dataset(train_idx, shuffle=True)
    val_ds = _make_dataset(val_idx, shuffle=False)
    print(f"Train patches: {len(train_idx)}, validation patches: {len(val_idx)}")

    model = build_cnn_regressor(X.shape[1:])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        weighted_metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop])

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Best val_loss (MSE): {min(history.history['val_loss']):.4f}")
    return model, history


def run_cnn_inference(model, X) -> np.ndarray:
    """Returns (n_patches, patch, patch) predicted LST -- drops the trailing
    channel-1 dim the model outputs."""
    return model.predict(X)[..., 0]


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
