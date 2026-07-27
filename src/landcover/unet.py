"""U-Net land-cover classifier (Track B / UN1): same 4-class scheme, same
feature bands, same season window and same validation-point exclusion as
`src/landcover/rf_baseline.py` — kept identical deliberately, since the
point of comparing RF vs U-Net is isolating the effect of model choice, not
accidentally comparing different inputs too.

Caching: training is skipped if `models/unet_landcover.keras` already
exists (pass `force_retrain=True` to override) — the one part of the
original notebook that was already local-file-friendly, just missing the
"skip if present" check.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings
from config.settings import (
    ALL_FEATURE_BANDS,
    GEE_EXPORT_BUCKET,
    INTERIM_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    S2_UTM_CRS,
    TARGET_SCALE_M,
)
from src.ingest.gee import export_patches_to_gcs
from src.ingest.worldcover import BUCKET_NAMES

MODEL_SAVE_PATH = MODELS_DIR / "unet_landcover.keras"
CLASSIFIED_RASTER_PATH = PROCESSED_DIR / "landcover" / "unet_landcover.tif"
TRAIN_PATCH_DIR = INTERIM_DIR / "unet_patches" / "train"
INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"

N_CLASSES = len(BUCKET_NAMES)


def build_training_stack(feature_image, wc_bucket_image, training_region):
    """wc_class values are 1-4 (buckets); pixels outside the mask get filled
    with 0 ("no valid label") so the exported patch always has a defined
    value per pixel — the sample-weight mask in `parse_training_patches`
    uses this to exclude those pixels from the loss."""
    return (
        feature_image.addBands(wc_bucket_image.select("wc_class").unmask(0))
        .clip(training_region)
    )


def export_training_patches(training_stack, training_region, patch_size=settings.UNET_PATCH_SIZE,
                             scale=TARGET_SCALE_M, crs=S2_UTM_CRS, bucket=GEE_EXPORT_BUCKET):
    return export_patches_to_gcs(
        training_stack, description="unet_train", bucket=bucket, prefix="unet_train_patches/unet_train",
        region=training_region, scale=scale, crs=crs, patch_dimensions=(patch_size, patch_size),
        out_dir=TRAIN_PATCH_DIR,
    )


def export_inference_patches(feature_image, boundary, patch_size=settings.UNET_PATCH_SIZE,
                              scale=TARGET_SCALE_M, crs=S2_UTM_CRS, bucket=GEE_EXPORT_BUCKET):
    return export_patches_to_gcs(
        feature_image.clip(boundary), description="unet_inference", bucket=bucket,
        prefix="unet_inference_patches/unet_inference", region=boundary, scale=scale, crs=crs,
        patch_dimensions=(patch_size, patch_size), out_dir=INFERENCE_PATCH_DIR,
    )


def parse_training_patches(patch_dir, patch_size=settings.UNET_PATCH_SIZE,
                            feature_bands=ALL_FEATURE_BANDS, n_classes=N_CLASSES,
                            train_val_split=settings.UNET_TRAIN_VAL_SPLIT, seed=42, batch_size=settings.UNET_BATCH_SIZE):
    import tensorflow as tf

    tfrecord_files = sorted(glob.glob(str(Path(patch_dir) / "*.tfrecord.gz")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {patch_dir} — check the export completed.")

    all_bands = feature_bands + ["wc_class"]
    feature_description = {
        band: tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for band in all_bands
    }

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        features = tf.stack([parsed[b] for b in feature_bands], axis=-1)
        label_raw = parsed["wc_class"]
        weight = tf.cast(label_raw > 0, tf.float32)
        label_idx = tf.cast(tf.maximum(label_raw - 1, 0), tf.int32)
        label_onehot = tf.one_hot(label_idx, depth=n_classes)
        return features, label_onehot, weight

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    parsed_dataset = parsed_dataset.filter(lambda f, l, w: tf.reduce_sum(w) > 0)

    all_patches = list(parsed_dataset.as_numpy_iterator())
    n_patches = len(all_patches)
    print(f"Usable patches (after dropping all-nodata ones): {n_patches}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_patches)
    split_idx = int(n_patches * train_val_split)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    def _make_dataset(idx_list, shuffle):
        features = np.stack([all_patches[i][0] for i in idx_list])
        labels = np.stack([all_patches[i][1] for i in idx_list])
        weights = np.stack([all_patches[i][2] for i in idx_list])
        ds = tf.data.Dataset.from_tensor_slices((features, labels, weights))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(idx_list), seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = _make_dataset(train_indices, shuffle=True)
    val_ds = _make_dataset(val_indices, shuffle=False)
    print(f"Train patches: {len(train_indices)}, validation patches: {len(val_indices)}")
    return train_ds, val_ds, n_patches


def build_unet(input_shape, n_classes=N_CLASSES, base_filters=settings.UNET_BASE_FILTERS):
    from tensorflow.keras import layers, models

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        return x

    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(2)(c3)

    b = conv_block(p3, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    d3 = conv_block(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = conv_block(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)
    return models.Model(inputs, outputs, name="unet_landcover")


def train_unet(train_ds, val_ds, patch_size=settings.UNET_PATCH_SIZE, n_feature_bands=len(ALL_FEATURE_BANDS),
               epochs=settings.UNET_EPOCHS, learning_rate=settings.UNET_LEARNING_RATE,
               patience=settings.UNET_EARLY_STOP_PATIENCE, model_save_path=MODEL_SAVE_PATH,
               force_retrain: bool = False):
    """Skip training entirely if a saved model already exists, unless
    `force_retrain=True` — the caching behaviour the original notebook was
    missing."""
    import tensorflow as tf

    if model_save_path.exists() and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        return tf.keras.models.load_model(model_save_path), None

    model = build_unet((patch_size, patch_size, n_feature_bands))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop])

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    return model, history


def run_inference_and_reconstruct(model, inference_patch_dir, patch_size=settings.UNET_PATCH_SIZE,
                                   feature_bands=ALL_FEATURE_BANDS, out_path=CLASSIFIED_RASTER_PATH):
    """Reads mixer.json (patch grid layout + georeferencing, written
    automatically alongside the patches) to lay predicted patches back into
    a single georeferenced raster. Known limitation carried over from the
    notebook: non-overlapping patches can leave minor discontinuities at
    patch boundaries ("tile seams") — acceptable for a baseline comparison."""
    import rasterio
    import tensorflow as tf
    from rasterio.transform import Affine

    inference_dir = Path(inference_patch_dir)
    tfrecord_files = sorted(glob.glob(str(inference_dir / "*.tfrecord.gz")))
    mixer_files = sorted(glob.glob(str(inference_dir / "*mixer.json")))
    if not tfrecord_files or not mixer_files:
        raise FileNotFoundError(f"Missing TFRecord/mixer.json in {inference_dir}.")

    with open(mixer_files[0]) as f:
        mixer = json.load(f)

    patches_per_row = mixer["patchesPerRow"]
    patches_per_col = mixer["totalPatches"] // patches_per_row
    proj = mixer["projection"]

    feature_description = {
        band: tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for band in feature_bands
    }

    def _parse(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return tf.stack([parsed[b] for b in feature_bands], axis=-1)

    raw = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
    patches = list(raw.map(_parse, num_parallel_calls=tf.data.AUTOTUNE).as_numpy_iterator())
    print(f"Loaded {len(patches)} inference patches for prediction.")

    full_raster = np.zeros((patches_per_col * patch_size, patches_per_row * patch_size), dtype=np.uint8)
    for idx, patch in enumerate(patches):
        pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
        pred_class = np.argmax(pred, axis=-1).astype(np.uint8) + 1
        row, col = idx // patches_per_row, idx % patches_per_row
        full_raster[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = pred_class
        if (idx + 1) % 50 == 0:
            print(f"  ...predicted {idx + 1}/{len(patches)} patches")

    crs_str = proj["crs"]
    affine_params = proj.get("affine", {}).get("doubleMatrix")
    if affine_params:
        transform = Affine(*affine_params)
    else:
        raise RuntimeError("mixer.json has no affine transform — cannot georeference the reconstructed raster.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=full_raster.shape[0], width=full_raster.shape[1],
        count=1, dtype=full_raster.dtype, crs=crs_str, transform=transform,
    ) as dst:
        dst.write(full_raster, 1)
    print(f"✅ Classified raster written to {out_path}")
    return out_path, crs_str


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
