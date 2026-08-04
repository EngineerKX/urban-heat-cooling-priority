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
import hashlib
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
PROB_RASTER_PATH = PROCESSED_DIR / "landcover" / "unet_landcover_prob.tif"
TRAIN_PATCH_DIR = INTERIM_DIR / "unet_patches" / "train"
INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
_FINGERPRINT_FILENAME = "source_fingerprint.txt"

N_CLASSES = len(BUCKET_NAMES)


def build_training_stack(feature_image, wc_bucket_image, training_region):
    """wc_class values are 1-4 (buckets); pixels outside the mask get filled
    with 0 ("no valid label") so the exported patch always has a defined
    value per pixel — the sample-weight mask in `parse_training_patches`
    uses this to exclude those pixels from the loss.

    `.toFloat()` matters here: GEE's TFRecord patch export encodes each
    band independently based on its own pixel type, and a non-float band
    mixed in with float bands (every S2/index band already is float) gets
    written as a single opaque bytes blob instead of a flat per-pixel
    float array — silently breaking `parse_training_patches`'s fixed-shape
    float32 parsing for that one band. Casting keeps every band in the
    stack encoded the same way.
    """
    return (
        feature_image.addBands(wc_bucket_image.select("wc_class").unmask(0).toFloat())
        .clip(training_region)
    )


def _patches_already_downloaded(out_dir: Path) -> bool:
    """True if `out_dir` already has both the TFRecord shard(s) and the
    mixer/metadata JSON from a previous export — lets re-runs skip the
    expensive GCS export + download instead of redoing it every time."""
    out_dir = Path(out_dir)
    return bool(list(out_dir.glob("*.tfrecord.gz"))) and bool(list(out_dir.glob("*.json")))


def _clear_patch_dir(out_dir: Path) -> None:
    """Remove old shard/mixer files before a re-export. Needed because
    export_patches_to_gcs downloads by matching blob name -- if a re-export
    produces a different shard count than what's cached locally (e.g. the
    training_region's shape changed because the validation sample's
    exclusion zone moved), leftover shards from the OLD export wouldn't be
    overwritten and would silently keep contaminating training with
    patches from the wrong region."""
    out_dir = Path(out_dir)
    for pattern in ("*.tfrecord.gz", "*.json"):
        for f in out_dir.glob(pattern):
            f.unlink()


def training_fingerprint(validation_csv_path) -> str:
    """Hash of the validation CSV's bytes -- changes iff the labels/points
    that define training_region's exclusion zone change. Comparing this
    (not just "do patch files exist") is what lets the cache tell the
    difference between "nothing changed, reuse freely" and "labels
    changed, this cache is now wrong" without relying on a human to
    remember to pass a --force flag."""
    return hashlib.sha256(Path(validation_csv_path).read_bytes()).hexdigest()


def _cached_fingerprint(out_dir: Path):
    fp_path = Path(out_dir) / _FINGERPRINT_FILENAME
    return fp_path.read_text().strip() if fp_path.exists() else None


def _write_fingerprint(out_dir: Path, fingerprint: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / _FINGERPRINT_FILENAME).write_text(fingerprint)


def export_training_patches(training_stack, training_region, validation_csv_path,
                             patch_size=settings.UNET_PATCH_SIZE, scale=TARGET_SCALE_M, crs=S2_UTM_CRS,
                             bucket=GEE_EXPORT_BUCKET, force: bool = False):
    """Re-exports automatically whenever validation_csv_path's contents
    have changed since the patches currently on disk were exported
    (fingerprint mismatch) -- so relabeling the validation sample can no
    longer silently train on a stale, wrongly-excluded region, and a plain
    retrain (different hyperparameters, same labels) can reuse the cached
    patches for free instead of re-exporting for no reason. `force=True` is
    a manual override to redo the export even when the fingerprint still
    matches."""
    fingerprint = training_fingerprint(validation_csv_path)
    cache_valid = _patches_already_downloaded(TRAIN_PATCH_DIR) and _cached_fingerprint(TRAIN_PATCH_DIR) == fingerprint

    if not force and cache_valid:
        print(f"Training patches at {TRAIN_PATCH_DIR} match the current validation CSV — skipping export (pass force=True to redo anyway).")
        return TRAIN_PATCH_DIR
    if not cache_valid and _patches_already_downloaded(TRAIN_PATCH_DIR):
        print("Cached training patches don't match the current validation CSV (labels changed since last export) — re-exporting.")

    _clear_patch_dir(TRAIN_PATCH_DIR)
    out_dir = export_patches_to_gcs(
        training_stack, description="unet_train", bucket=bucket, prefix="unet_train_patches/unet_train",
        region=training_region, scale=scale, crs=crs, patch_dimensions=(patch_size, patch_size),
        out_dir=TRAIN_PATCH_DIR,
    )
    _write_fingerprint(out_dir, fingerprint)
    return out_dir


def export_inference_patches(feature_image, boundary, patch_size=settings.UNET_PATCH_SIZE,
                              scale=TARGET_SCALE_M, crs=S2_UTM_CRS, bucket=GEE_EXPORT_BUCKET, force: bool = False):
    if not force and _patches_already_downloaded(INFERENCE_PATCH_DIR):
        print(f"Inference patches already downloaded at {INFERENCE_PATCH_DIR} — skipping export (pass force=True to redo).")
        return INFERENCE_PATCH_DIR
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


def build_unet_backbone(input_shape, base_filters=settings.UNET_BASE_FILTERS):
    """3-level encoder-decoder with skip connections, WITHOUT an output
    head -- shared by build_unet's softmax classification head and
    src/heat_model/cnn_patches.py::build_cnn_regressor's linear regression
    head, so the two models stay identical apart from what they predict.
    Returns (inputs, final_decoder_features) for a caller to attach
    `layers.Conv2D(n_outputs, 1, activation=...)` onto."""
    from tensorflow.keras import layers

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

    return inputs, d1


def build_unet(input_shape, n_classes=N_CLASSES, base_filters=settings.UNET_BASE_FILTERS):
    from tensorflow.keras import layers, models

    inputs, d1 = build_unet_backbone(input_shape, base_filters)
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)
    return models.Model(inputs, outputs, name="unet_landcover")


def train_unet(train_ds, val_ds, patch_size=settings.UNET_PATCH_SIZE, n_feature_bands=len(ALL_FEATURE_BANDS),
               epochs=settings.UNET_EPOCHS, learning_rate=settings.UNET_LEARNING_RATE,
               patience=settings.UNET_EARLY_STOP_PATIENCE, model_save_path=MODEL_SAVE_PATH,
               data_fingerprint: str = None, force_retrain: bool = False):
    """Skip training entirely if a saved model already exists AND (when
    `data_fingerprint` is given) it was trained on the same data -- a
    fingerprint mismatch means the patches this model saw have since been
    replaced (e.g. the validation sample got relabeled and training_region
    moved), so silently reusing it would be training on stale, wrongly-
    excluded data one level removed from the patch cache itself.
    `data_fingerprint=None` (the default) skips this check entirely, same
    as before. `force_retrain=True` always retrains regardless."""
    import tensorflow as tf

    fingerprint_path = model_save_path.parent / f"{model_save_path.stem}_fingerprint.txt"
    cached_fingerprint = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
    cache_valid = model_save_path.exists() and (data_fingerprint is None or cached_fingerprint == data_fingerprint)

    if cache_valid and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        return tf.keras.models.load_model(model_save_path), None
    if model_save_path.exists() and not cache_valid:
        print(f"Cached model at {model_save_path} was trained on different data (fingerprint mismatch) — retraining.")

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
    if data_fingerprint is not None:
        fingerprint_path.write_text(data_fingerprint)
    print(f"Model saved to {model_save_path}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    return model, history


def run_inference_and_reconstruct(model, inference_patch_dir, boundary, patch_size=settings.UNET_PATCH_SIZE,
                                   feature_bands=ALL_FEATURE_BANDS, out_path=CLASSIFIED_RASTER_PATH,
                                   also_write_probabilities: bool = False, prob_out_path=PROB_RASTER_PATH):
    """Reads mixer.json (patch grid layout + georeferencing, written
    automatically alongside the patches) to lay predicted patches back into
    a single georeferenced raster. Known limitation carried over from the
    notebook: non-overlapping patches can leave minor discontinuities at
    patch boundaries ("tile seams") — acceptable for a baseline comparison.

    Unlike `ee.Classifier.classify()` (used by the RF baseline), which
    automatically carries the input mask through to the output, per-patch
    CNN inference has no notion of "outside the mask" — every patch gets a
    real class prediction, including the padding area between Singapore's
    actual coastline and the rectangular patch grid's bounding box (the
    model tends to predict "water" for that blank/near-zero input, which
    silently inflated the water class). `boundary` is used to mask those
    pixels back to nodata (0) after reconstruction, matching RF's behavior.

    `also_write_probabilities=True` persists the per-class softmax array
    (`pred`, already computed below for the argmax) into a second 5-band
    GeoTIFF instead of discarding it — needed for the soft-voting ensemble.
    """
    import rasterio
    import tensorflow as tf
    from rasterio.transform import Affine

    inference_dir = Path(inference_patch_dir)
    tfrecord_files = sorted(glob.glob(str(inference_dir / "*.tfrecord.gz")))
    # GCS exports (Export.image.toCloudStorage, used here) name the mixer
    # file "{prefix}.json" — Drive exports name it "{prefix}mixer.json"
    # instead (the pattern the original notebook was written against). A
    # plain "*.json" glob matches either, since it's the only .json file
    # alongside the .tfrecord.gz shards either way.
    mixer_files = sorted(glob.glob(str(inference_dir / "*.json")))
    if not tfrecord_files or not mixer_files:
        raise FileNotFoundError(f"Missing TFRecord/mixer JSON in {inference_dir}.")

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
    full_prob = (
        np.zeros((patches_per_col * patch_size, patches_per_row * patch_size, N_CLASSES), dtype=np.float32)
        if also_write_probabilities else None
    )
    for idx, patch in enumerate(patches):
        pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
        pred_class = np.argmax(pred, axis=-1).astype(np.uint8) + 1
        row, col = idx // patches_per_row, idx % patches_per_row
        row_slice = slice(row * patch_size, (row + 1) * patch_size)
        col_slice = slice(col * patch_size, (col + 1) * patch_size)
        full_raster[row_slice, col_slice] = pred_class
        if full_prob is not None:
            full_prob[row_slice, col_slice, :] = pred
        if (idx + 1) % 50 == 0:
            print(f"  ...predicted {idx + 1}/{len(patches)} patches")

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
