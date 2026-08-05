"""U-Net data I/O: building the GEE training-stack image, exporting/caching
TFRecord patches, and parsing them into PyTorch DataLoaders. Split out of
the old combined `unet.py` (see `snapshot/tensorflow`) so the Colab training
notebooks can import just architecture+data+train, without pulling in
`unet_infer.py`'s rasterio/reconstruction code they never run.

Two caching layers on the patch exports, stacked:
1. Local disk (`_patches_already_downloaded` / fingerprint sidecar) — skip
   entirely if the right patches are already sitting in `data/interim/`.
2. GCS (`blobs_exist_with_prefix` / `download_blobs_with_prefix`) — skip the
   *GEE export itself* if another machine/session already produced the same
   patches, downloading from GCS instead. This matters specifically because
   Colab's `/content` disk is empty every session, so layer 1 alone would
   force a full multi-minute-plus GEE export on every single Colab run.
   Training patches are keyed by a fingerprint of the validation CSV (so a
   relabel correctly invalidates the cache); inference patches have no such
   natural fingerprint and are keyed by plain existence instead, matching
   local disk's existing "does it exist" semantics, just GCS-shared.
"""

import glob
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import settings
from config.settings import (
    ALL_FEATURE_BANDS,
    GCS_MODEL_BUCKET,
    GEE_EXPORT_BUCKET,
    S2_UTM_CRS,
    TARGET_SCALE_M,
    UNET_INFERENCE_PATCH_DIR,
    UNET_INFERENCE_PATCHES_GCS_PREFIX,
    UNET_TRAIN_PATCH_DIR,
    UNET_TRAIN_PATCHES_GCS_PREFIX,
)
from src.ingest.gee import export_patches_to_gcs
from src.ingest.worldcover import BUCKET_NAMES
from src.utils.gcs import blobs_exist_with_prefix, download_blobs_with_prefix

TRAIN_PATCH_DIR = UNET_TRAIN_PATCH_DIR
INFERENCE_PATCH_DIR = UNET_INFERENCE_PATCH_DIR
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
    float array — silently breaking `read_raw_patches`'s fixed-shape
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
    """Remove old shard/mixer files before a re-export. Needed because a
    re-export producing a different shard count than what's cached locally
    (e.g. the training_region's shape changed because the validation
    sample's exclusion zone moved) wouldn't overwrite leftover shards from
    the OLD export, silently contaminating training with patches from the
    wrong region."""
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
    remember to pass a --force flag. Also used to key the GCS cache
    prefix, so a relabel correctly triggers a fresh export there too."""
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
                             bucket=GEE_EXPORT_BUCKET, force: bool = False, force_export: bool = False):
    """Re-exports automatically whenever validation_csv_path's contents
    have changed since the patches currently on disk were exported
    (fingerprint mismatch) -- so relabeling the validation sample can no
    longer silently train on a stale, wrongly-excluded region, and a plain
    retrain (different hyperparameters, same labels) can reuse the cached
    patches for free instead of re-exporting for no reason. `force=True`
    bypasses the local-disk cache (but still checks GCS first);
    `force_export=True` bypasses GCS too, forcing a genuine new GEE job."""
    fingerprint = training_fingerprint(validation_csv_path)
    cache_valid = _patches_already_downloaded(TRAIN_PATCH_DIR) and _cached_fingerprint(TRAIN_PATCH_DIR) == fingerprint

    if not force and cache_valid:
        print(f"Training patches at {TRAIN_PATCH_DIR} match the current validation CSV — skipping export (pass force=True to redo anyway).")
        return TRAIN_PATCH_DIR
    if not cache_valid and _patches_already_downloaded(TRAIN_PATCH_DIR):
        print("Cached training patches don't match the current validation CSV (labels changed since last export) — re-exporting.")

    _clear_patch_dir(TRAIN_PATCH_DIR)

    gcs_prefix = f"{UNET_TRAIN_PATCHES_GCS_PREFIX}/{fingerprint}/unet_train"
    if not force_export and blobs_exist_with_prefix(bucket, gcs_prefix):
        print(f"Training patches for this fingerprint already exist at gs://{bucket}/{gcs_prefix} — downloading instead of re-exporting from GEE.")
        download_blobs_with_prefix(bucket, gcs_prefix, TRAIN_PATCH_DIR)
    else:
        export_patches_to_gcs(
            training_stack, description="unet_train", bucket=bucket, prefix=gcs_prefix,
            region=training_region, scale=scale, crs=crs, patch_dimensions=(patch_size, patch_size),
            out_dir=TRAIN_PATCH_DIR,
        )
    _write_fingerprint(TRAIN_PATCH_DIR, fingerprint)
    return TRAIN_PATCH_DIR


def export_inference_patches(feature_image, boundary, patch_size=settings.UNET_PATCH_SIZE,
                              scale=TARGET_SCALE_M, crs=S2_UTM_CRS, bucket=GEE_EXPORT_BUCKET,
                              force: bool = False, force_export: bool = False):
    """No natural fingerprint input for inference patches (season window and
    region are fixed settings, not a hashable file) — cached by plain
    existence instead, both locally and (new) in GCS."""
    if not force and _patches_already_downloaded(INFERENCE_PATCH_DIR):
        print(f"Inference patches already downloaded at {INFERENCE_PATCH_DIR} — skipping export (pass force=True to redo).")
        return INFERENCE_PATCH_DIR

    gcs_prefix = f"{UNET_INFERENCE_PATCHES_GCS_PREFIX}/unet_inference"
    if not force_export and blobs_exist_with_prefix(bucket, gcs_prefix):
        print(f"Inference patches already exist at gs://{bucket}/{gcs_prefix} — downloading instead of re-exporting from GEE.")
        download_blobs_with_prefix(bucket, gcs_prefix, INFERENCE_PATCH_DIR)
        return INFERENCE_PATCH_DIR

    return export_patches_to_gcs(
        feature_image.clip(boundary), description="unet_inference", bucket=bucket,
        prefix=gcs_prefix, region=boundary, scale=scale, crs=crs,
        patch_dimensions=(patch_size, patch_size), out_dir=INFERENCE_PATCH_DIR,
    )


def read_raw_patches(patch_dir, bands, patch_size=settings.UNET_PATCH_SIZE) -> np.ndarray:
    """Reads every `*.tfrecord.gz` shard in `patch_dir` via the pure-Python
    `tfrecord` package (no tensorflow dependency — validated standalone
    against these exact files before this module was written: patch counts,
    shapes, dtypes, and band value ranges all matched what
    `tf.io.parse_single_example` produced). Returns a
    `(n_patches, len(bands), patch_size, patch_size)` float32 array,
    channels in `bands` order and patches in on-disk shard/record order
    (row-major GEE patch-grid order — `unet_infer.py`'s tile reconstruction
    relies on this). Shared by `parse_training_patches` (below),
    `unet_infer.py`, and `src/heat_model/cnn_data.py` — previously
    duplicated 3x as near-identical `tf.io` blocks before this migration.
    """
    from tfrecord.torch.dataset import TFRecordDataset

    tfrecord_files = sorted(glob.glob(str(Path(patch_dir) / "*.tfrecord.gz")))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {patch_dir} — check the export completed.")

    description = {band: "float" for band in bands}
    patches = []
    for path in tfrecord_files:
        dataset = TFRecordDataset(path, index_path=None, description=description, compression_type="gzip")
        for record in dataset:
            stacked = np.stack(
                [np.asarray(record[band], dtype=np.float32).reshape(patch_size, patch_size) for band in bands],
                axis=0,
            )
            patches.append(stacked)
    return np.stack(patches, axis=0)


def parse_training_patches(patch_dir, patch_size=settings.UNET_PATCH_SIZE,
                            feature_bands=ALL_FEATURE_BANDS, n_classes=N_CLASSES,
                            train_val_split=settings.UNET_TRAIN_VAL_SPLIT, seed=42,
                            batch_size=settings.UNET_BATCH_SIZE):
    all_bands = feature_bands + ["wc_class"]
    raw = read_raw_patches(patch_dir, all_bands, patch_size)  # (n, len(all_bands), H, W)

    features = raw[:, : len(feature_bands)]
    label_raw = raw[:, len(feature_bands)]
    weight = (label_raw > 0).astype(np.float32)

    keep = weight.sum(axis=(1, 2)) > 0  # drop all-nodata patches
    features, label_raw, weight = features[keep], label_raw[keep], weight[keep]
    n_patches = features.shape[0]
    print(f"Usable patches (after dropping all-nodata ones): {n_patches}")

    # wc_class is 1..n_classes; CrossEntropyLoss wants 0-indexed class ids.
    # Pixels with label_raw==0 (weight==0) get clamped to a valid-but-unused
    # index — their loss contribution is zeroed by `weight` regardless.
    label_idx = np.clip(label_raw - 1, 0, None).astype(np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_patches)
    split_idx = int(n_patches * train_val_split)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    def _make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(features[idx]),
            torch.from_numpy(label_idx[idx]),
            torch.from_numpy(weight[idx]),
        )
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)

    train_loader = _make_loader(train_indices, shuffle=True)
    val_loader = _make_loader(val_indices, shuffle=False)
    print(f"Train patches: {len(train_indices)}, validation patches: {len(val_indices)}")
    return train_loader, val_loader, n_patches
