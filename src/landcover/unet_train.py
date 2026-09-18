"""U-Net training — meant to run in Colab (GPU), called from
`notebooks/colab_training/train_unet.ipynb`. Imports only `unet_model` +
`unet_data`, deliberately not `unet_infer.py` (rasterio reconstruction
never runs in Colab, see the migration plan).
"""

from pathlib import Path

import tensorflow as tf

from config import settings
from config.settings import ALL_FEATURE_BANDS, UNET_MODEL_SAVE_PATH
from src.landcover.unet_data import N_CLASSES
from src.landcover.unet_model import build_unet


def train_unet(train_ds, val_ds, patch_size=settings.UNET_PATCH_SIZE,
                n_feature_bands=len(ALL_FEATURE_BANDS), n_classes=N_CLASSES,
                epochs=settings.UNET_EPOCHS, learning_rate=settings.UNET_LEARNING_RATE,
                patience=settings.UNET_EARLY_STOP_PATIENCE, base_filters=settings.UNET_BASE_FILTERS,
                model_save_path=UNET_MODEL_SAVE_PATH, data_fingerprint: str | None = None,
                force_retrain: bool = False):
    """Skip training entirely if a saved model already exists AND (when
    `data_fingerprint` is given) it was trained on the same data -- a
    fingerprint mismatch means the patches this model saw have since been
    replaced (e.g. the validation sample got relabeled and training_region
    moved), so silently reusing it would be training on stale, wrongly-
    excluded data one level removed from the patch cache itself.
    `data_fingerprint=None` (the default) skips this check entirely, same
    as before. `force_retrain=True` always retrains regardless."""
    model_save_path = Path(model_save_path)
    fingerprint_path = model_save_path.parent / f"{model_save_path.stem}_fingerprint.txt"
    cached_fingerprint = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
    cache_valid = model_save_path.exists() and (data_fingerprint is None or cached_fingerprint == data_fingerprint)

    if cache_valid and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        return tf.keras.models.load_model(model_save_path), None
    if model_save_path.exists() and not cache_valid:
        print(f"Cached model at {model_save_path} was trained on different data (fingerprint mismatch) — retraining.")

    model = build_unet((patch_size, patch_size, n_feature_bands), n_classes=n_classes, base_filters=base_filters)
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
    # Normalize Keras's {"loss","accuracy","val_loss","val_accuracy"} history
    # keys to the {"train_loss","train_metric","val_loss","val_metric"} shape
    # the training notebooks' logging cell and export_run_summary expect.
    normalized_history = {
        "train_loss": history.history["loss"],
        "train_metric": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_metric": history.history["val_accuracy"],
    }
    return model, normalized_history
