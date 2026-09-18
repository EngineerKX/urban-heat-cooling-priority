"""S5 (C2) CNN heat-model training — meant to run in Colab (GPU), called
from `notebooks/colab_training/train_heat_cnn.ipynb`.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

from config.settings import (
    CNN_BASE_FILTERS,
    CNN_BATCH_SIZE,
    CNN_EARLY_STOP_PATIENCE,
    CNN_EPOCHS,
    CNN_LEARNING_RATE,
    CNN_MODEL_SAVE_PATH,
    CNN_TRAIN_VAL_SPLIT,
    RANDOM_SEED,
)
from src.heat_model.cnn_model import build_cnn_regressor


def train_cnn_regressor(
    X, y, valid_mask, model_save_path=CNN_MODEL_SAVE_PATH, epochs=CNN_EPOCHS,
    learning_rate=CNN_LEARNING_RATE, patience=CNN_EARLY_STOP_PATIENCE,
    train_val_split=CNN_TRAIN_VAL_SPLIT, batch_size=CNN_BATCH_SIZE, seed=RANDOM_SEED,
    base_filters=CNN_BASE_FILTERS, force_retrain: bool = False,
):
    """Same "skip if a saved model already exists" caching convention as
    `src/landcover/unet_train.py::train_unet`. No content-fingerprint cache
    here -- unlike U-Net's training labels, this model's target/features
    come from already-cached rasters rather than a validation CSV that gets
    relabeled, so there's no separate "did the underlying data change"
    signal to hash."""
    model_save_path = Path(model_save_path)

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

    model = build_cnn_regressor(X.shape[1:], base_filters=base_filters)
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
    # Normalize Keras's {"loss","rmse","val_loss","val_rmse"} history keys to
    # the {"train_loss","train_metric","val_loss","val_metric"} shape the
    # training notebook's logging cell and export_run_summary expect.
    normalized_history = {
        "train_loss": history.history["loss"],
        "train_metric": history.history["rmse"],
        "val_loss": history.history["val_loss"],
        "val_metric": history.history["val_rmse"],
    }
    return model, normalized_history
