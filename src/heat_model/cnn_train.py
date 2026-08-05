"""S5 (C2) CNN heat-model training — meant to run in Colab (GPU), called
from `notebooks/colab_training/train_heat_cnn.ipynb`.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    CNN_MODEL_SAVE_PATH,
    RANDOM_SEED,
    UNET_BASE_FILTERS,
    UNET_BATCH_SIZE,
    UNET_EARLY_STOP_PATIENCE,
    UNET_EPOCHS,
    UNET_LEARNING_RATE,
    UNET_TRAIN_VAL_SPLIT,
)
from src.heat_model.cnn_model import CNNRegressor
from src.utils.torch_train import default_device, train_loop


def _weighted_rmse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> float:
    """Matches Keras's `weighted_metrics=[RootMeanSquaredError()]`."""
    se = (pred.squeeze(1) - target) ** 2 * weight
    mse = se.sum() / weight.sum().clamp(min=1e-8)
    return torch.sqrt(mse).item()


def train_cnn_regressor(
    X, y, valid_mask, model_save_path=CNN_MODEL_SAVE_PATH, epochs=UNET_EPOCHS,
    learning_rate=UNET_LEARNING_RATE, patience=UNET_EARLY_STOP_PATIENCE,
    train_val_split=UNET_TRAIN_VAL_SPLIT, batch_size=UNET_BATCH_SIZE, seed=RANDOM_SEED,
    base_filters=UNET_BASE_FILTERS, force_retrain: bool = False, device: torch.device | None = None,
):
    """Same "skip if a saved model already exists" caching convention as
    `src/landcover/unet_train.py::train_unet`. No content-fingerprint cache
    here -- unlike U-Net's training labels, this model's target/features
    come from already-cached rasters rather than a validation CSV that gets
    relabeled, so there's no separate "did the underlying data change"
    signal to hash."""
    model_save_path = Path(model_save_path)
    n_bands = X.shape[1]
    model = CNNRegressor(in_channels=n_bands, base_filters=base_filters)

    if model_save_path.exists() and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        return model, None

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    split_idx = int(n * train_val_split)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    weight = valid_mask.astype(np.float32)

    def _make_loader(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]), torch.from_numpy(weight[idx]))
        generator = torch.Generator().manual_seed(seed) if shuffle else None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)

    train_loader = _make_loader(train_idx, True)
    val_loader = _make_loader(val_idx, False)
    print(f"Train patches: {len(train_idx)}, validation patches: {len(val_idx)}")

    device = device or default_device()
    print(model)
    print(f"Training on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_fn(pred, target):
        return (pred.squeeze(1) - target) ** 2  # per-pixel squared error, reduction="none"

    history = train_loop(
        model, train_loader, val_loader, loss_fn, _weighted_rmse, optimizer,
        epochs=epochs, patience=patience, device=device,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Best val_loss (MSE): {min(history['val_loss']):.4f}")
    return model, history
