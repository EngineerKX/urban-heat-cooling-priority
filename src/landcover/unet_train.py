"""U-Net training — meant to run in Colab (GPU), called from
`notebooks/colab_training/train_unet.ipynb`. Imports only `unet_model` +
`unet_data` + `src.utils.torch_train`, deliberately not `unet_infer.py`
(rasterio reconstruction never runs in Colab, see the migration plan).
"""

from pathlib import Path

import torch
from torch import nn

from config import settings
from config.settings import ALL_FEATURE_BANDS, UNET_MODEL_SAVE_PATH
from src.landcover.unet_data import N_CLASSES
from src.landcover.unet_model import UNet
from src.utils.torch_train import default_device, train_loop


def _weighted_accuracy(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> float:
    """Matches Keras's `weighted_metrics=[CategoricalAccuracy()]` — mean
    accuracy over valid (weight>0) pixels only."""
    preds = pred.argmax(dim=1)
    correct = (preds == target).float() * weight
    return (correct.sum() / weight.sum().clamp(min=1e-8)).item()


def train_unet(train_loader, val_loader, patch_size=settings.UNET_PATCH_SIZE,
                n_feature_bands=len(ALL_FEATURE_BANDS), n_classes=N_CLASSES,
                epochs=settings.UNET_EPOCHS, learning_rate=settings.UNET_LEARNING_RATE,
                patience=settings.UNET_EARLY_STOP_PATIENCE, base_filters=settings.UNET_BASE_FILTERS,
                model_save_path=UNET_MODEL_SAVE_PATH, data_fingerprint: str | None = None,
                force_retrain: bool = False, device: torch.device | None = None):
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

    model = UNet(in_channels=n_feature_bands, n_classes=n_classes, base_filters=base_filters)

    if cache_valid and not force_retrain:
        print(f"Loading cached model from {model_save_path} (pass force_retrain=True to retrain).")
        model.load_state_dict(torch.load(model_save_path, map_location="cpu"))
        return model, None
    if model_save_path.exists() and not cache_valid:
        print(f"Cached model at {model_save_path} was trained on different data (fingerprint mismatch) — retraining.")

    device = device or default_device()
    print(model)
    print(f"Training on device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    history = train_loop(
        model, train_loader, val_loader, loss_fn, _weighted_accuracy, optimizer,
        epochs=epochs, patience=patience, device=device,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    if data_fingerprint is not None:
        fingerprint_path.write_text(data_fingerprint)
    print(f"Model saved to {model_save_path}")
    print(f"Best val_loss: {min(history['val_loss']):.4f}")
    return model, history
