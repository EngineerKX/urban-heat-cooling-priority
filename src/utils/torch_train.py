"""Shared PyTorch training loop for `src/landcover/unet_train.py` and
`src/heat_model/cnn_train.py`. Keras's `.fit(..., callbacks=[EarlyStopping(
restore_best_weights=True)])` has no direct PyTorch equivalent — this hand-
rolls the epoch/batch loop, per-pixel-weighted loss (weighting is applied
here, not via a loss-level `weight=` kwarg, since the weighting is per-pixel
sample masking — "ignore this pixel", not a per-class rebalancing term), and
best-checkpoint tracking once, so both models share one tested implementation
instead of two near-duplicates.
"""

import copy
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

_EPS = 1e-8


def default_device() -> torch.device:
    """Colab (GPU training) vs. local (CPU inference/no training) — auto-pick
    so callers never hardcode a device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    metric_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    patience: int,
    device: torch.device | None = None,
) -> dict:
    """Every batch from both loaders is a `(features, target, weight)`
    triple. `loss_fn(pred, target)` must return a per-element loss tensor
    (i.e. built with `reduction="none"`) broadcastable against `weight` —
    the weighted mean `(loss * weight).sum() / weight.sum()` is computed
    here, uniformly for both classification and regression. `metric_fn(pred,
    target, weight)` returns a single float, printed per epoch (e.g.
    weighted accuracy or weighted RMSE) but not used for early stopping.

    Early stopping + "restore_best_weights" both track `val_loss`: training
    stops once `patience` epochs pass with no improvement, and the model's
    weights are reset to whichever epoch had the lowest `val_loss` seen,
    matching Keras's `EarlyStopping(monitor="val_loss", restore_best_weights=True)`.
    """
    device = device or default_device()
    model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss, n_train_batches = 0.0, 0
        for features, target, weight in train_loader:
            features, target, weight = features.to(device), target.to(device), weight.to(device)

            optimizer.zero_grad()
            pred = model(features)
            raw_loss = loss_fn(pred, target)
            weight_sum = weight.sum().clamp(min=_EPS)
            loss = (raw_loss * weight).sum() / weight_sum
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_train_batches += 1

        model.eval()
        running_val_loss, running_val_metric, n_val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for features, target, weight in val_loader:
                features, target, weight = features.to(device), target.to(device), weight.to(device)
                pred = model(features)
                raw_loss = loss_fn(pred, target)
                weight_sum = weight.sum().clamp(min=_EPS)
                val_loss = ((raw_loss * weight).sum() / weight_sum).item()
                running_val_loss += val_loss
                running_val_metric += metric_fn(pred, target, weight)
                n_val_batches += 1

        train_loss = running_loss / max(n_train_batches, 1)
        val_loss = running_val_loss / max(n_val_batches, 1)
        val_metric = running_val_metric / max(n_val_batches, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_metric: {val_metric:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no val_loss improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
