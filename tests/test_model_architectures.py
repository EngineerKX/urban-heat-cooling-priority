#!/usr/bin/env python
"""Standalone shape/wiring check for the PyTorch U-Net and CNN heat-model
architectures -- no trained weights, no GEE, no local patch data required.
Validates the architecture port is structurally correct (encoder/decoder
channel counts, skip connections, output shapes) independent of whether any
model has actually been trained yet. Same runnable-script + printed
pass/fail style as the rest of tests/ (no pytest in this repo).

Usage: python tests/test_model_architectures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.settings import ALL_FEATURE_BANDS
from src.heat_model.cnn_model import N_LANDCOVER_CLASSES, CNNRegressor
from src.landcover.unet_data import N_CLASSES
from src.landcover.unet_model import UNet, UNetBackbone

PATCH_SIZE = 128
N_FEATURE_BANDS = len(ALL_FEATURE_BANDS)


def test_unet_output_shape():
    model = UNet(in_channels=N_FEATURE_BANDS, n_classes=N_CLASSES, base_filters=32)
    model.eval()
    x = torch.randn(2, N_FEATURE_BANDS, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        out = model(x)
    expected = (2, N_CLASSES, PATCH_SIZE, PATCH_SIZE)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"
    print(f"PASS: UNet forward pass produces logits of shape {tuple(out.shape)}")


def test_unet_backbone_reused_by_cnn_head():
    """The CNN regressor reuses UNetBackbone directly (not a re-implementation)
    -- confirm it accepts the CNN's wider input (feature bands + one-hot
    land-cover channels) and produces the expected feature-map width."""
    in_channels = N_FEATURE_BANDS + N_LANDCOVER_CLASSES
    backbone = UNetBackbone(in_channels=in_channels, base_filters=32)
    backbone.eval()
    x = torch.randn(1, in_channels, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        feat = backbone(x)
    assert feat.shape == (1, 32, PATCH_SIZE, PATCH_SIZE), f"got {tuple(feat.shape)}"
    print(f"PASS: UNetBackbone accepts CNN's {in_channels}-channel input, outputs {tuple(feat.shape)}")


def test_cnn_regressor_output_shape():
    in_channels = N_FEATURE_BANDS + N_LANDCOVER_CLASSES
    model = CNNRegressor(in_channels=in_channels, base_filters=32)
    model.eval()
    x = torch.randn(2, in_channels, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        out = model(x)
    expected = (2, 1, PATCH_SIZE, PATCH_SIZE)
    assert out.shape == expected, f"expected {expected}, got {tuple(out.shape)}"
    print(f"PASS: CNNRegressor forward pass produces predictions of shape {tuple(out.shape)}")


def test_models_are_on_cpu_by_default():
    """Local inference must work with no GPU present (the partner's
    machine) -- a freshly constructed model's parameters should live on CPU
    until explicitly moved, never assume CUDA is available."""
    model = UNet(in_channels=N_FEATURE_BANDS, n_classes=N_CLASSES)
    for p in model.parameters():
        assert p.device.type == "cpu", f"parameter unexpectedly on {p.device}"
    print("PASS: freshly constructed UNet's parameters default to CPU")


def main():
    test_unet_output_shape()
    test_unet_backbone_reused_by_cnn_head()
    test_cnn_regressor_output_shape()
    test_models_are_on_cpu_by_default()
    print("\nAll model architecture checks passed.")


if __name__ == "__main__":
    main()
