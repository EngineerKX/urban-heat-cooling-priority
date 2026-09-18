#!/usr/bin/env python
"""Standalone shape/wiring check for the Keras U-Net and CNN heat-model
architectures -- no trained weights, no GEE, no local patch data required.
Validates the architecture is structurally correct (encoder/decoder channel
counts, skip connections, output shapes) independent of whether any model
has actually been trained yet. Same runnable-script + printed pass/fail
style as the rest of tests/ (no pytest in this repo).

Usage: python tests/test_model_architectures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import tensorflow as tf

from config.settings import ALL_FEATURE_BANDS
from src.heat_model.cnn_model import N_LANDCOVER_CLASSES, build_cnn_regressor
from src.landcover.unet_data import N_CLASSES
from src.landcover.unet_model import build_unet, build_unet_backbone

PATCH_SIZE = 128
N_FEATURE_BANDS = len(ALL_FEATURE_BANDS)


def test_unet_output_shape():
    model = build_unet((PATCH_SIZE, PATCH_SIZE, N_FEATURE_BANDS), n_classes=N_CLASSES, base_filters=32)
    x = tf.random.normal((2, PATCH_SIZE, PATCH_SIZE, N_FEATURE_BANDS))
    out = model(x, training=False)
    expected = (2, PATCH_SIZE, PATCH_SIZE, N_CLASSES)
    assert tuple(out.shape) == expected, f"expected {expected}, got {tuple(out.shape)}"
    assert np.allclose(np.asarray(out).sum(axis=-1), 1.0, atol=1e-4), "softmax head should sum to 1 across classes"
    print(f"PASS: U-Net forward pass produces softmax probabilities of shape {tuple(out.shape)}")


def test_unet_backbone_reused_by_cnn_head():
    """The CNN regressor reuses build_unet_backbone directly (not a
    re-implementation) -- confirm it accepts the CNN's wider input (feature
    bands + one-hot land-cover channels) and produces the expected
    feature-map width."""
    in_channels = N_FEATURE_BANDS + N_LANDCOVER_CLASSES
    inputs, features = build_unet_backbone((PATCH_SIZE, PATCH_SIZE, in_channels), base_filters=32)
    backbone = tf.keras.Model(inputs, features)
    x = tf.random.normal((1, PATCH_SIZE, PATCH_SIZE, in_channels))
    feat = backbone(x, training=False)
    assert tuple(feat.shape) == (1, PATCH_SIZE, PATCH_SIZE, 32), f"got {tuple(feat.shape)}"
    print(f"PASS: build_unet_backbone accepts CNN's {in_channels}-channel input, outputs {tuple(feat.shape)}")


def test_cnn_regressor_output_shape():
    in_channels = N_FEATURE_BANDS + N_LANDCOVER_CLASSES
    model = build_cnn_regressor((PATCH_SIZE, PATCH_SIZE, in_channels), base_filters=32)
    x = tf.random.normal((2, PATCH_SIZE, PATCH_SIZE, in_channels))
    out = model(x, training=False)
    expected = (2, PATCH_SIZE, PATCH_SIZE, 1)
    assert tuple(out.shape) == expected, f"expected {expected}, got {tuple(out.shape)}"
    print(f"PASS: CNNRegressor forward pass produces predictions of shape {tuple(out.shape)}")


def test_models_run_on_cpu():
    """Local inference must work with no GPU present -- a forward pass
    should succeed even when pinned to CPU explicitly."""
    with tf.device("/CPU:0"):
        model = build_unet((PATCH_SIZE, PATCH_SIZE, N_FEATURE_BANDS), n_classes=N_CLASSES)
        x = tf.random.normal((1, PATCH_SIZE, PATCH_SIZE, N_FEATURE_BANDS))
        out = model(x, training=False)
    assert tuple(out.shape) == (1, PATCH_SIZE, PATCH_SIZE, N_CLASSES)
    print("PASS: U-Net forward pass succeeds pinned to /CPU:0")


def main():
    test_unet_output_shape()
    test_unet_backbone_reused_by_cnn_head()
    test_cnn_regressor_output_shape()
    test_models_run_on_cpu()
    print("\nAll model architecture checks passed.")


if __name__ == "__main__":
    main()
