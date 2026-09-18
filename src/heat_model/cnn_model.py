"""S5 (C2) CNN heat-model architecture only — pure Keras functional-API
builder. Reuses `src.landcover.unet_model.build_unet_backbone` (same
encoder-decoder as the land-cover U-Net) with a linear regression head
instead of a softmax classification head, so the two models stay identical
apart from what they predict.
"""

from tensorflow.keras import layers, models

from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.unet_model import build_unet_backbone

N_LANDCOVER_CLASSES = len(BUCKET_NAMES)


def build_cnn_regressor(input_shape, base_filters: int = 32):
    """`build_unet_backbone` + a 1x1-conv linear regression head. Output
    shape `(N, H, W, 1)` — predicted `lst_bicubic10`."""
    inputs, d1 = build_unet_backbone(input_shape, base_filters)
    outputs = layers.Conv2D(1, 1, activation="linear")(d1)
    return models.Model(inputs, outputs, name="heat_cnn_regressor")
