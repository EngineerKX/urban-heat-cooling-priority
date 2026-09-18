"""U-Net architecture only — pure Keras functional-API builders, zero
I/O/GEE/rasterio dependencies. Same 3-level encoder-decoder with skip
connections, same `base_filters` doubling per level, same bottleneck depth
as the pre-PyTorch-migration version (`snapshot/tensorflow`'s
`src/landcover/unet.py`).

Channels-last `(N, H, W, C)` tensors, matching Keras convention — callers
building input tensors (`unet_data.py`) must stack feature bands on the
last axis.
"""

from tensorflow.keras import layers, models


def _conv_block(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_unet_backbone(input_shape, base_filters: int = 32):
    """3-level encoder-decoder with skip connections, WITHOUT an output
    head — shared by `build_unet`'s softmax classification head and
    `src/heat_model/cnn_model.py::build_cnn_regressor`'s linear regression
    head, so the two models stay identical apart from what they predict.
    Returns `(inputs, final_decoder_features)` for a caller to attach
    `layers.Conv2D(n_outputs, 1, activation=...)` onto."""
    inputs = layers.Input(shape=input_shape)

    c1 = _conv_block(inputs, base_filters)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = _conv_block(p1, base_filters * 2)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = _conv_block(p2, base_filters * 4)
    p3 = layers.MaxPooling2D(2)(c3)

    b = _conv_block(p3, base_filters * 8)

    u3 = layers.Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(b)
    u3 = layers.Concatenate()([u3, c3])
    d3 = _conv_block(u3, base_filters * 4)

    u2 = layers.Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = _conv_block(u2, base_filters * 2)

    u1 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = _conv_block(u1, base_filters)

    return inputs, d1


def build_unet(input_shape, n_classes: int, base_filters: int = 32):
    """`build_unet_backbone` + a 1x1-conv softmax classification head.
    Output shape `(N, H, W, n_classes)`."""
    inputs, d1 = build_unet_backbone(input_shape, base_filters)
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)
    return models.Model(inputs, outputs, name="unet_landcover")
