"""U-Net architecture only — pure `torch.nn.Module`, zero I/O/GEE/rasterio
dependencies. Ported from the Keras version's `build_unet_backbone`/
`build_unet` (see `snapshot/tensorflow`'s `src/landcover/unet.py`): same
3-level encoder-decoder with skip connections, same `base_filters` doubling
per level, same bottleneck depth.

Two ports-not-bit-identical decisions, both standard/idiomatic PyTorch:
- Channels-first `(N, C, H, W)` tensors, not Keras's channels-last — callers
  building input tensors (`unet_data.py`) must stack feature bands on dim 1.
- `UNet.forward()` returns raw logits, no softmax layer. `nn.CrossEntropyLoss`
  (used by `unet_train.py`) applies log-softmax internally and is more
  numerically stable than a separate softmax + categorical-crossentropy —
  callers that need class *probabilities* (e.g. `unet_infer.py`'s output
  raster) apply `torch.softmax(logits, dim=1)` explicitly.
"""

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
        nn.ReLU(inplace=True),
    )


class UNetBackbone(nn.Module):
    """Encoder-decoder with skip connections, WITHOUT an output head —
    shared by `UNet`'s classification head and
    `src/heat_model/cnn_model.py::CNNRegressor`'s regression head, so the
    two models stay identical apart from what they predict. `forward()`
    returns the final decoder feature map, `out_channels` wide."""

    def __init__(self, in_channels: int, base_filters: int = 32):
        super().__init__()
        self.out_channels = base_filters

        self.enc1 = _conv_block(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = _conv_block(base_filters * 4, base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_filters * 8, base_filters * 4)
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_filters * 4, base_filters * 2)
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_filters * 2, base_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        b = self.bottleneck(self.pool3(c3))

        d3 = self.dec3(torch.cat([self.up3(b), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return d1


class UNet(nn.Module):
    """`UNetBackbone` + a 1x1-conv classification head. `forward()` returns
    raw logits of shape `(N, n_classes, H, W)` — see module docstring."""

    def __init__(self, in_channels: int, n_classes: int, base_filters: int = 32):
        super().__init__()
        self.backbone = UNetBackbone(in_channels, base_filters)
        self.head = nn.Conv2d(self.backbone.out_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
