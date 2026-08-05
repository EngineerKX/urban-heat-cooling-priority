"""S5 (C2) CNN heat-model architecture only — pure `torch.nn.Module`.
Reuses `src.landcover.unet_model.UNetBackbone` (same encoder-decoder as the
land-cover U-Net) with a linear regression head instead of a softmax
classification head, so the two models stay identical apart from what they
predict.
"""

from torch import nn

from src.ingest.worldcover import BUCKET_NAMES
from src.landcover.unet_model import UNetBackbone

N_LANDCOVER_CLASSES = len(BUCKET_NAMES)


class CNNRegressor(nn.Module):
    """`forward()` returns raw linear output of shape `(N, 1, H, W)` —
    predicted `lst_bicubic10`."""

    def __init__(self, in_channels: int, base_filters: int = 32):
        super().__init__()
        self.backbone = UNetBackbone(in_channels, base_filters)
        self.head = nn.Conv2d(self.backbone.out_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.head(self.backbone(x))
