"""Training-label source for the land-cover classifiers, resolved from
config.settings.LANDCOVER_LABEL_SOURCE so the RF script and the U-Net
notebook can't disagree about which product they learn from.

  "dynamicworld"  Dynamic World mode class over DW_TRAIN_START..DW_TRAIN_END
                  (production since 2026-09-29) -> band `dw_class`
  "worldcover"    ESA WorldCover v200 (2021), the original source, kept as the
                  comparison -> band `wc_class`

Both come back collapsed to the same 4 buckets (src/ingest/worldcover.py,
src/ingest/dynamic_world.py). `label_source_tag` is what keys every cache that
depends on the labels (the U-Net training-patch fingerprint and the RF GEE
classifier asset), so changing the source or its window can never silently
reuse a model or patches built from different labels.
"""

import hashlib
from pathlib import Path

from config.settings import DW_TRAIN_END, DW_TRAIN_START, LANDCOVER_LABEL_SOURCE

LABEL_SOURCES = ("dynamicworld", "worldcover")


def _check(source: str) -> str:
    if source not in LABEL_SOURCES:
        raise ValueError(f"Unknown land-cover label source '{source}', expected one of {LABEL_SOURCES}.")
    return source


def get_training_label_image(boundary, crs: str, scale: int, valid_mask=None, source: str = LANDCOVER_LABEL_SOURCE):
    """Returns (label_image, class_band) for `source`, masked to `valid_mask`."""
    if _check(source) == "dynamicworld":
        from src.ingest.dynamic_world import get_dynamicworld_bucket_image
        image = get_dynamicworld_bucket_image(boundary, crs, scale, DW_TRAIN_START, DW_TRAIN_END, valid_mask=valid_mask)
        return image, "dw_class"
    from src.ingest.worldcover import get_worldcover_bucket_image
    return get_worldcover_bucket_image(boundary, crs, scale, valid_mask=valid_mask), "wc_class"


def label_source_tag(source: str = LANDCOVER_LABEL_SOURCE) -> str:
    """Stable identifier for the labels, including the Dynamic World window.
    "worldcover" stays exactly "worldcover" so existing WorldCover caches keep
    their keys (src/landcover/unet_data.py::training_fingerprint)."""
    if _check(source) == "dynamicworld":
        return f"dynamicworld_{DW_TRAIN_START}_{DW_TRAIN_END}"
    return "worldcover"


def rf_classifier_asset_name(validation_csv_path, source: str = LANDCOVER_LABEL_SOURCE) -> str:
    """GEE asset name for a cached RF classifier, keyed on the label source AND
    the validation points its training region excludes. The old fixed name
    ("rf_landcover_classifier") let a classifier trained against an earlier
    200-point validation sample be reused silently after the sample changed."""
    csv_hash = hashlib.sha256(Path(validation_csv_path).read_bytes()).hexdigest()[:10]
    tag = label_source_tag(source).replace("-", "")
    return f"rf_landcover_classifier_{tag}_val{csv_hash}"
