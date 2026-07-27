"""ESA WorldCover, collapsed to the locked 4-class gate-review scheme
(vegetation / built_up / bare / water). This exact remap table used to be
copy-pasted into `generate_validation_sample.ipynb`, `train_rf_baseline.ipynb`
and `train_unet.ipynb` independently — one copy here now.

Scope reminder carried over from the notebooks: WorldCover is TRAINING data
only. It is never the validation answer key — validation uses Dynamic World
plus the independent hand-labeled sample (see
validation/input_validation/labeling_sample.py).
"""

import ee

from config.settings import (
    BUCKET_NAMES,
    ORIGINAL_WC_NAMES,
    WC_TO_BUCKET_FROM,
    WC_TO_BUCKET_TO,
    WORLDCOVER_ASSET,
)

__all__ = ["BUCKET_NAMES", "ORIGINAL_WC_NAMES", "get_worldcover_bucket_image"]


def get_worldcover_bucket_image(boundary, crs: str, scale: int, valid_mask=None):
    """WorldCover, reprojected onto `crs`/`scale`, clipped to `boundary`,
    collapsed to the 4-class bucket scheme. Returns an image with bands
    `wc_class` (bucket id, masked out where snow/ice, out of Singapore, or —
    if `valid_mask` is given — wherever the caller's own composite has no
    valid pixel, so a label never survives where there's no feature data to
    pair it with) and `wc_raw` (original fine-grained WorldCover code,
    audit-trail only, unaffected by `valid_mask`).
    """
    worldcover_raw = ee.Image(WORLDCOVER_ASSET).select("Map")
    worldcover = worldcover_raw.reproject(crs=crs, scale=scale).clip(boundary)

    wc_bucket = worldcover.remap(WC_TO_BUCKET_FROM, WC_TO_BUCKET_TO, 0).rename("wc_class")
    wc_bucket = wc_bucket.updateMask(wc_bucket.neq(0))
    if valid_mask is not None:
        wc_bucket = wc_bucket.updateMask(valid_mask)

    return ee.Image.cat([wc_bucket, worldcover.rename("wc_raw")])
