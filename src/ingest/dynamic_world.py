"""Google Dynamic World, collapsed to the same locked 4-class gate-review
scheme as ESA WorldCover (vegetation / built_up / bare / water) -- an
alternative TRAINING label source, for the RF trial in
scripts/train_landcover_rf_dynamicworld.py (checkpoint_WorldCover branch).

Mirrors src/ingest/worldcover.py's structure and output band convention on
purpose, so both label sources can be dropped into the same RF training code
almost unchanged (see src/landcover/rf_baseline.py's `class_band` parameter).
"""

import ee

from config.settings import DW_TO_BUCKET_FROM, DW_TO_BUCKET_TO

DW_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"

__all__ = ["get_dynamicworld_bucket_image"]


def get_dynamicworld_bucket_image(boundary, crs: str, scale: int, start: str, end: str, valid_mask=None):
    """Dynamic World's most common ("mode") class per pixel over [start, end),
    reprojected onto `crs`/`scale`, clipped to `boundary`, collapsed to the
    4-class bucket scheme. Returns an image with band `dw_class` (bucket id,
    masked out where snow/ice, out of Singapore, or -- if `valid_mask` is
    given -- wherever the caller's own composite has no valid pixel, same
    "a label never survives where there's no feature data to pair it with"
    rule as `get_worldcover_bucket_image`).

    Unlike WorldCover (one fixed annual release), Dynamic World labels every
    Sentinel-2 scene, so a window has to be reduced to one "typical" class
    per pixel first -- `ee.Reducer.mode()`, the same choice
    `validation/input_validation/land_change.py` makes, robust to the
    occasional misclassified scene.
    """
    col = ee.ImageCollection(DW_COLLECTION).filterBounds(boundary).select("label").filterDate(start, end)
    n_images = col.size().getInfo()
    print(f"Dynamic World images {start} to {end}: {n_images}")
    if n_images == 0:
        raise RuntimeError(f"No Dynamic World images between {start} and {end} over this boundary.")

    dw_native = col.reduce(ee.Reducer.mode()).rename("dw_raw")
    dw = dw_native.reproject(crs=crs, scale=scale).clip(boundary)

    dw_bucket = dw.remap(DW_TO_BUCKET_FROM, DW_TO_BUCKET_TO, 0).rename("dw_class")
    dw_bucket = dw_bucket.updateMask(dw_bucket.neq(0))
    if valid_mask is not None:
        dw_bucket = dw_bucket.updateMask(valid_mask)

    return ee.Image.cat([dw_bucket, dw.rename("dw_raw")])
