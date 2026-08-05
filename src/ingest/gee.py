"""Earth Engine session bootstrap + the Landsat/Sentinel-2 masking, scaling
and season-filtering helpers that used to be pasted into nearly every
notebook verbatim.

Auth: every notebook did `ee.Initialize()` and fell back to an interactive
`ee.Authenticate()` browser popup. That can't run unattended in a script, so
`init_ee()` instead authenticates non-interactively via the service account
already anticipated by `.env.example` (`GEE_SERVICE_ACCOUNT` /
`GEE_PRIVATE_KEY_PATH`).
"""

import ee

from config.settings import (
    GEE_PRIVATE_KEY_PATH,
    GEE_PROJECT_ID,
    GEE_SERVICE_ACCOUNT,
)
from src.utils.gcs import download_blob, download_blobs_with_prefix

_initialized = False


def init_ee(force: bool = False) -> None:
    """Authenticate + initialize Earth Engine via service account. Safe to
    call repeatedly — a no-op after the first successful call unless
    `force=True`."""
    global _initialized
    if _initialized and not force:
        return

    if not GEE_SERVICE_ACCOUNT or not GEE_PRIVATE_KEY_PATH:
        raise RuntimeError(
            "GEE_SERVICE_ACCOUNT / GEE_PRIVATE_KEY_PATH are not set. Copy .env.example "
            "to .env and fill in your service account details, then re-run."
        )

    credentials = ee.ServiceAccountCredentials(GEE_SERVICE_ACCOUNT, GEE_PRIVATE_KEY_PATH)
    ee.Initialize(credentials, project=GEE_PROJECT_ID)
    _initialized = True
    print(f"EE initialized OK (service account), project: {GEE_PROJECT_ID}")


def date_filter_for_years_months(collection, years, months):
    """Union filter: keep images that fall in ANY (year, month) combo — used
    to build season-controlled, multi-year composites (C4). A plain
    filterDate(start, end) would mix wet/dry-season conditions."""
    filters = []
    for y in years:
        for m in months:
            start = ee.Date.fromYMD(y, m, 1)
            end = start.advance(1, "month")
            filters.append(ee.Filter.date(start, end))
    return collection.filter(ee.Filter.Or(*filters))


# --- Landsat 8/9 Collection 2 Level 2 ---------------------------------------

def mask_landsat_c2l2(image):
    """Cloud/shadow/snow mask using QA_PIXEL bits, per USGS C2 L2 spec."""
    qa = image.select("QA_PIXEL")
    dilated_cloud = 1 << 1
    cirrus = 1 << 2
    cloud = 1 << 3
    shadow = 1 << 4
    snow = 1 << 5
    mask = (
        qa.bitwiseAnd(dilated_cloud).eq(0)
        .And(qa.bitwiseAnd(cirrus).eq(0))
        .And(qa.bitwiseAnd(cloud).eq(0))
        .And(qa.bitwiseAnd(shadow).eq(0))
        .And(qa.bitwiseAnd(snow).eq(0))
    )
    sat_mask = image.select("QA_RADSAT").eq(0)
    return image.updateMask(mask).updateMask(sat_mask)


def scale_landsat_c2l2(image):
    """Official C2 L2 scale/offset factors: optical -> reflectance, ST_B10 -> Kelvin."""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermal = image.select("ST_B10").multiply(0.00341802).add(149.0)
    return image.addBands(optical, None, True).addBands(thermal, None, True)


def scale_landsat_st_celsius(image):
    """ST_B10 -> Celsius, renamed LST_C (the band name every downscaling/
    pillar module downstream expects)."""
    lst_c = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
        .rename("LST_C")
    )
    return image.addBands(lst_c, overwrite=True)


def fetch_landsat_lst_collection(sg_bbox, years, months, cloud_cover_max):
    """Landsat 8+9, season-filtered, cloud-masked, LST in Celsius (LST_C band)."""
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(sg_bbox)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))
    )
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(sg_bbox)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_cover_max))
    )
    l89 = date_filter_for_years_months(l8.merge(l9), years, months)
    return l89.map(mask_landsat_c2l2).map(scale_landsat_st_celsius)


# --- Sentinel-2 L2A ----------------------------------------------------------

def mask_s2_cloud_prob(cloud_prob_image, prob_threshold):
    """s2cloudless probability mask for Sentinel-2 L2A."""
    return cloud_prob_image.select("probability").lt(prob_threshold)


def join_s2_with_cloud_prob(s2_sr, s2_cloud_prob):
    return ee.Join.saveFirst("cloud_mask").apply(
        primary=s2_sr,
        secondary=s2_cloud_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )


def apply_s2_cloud_mask(joined_collection, prob_threshold, scl_mask=False):
    """Map cloud-probability masking (and, optionally, SCL-class masking —
    drop shadow/cloud-med/cloud-high/cirrus/snow classes) over a collection
    already joined via `join_s2_with_cloud_prob`."""

    def _mask(img):
        img = ee.Image(img)
        cloud_img = ee.Image(img.get("cloud_mask"))
        clear_mask = mask_s2_cloud_prob(cloud_img, prob_threshold)
        img = img.updateMask(clear_mask)
        if scl_mask:
            scl = img.select("SCL")
            valid_scl = (
                scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            )
            img = img.updateMask(valid_scl)
        return img

    return ee.ImageCollection(joined_collection).map(_mask)


def fetch_sentinel2_collection(sg_bbox, years, months, cloud_prob_max, scl_mask=False):
    """Sentinel-2 L2A, season-filtered, cloud-masked (bands still at their
    native reflectance scale — caller adds indices as needed)."""
    s2_sr = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(sg_bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
    )
    s2_sr = date_filter_for_years_months(s2_sr, years, months)

    s2_cloud_prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterBounds(sg_bbox)
    s2_cloud_prob = date_filter_for_years_months(s2_cloud_prob, years, months)

    joined = join_s2_with_cloud_prob(s2_sr, s2_cloud_prob)
    return apply_s2_cloud_mask(joined, cloud_prob_max, scl_mask=scl_mask)


def add_spectral_indices(image):
    """NDVI/NDBI/NDWI from Sentinel-2 bands (B8=NIR, B4=Red, B11=SWIR1, B3=Green)."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndvi, ndbi, ndwi])


def export_patches_to_gcs(image, description: str, bucket: str, prefix: str, region,
                           scale: int, crs: str, patch_dimensions, out_dir):
    """Export `image` as compressed TFRecord patches to Cloud Storage, poll
    until done, then download every resulting blob (the .tfrecord.gz shards
    plus mixer.json) to `out_dir`. This is the local-disk-friendly
    replacement for the notebooks' `Export.image.toDrive(...)` + manual
    Drive-mount step — GEE's patch-export mechanism only supports Drive or
    GCS as a destination, there's no direct synchronous local download for
    it, so GCS + immediate download is the closest thing to "just save it
    locally" for this specific operation.
    """
    import time
    from pathlib import Path

    if not bucket:
        raise RuntimeError(
            "GEE_EXPORT_BUCKET is not set in .env — a GCS bucket is required for TFRecord "
            "patch export (Drive/GCS are the only export destinations GEE supports for this)."
        )

    task = ee.batch.Export.image.toCloudStorage(
        image=image, description=description, bucket=bucket, fileNamePrefix=prefix,
        region=region, scale=scale, crs=crs, maxPixels=1e13, fileFormat="TFRecord",
        formatOptions={"patchDimensions": list(patch_dimensions), "compressed": True},
    )
    task.start()
    print(f"Patch export started: gs://{bucket}/{prefix}")
    while task.active():
        print(f"  ...{task.status()['state']}")
        time.sleep(20)
    status = task.status()
    if status["state"] != "COMPLETED":
        raise RuntimeError(f"Patch export did not complete: {status}")
    print("✅ Export completed — downloading blobs locally.")

    downloaded = download_blobs_with_prefix(bucket, prefix, out_dir)
    print(f"Downloaded {len(downloaded)} file(s) to {out_dir}")
    return Path(out_dir)


def download_small_image(image, region, scale: int, crs: str, out_path, bands=None) -> "Path":
    """Synchronous download of a SMALL image region as a local GeoTIFF, via
    `ee.Image.getDownloadURL` (a zipped GeoTIFF) + `requests` — no extra
    dependency needed beyond what's already in requirements.txt.

    Deliberately not using `geemap.ee_export_image` here: geemap's package
    `__init__.py` unconditionally imports the full interactive-widget stack
    (ipyleaflet, bqplot, anywidget, ...) even though we only want this one
    export function, and on Windows that dependency chain failed to install
    outright (ipyleaflet ships jupyterlab labextension assets with paths
    long enough to trip the default 260-character Windows path limit).

    Only suitable for small regions (a test tile, not all of Singapore) —
    `getDownloadURL` has an undocumented but real response-size cap. For
    full-Singapore-scale rasters, use `export_geotiff_to_gcs` instead.
    """
    import zipfile
    from io import BytesIO
    from pathlib import Path

    import requests

    params = {"region": region, "scale": scale, "crs": crs, "format": "GEO_TIFF"}
    if bands:
        image = image.select(list(bands))
    url = image.getDownloadURL(params)

    resp = requests.get(url)
    resp.raise_for_status()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    content_type = resp.headers.get("content-type", "")
    if "zip" in content_type or resp.content[:2] == b"PK":
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            tif_names = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            with zf.open(tif_names[0]) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    else:
        with open(out_path, "wb") as f:
            f.write(resp.content)

    print(f"Downloaded image -> {out_path}")
    return out_path


def export_geotiff_to_gcs(image, description: str, bucket: str, prefix: str, region,
                           scale: int, crs: str, out_path):
    """Export a full-size image as a GeoTIFF via Cloud Storage (async task,
    same reasoning as `export_patches_to_gcs`: GEE's export mechanism for
    anything beyond a small synchronous download requires Drive or GCS as a
    destination), then download the result straight to local disk."""
    import time
    from pathlib import Path

    if not bucket:
        raise RuntimeError(
            "GEE_EXPORT_BUCKET is not set in .env — a GCS bucket is required to export "
            "rasters too large for a synchronous download."
        )

    task = ee.batch.Export.image.toCloudStorage(
        image=image, description=description, bucket=bucket, fileNamePrefix=prefix,
        region=region, scale=scale, crs=crs, maxPixels=1e13, fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Raster export started: gs://{bucket}/{prefix}.tif")
    while task.active():
        print(f"  ...{task.status()['state']}")
        time.sleep(20)
    status = task.status()
    if status["state"] != "COMPLETED":
        raise RuntimeError(f"Raster export did not complete: {status}")

    out_path = download_blob(bucket, f"{prefix}.tif", out_path)
    print(f"✅ Downloaded raster -> {out_path}")
    return out_path


def fetch_modis_lst_collection(sg_bbox, years, months, day_or_night: str = "day"):
    """MODIS/061/MOD11A2, 8-day 1km LST composites, season-filtered (reuses
    date_filter_for_years_months unmodified) -- a genuine LST product
    (same physical quantity as Landsat's thermal band), unlike NEA's
    air-temperature proxy, at the cost of much coarser 1km resolution.
    Used as the S5-adjacent secondary LST cross-check
    (validation/input_validation/modis_heldout.py). 0 is this product's
    fill value, not a legitimate 0 Kelvin reading -- masked out before the
    Kelvin*0.02 -> Celsius scale conversion."""
    band = "LST_Day_1km" if day_or_night == "day" else "LST_Night_1km"
    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(sg_bbox)
        .select(band)
    )
    collection = date_filter_for_years_months(collection, years, months)

    def _scale_and_mask(image):
        valid = image.select(band).gt(0)
        lst_c = image.select(band).multiply(0.02).subtract(273.15).rename("LST_C")
        return lst_c.updateMask(valid)

    return collection.map(_scale_and_mask)


def coverage_fraction(image, band_name, aoi, scale) -> float:
    """Fraction of `aoi` pixels that have valid (unmasked) data in `image`.
    Low coverage means the compositing window/threshold needs widening."""
    mask_img = image.select(band_name).mask()
    stats = mask_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=scale, maxPixels=1e10, bestEffort=True,
    )
    return float(ee.Number(stats.get(band_name)).getInfo())
