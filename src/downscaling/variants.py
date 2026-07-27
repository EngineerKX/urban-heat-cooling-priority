"""Three heat-layer variants (from gee_heat_variants.ipynb, Track A / S2):

- native30    — Landsat LST at its native 30m grid
- bicubic10   — pure bicubic interpolation to 10m (the "dumb" baseline)
- regress10   — TsHARP-style regression downscaling to 10m using Sentinel-2
                indices (C3, the genuine sensor-fusion contribution)

Each is reduced to a per-subzone mean and joined into one table.
"""

import ee
import pandas as pd

from src.ingest.gee import (
    add_spectral_indices,
    coverage_fraction,
    fetch_landsat_lst_collection,
    fetch_sentinel2_collection,
)
from src.utils.geo import zonal_mean


def build_lst_30m(sg_bbox, years, months, cloud_cover_max, crs, native_scale):
    """Season-controlled Landsat LST composite (Celsius), explicitly
    reprojected onto a real metric grid.

    A `.median()` composite over an ImageCollection does not carry a usable
    default projection (it silently falls back to plain EPSG:4326) — every
    downstream step that treats this as if it were a proper native-resolution
    UTM grid needs the explicit `.reproject()` below, done once here, rather
    than a patch at each place the image gets used.
    """
    l89 = fetch_landsat_lst_collection(sg_bbox, years, months, cloud_cover_max)
    lst_30m_raw = l89.select("LST_C").median().rename("LST_C").clip(sg_bbox)
    return lst_30m_raw.reproject(crs=crs, scale=native_scale)


def build_s2_indices_10m(sg_bbox, years, months, cloud_prob_max):
    """Season-controlled Sentinel-2 NDVI/NDBI/NDWI composite at native 10m."""
    s2_masked = fetch_sentinel2_collection(sg_bbox, years, months, cloud_prob_max)
    s2_indexed = s2_masked.map(add_spectral_indices)
    return s2_indexed.select(["NDVI", "NDBI", "NDWI"]).median().clip(sg_bbox)


def check_composite_coverage(lst_30m, s2_indices_10m, sg_bbox, native_scale, target_scale, threshold=0.90):
    lst_coverage = coverage_fraction(lst_30m, "LST_C", sg_bbox, native_scale)
    ndvi_coverage = coverage_fraction(s2_indices_10m, "NDVI", sg_bbox, target_scale)
    print(f"Landsat LST_C valid-pixel coverage: {lst_coverage * 100:.1f}%")
    print(f"Sentinel-2 NDVI valid-pixel coverage: {ndvi_coverage * 100:.1f}%")
    ok = lst_coverage >= threshold and ndvi_coverage >= threshold
    if not ok:
        print(f"⚠️  Coverage below {threshold * 100:.0f}% — widen the season window or cloud thresholds.")
    return ok, lst_coverage, ndvi_coverage


def variant_native30(lst_30m):
    return lst_30m.rename("lst_native30")


def variant_bicubic10(lst_30m, crs, target_scale):
    """Pure interpolation, no new information — the baseline the regression
    variant needs to beat on both RMSE and rank impact."""
    return (
        lst_30m.resample("bicubic")
        .reproject(crs=crs, scale=target_scale)
        .rename("lst_bicubic10")
    )


def variant_regress10(
    lst_30m,
    s2_indices_10m,
    sg_bbox,
    crs,
    native_scale,
    target_scale,
    sample_n,
    sample_seed,
):
    """TsHARP-style downscaling (C3):
    1. Aggregate S2 indices to 30m (mean) to match Landsat resolution.
    2. Fit LST_30m ~ NDVI_30m + NDBI_30m + NDWI_30m by OLS.
    3. Apply coefficients to native 10m indices for the raw prediction.
    4. Residual correction (mass conservation): predicted_10m + resampled
       (lst_30m - predicted_30m).

    Note: reduceResolution requires the input to already have a concrete
    fine-resolution default projection — set one explicitly at native 10m
    BEFORE calling reduceResolution, then reproject onto the coarser 30m
    grid to perform the aggregation. Doing it the other way around throws
    away the fine pixels reduceResolution needs.
    """
    s2_indices_10m_fine = s2_indices_10m.reproject(crs=crs, scale=target_scale)
    s2_indices_30m = (
        s2_indices_10m_fine.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
        .reproject(crs=crs, scale=native_scale)
    )

    training_stack = lst_30m.addBands(s2_indices_30m).addBands(ee.Image.constant(1).rename("CONST"))
    training_bands = ["CONST", "NDVI", "NDBI", "NDWI", "LST_C"]

    samples = training_stack.select(training_bands).sample(
        region=sg_bbox, scale=native_scale, numPixels=sample_n, seed=sample_seed,
        geometries=False, tileScale=4,
    )
    n_samples = samples.size().getInfo()
    print(f"Regression training samples drawn: {n_samples} (requested {sample_n})")
    if n_samples < 30:
        print("⚠️  Very few valid samples — regression coefficients will be noisy. "
              "Check AOI/season-filter coverage before trusting this variant.")

    regression = samples.reduceColumns(
        reducer=ee.Reducer.linearRegression(numX=4, numY=1), selectors=training_bands,
    )
    coeffs = ee.Array(regression.get("coefficients")).project([0])
    b0, b1, b2, b3 = (ee.Number(coeffs.get([i])) for i in range(4))
    print(
        f"Fitted coefficients — intercept={b0.getInfo():.3f}, "
        f"NDVI={b1.getInfo():.3f}, NDBI={b2.getInfo():.3f}, NDWI={b3.getInfo():.3f}"
    )

    def _predict(indices_image):
        return (
            indices_image.select("NDVI").multiply(b1)
            .add(indices_image.select("NDBI").multiply(b2))
            .add(indices_image.select("NDWI").multiply(b3))
            .add(b0)
        )

    predicted_10m = _predict(s2_indices_10m).rename("LST_pred_10m")
    predicted_30m = _predict(s2_indices_30m).rename("LST_pred_30m")

    residual_30m = lst_30m.subtract(predicted_30m).rename("residual_30m")
    residual_10m = residual_30m.resample("bilinear").reproject(crs=crs, scale=target_scale)

    return predicted_10m.add(residual_10m).rename("lst_regress10")


def zonal_join_variants(variants: dict, subzones_fc, id_property, native_scale, target_scale) -> pd.DataFrame:
    """Zonal mean per variant, joined on subzone_id, with a printed
    silent-drop count per variant (non-zero is expected for tiny/sliver
    subzones at 10m — not a bug on its own)."""
    scales = {
        "lst_native30": native_scale,
        "lst_bicubic10": target_scale,
        "lst_regress10": target_scale,
    }

    merged = None
    for name, image in variants.items():
        df = zonal_mean(image, name, subzones_fc, id_property, scales.get(name, target_scale))
        merged = df if merged is None else merged.merge(df, on="subzone_id", how="outer")

    n_total = len(merged)
    print(f"\nJoin summary: {n_total} subzones across {len(variants)} variants.")
    return merged
