#!/usr/bin/env python
"""Standalone verification for src/landcover/zonal.py against a small
synthetic in-memory GeoTIFF -- no GEE/GPU required. Same style as
tests/test_landcover_ensemble.py: runnable script, plain asserts, printed
pass/fail (this repo has no pytest convention).

Usage: python tests/test_landcover_zonal.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from shapely.geometry import box

from src.landcover.zonal import zonal_class_fractions

CRS = "EPSG:32648"
TRANSFORM = Affine(10.0, 0.0, 100000.0, 0.0, -10.0, 200000.0)
CLASS_NAMES = {1: "vegetation", 2: "built_up", 3: "bare", 4: "water"}


def _write_label_raster(path):
    """4x4 uint8 raster: top-left 2x2 = vegetation, top-right 2x2 = built_up,
    bottom half (2x4) = 0 (nodata), matching the ensemble raster's own
    0=nodata / 1..4=bucket convention."""
    band = np.zeros((4, 4), dtype=np.uint8)
    band[0:2, 0:2] = 1  # vegetation
    band[0:2, 2:4] = 2  # built_up
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype=np.uint8, crs=CRS, transform=TRANSFORM,
    ) as dst:
        dst.write(band, 1)


def _make_gdf(rows, crs=CRS):
    return gpd.GeoDataFrame(
        {"SUBZONE_N": [r[0] for r in rows]},
        geometry=[r[1] for r in rows],
        crs=crs,
    )


def test_zonal_class_fractions_pure_and_mixed_and_offraster(tmp_dir):
    raster_path = tmp_dir / "labels.tif"
    _write_label_raster(raster_path)

    gdf = _make_gdf([
        # Pixel centers at x=100005/100015, y=199995/199985 -- comfortably
        # inside this box, so exact edge/reprojection precision can't flip
        # which pixels are captured.
        ("SZ_PURE_VEG", box(100000, 199980, 100020, 200000)),
        # Covers the full raster: 4 vegetation + 4 built_up + 8 nodata.
        ("SZ_MIXED", box(100000, 199960, 100040, 200000)),
        # Nowhere near the raster.
        ("SZ_OFFRASTER", box(200000, 200000, 200010, 200010)),
    ])

    df = zonal_class_fractions(raster_path, gdf, id_property="SUBZONE_N", class_names=CLASS_NAMES)
    by_id = df.set_index("subzone_id")

    assert by_id.loc["SZ_PURE_VEG", "n_valid_pixels"] == 4
    assert np.isclose(by_id.loc["SZ_PURE_VEG", "fraction_vegetation"], 1.0)
    assert np.isclose(by_id.loc["SZ_PURE_VEG", "fraction_built_up"], 0.0)
    print("PASS: pure-vegetation subzone -> fraction_vegetation=1.0")

    assert by_id.loc["SZ_MIXED", "n_valid_pixels"] == 8, "nodata pixels must be excluded from n_valid_pixels"
    assert np.isclose(by_id.loc["SZ_MIXED", "fraction_vegetation"], 0.5)
    assert np.isclose(by_id.loc["SZ_MIXED", "fraction_built_up"], 0.5)
    assert np.isclose(by_id.loc["SZ_MIXED", "fraction_bare"], 0.0)
    print("PASS: mixed subzone -> fractions computed over valid pixels only, nodata excluded from denominator")

    assert by_id.loc["SZ_OFFRASTER", "n_valid_pixels"] == 0
    assert np.isnan(by_id.loc["SZ_OFFRASTER", "fraction_vegetation"])
    print("PASS: off-raster subzone -> n_valid_pixels=0, fractions NaN (not a silent 0.0)")


def test_zonal_class_fractions_reprojects_mismatched_crs(tmp_dir):
    """Production callers (pillars.py) pass subzones in EPSG:4326 while the
    ensemble raster is EPSG:32648 -- this exercises that exact reprojection
    path and checks it recovers the same result as passing the raster's
    native CRS directly."""
    raster_path = tmp_dir / "labels_reproj.tif"
    _write_label_raster(raster_path)

    gdf_native = _make_gdf([("SZ_PURE_VEG", box(100000, 199980, 100020, 200000))], crs=CRS)
    gdf_4326 = gdf_native.to_crs("EPSG:4326")
    assert str(gdf_4326.crs) != CRS

    df = zonal_class_fractions(raster_path, gdf_4326, id_property="SUBZONE_N", class_names=CLASS_NAMES)
    assert df.iloc[0]["n_valid_pixels"] == 4
    assert np.isclose(df.iloc[0]["fraction_vegetation"], 1.0)
    print("PASS: mismatched input CRS is reprojected onto the raster's CRS before masking")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        test_zonal_class_fractions_pure_and_mixed_and_offraster(tmp_dir)
        test_zonal_class_fractions_reprojects_mismatched_crs(tmp_dir)
    print("\nAll zonal.py checks passed.")


if __name__ == "__main__":
    main()
