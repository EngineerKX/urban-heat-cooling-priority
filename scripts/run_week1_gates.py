#!/usr/bin/env python
"""Week-1 verification gates (G1-G5): GEE composite pull, NEA API depth,
end-to-end smoke test, downscaling sanity check, labeling agreement.
Replaces urban_heat_sg_week1_gates.ipynb — re-implemented on top of the
consolidated src.ingest / src.downscaling helpers instead of re-pasting the
raw GEE calls, but keeps the original gate structure and verdict-dict
pattern.

Usage: python scripts/run_week1_gates.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import numpy as np
import pandas as pd

from config.settings import (
    DIAGNOSTICS_DIR,
    DRY_SEASON_MONTHS,
    INTERIM_DIR,
    LANDSAT_CLOUD_COVER_MAX,
    RANDOM_SEED,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    SUBZONE_ID_PROPERTY,
    YEARS,
)
from src.ingest.gee import (
    coverage_fraction,
    download_small_image,
    fetch_landsat_lst_collection,
    fetch_sentinel2_collection,
    init_ee,
)
from src.ingest.nea import fetch_air_temp, fetch_station_metadata
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from validation.input_validation.labeling_agreement import agreement_report, generate_shared_points

TILE_DIR = INTERIM_DIR / "week1_gates_tile"
gate_results = {}


def gate_g1():
    print("\n" + "=" * 70 + "\nG1 — GEE composite pull + coverage\n" + "=" * 70)
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    lst = fetch_landsat_lst_collection(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX).select("LST_C").median().clip(sg_bbox)
    s2 = fetch_sentinel2_collection(sg_bbox, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX).select("B4").median().clip(sg_bbox)

    lst_cov = coverage_fraction(lst, "LST_C", sg_bbox, 30)
    s2_cov = coverage_fraction(s2, "B4", sg_bbox, 10)
    ok = lst_cov >= 0.90 and s2_cov >= 0.90
    print(f"Landsat LST coverage: {lst_cov * 100:.1f}% | Sentinel-2 coverage: {s2_cov * 100:.1f}%")
    gate_results["G1"] = {
        "status": "PASS" if ok else "FAIL",
        "checks": {"Landsat coverage >= 90%": lst_cov >= 0.90, "Sentinel-2 coverage >= 90%": s2_cov >= 0.90},
    }
    print(f"{'✅ G1 PASS' if ok else '⚠️  G1 FAIL — widen season window / cloud thresholds'}")


def gate_g2(api_key: str = ""):
    print("\n" + "=" * 70 + "\nG2 — NEA API depth check\n" + "=" * 70)
    try:
        stations_df = fetch_station_metadata(api_key=api_key)
    except Exception as e:
        print(f"⚠️  {e}")
        gate_results["G2"] = {"status": "FAIL", "checks": {"API reachable and authenticated": False}}
        return

    import time
    from datetime import datetime, timedelta

    n_days_ok = 0
    for i in range(7):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        r = fetch_air_temp(date_str=d, api_key=api_key)
        if r.status_code == 200:
            n_days_ok += 1
        time.sleep(0.3)

    checks = {
        "Station count usable (>=10)": len(stations_df) >= 10,
        "Historical depth: most of last 7 days retrievable (>=5/7)": n_days_ok >= 5,
    }
    ok = all(checks.values())
    gate_results["G2"] = {"status": "PASS" if ok else "FAIL", "checks": checks}
    print(f"Stations: {len(stations_df)} | Days retrievable: {n_days_ok}/7")
    print(f"{'✅ G2 PASS' if ok else '⚠️  G2 FAIL'}")


def gate_g3():
    """Tiny composite -> toy downscale -> toy land cover -> subzone join -> dummy score."""
    print("\n" + "=" * 70 + "\nG3 — End-to-end smoke test\n" + "=" * 70)
    from rasterio.features import geometry_mask
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import box
    from sklearn.linear_model import LinearRegression

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    tile_aoi = ee.Geometry.Rectangle([103.925, 1.345, 103.955, 1.375])  # ~3km x 3km, Tampines

    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    landsat_composite = fetch_landsat_lst_collection(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX).median().clip(sg_bbox)
    s2_composite = fetch_sentinel2_collection(sg_bbox, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX).median().clip(sg_bbox)

    landsat_tile_path = TILE_DIR / "landsat_tile.tif"
    s2_tile_path = TILE_DIR / "s2_tile.tif"
    download_small_image(
        landsat_composite, region=tile_aoi, scale=30, crs=S2_UTM_CRS, out_path=landsat_tile_path,
        bands=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "LST_C"],
    )
    download_small_image(
        s2_composite, region=tile_aoi, scale=10, crs=S2_UTM_CRS, out_path=s2_tile_path,
        bands=["B2", "B3", "B4", "B8", "B11"],
    )
    landsat_tile_path, s2_tile_path = str(landsat_tile_path), str(s2_tile_path)

    import rasterio
    with rasterio.open(landsat_tile_path) as src:
        landsat_arr, landsat_crs, landsat_transform, landsat_shape, landsat_bounds = src.read(), src.crs, src.transform, src.shape, src.bounds
    with rasterio.open(s2_tile_path) as src:
        s2_arr, s2_crs, s2_transform = src.read(), src.crs, src.transform

    blue, green, red, nir, swir1 = s2_arr[0], s2_arr[1], s2_arr[2], s2_arr[3], s2_arr[4]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndbi = (swir1 - nir) / (swir1 + nir + eps)
    ndwi = (green - nir) / (green + nir + eps)

    def resample_to_match(src_arr):
        dst = np.zeros(landsat_shape, dtype=np.float32)
        reproject(source=src_arr, destination=dst, src_transform=s2_transform, src_crs=s2_crs,
                  dst_transform=landsat_transform, dst_crs=landsat_crs, resampling=Resampling.average)
        return dst

    ndvi_30, ndbi_30, ndwi_30 = resample_to_match(ndvi), resample_to_match(ndbi), resample_to_match(ndwi)
    lst_30m = landsat_arr[4]
    valid = (lst_30m > 0) & np.isfinite(ndvi_30) & np.isfinite(ndbi_30) & np.isfinite(ndwi_30)
    n_valid = valid.sum()
    print(f"Valid 30m training pixels: {n_valid}")

    downscale_ok = n_valid >= 10
    if downscale_ok:
        X_30 = np.stack([ndvi_30[valid], ndbi_30[valid], ndwi_30[valid]], axis=1)
        LinearRegression().fit(X_30, lst_30m[valid])

    landcover_toy = np.full(ndvi.shape, -1, dtype=np.int8)
    landcover_toy[ndwi > 0.1] = 0
    landcover_toy[(ndwi <= 0.1) & (ndvi > 0.35)] = 1
    landcover_toy[(ndwi <= 0.1) & (ndvi <= 0.35) & (ndbi > 0.0)] = 2
    landcover_toy[(ndwi <= 0.1) & (ndvi <= 0.35) & (ndbi <= 0.0)] = 3

    subzones_gdf = as_geodataframe(fetch_subzones_geojson())
    if subzones_gdf.crs != landsat_crs:
        subzones_gdf = subzones_gdf.to_crs(landsat_crs)
    tile_bounds_geom = box(*landsat_bounds)
    subzones_in_tile = subzones_gdf[subzones_gdf.intersects(tile_bounds_geom)].copy()

    records = []
    n_dropped = 0
    veg_mask_10m = (landcover_toy == 1).astype(np.float32)
    veg_frac_30 = resample_to_match(veg_mask_10m)
    for _, row in subzones_in_tile.iterrows():
        try:
            mask = geometry_mask([row.geometry.__geo_interface__], out_shape=landsat_shape, transform=landsat_transform, invert=True)
        except Exception:
            n_dropped += 1
            continue
        if mask.sum() == 0:
            n_dropped += 1
            continue
        lst_vals = lst_30m[mask]
        lst_vals = lst_vals[lst_vals > 0]
        records.append({
            "subzone": row[SUBZONE_ID_PROPERTY],
            "mean_lst_c": float(np.nanmean(lst_vals)) if len(lst_vals) else np.nan,
            "greenery_fraction": float(np.nanmean(veg_frac_30[mask])),
        })

    df = pd.DataFrame(records)
    if len(df):
        np.random.seed(RANDOM_SEED)
        df["sensitivity_dummy"] = np.random.uniform(0.3, 1.0, size=len(df))

        def normalize(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0

        df["dummy_score"] = (
            normalize(df["mean_lst_c"]) + normalize(df["sensitivity_dummy"]) + normalize(1 - df["greenery_fraction"])
        ) / 3

    checks = {
        "Subzones intersect tile": len(subzones_in_tile) > 0,
        "Zonal join retained subzones": len(df) > 0,
        "No majority silent-drop": n_dropped < len(subzones_in_tile) * 0.5 if len(subzones_in_tile) else False,
        "Downscaling wiring OK (n_valid>=10)": downscale_ok,
        "Score computed for all retained subzones": df["dummy_score"].notna().all() if len(df) else False,
    }
    ok = all(checks.values())
    gate_results["G3"] = {"status": "PASS" if ok else "FAIL", "checks": checks}
    print(f"{'✅ G3 PASS' if ok else '⚠️  G3 FAIL'}")


def gate_g4():
    print("\n" + "=" * 70 + "\nG4 — Downscaling sanity check\n" + "=" * 70)
    landsat_tile_path, s2_tile_path = str(TILE_DIR / "landsat_tile.tif"), str(TILE_DIR / "s2_tile.tif")
    if not Path(landsat_tile_path).exists():
        print("⚠️  G4 depends on G3's tile exports — run gate_g3() first. Skipping.")
        gate_results["G4"] = {"status": "FAIL", "checks": {"G3 tiles available": False}}
        return

    import rasterio
    from rasterio.warp import Resampling, reproject
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    with rasterio.open(landsat_tile_path) as src:
        landsat_arr, landsat_crs, landsat_transform, landsat_shape = src.read(), src.crs, src.transform, src.shape
    with rasterio.open(s2_tile_path) as src:
        s2_arr, s2_crs, s2_transform = src.read(), src.crs, src.transform

    blue, green, red, nir, swir1 = s2_arr[0], s2_arr[1], s2_arr[2], s2_arr[3], s2_arr[4]
    eps = 1e-6
    ndvi, ndbi, ndwi = (nir - red) / (nir + red + eps), (swir1 - nir) / (swir1 + nir + eps), (green - nir) / (green + nir + eps)

    def resample_to_match(src_arr):
        dst = np.zeros(landsat_shape, dtype=np.float32)
        reproject(source=src_arr, destination=dst, src_transform=s2_transform, src_crs=s2_crs,
                  dst_transform=landsat_transform, dst_crs=landsat_crs, resampling=Resampling.average)
        return dst

    ndvi_30, ndbi_30, ndwi_30 = resample_to_match(ndvi), resample_to_match(ndbi), resample_to_match(ndwi)
    lst_30m = landsat_arr[4]
    valid = (lst_30m > 0) & np.isfinite(ndvi_30) & np.isfinite(ndbi_30) & np.isfinite(ndwi_30)
    X_all, y_all = np.stack([ndvi_30[valid], ndbi_30[valid], ndwi_30[valid]], axis=1), lst_30m[valid]

    if len(y_all) < 30:
        print("⚠️  Very few valid pixels — result will be noisy.")

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=RANDOM_SEED)
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_r2 = r2_score(y_test, baseline_pred)

    lr = LinearRegression().fit(X_train, y_train)
    lr_r2 = r2_score(y_test, lr.predict(X_test))
    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=RANDOM_SEED).fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    print(f"Baseline R²={baseline_r2:.3f} | Linear R²={lr_r2:.3f} | RF R²={rf_r2:.3f}")
    checks = {
        "Enough valid pixels (n>=30)": len(y_all) >= 30,
        "Linear Regression beats baseline": lr_r2 > baseline_r2,
        "Random Forest beats baseline": rf_r2 > baseline_r2,
        "Best model R² exceeds feasibility bar (0.3)": max(lr_r2, rf_r2) > 0.3,
    }
    ok = all(checks.values())
    gate_results["G4"] = {"status": "PASS" if ok else "FAIL", "checks": checks, "best_r2": max(lr_r2, rf_r2)}
    print(f"{'✅ G4 PASS' if ok else '⚠️  G4 FAIL/MARGINAL'}")


def gate_g5(member_a_labels: list, member_b_labels: list):
    """Both members must label the SAME shared points independently before
    calling this — see validation/input_validation/labeling_agreement.py."""
    print("\n" + "=" * 70 + "\nG5 — Labeling agreement\n" + "=" * 70)
    class_names = {0: "water", 1: "vegetation", 2: "built-up", 3: "bare/other"}
    sg_boundary = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Singapore")).geometry()
    points_df = generate_shared_points(sg_boundary, n_points=len(member_a_labels), seed=RANDOM_SEED)

    report = agreement_report(member_a_labels, member_b_labels, list(points_df["point_id"]), class_names)
    ok = report["passed"]
    gate_results["G5"] = {"status": "PASS" if ok else "FAIL", "checks": {"kappa >= 0.60": ok}, "kappa": report["kappa"]}
    print(f"{'✅ G5 PASS' if ok else '⚠️  G5 FAIL/MARGINAL'}")


def compile_summary():
    print("\n" + "=" * 70 + "\nWEEK-1 VERIFICATION GATES — SUMMARY\n" + "=" * 70)
    rows = []
    for g in ["G1", "G2", "G3", "G4", "G5"]:
        if g not in gate_results:
            rows.append({"gate": g, "status": "NOT RUN"})
            continue
        rows.append({"gate": g, "status": gate_results[g]["status"]})
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    out_path = DIAGNOSTICS_DIR / "week1_gate_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    init_ee()
    gate_g1()
    gate_g2()
    gate_g3()
    gate_g4()
    # G5 needs two real, independently-collected label lists — placeholder
    # example values below, replace with your own before trusting the result.
    gate_g5(
        member_a_labels=[1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2],
        member_b_labels=[1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3, 2, 2, 1, 3, 1, 2, 3, 2],
    )
    compile_summary()
