#!/usr/bin/env python
"""Read-only diagnostic for S5 (C2): recomputes the XGBoost test metrics,
and -- if the CNN model exists -- runs both counterfactual paths on a
handful of real high-priority subzones and checks they agree on
direction. The saved examples double as a reference/fallback example
library for app/pages/4_Counterfactual_Greening.py (which now also runs
the CNN live via PyTorch CPU inference -- these precomputed examples are
kept as a known-good reference, not a workaround for a broken model
anymore). Same "build vs. diagnose" split as
scripts/diagnose_heat_variants.py; deliberately does NOT import from
scripts/run_counterfactual.py (this repo's scripts/ has no __init__.py --
scripts import from src/config/validation only, not from each other).

Usage: python scripts/diagnose_heat_model.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from config.settings import (
    BUCKET_VEGETATION,
    CNN_MODEL_SAVE_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_UTM_CRS,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    UNET_PATCH_SIZE,
)
from src.heat_model.tabular import (
    XGB_FEATURE_COLUMNS,
    XGB_TARGET_COLUMN,
    build_xgb_training_table,
    fit_ndvi_vegetation_slope,
    predict_counterfactual_subzone,
)
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
XGB_MODEL_PATH = PROCESSED_DIR / "heat_model" / "xgb_model.pkl"
CANNED_EXAMPLES_OUT_PATH = PROCESSED_DIR / "heat_model" / "canned_counterfactual_examples.json"

INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
MIXER_JSON_PATH = INFERENCE_PATCH_DIR / "unet_inference.json"
LST_BICUBIC10_PATH = INTERIM_DIR / "lst_bicubic10_full.tif"

# The top 3 highest-priority (production PCA score) subzones as of the
# 2026-08-04 run -- not hand-picked for any narrative, just representative
# real hotspots to demo the counterfactual mechanism on.
DEMO_SUBZONES = ["TUAS NORTH", "GUL CIRCLE", "CHIN BEE"]
DEMO_DELTA_VEGETATION = 0.15
DEMO_RADIUS_M = 50.0


def _subzone_centroid_lonlat(subzones_gdf: gpd.GeoDataFrame, subzone_id: str):
    """Centroid computed in a projected CRS (not raw lat/lon degrees, which
    distorts area/centroid math), then converted back to WGS84 lon/lat."""
    match = subzones_gdf[subzones_gdf[SUBZONE_ID_PROPERTY] == subzone_id]
    if match.empty:
        return None, None
    centroid_utm = match.to_crs(S2_UTM_CRS).geometry.centroid.iloc[0]
    centroid_lonlat = gpd.GeoSeries([centroid_utm], crs=S2_UTM_CRS).to_crs("EPSG:4326").iloc[0]
    return centroid_lonlat.x, centroid_lonlat.y


def main():
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"{XGB_MODEL_PATH} not found — run scripts/train_heat_model_xgboost.py first.")

    df = build_xgb_training_table(HEAT_CSV_PATH, HOTSPOT_CLUSTERS_CSV_PATH, SENSITIVITY_CSV_PATH)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    ndvi_slope = fit_ndvi_vegetation_slope(df)

    print("--- XGBoost (recomputed on full table, not held-out — see training run's own printed test RMSE/R²) ---")
    all_pred = xgb_model.predict(df[XGB_FEATURE_COLUMNS])
    full_rmse = float(np.sqrt(np.mean((all_pred - df[XGB_TARGET_COLUMN]) ** 2)))
    print(f"Full-table RMSE (in-sample + held-out mixed, informal): {full_rmse:.3f}°C "
          f"(see scripts/train_heat_model_xgboost.py's own run for the honest held-out test RMSE/R²)")

    cnn_available = CNN_MODEL_SAVE_PATH.exists() and MIXER_JSON_PATH.exists() and LST_BICUBIC10_PATH.exists()
    print(f"\nCNN model available: {cnn_available}"
          + ("" if cnn_available else " — train it via notebooks/colab_training/train_heat_cnn.ipynb, then `python scripts/pull_models.py --model cnn`."))

    subzones_gdf = as_geodataframe(fetch_subzones_geojson())

    if cnn_available:
        from src.heat_model.cnn_data import build_local_feature_target_patches
        from src.heat_model.cnn_infer import load_cnn_regressor, locate_patch_and_pixel
        from src.heat_model.counterfactual import class_mean_feature_vectors, rescale_subzone_delta, run_patch_counterfactual

        print("\nBuilding local feature/target patches for the demo examples ...")
        X, _y, valid_mask = build_local_feature_target_patches(
            INFERENCE_PATCH_DIR, MIXER_JSON_PATH, ENSEMBLE_RASTER_PATH, LST_BICUBIC10_PATH,
        )
        class_means = class_mean_feature_vectors(X, valid_mask=valid_mask)
        cnn_model = load_cnn_regressor(CNN_MODEL_SAVE_PATH)
        subzones_gdf_utm = subzones_gdf.to_crs(S2_UTM_CRS)

    examples = []
    for subzone_id in DEMO_SUBZONES:
        row_df = df[df["subzone_id"] == subzone_id]
        if row_df.empty:
            print(f"\n⚠️  '{subzone_id}' not found — skipping.")
            continue

        xgb_result = predict_counterfactual_subzone(xgb_model, row_df, DEMO_DELTA_VEGETATION, ndvi_slope)
        example = {"subzone_id": subzone_id, "delta_fraction_vegetation": DEMO_DELTA_VEGETATION, "xgboost": xgb_result}
        print(f"\n--- {subzone_id} ---")
        print(f"  XGBoost: {xgb_result['original_lst']:.2f}°C -> {xgb_result['counterfactual_lst']:.2f}°C "
              f"(Δ={xgb_result['delta_lst']:+.3f}°C) for +{DEMO_DELTA_VEGETATION:.2f} vegetation fraction")

        if cnn_available:
            lon, lat = _subzone_centroid_lonlat(subzones_gdf, subzone_id)
            patch_idx, local_row, local_col = locate_patch_and_pixel(lon, lat, MIXER_JSON_PATH, UNET_PATCH_SIZE)
            if 0 <= patch_idx < X.shape[0]:
                radius_px = DEMO_RADIUS_M / TARGET_SCALE_M
                yy, xx = np.mgrid[0:UNET_PATCH_SIZE, 0:UNET_PATCH_SIZE]
                edit_mask = ((yy - local_row) ** 2 + (xx - local_col) ** 2) <= radius_px ** 2

                cnn_result = run_patch_counterfactual(cnn_model, X[patch_idx], edit_mask, BUCKET_VEGETATION, class_means)
                edit_area_m2 = cnn_result["n_edited_pixels"] * (TARGET_SCALE_M ** 2)
                match = subzones_gdf_utm[subzones_gdf_utm[SUBZONE_ID_PROPERTY] == subzone_id]
                subzone_area_m2 = float(match.geometry.area.iloc[0]) if not match.empty else None
                rescaled = (
                    rescale_subzone_delta(cnn_result["mean_delta_lst_in_edit_area"], edit_area_m2, subzone_area_m2)
                    if subzone_area_m2 else None
                )

                same_sign = np.sign(xgb_result["delta_lst"]) == np.sign(cnn_result["mean_delta_lst_in_edit_area"])
                print(f"  CNN (centroid, r={DEMO_RADIUS_M:.0f}m): mean ΔLST in edit area = "
                      f"{cnn_result['mean_delta_lst_in_edit_area']:+.3f}°C "
                      f"({'✅ same direction as XGBoost' if same_sign else '⚠️  OPPOSITE direction from XGBoost'})")

                example["cnn"] = {
                    "lon": lon, "lat": lat, "radius_m": DEMO_RADIUS_M,
                    "n_edited_pixels": cnn_result["n_edited_pixels"],
                    "mean_delta_lst_in_edit_area": cnn_result["mean_delta_lst_in_edit_area"],
                    "rescaled_subzone_equivalent_delta": rescaled,
                    "agrees_with_xgboost_direction": bool(same_sign),
                }
            else:
                print(f"  ⚠️  Centroid of {subzone_id} fell outside the CNN patch grid — skipping patch-level example.")

        examples.append(example)

    CANNED_EXAMPLES_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANNED_EXAMPLES_OUT_PATH, "w") as f:
        json.dump(examples, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nSaved: {CANNED_EXAMPLES_OUT_PATH}")


if __name__ == "__main__":
    main()
