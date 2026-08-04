#!/usr/bin/env python
"""CLI counterfactual query tool for S5 (C2): "what if this area were
greener?" Two independent counterfactual paths that cross-check each other
(see src/heat_model/counterfactual.py's module docstring) rather than one
combined model:

- Subzone-level (XGBoost): --subzone-id + --delta-vegetation. Always runs
  if the XGBoost model exists.
- Patch-level (CNN): additionally pass --lon/--lat/--radius-m to edit a
  real 10m patch around that point. Requires the CNN model
  (scripts/train_heat_model_cnn.py, WSL2-only) -- skipped with a printed
  note if it doesn't exist, rather than failing the whole query.

No --force flag -- this is a query tool, not a build-once cache (same
category as scripts/run_week1_gates.py / scripts/evaluate_landcover_classifiers.py).

Usage:
  python scripts/run_counterfactual.py --subzone-id "TUAS NORTH" --delta-vegetation 0.2
  python scripts/run_counterfactual.py --subzone-id "TUAS NORTH" --delta-vegetation 0.2 \
      --lon 103.63 --lat 1.327 --radius-m 100 --target-class vegetation
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config.settings import (
    CNN_MODEL_SAVE_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_UTM_CRS,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
)
from src.heat_model.tabular import (
    XGB_TARGET_COLUMN,
    build_xgb_training_table,
    fit_ndvi_vegetation_slope,
    predict_counterfactual_subzone,
)
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.ingest.worldcover import BUCKET_NAMES

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
XGB_MODEL_PATH = PROCESSED_DIR / "heat_model" / "xgb_model.pkl"

BUCKET_NAME_TO_ID = {v: k for k, v in BUCKET_NAMES.items()}


def run_xgb_counterfactual(subzone_id: str, delta_fraction_vegetation: float):
    if not XGB_MODEL_PATH.exists():
        print(f"⚠️  {XGB_MODEL_PATH} not found — run scripts/train_heat_model_xgboost.py first. Skipping subzone-level counterfactual.")
        return None

    df = build_xgb_training_table(HEAT_CSV_PATH, HOTSPOT_CLUSTERS_CSV_PATH, SENSITIVITY_CSV_PATH)
    row_df = df[df["subzone_id"] == subzone_id]
    if row_df.empty:
        print(f"⚠️  subzone_id '{subzone_id}' not found in the XGBoost training table.")
        return None

    with open(XGB_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    ndvi_slope = fit_ndvi_vegetation_slope(df)

    result = predict_counterfactual_subzone(model, row_df, delta_fraction_vegetation, ndvi_slope)
    print(f"\n--- XGBoost subzone-level counterfactual: {subzone_id} ---")
    print(f"  Current {XGB_TARGET_COLUMN}: {result['original_lst']:.2f}°C")
    print(f"  Requested vegetation-fraction delta: +{delta_fraction_vegetation:.2f} "
          f"(actual applied: +{result['actual_delta_vegetation']:.3f}, clamped at [0,1])")
    print(f"  Predicted {XGB_TARGET_COLUMN} after greening: {result['counterfactual_lst']:.2f}°C "
          f"(Δ={result['delta_lst']:+.3f}°C)")
    if delta_fraction_vegetation > 0.15:
        print("  ⚠️  Large deltas can push tree-model predictions outside the training density and produce "
              "non-monotonic responses -- treat deltas beyond ~0.15 as directionally indicative, not precise.")
    return result


def run_cnn_counterfactual(subzone_id: str, lon: float, lat: float, radius_m: float, target_class_name: str):
    import tensorflow as tf

    from config.settings import UNET_PATCH_SIZE
    from src.heat_model.cnn_patches import build_local_feature_target_patches, locate_patch_and_pixel
    from src.heat_model.counterfactual import class_mean_feature_vectors, rescale_subzone_delta, run_patch_counterfactual
    from src.landcover.ensemble import ENSEMBLE_RASTER_PATH

    if not CNN_MODEL_SAVE_PATH.exists():
        print(f"⚠️  {CNN_MODEL_SAVE_PATH} not found — run scripts/train_heat_model_cnn.py (WSL2) first. Skipping patch-level counterfactual.")
        return None

    inference_patch_dir = INTERIM_DIR / "unet_patches" / "inference"
    mixer_json_path = inference_patch_dir / "unet_inference.json"
    lst_bicubic10_path = INTERIM_DIR / "lst_bicubic10_full.tif"
    if not (mixer_json_path.exists() and lst_bicubic10_path.exists() and ENSEMBLE_RASTER_PATH.exists()):
        print("⚠️  Missing patch/raster inputs for the CNN counterfactual — skipping.")
        return None

    target_class = BUCKET_NAME_TO_ID.get(target_class_name)
    if target_class is None:
        print(f"⚠️  Unknown --target-class '{target_class_name}', expected one of {list(BUCKET_NAME_TO_ID)}. Skipping.")
        return None

    print("\nBuilding local feature/target patches (rasterio reprojection, no GEE call) ...")
    X, _y, valid_mask = build_local_feature_target_patches(
        inference_patch_dir, mixer_json_path, ENSEMBLE_RASTER_PATH, lst_bicubic10_path,
    )
    class_means = class_mean_feature_vectors(X, valid_mask=valid_mask)

    patch_idx, local_row, local_col = locate_patch_and_pixel(lon, lat, mixer_json_path, UNET_PATCH_SIZE)
    if not (0 <= patch_idx < X.shape[0]):
        print(f"⚠️  (lon={lon}, lat={lat}) falls outside the patch grid — skipping.")
        return None

    radius_px = radius_m / TARGET_SCALE_M
    yy, xx = np.mgrid[0:UNET_PATCH_SIZE, 0:UNET_PATCH_SIZE]
    edit_mask = ((yy - local_row) ** 2 + (xx - local_col) ** 2) <= radius_px ** 2

    model = tf.keras.models.load_model(CNN_MODEL_SAVE_PATH)
    result = run_patch_counterfactual(model, X[patch_idx], edit_mask, target_class, class_means)

    edit_area_m2 = result["n_edited_pixels"] * (TARGET_SCALE_M ** 2)
    subzones_gdf = as_geodataframe(fetch_subzones_geojson()).to_crs(S2_UTM_CRS)
    match = subzones_gdf[subzones_gdf[SUBZONE_ID_PROPERTY] == subzone_id]
    subzone_area_m2 = float(match.geometry.area.iloc[0]) if not match.empty else None

    print(f"\n--- CNN patch-level counterfactual: ({lon}, {lat}), radius={radius_m}m -> {target_class_name} ---")
    print(f"  Edited pixels: {result['n_edited_pixels']} (~{edit_area_m2:,.0f} m²)")
    print(f"  Mean ΔLST within edited area: {result['mean_delta_lst_in_edit_area']:+.3f}°C")

    if subzone_area_m2:
        rescaled = rescale_subzone_delta(result["mean_delta_lst_in_edit_area"], edit_area_m2, subzone_area_m2)
        print(f"  Naive area-weighted equivalent subzone-wide Δ: {rescaled:+.4f}°C "
              f"(edited area is {edit_area_m2 / subzone_area_m2 * 100:.3f}% of {subzone_id}'s area)")
        print("  Cross-check against the XGBoost subzone-level result above -- same sign expected, "
              "not the same magnitude (these operate at very different spatial scales).")
    else:
        print(f"  ⚠️  Could not find '{subzone_id}' in the subzones geodataframe — skipping area rescaling.")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subzone-id", required=True)
    parser.add_argument("--delta-vegetation", type=float, default=0.2,
                         help="Vegetation-fraction delta for the XGBoost subzone-level counterfactual (default 0.2).")
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--target-class", default="vegetation", choices=list(BUCKET_NAME_TO_ID))
    args = parser.parse_args()

    run_xgb_counterfactual(args.subzone_id, args.delta_vegetation)

    if args.lon is not None and args.lat is not None:
        run_cnn_counterfactual(args.subzone_id, args.lon, args.lat, args.radius_m, args.target_class)
    else:
        print("\n(Pass --lon/--lat/--radius-m to also run the CNN patch-level counterfactual.)")


if __name__ == "__main__":
    main()
