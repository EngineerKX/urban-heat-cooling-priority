"""S7 counterfactual greening page (C2): "what if this area were greener?"
Two paths, matching src/heat_model/counterfactual.py's module docstring --
they cross-check each other rather than being one combined model:

- Live XGBoost subzone-level slider -- runs natively on Windows (no GPU
  needed), fully interactive.
- Live CNN patch-level slider -- since the PyTorch migration, CPU inference
  works natively on Windows (the old blocker, native Windows TensorFlow
  being broken, no longer applies), so this now runs live too instead of
  showing precomputed canned examples. The one-time patch-building cost is
  cached via @st.cache_resource so it only pays once per session, not per
  slider move.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import numpy as np
import pickle
import streamlit as st

from config.settings import (
    CNN_MODEL_SAVE_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_UTM_CRS,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    UNET_PATCH_SIZE,
)
from src.heat_model.tabular import build_xgb_training_table, fit_ndvi_vegetation_slope, predict_counterfactual_subzone
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.ingest.worldcover import BUCKET_NAMES

st.set_page_config(page_title="Counterfactual Greening — Urban Heat & Cooling Priority", page_icon="🌱", layout="wide")
st.title("🌱 Counterfactual greening: \"what if this area were greener?\"")

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
XGB_MODEL_PATH = PROCESSED_DIR / "heat_model" / "xgb_model.pkl"

CNN_INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
CNN_MIXER_JSON_PATH = CNN_INFERENCE_PATCH_DIR / "unet_inference.json"
ENSEMBLE_RASTER_PATH = PROCESSED_DIR / "landcover" / "ensemble_landcover.tif"
LST_BICUBIC10_PATH = INTERIM_DIR / "lst_bicubic10_full.tif"

st.subheader("Subzone-level (XGBoost) — live")

if not XGB_MODEL_PATH.exists():
    st.warning("XGBoost heat model not trained yet — run `python scripts/train_heat_model_xgboost.py` first.")
else:
    @st.cache_data
    def load_xgb_assets():
        df = build_xgb_training_table(HEAT_CSV_PATH, HOTSPOT_CLUSTERS_CSV_PATH, SENSITIVITY_CSV_PATH)
        ndvi_slope = fit_ndvi_vegetation_slope(df)
        return df, ndvi_slope

    @st.cache_resource
    def load_xgb_model():
        with open(XGB_MODEL_PATH, "rb") as f:
            return pickle.load(f)

    xgb_df, ndvi_slope = load_xgb_assets()
    xgb_model = load_xgb_model()

    subzone_ids = sorted(xgb_df["subzone_id"].astype(str).unique())
    preselected = st.session_state.get("selected_subzone_id")
    default_idx = subzone_ids.index(preselected) if preselected in subzone_ids else 0
    chosen = st.selectbox("Subzone", subzone_ids, index=default_idx, key="cf_subzone_select")

    row_df = xgb_df[xgb_df["subzone_id"] == chosen]
    current_veg = float(row_df["fraction_vegetation"].iloc[0])
    st.caption(f"Current vegetation fraction: {current_veg:.2f}")

    max_delta = max(0.0, 1.0 - current_veg)
    delta = st.slider(
        "Increase vegetation fraction by", min_value=0.0, max_value=float(max_delta),
        value=float(min(0.15, max_delta)), step=0.01,
    )

    result = predict_counterfactual_subzone(xgb_model, row_df, delta, ndvi_slope)

    m1, m2, m3 = st.columns(3)
    m1.metric("Current LST", f"{result['original_lst']:.2f}°C")
    m2.metric(
        "Predicted LST after greening", f"{result['counterfactual_lst']:.2f}°C", delta=f"{result['delta_lst']:+.2f}°C",
        delta_color="inverse",  # cooling (a negative delta) is the desired outcome here, not a regression
    )
    m3.metric("Actual vegetation delta applied", f"+{result['actual_delta_vegetation']:.3f}")
    if delta > 0.15:
        st.caption("⚠️ Large deltas can push tree-model predictions outside the training density and produce "
                   "non-monotonic responses — treat deltas beyond ~0.15 as directionally indicative, not precise.")

st.divider()
st.subheader("Patch-level (CNN) — live")

cnn_available = (
    CNN_MODEL_SAVE_PATH.exists() and CNN_MIXER_JSON_PATH.exists()
    and ENSEMBLE_RASTER_PATH.exists() and LST_BICUBIC10_PATH.exists()
)

if not cnn_available:
    st.info(
        "CNN heat model or its input rasters aren't available yet — train it via "
        "`notebooks/colab_training/train_heat_cnn.ipynb`, then "
        "`python scripts/pull_models.py --model cnn`."
    )
elif not XGB_MODEL_PATH.exists():
    st.info("Train the XGBoost model above first — the CNN demo reuses the subzone you pick there.")
else:
    from src.heat_model.cnn_data import build_local_feature_target_patches
    from src.heat_model.cnn_infer import load_cnn_regressor, locate_patch_and_pixel
    from src.heat_model.counterfactual import class_mean_feature_vectors, rescale_subzone_delta, run_patch_counterfactual

    @st.cache_resource
    def load_cnn_assets():
        """Reads all ~1000 patches + reprojects the ensemble/LST rasters onto
        each -- a real one-time cost (tens of seconds), so cached per session
        rather than re-run on every slider move."""
        X, _y, valid_mask = build_local_feature_target_patches(
            CNN_INFERENCE_PATCH_DIR, CNN_MIXER_JSON_PATH, ENSEMBLE_RASTER_PATH, LST_BICUBIC10_PATH,
        )
        class_means = class_mean_feature_vectors(X, valid_mask=valid_mask)
        model = load_cnn_regressor(CNN_MODEL_SAVE_PATH)
        return X, class_means, model

    @st.cache_data
    def load_subzones_utm():
        return as_geodataframe(fetch_subzones_geojson()).to_crs(S2_UTM_CRS)

    X, class_means, cnn_model = load_cnn_assets()
    subzones_utm = load_subzones_utm()

    match = subzones_utm[subzones_utm[SUBZONE_ID_PROPERTY] == chosen]
    if match.empty:
        st.warning(f"'{chosen}' not found in the subzones geodataframe — can't locate its centroid.")
    else:
        centroid_utm = match.geometry.centroid.iloc[0]
        centroid_lonlat = gpd.GeoSeries([centroid_utm], crs=S2_UTM_CRS).to_crs("EPSG:4326").iloc[0]
        lon, lat = centroid_lonlat.x, centroid_lonlat.y
        subzone_area_m2 = float(match.geometry.area.iloc[0])

        patch_idx, local_row, local_col = locate_patch_and_pixel(lon, lat, CNN_MIXER_JSON_PATH, UNET_PATCH_SIZE)
        if not (0 <= patch_idx < X.shape[0]):
            st.warning(f"'{chosen}'s centroid falls outside the CNN patch grid — no live example available for it.")
        else:
            radius_m = st.slider("Edit radius around subzone centroid (m)", 10.0, 200.0, 50.0, step=10.0)
            target_class_name = st.selectbox("Convert edited area to", list(BUCKET_NAMES.values()), index=0)
            bucket_name_to_id = {v: k for k, v in BUCKET_NAMES.items()}
            target_class = bucket_name_to_id[target_class_name]

            radius_px = radius_m / TARGET_SCALE_M
            yy, xx = np.mgrid[0:UNET_PATCH_SIZE, 0:UNET_PATCH_SIZE]
            edit_mask = ((yy - local_row) ** 2 + (xx - local_col) ** 2) <= radius_px ** 2

            cnn_result = run_patch_counterfactual(cnn_model, X[patch_idx], edit_mask, target_class, class_means)

            c1, c2 = st.columns(2)
            c1.metric(
                "Mean ΔLST in edited area", f"{cnn_result['mean_delta_lst_in_edit_area']:+.3f}°C",
                delta=f"{cnn_result['mean_delta_lst_in_edit_area']:+.3f}°C",
                delta_color="inverse",  # cooling (a negative delta) is the desired outcome here, not a regression
            )
            edit_area_m2 = cnn_result["n_edited_pixels"] * (TARGET_SCALE_M ** 2)
            c2.metric("Edited pixels", f"{cnn_result['n_edited_pixels']}", f"~{edit_area_m2:,.0f} m²")

            if subzone_area_m2 > 0 and edit_area_m2 > 0:
                rescaled = rescale_subzone_delta(cnn_result["mean_delta_lst_in_edit_area"], edit_area_m2, subzone_area_m2)
                st.caption(
                    f"Naive area-weighted subzone-wide equivalent: {rescaled:+.4f}°C "
                    f"(edited area is {edit_area_m2 / subzone_area_m2 * 100:.3f}% of {chosen}'s area). "
                    "Cross-check against the XGBoost result above — same sign expected, not the same magnitude "
                    "(these operate at very different spatial scales)."
                )

if st.button("← Back to map"):
    st.switch_page("pages/2_Island_Map.py")
