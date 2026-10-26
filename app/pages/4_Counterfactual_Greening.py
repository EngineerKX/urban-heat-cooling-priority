"""S7 counterfactual greening page (C2): "what if this area were greener?"
Two paths, matching src/heat_model/counterfactual.py's module docstring --
they cross-check each other rather than being one combined model:

- Live XGBoost subzone-level slider -- runs natively on Windows (no GPU
  needed), fully interactive. Shown with its response curve (cooling vs.
  added vegetation, swept from 0 to the maximum), so the curve's shape --
  including where it stops being monotonic -- is visible, not just stated.
- Live CNN patch-level edit -- since the PyTorch migration, CPU inference
  works natively on Windows (the old blocker, native Windows TensorFlow
  being broken, no longer applies), so this runs live too instead of
  showing precomputed canned examples. Shown as before / after / change
  maps of the patch around the edit. The one-time patch-building cost is
  cached via @st.cache_resource so it only pays once per session, not per
  slider move.

When both run and the CNN edit converts to vegetation, a verdict at the top
says whether the two models agree on the direction (cooling or not).

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import numpy as np
import pickle
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.settings import (
    LANDCOVER_RASTER_PATH,
    CNN_MODEL_SAVE_PATH,
    INTERIM_DIR,
    PROCESSED_DIR,
    S2_UTM_CRS,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    UNET_PATCH_SIZE,
)
from src.heat_model.tabular import build_xgb_training_table, fit_index_landcover_model, predict_counterfactual_subzone
from app.ui import LANDCOVER_LABELS, subzone_name
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.ingest.worldcover import BUCKET_NAMES

st.set_page_config(page_title="Counterfactual Greening — Urban Heat & Cooling Priority", page_icon="🌱", layout="wide")
st.title("🌱 Counterfactual greening: \"what if this area were greener?\"")
st.caption(
    "Two heat models answer the same question at different scales: XGBoost for a whole subzone, a CNN for the "
    "street blocks around its centre. They are a cross-check on each other, not one combined estimate."
)

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
XGB_MODEL_PATH = PROCESSED_DIR / "heat_model" / "xgb_model.pkl"

CNN_INFERENCE_PATCH_DIR = INTERIM_DIR / "unet_patches" / "inference"
CNN_MIXER_JSON_PATH = CNN_INFERENCE_PATCH_DIR / "unet_inference.json"
LST_BICUBIC10_PATH = INTERIM_DIR / "lst_bicubic10_full.tif"

RELIABLE_DELTA = 0.15  # beyond this, tree-model predictions leave the training density (see the warning below)
COOLING_COLOR, WARMING_COLOR = "#2166ac", "#b2182b"

if not XGB_MODEL_PATH.exists():
    st.warning("XGBoost heat model not trained yet — run `python scripts/train_heat_model_xgboost.py` first.")
    st.stop()


@st.cache_data
def load_xgb_assets():
    df = build_xgb_training_table(HEAT_CSV_PATH, HOTSPOT_CLUSTERS_CSV_PATH, SENSITIVITY_CSV_PATH)
    index_model = fit_index_landcover_model(df)
    return df, index_model


@st.cache_resource
def load_xgb_model():
    with open(XGB_MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Drawing the cooling curve…")
def xgb_response_curve(subzone_id: str, max_delta: float, _df, _model, _index_model, n_points: int = 25):
    """Predicted LST change for added vegetation from 0 to max_delta (one
    subzone; cached per subzone so the slider stays instant)."""
    row_df = _df[_df["subzone_id"] == subzone_id]
    deltas = np.linspace(0.0, max_delta, n_points)
    return deltas, np.array([predict_counterfactual_subzone(_model, row_df, d, _index_model)["delta_lst"] for d in deltas])


xgb_df, index_model = load_xgb_assets()
xgb_model = load_xgb_model()

subzone_ids = sorted(xgb_df["subzone_id"].astype(str).unique(), key=subzone_name)
preselected = st.session_state.get("selected_subzone_id")
default_idx = subzone_ids.index(preselected) if preselected in subzone_ids else 0
chosen = st.selectbox("Subzone", subzone_ids, index=default_idx, format_func=subzone_name)
st.session_state["selected_subzone_id"] = chosen  # carry the choice back to the map and breakdown pages
verdict_slot = st.container()  # filled at the end, once both models have run

# --- Whole subzone (XGBoost) ----------------------------------------------------
st.subheader("Whole subzone (XGBoost)")
row_df = xgb_df[xgb_df["subzone_id"] == chosen]
current_veg = float(row_df["fraction_vegetation"].iloc[0])
max_delta = max(0.0, 1.0 - current_veg)

control_col, curve_col = st.columns([1, 1.4], gap="large")
with control_col:
    st.caption(f"Vegetation today: {current_veg:.0%} of {subzone_name(chosen)}.")
    delta = st.slider(
        "Add vegetation (share of the subzone)", min_value=0.0, max_value=float(max_delta),
        value=float(min(0.15, max_delta)), step=0.01, format="%.2f",
    )
    result = predict_counterfactual_subzone(xgb_model, row_df, delta, index_model)
    m1, m2 = st.columns(2)
    m1.metric("LST today", f"{result['original_lst']:.1f}°C")
    m2.metric(
        "After greening", f"{result['counterfactual_lst']:.1f}°C", delta=f"{result['delta_lst']:+.2f}°C",
        delta_color="inverse",  # cooling (a negative delta) is the desired outcome here, not a regression
    )
    st.caption(
        f"Vegetation actually added: +{result['actual_delta_vegetation']:.0%} (taken from built-up and bare land)."
    )
    if delta > RELIABLE_DELTA:
        st.caption(f"⚠️ Beyond +{RELIABLE_DELTA:.0%} the prediction leaves the range the model was trained on — "
                   "read it as a direction, not a precise number.")

with curve_col:
    if max_delta > 0:
        sweep, sweep_lst = xgb_response_curve(chosen, max_delta, xgb_df, xgb_model, index_model)
        fig = go.Figure()
        if max_delta > RELIABLE_DELTA:
            fig.add_vrect(x0=RELIABLE_DELTA * 100, x1=max_delta * 100, fillcolor="#9e9e9e", opacity=0.12, line_width=0,
                          annotation_text="indicative only", annotation_position="top left")
        fig.add_trace(go.Scatter(x=sweep * 100, y=sweep_lst, mode="lines", line=dict(color=COOLING_COLOR, width=2.5),
                                 hovertemplate="+%{x:.0f}% vegetation → %{y:+.2f}°C<extra></extra>"))
        fig.add_trace(go.Scatter(x=[delta * 100], y=[result["delta_lst"]], mode="markers",
                                 marker=dict(size=12, color=COOLING_COLOR, line=dict(color="white", width=2)),
                                 hoverinfo="skip"))
        fig.add_hline(y=0, line_color="#888", line_width=1)
        fig.update_layout(
            height=300, margin=dict(l=60, r=10, t=30, b=40), showlegend=False,
            xaxis_title="Vegetation added (percentage points)", yaxis_title="Predicted LST change (°C)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Cooling for every amount of greening, with the slider's choice marked. Small bumps are normal "
                   "for a tree model: it predicts in steps, not along a smooth line.")
    else:
        st.caption("This subzone is already fully vegetated — there's nothing to add.")

# --- Street blocks around the centre (CNN) -----------------------------------------
st.divider()
st.subheader("Street blocks around the subzone centre (CNN)")

cnn_available = (
    CNN_MODEL_SAVE_PATH.exists() and CNN_MIXER_JSON_PATH.exists()
    and LANDCOVER_RASTER_PATH.exists() and LST_BICUBIC10_PATH.exists()
)
cnn_rescaled, cnn_target = None, None

if not cnn_available:
    st.info(
        "CNN heat model or its input rasters aren't available yet — train it via "
        "`notebooks/colab_training/train_heat_cnn.ipynb`, then "
        "`python scripts/pull_models.py --model cnn`."
    )
else:
    from src.heat_model.cnn_data import build_local_feature_target_patches
    from src.heat_model.cnn_infer import load_cnn_regressor, locate_patch_and_pixel
    from src.heat_model.counterfactual import class_mean_feature_vectors, rescale_subzone_delta, run_patch_counterfactual

    @st.cache_resource(show_spinner="Loading the CNN and its ~1,000 image patches (one-time, can take ~30 s)…")
    def load_cnn_assets():
        """Reads all ~1000 patches + reprojects the land-cover/LST rasters onto
        each -- a real one-time cost (tens of seconds), so cached per session
        rather than re-run on every slider move."""
        X, _y, valid_mask = build_local_feature_target_patches(
            CNN_INFERENCE_PATCH_DIR, CNN_MIXER_JSON_PATH, LANDCOVER_RASTER_PATH, LST_BICUBIC10_PATH,
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
        st.warning(f"'{subzone_name(chosen)}' not found in the subzones geodataframe — can't locate its centroid.")
    else:
        centroid_utm = match.geometry.centroid.iloc[0]
        centroid_lonlat = gpd.GeoSeries([centroid_utm], crs=S2_UTM_CRS).to_crs("EPSG:4326").iloc[0]
        lon, lat = centroid_lonlat.x, centroid_lonlat.y
        subzone_area_m2 = float(match.geometry.area.iloc[0])

        patch_idx, local_row, local_col = locate_patch_and_pixel(lon, lat, CNN_MIXER_JSON_PATH, UNET_PATCH_SIZE)
        if not (0 <= patch_idx < X.shape[0]):
            st.warning(f"'{subzone_name(chosen)}'s centroid falls outside the CNN patch grid — no live example available for it.")
        else:
            c_radius, c_target = st.columns(2)
            radius_m = c_radius.slider("Edit a circle around the centre, radius (m)", 10.0, 200.0, 50.0, step=10.0)
            bucket_name_to_id = {v: k for k, v in BUCKET_NAMES.items()}
            target_class_name = c_target.selectbox(
                "Turn the circle into", list(BUCKET_NAMES.values()), index=0,
                format_func=lambda name: LANDCOVER_LABELS.get(name, name),
            )
            target_class = bucket_name_to_id[target_class_name]

            radius_px = radius_m / TARGET_SCALE_M
            yy, xx = np.mgrid[0:UNET_PATCH_SIZE, 0:UNET_PATCH_SIZE]
            edit_mask = ((yy - local_row) ** 2 + (xx - local_col) ** 2) <= radius_px ** 2

            cnn_result = run_patch_counterfactual(cnn_model, X[patch_idx], edit_mask, target_class, class_means)
            edit_area_m2 = cnn_result["n_edited_pixels"] * (TARGET_SCALE_M ** 2)

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "LST change inside the circle", f"{cnn_result['mean_delta_lst_in_edit_area']:+.2f}°C",
                delta=f"{cnn_result['mean_delta_lst_in_edit_area']:+.2f}°C",
                delta_color="inverse",  # cooling (a negative delta) is the desired outcome here, not a regression
            )
            c2.metric("Area edited", f"{edit_area_m2:,.0f} m²", f"{cnn_result['n_edited_pixels']} pixels", delta_color="off",
                      delta_arrow="off")
            if subzone_area_m2 > 0 and edit_area_m2 > 0:
                cnn_rescaled = rescale_subzone_delta(cnn_result["mean_delta_lst_in_edit_area"], edit_area_m2, subzone_area_m2)
                cnn_target = target_class_name
                c3.metric(
                    "Spread over the whole subzone", f"{cnn_rescaled:+.3f}°C",
                    help=f"The circle's change, scaled by its share of the subzone's area "
                         f"({edit_area_m2 / subzone_area_m2:.2%}). A rough, area-weighted figure for comparing "
                         f"direction with XGBoost — not the same magnitude, since the two work at very different scales.",
                )

            # Before / after / change, cropped to a window around the edit so a
            # small circle is still visible (the full patch is ~1.3 km across).
            half = int(max(25, 3 * radius_px))
            r0, r1 = max(0, local_row - half), min(UNET_PATCH_SIZE, local_row + half)
            q0, q1 = max(0, local_col - half), min(UNET_PATCH_SIZE, local_col + half)
            before = cnn_result["original_lst_patch"][r0:r1, q0:q1]
            after = cnn_result["counterfactual_lst_patch"][r0:r1, q0:q1]
            change = cnn_result["delta_lst_patch"][r0:r1, q0:q1]
            lst_lo, lst_hi = np.nanpercentile(before, [2, 98])
            # 98th percentile, not the max: a few extreme pixels would otherwise wash out the rest.
            change_lim = max(0.05, float(np.nanpercentile(np.abs(change), 98)))

            maps = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                                 subplot_titles=("LST today", "LST after the edit", "Change"))
            for col, grid in ((1, before), (2, after)):
                maps.add_trace(go.Heatmap(z=grid, coloraxis="coloraxis", hovertemplate="%{z:.1f}°C<extra></extra>"), 1, col)
            maps.add_trace(go.Heatmap(z=change, coloraxis="coloraxis2", hovertemplate="%{z:+.2f}°C<extra></extra>"), 1, 3)
            cy, cx = local_row - r0, local_col - q0
            for col in (1, 2, 3):
                maps.add_shape(type="circle", x0=cx - radius_px, x1=cx + radius_px, y0=cy - radius_px, y1=cy + radius_px,
                               line=dict(color="#111", width=1.5, dash="dot"), row=1, col=col)
                axis = "" if col == 1 else str(col)
                maps.update_layout({
                    f"xaxis{axis}": dict(showticklabels=False, ticks="", showgrid=False, zeroline=False, constrain="domain"),
                    f"yaxis{axis}": dict(showticklabels=False, ticks="", showgrid=False, zeroline=False, autorange="reversed",
                                         scaleanchor=f"x{axis}", constrain="domain"),
                })
            maps.update_layout(
                height=330, margin=dict(l=10, r=10, t=40, b=10),
                coloraxis=dict(colorscale="YlOrRd", cmin=lst_lo, cmax=lst_hi,
                               colorbar=dict(title="°C", x=0.645, len=0.85, thickness=12)),
                coloraxis2=dict(colorscale="RdBu_r", cmin=-change_lim, cmax=change_lim,
                                colorbar=dict(title="Δ°C", x=1.0, len=0.85, thickness=12)),
            )
            st.plotly_chart(maps, width="stretch")
            st.caption(
                f"A {(r1 - r0) * TARGET_SCALE_M:,} m × {(q1 - q0) * TARGET_SCALE_M:,} m window around "
                f"{subzone_name(chosen)}'s centre; the dotted circle is the edited area. Blue in the change map is cooling. "
                "The CNN also changes LST just outside the circle, because it reads each pixel's neighbourhood."
            )

# --- Verdict: do the two models agree on the direction? ------------------------------
with verdict_slot:
    if cnn_rescaled is not None and cnn_target == "vegetation" and delta > 0:
        xgb_cools, cnn_cools = result["delta_lst"] < 0, cnn_rescaled < 0
        if xgb_cools and cnn_cools:
            st.success("Both models predict cooling from greening this subzone.", icon=":material/check_circle:")
        elif not xgb_cools and not cnn_cools:
            st.warning("Neither model predicts cooling here.", icon=":material/info:")
        else:
            st.warning(
                f"The models disagree on direction: XGBoost predicts {'cooling' if xgb_cools else 'warming'}, "
                f"the CNN {'cooling' if cnn_cools else 'warming'}. Treat this subzone's result with care.",
                icon=":material/warning:",
            )

if st.button("← Back to map"):
    st.switch_page("pages/2_Island_Map.py")
