"""S7 counterfactual greening page (C2): "what if this area were greener?"
Two paths, matching src/heat_model/counterfactual.py's module docstring --
they cross-check each other rather than being one combined model:

- Live XGBoost subzone-level slider -- runs natively on Windows (no GPU
  needed), fully interactive.
- A precomputed CNN patch-level example gallery
  (data/processed/heat_model/canned_counterfactual_examples.json, built by
  scripts/diagnose_heat_model.py) -- native Windows TensorFlow is
  confirmed broken in this repo, so the CNN can't run live from this app;
  shipping canned real examples is the documented workaround rather than
  silently omitting the CNN half from the UI entirely.

Run the whole app with: streamlit run app/Home.py
"""

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from config.settings import INTERIM_DIR, PROCESSED_DIR
from src.heat_model.tabular import build_xgb_training_table, fit_ndvi_vegetation_slope, predict_counterfactual_subzone

st.set_page_config(page_title="Counterfactual Greening — Urban Heat & Cooling Priority", page_icon="🌱", layout="wide")
st.title("🌱 Counterfactual greening: \"what if this area were greener?\"")

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
HOTSPOT_CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
XGB_MODEL_PATH = PROCESSED_DIR / "heat_model" / "xgb_model.pkl"
CANNED_EXAMPLES_PATH = PROCESSED_DIR / "heat_model" / "canned_counterfactual_examples.json"

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
st.subheader("Patch-level (CNN) — precomputed examples")
st.caption(
    "Native Windows TensorFlow is confirmed broken in this repo, so the CNN can't run live from this "
    "Streamlit app. These examples were precomputed in WSL2 via `python scripts/diagnose_heat_model.py` "
    "and cross-check the XGBoost result above at a much finer spatial scale (a real 10m patch around each "
    "subzone's centroid) rather than being silently omitted from the UI."
)

if not CANNED_EXAMPLES_PATH.exists():
    st.info(
        "No precomputed CNN examples yet — run `python scripts/train_heat_model_cnn.py` (WSL2) then "
        "`python scripts/diagnose_heat_model.py` to generate them."
    )
else:
    with open(CANNED_EXAMPLES_PATH) as f:
        examples = json.load(f)

    for example in examples:
        with st.container(border=True):
            st.markdown(f"**{example['subzone_id']}**")
            xgb = example["xgboost"]
            st.write(
                f"XGBoost (Δveg=+{example['delta_fraction_vegetation']:.2f}): "
                f"{xgb['original_lst']:.2f}°C → {xgb['counterfactual_lst']:.2f}°C (Δ={xgb['delta_lst']:+.3f}°C)"
            )
            cnn = example.get("cnn")
            if cnn:
                agree = "✅ same direction as XGBoost" if cnn["agrees_with_xgboost_direction"] else "⚠️ opposite direction from XGBoost"
                st.write(
                    f"CNN (r={cnn['radius_m']:.0f}m patch, {cnn['n_edited_pixels']} px edited): "
                    f"mean ΔLST in edited area = {cnn['mean_delta_lst_in_edit_area']:+.3f}°C ({agree})"
                )
                if cnn.get("rescaled_subzone_equivalent_delta") is not None:
                    st.caption(f"Naive area-weighted subzone-wide equivalent: {cnn['rescaled_subzone_equivalent_delta']:+.4f}°C")
            else:
                st.caption("No CNN example for this subzone (its centroid may have fallen outside the patch grid).")

if st.button("← Back to map"):
    st.switch_page("pages/2_Island_Map.py")
