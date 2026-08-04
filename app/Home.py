"""Streamlit entry point (S7). Landing page: quick status check of what
pipeline outputs already exist on disk, plus links to every page —
the validation-labeling tool, the island map, per-subzone breakdown,
counterfactual greening, and the validation dashboard.

Run with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from config.settings import CNN_MODEL_SAVE_PATH, INTERIM_DIR, PROCESSED_DIR

st.set_page_config(page_title="Urban Heat & Cooling-Priority Mapping", page_icon="🌡️", layout="wide")

st.title("🌡️ Urban Heat & Cooling-Priority Mapping — Singapore")
st.caption("NUS-ISS Practice Module, Graduate Certificate in Pattern Recognition Systems")

st.markdown("Use the sidebar, or the links below, to explore the runnable system:")

nav1, nav2, nav3, nav4, nav5 = st.columns(5)
with nav1:
    st.page_link("pages/1_Label_Validation_Points.py", label="Label Validation Points", icon="🌿")
with nav2:
    st.page_link("pages/2_Island_Map.py", label="Island Map", icon="🗺️")
with nav3:
    st.page_link("pages/3_Subzone_Breakdown.py", label="Subzone Breakdown", icon="📊")
with nav4:
    st.page_link("pages/4_Counterfactual_Greening.py", label="Counterfactual Greening", icon="🌱")
with nav5:
    st.page_link("pages/5_Validation_Dashboard.py", label="Validation Dashboard", icon="✅")

st.subheader("Pipeline status")

checks = [
    ("Heat-layer variants", INTERIM_DIR / "heat_variants_subzone.csv"),
    ("Sensitivity pillar", INTERIM_DIR / "sensitivity_pillar.csv"),
    ("Adaptive-capacity pillar", INTERIM_DIR / "adaptive_capacity_pillar.csv"),
    ("NEA held-out LST", INTERIM_DIR / "nea_heldout_lst.csv"),
    ("MODIS held-out LST", INTERIM_DIR / "modis_heldout_lst.csv"),
    ("Land-change flags", INTERIM_DIR / "land_change_flags.csv"),
    ("Validation sample (drawn)", INTERIM_DIR / "validation_sample" / "validation_sample_200.csv"),
    ("Validation sample (labeled)", INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"),
    ("RF land cover raster", PROCESSED_DIR / "landcover" / "rf_landcover.tif"),
    ("U-Net land cover raster", PROCESSED_DIR / "landcover" / "unet_landcover.tif"),
    ("Land-cover ensemble raster", PROCESSED_DIR / "landcover" / "ensemble_landcover.tif"),
    ("S4 hotspot clusters", PROCESSED_DIR / "hotspot_clusters.csv"),
    ("S5 XGBoost heat model", PROCESSED_DIR / "heat_model" / "xgb_model.pkl"),
    ("S5 CNN heat model", CNN_MODEL_SAVE_PATH),
    ("Production priority score", PROCESSED_DIR / "priority_score.csv"),
    ("S6 confidence bands", PROCESSED_DIR / "priority_score_confidence_bands.csv"),
]

for label, path in checks:
    status = "✅ ready" if path.exists() else "⬜ not yet built"
    st.write(f"{status} — **{label}** (`{path}`)")
