"""Streamlit entry point (S7). The full runnable system — interactive
island map, per-subzone score breakdown, counterfactual greening slider,
validation dashboard — is not built yet (no source notebook exists for it).
This page is a landing page plus a quick status check of what pipeline
outputs already exist on disk, and a link to the validation-labeling tool
(the one piece of S7-adjacent UI that does exist today).

Run with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from config.settings import INTERIM_DIR, PROCESSED_DIR

st.set_page_config(page_title="Urban Heat & Cooling-Priority Mapping", page_icon="🌡️", layout="wide")

st.title("🌡️ Urban Heat & Cooling-Priority Mapping — Singapore")
st.caption("NUS-ISS Practice Module, Graduate Certificate in Pattern Recognition Systems")

st.markdown(
    """
Use the sidebar to open **Label Validation Points** — the joint hand-labeling
tool for the 200-point stratified validation sample.

The rest of the runnable system (island-wide priority map, per-subzone score
breakdown, counterfactual greening slider) lands here once S4-S7 are built.
"""
)

st.subheader("Pipeline status")

checks = [
    ("Heat-layer variants", INTERIM_DIR / "heat_variants_subzone.csv"),
    ("Sensitivity pillar", INTERIM_DIR / "sensitivity_pillar.csv"),
    ("Adaptive-capacity pillar", INTERIM_DIR / "adaptive_capacity_pillar.csv"),
    ("NEA held-out LST", INTERIM_DIR / "nea_heldout_lst.csv"),
    ("Land-change flags", INTERIM_DIR / "land_change_flags.csv"),
    ("Validation sample (drawn)", INTERIM_DIR / "validation_sample" / "validation_sample_200.csv"),
    ("Validation sample (labeled)", INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"),
    ("RF land cover raster", PROCESSED_DIR / "landcover" / "rf_landcover.tif"),
    ("U-Net land cover raster", PROCESSED_DIR / "landcover" / "unet_landcover.tif"),
    ("Production priority score", PROCESSED_DIR / "priority_score.csv"),
]

for label, path in checks:
    status = "✅ ready" if path.exists() else "⬜ not yet built"
    st.write(f"{status} — **{label}** (`{path}`)")
