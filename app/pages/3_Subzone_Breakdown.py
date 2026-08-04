"""S7 per-subzone score breakdown. Reloads the same joined pillar table
and build_score() used by scripts/build_priority_score.py (pure CSV +
sklearn, no GEE call -- safe to run live in the app) so the pillar
contribution chart shown here always matches the production score exactly,
rather than recomputing it a different way.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import PROCESSED_DIR, REFERENCE_VARIANT
from src.priority_score.io import load_and_join
from src.priority_score.score import build_score
from src.utils.geo import normalize

st.set_page_config(page_title="Subzone Breakdown — Urban Heat & Cooling Priority", page_icon="📊", layout="wide")
st.title("📊 Per-subzone score breakdown")

HOTSPOT_CLUSTERS_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
BANDS_PATH = PROCESSED_DIR / "priority_score_confidence_bands.csv"

if not (PROCESSED_DIR / "priority_score.csv").exists():
    st.warning("Priority score not built yet — run `python scripts/build_priority_score.py` first.")
    st.stop()


@st.cache_data
def load_score_table():
    df, _heldout = load_and_join(toy_mode=False)
    score, weights = build_score(df, REFERENCE_VARIANT, "pca")
    return df.assign(priority_score=score), weights


df, weights = load_score_table()
subzone_ids = sorted(df["subzone_id"].astype(str).unique())

selected = st.session_state.get("selected_subzone_id")
if selected not in subzone_ids:
    selected = None

chosen = st.selectbox("Subzone", subzone_ids, index=subzone_ids.index(selected) if selected else 0)
st.session_state["selected_subzone_id"] = chosen
if selected is None:
    st.caption("No subzone was pre-selected from the map — pick one above, or go to **Island Map** and click one.")

row = df[df["subzone_id"] == chosen].iloc[0]
idx = row.name

col1, col2, col3 = st.columns(3)
col1.metric("Priority score", f"{row['priority_score']:.3f}")
col2.metric("Exposure (LST)", f"{row[REFERENCE_VARIANT]:.1f}°C")
col3.metric("Greenery fraction", f"{row['greenery_fraction']:.2f}")

if BANDS_PATH.exists():
    bands_df = pd.read_csv(BANDS_PATH)
    band_row = bands_df[bands_df["subzone_id"] == chosen]
    if not band_row.empty:
        b = band_row.iloc[0]
        st.subheader("Confidence band")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[b["priority_score_p50"]], y=["Priority score"], mode="markers",
            marker=dict(size=14, color="#2f6fed"),
            error_x=dict(
                type="data", symmetric=False,
                array=[b["priority_score_p95"] - b["priority_score_p50"]],
                arrayminus=[b["priority_score_p50"] - b["priority_score_p05"]],
                color="#2f6fed", thickness=2, width=8,
            ),
            name="p05 – p50 – p95",
        ))
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=30), showlegend=False, xaxis_title="Priority score")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Band width (p95−p05): {b['band_width']:.3f}. See the Validation Dashboard page for this "
                   f"bootstrap's stated limitations (reflects exposure + adaptive-capacity uncertainty only).")
else:
    st.caption(f"`{BANDS_PATH}` not found — run `python scripts/build_priority_score_confidence_bands.py` to enable this section.")

st.subheader("Pillar contribution")
exposure_norm = normalize(df[REFERENCE_VARIANT])
sensitivity_norm = normalize(df["sensitivity_raw"])
adaptive_deficit_norm = normalize(1 - df["greenery_fraction"])

pillar_names = ["Exposure", "Sensitivity", "Adaptive deficit"]
pillar_values = [exposure_norm.loc[idx], sensitivity_norm.loc[idx], adaptive_deficit_norm.loc[idx]]
pillar_weights = [weights["exposure"], weights["sensitivity"], weights["adaptive_deficit"]]
pillar_contribution = [v * w for v, w in zip(pillar_values, pillar_weights)]
PILLAR_COLORS = ["#2f6fed", "#f2994a", "#27ae60"]  # fixed order: exposure, sensitivity, adaptive deficit

fig2 = go.Figure(go.Bar(
    x=pillar_contribution, y=pillar_names, orientation="h", marker_color=PILLAR_COLORS,
    text=[f"{v:.3f} (weight={w:.2f})" for v, w in zip(pillar_contribution, pillar_weights)], textposition="auto",
))
fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=30), xaxis_title="Weighted contribution to priority score")
st.plotly_chart(fig2, use_container_width=True)

if HOTSPOT_CLUSTERS_PATH.exists():
    hs_df = pd.read_csv(HOTSPOT_CLUSTERS_PATH)
    hs_row = hs_df[hs_df["subzone_id"] == chosen]
    if not hs_row.empty:
        hs = hs_row.iloc[0]
        st.subheader("Land cover & hotspot typology")
        lc_col, cl_col = st.columns(2)
        with lc_col:
            classes = ["vegetation", "built_up", "bare", "water"]
            LC_COLORS = {"vegetation": "#27ae60", "built_up": "#8c8c8c", "bare": "#c9a35d", "water": "#2f80ed"}
            lc_fig = go.Figure(go.Bar(
                x=classes, y=[hs[f"fraction_{c}"] for c in classes],
                marker_color=[LC_COLORS[c] for c in classes],
            ))
            lc_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=30), yaxis_title="Fraction of subzone area")
            st.plotly_chart(lc_fig, use_container_width=True)
        with cl_col:
            st.metric("Hotspot cluster", f"Cluster {int(hs['primary_cluster'])} ({hs['primary_cluster_method']})")
            st.metric("Dry-season LST", f"{hs['lst_dry']:.1f}°C")
            st.metric("Wet-season LST", f"{hs['lst_wet']:.1f}°C")
            st.caption("See the Validation Dashboard page for cluster-quality metrics and the land-cover coherence check.")
    else:
        st.caption("No S4 hotspot-cluster data for this subzone yet.")
else:
    st.caption(f"`{HOTSPOT_CLUSTERS_PATH}` not found — run `python scripts/build_hotspot_clusters.py` to enable this section.")

nav1, nav2 = st.columns(2)
with nav1:
    if st.button("← Back to map"):
        st.switch_page("pages/2_Island_Map.py")
with nav2:
    if st.button("Explore counterfactual greening →"):
        st.switch_page("pages/4_Counterfactual_Greening.py")
