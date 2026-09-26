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

from config.settings import PROCESSED_DIR, REFERENCE_VARIANT, TOP_N
from src.priority_score.io import load_and_join
from src.priority_score.lenses import DEFAULT_LENS, LENSES
from src.priority_score.score import build_score
from src.utils.geo import normalize

st.set_page_config(page_title="Subzone Breakdown — Urban Heat & Cooling Priority", page_icon="📊", layout="wide")
st.title("📊 Per-subzone score breakdown")

HOTSPOT_CLUSTERS_PATH = PROCESSED_DIR / "hotspot_clusters.csv"

# Same view selector and session key as the Island Map page (see
# src/priority_score/lenses.py): the map's choice carries over to this page.
lens_keys = list(LENSES)
remembered = st.session_state.get("priority_lens", DEFAULT_LENS)
lens_key = st.radio(
    "View", lens_keys, index=lens_keys.index(remembered if remembered in lens_keys else DEFAULT_LENS),
    format_func=lambda key: LENSES[key]["label"], horizontal=True,
)
st.session_state["priority_lens"] = lens_key
lens = LENSES[lens_key]
st.caption(lens["description"])
BANDS_PATH = lens["bands_csv"]

if not lens["score_csv"].exists():
    st.warning("Priority score not built yet — run `python scripts/build_priority_score.py` first.")
    st.stop()


@st.cache_data
def load_score_table(lens_key: str):
    view = LENSES[lens_key]
    df, _heldout = load_and_join(toy_mode=False, include_unranked=view["include_unranked"])
    score, weights = build_score(df, REFERENCE_VARIANT, view["weighting"])
    return df.assign(priority_score=score), weights


df, weights = load_score_table(lens_key)
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Priority score", f"{row['priority_score']:.3f}")
col2.metric("Exposure (LST)", f"{row[REFERENCE_VARIANT]:.1f}°C")
col3.metric("Greenery fraction", f"{row['greenery_fraction']:.2f}")
if "population_total" in row.index and pd.notna(row["population_total"]):
    col4.metric("Residents", f"{int(row['population_total']):,}")

if BANDS_PATH.exists():
    bands_df = pd.read_csv(BANDS_PATH)
    band_row = bands_df[bands_df["subzone_id"] == chosen]
    if not band_row.empty:
        b = band_row.iloc[0]
        st.subheader("Confidence band")
        p_col = f"p_top{TOP_N}"
        if p_col in b.index:
            st.metric(
                f"Chance of being in the top {TOP_N}", f"{b[p_col]:.0%}",
                help=f"Share of bootstrap draws in which this subzone lands among the {TOP_N} highest-priority "
                     f"subzones. A rank alone can't tell you how reliable it is; this can.",
            )
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
pillar_norms = {
    "Exposure": (normalize(df[REFERENCE_VARIANT]), weights["exposure"], "#2f6fed"),
    "Sensitivity": (normalize(df["sensitivity_raw"]), weights["sensitivity"], "#f2994a"),
    "Adaptive deficit": (normalize(1 - df["greenery_fraction"]), weights["adaptive_deficit"], "#27ae60"),
}
if lens["weighting"] == "heat_greenery":
    pillar_norms.pop("Sensitivity")  # the all-places view has no sensitivity term (and no value for small subzones)

pillar_names = list(pillar_norms)
pillar_values = [norm.loc[idx] for norm, _w, _c in pillar_norms.values()]
pillar_weights = [w for _norm, w, _c in pillar_norms.values()]
pillar_contribution = [v * w for v, w in zip(pillar_values, pillar_weights)]
PILLAR_COLORS = [c for _norm, _w, c in pillar_norms.values()]  # fixed colour per pillar

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
