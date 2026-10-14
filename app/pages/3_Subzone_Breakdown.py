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
from app.ui import CLUSTER_NAMES, LANDCOVER_COLORS, LANDCOVER_LABELS, PILLAR_COLORS, PILLAR_LABELS, cluster_name, subzone_name
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

chosen = st.selectbox(
    "Subzone", subzone_ids, index=subzone_ids.index(selected) if selected else 0, format_func=subzone_name,
)
st.session_state["selected_subzone_id"] = chosen
if selected is None:
    st.caption("No subzone was pre-selected from the map — pick one above, or go to **Island Map** and click one.")

row = df[df["subzone_id"] == chosen].iloc[0]
idx = row.name

P_TOP_COL = f"p_top{TOP_N}"
bands_df = pd.read_csv(BANDS_PATH) if BANDS_PATH.exists() else None
band_row = bands_df[bands_df["subzone_id"] == chosen] if bands_df is not None else None

# --- Headline: where it ranks, and how it compares with the median subzone ---
ranks = df["priority_score"].rank(ascending=False, method="first").astype(int)
rank, n_ranked = int(ranks.loc[idx]), len(df)
median_lst, median_green = df[REFERENCE_VARIANT].median(), df["greenery_fraction"].median()

st.markdown(f"## {subzone_name(chosen)}")
h1, h2, h3, h4 = st.columns(4)
position = (f"top {max(1, round(rank / n_ranked * 100))}%" if rank <= n_ranked / 2
            else f"bottom {max(1, round((n_ranked - rank + 1) / n_ranked * 100))}%")
h1.metric("Rank", f"{rank} of {n_ranked}", position, delta_color="off", delta_arrow="off")
h2.metric("Priority score", f"{row['priority_score']:.2f}",
          help="Weighted sum of the 0–1 pillars below; higher means a stronger case for cooling.")
if band_row is not None and not band_row.empty and P_TOP_COL in band_row.columns:
    h3.metric(f"Chance of top {TOP_N}", f"{band_row.iloc[0][P_TOP_COL]:.0%}",
              help=f"Share of bootstrap draws in which this subzone lands among the {TOP_N} highest-priority "
                   f"subzones. A rank alone can't tell you how reliable it is; this can.")
if "population_total" in row.index and pd.notna(row["population_total"]):
    h4.metric("Residents", f"{int(row['population_total']):,}")

d1, d2, _d3, _d4 = st.columns(4)
d1.metric("Land surface temperature", f"{row[REFERENCE_VARIANT]:.1f}°C",
          f"{row[REFERENCE_VARIANT] - median_lst:+.1f}°C vs median subzone", delta_color="inverse")
d2.metric("Greenery", f"{row['greenery_fraction']:.1%}",
          f"{(row['greenery_fraction'] - median_green) * 100:+.1f} pts vs median subzone")

# --- Why it ranks here: pillar contributions vs the top N and the median ---
st.subheader("Why it ranks here")
pillar_keys = [k for k in ("exposure", "sensitivity", "adaptive_deficit") if weights.get(k, 0) > 0]
if lens["weighting"] == "heat_greenery":
    pillar_keys = [k for k in pillar_keys if k != "sensitivity"]  # no sensitivity term (or value) in the all-places view
pillar_norm = {
    "exposure": lambda: normalize(df[REFERENCE_VARIANT]),
    "sensitivity": lambda: normalize(df["sensitivity_raw"]),
    "adaptive_deficit": lambda: normalize(1 - df["greenery_fraction"]),
}
top_idx = df.nlargest(TOP_N, "priority_score").index
contrib = {"This subzone": [], f"Top-{TOP_N} average": [], "Median subzone": []}
for key in pillar_keys:
    norm, w = pillar_norm[key](), weights[key]
    contrib["This subzone"].append(norm.loc[idx] * w)
    contrib[f"Top-{TOP_N} average"].append(norm.loc[top_idx].mean() * w)
    contrib["Median subzone"].append(norm.median() * w)
labels = [PILLAR_LABELS[k] for k in pillar_keys]

fig2 = go.Figure()
for name, colors in (("Median subzone", "#d0d0d0"), (f"Top-{TOP_N} average", "#8c8c8c"),
                     ("This subzone", [PILLAR_COLORS[k] for k in pillar_keys])):
    fig2.add_trace(go.Bar(
        y=labels, x=contrib[name], name=name, orientation="h", marker_color=colors,
        text=[f"{v:.2f}" for v in contrib[name]] if name == "This subzone" else None, textposition="outside",
    ))
fig2.update_layout(
    barmode="group", height=110 + 70 * len(labels), margin=dict(l=10, r=30, t=10, b=30),
    xaxis_title="Contribution to priority score", yaxis=dict(autorange="reversed"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, traceorder="reversed"),
)
fig2.update_yaxes(automargin=True)
st.plotly_chart(fig2, width="stretch")
st.caption(
    "Each bar is the pillar's 0–1 value times its weight ("
    + ", ".join(f"{PILLAR_LABELS[k]} {weights[k]:.0%}" for k in pillar_keys)
    + "); this subzone's bars add up to its priority score. Grey bars show the same for the average top-"
    f"{TOP_N} subzone and the median subzone in this view."
)

# --- Confidence band: this subzone among its rank neighbours ---
st.subheader("How sure is this rank?")
if band_row is None:
    st.caption("Confidence bands haven't been built for this view yet.")
elif band_row.empty:
    st.caption("No confidence band for this subzone.")
else:
    ranked_bands = bands_df.sort_values("priority_score_point", ascending=False).reset_index(drop=True)
    pos = int(ranked_bands.index[ranked_bands["subzone_id"] == chosen][0])
    window = ranked_bands.iloc[max(0, pos - 6): pos + 7].copy()
    window["label"] = [f"{i + 1}. {subzone_name(z)}" for i, z in zip(window.index, window["subzone_id"])]
    is_chosen = window["subzone_id"] == chosen

    fig = go.Figure()
    for mask, color, size in ((~is_chosen, "#9e9e9e", 8), (is_chosen, PILLAR_COLORS["exposure"], 13)):
        part = window[mask]
        fig.add_trace(go.Scatter(
            x=part["priority_score_p50"], y=part["label"], mode="markers", marker=dict(size=size, color=color),
            error_x=dict(type="data", symmetric=False, array=part["priority_score_p95"] - part["priority_score_p50"],
                         arrayminus=part["priority_score_p50"] - part["priority_score_p05"], color=color, thickness=2, width=5),
            hovertemplate="%{y}<br>median %{x:.2f}<extra></extra>",
        ))
    if len(ranked_bands) > TOP_N:
        cutoff = ranked_bands["priority_score_point"].iloc[TOP_N - 1: TOP_N + 1].mean()
        fig.add_vline(x=cutoff, line_dash="dash", line_color="#555",
                      annotation_text=f"top-{TOP_N} cut-off", annotation_position="top")
    fig.update_layout(
        height=60 + 28 * len(window), margin=dict(l=10, r=10, t=30, b=30), showlegend=False,
        xaxis_title="Priority score (p05 – median – p95)",
        yaxis=dict(categoryorder="array", categoryarray=window["label"].tolist(), autorange="reversed"),
    )
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Each line spans the middle 90% of bootstrap draws for that subzone; overlapping lines mean their order "
        f"could swap. A line crossing the cut-off is why a subzone's chance of top {TOP_N} sits between 0% and 100%. "
        f"The bands reflect heat and greenery measurement noise only — see the Validation Dashboard."
    )

if HOTSPOT_CLUSTERS_PATH.exists():
    hs_df = pd.read_csv(HOTSPOT_CLUSTERS_PATH)
    hs_row = hs_df[hs_df["subzone_id"] == chosen]
    if not hs_row.empty:
        hs = hs_row.iloc[0]
        st.subheader("Land cover & hotspot typology")
        lc_col, cl_col = st.columns(2)
        with lc_col:
            classes = list(LANDCOVER_LABELS)
            lc_fig = go.Figure(go.Bar(
                x=[LANDCOVER_LABELS[c] for c in classes], y=[hs[f"fraction_{c}"] for c in classes],
                marker_color=[LANDCOVER_COLORS[c] for c in classes],
            ))
            lc_fig.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=30), yaxis_title="Share of subzone area", yaxis_tickformat=".0%",
            )
            st.plotly_chart(lc_fig, width="stretch")
        with cl_col:
            st.metric(
                "Hotspot type", cluster_name(hs["primary_cluster"]),
                help=f"Cluster {int(hs['primary_cluster'])} of {len(CLUSTER_NAMES)} "
                     f"({hs['primary_cluster_method']} on dry/wet-season LST, NDVI and NDBI).",
            )
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
