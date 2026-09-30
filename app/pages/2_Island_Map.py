"""S7 island-wide interactive map. Loads the URA subzone GeoJSON (already
cached locally, no GEE call needed to view it) + priority_score.csv (and
the confidence-band CSV, if built) and renders a folium choropleth,
matching the map conventions already established in
app/pages/1_Label_Validation_Points.py (Esri-style basemap, folium +
streamlit-folium). Click a subzone, then jump to its breakdown page.
In the residents view, three weight sliders (heat / vulnerable residents /
missing greenery) re-score the map live and count how many of the top-N move
compared with the measurement-noise floor.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import branca.colormap as cm
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from config.settings import (
    MIN_RESIDENTS_FOR_RANKING, PROCESSED_DIR, REFERENCE_VARIANT, SG_CENTER, SUBZONE_ID_PROPERTY, TOP_N,
)
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.priority_score.io import load_and_join
from src.priority_score.lenses import DEFAULT_LENS, LENSES
from src.priority_score.score import build_score
from validation.score_validation.decision_impact import noise_floor

st.set_page_config(page_title="Island Map — Urban Heat & Cooling Priority", page_icon="🗺️", layout="wide")
st.title("🗺️ Island-wide cooling-priority map")

# Which view of the score to map (see src/priority_score/lenses.py). Kept in
# session_state under our own key because a widget's own state is dropped when
# you navigate to another page; the breakdown page reads the same key.
lens_keys = list(LENSES)
remembered = st.session_state.get("priority_lens", DEFAULT_LENS)
lens_key = st.radio(
    "View", lens_keys, index=lens_keys.index(remembered if remembered in lens_keys else DEFAULT_LENS),
    format_func=lambda key: LENSES[key]["label"], horizontal=True,
)
st.session_state["priority_lens"] = lens_key
lens = LENSES[lens_key]
st.caption(lens["description"])

PRIORITY_SCORE_PATH = lens["score_csv"]
BANDS_PATH = lens["bands_csv"]
P_TOP_COL = f"p_top{TOP_N}"

if not PRIORITY_SCORE_PATH.exists():
    st.warning(f"`{PRIORITY_SCORE_PATH}` not found — run `python scripts/build_priority_score.py` first.")
    st.stop()


@st.cache_data
def load_map_data(score_path: str, bands_path: str):
    geojson = fetch_subzones_geojson()
    subzones_gdf = as_geodataframe(geojson)
    score_df = pd.read_csv(score_path)
    merged = subzones_gdf.merge(score_df, left_on=SUBZONE_ID_PROPERTY, right_on="subzone_id", how="left")
    if Path(bands_path).exists():
        bands_df = pd.read_csv(bands_path)
        merged = merged.merge(bands_df, on="subzone_id", how="left")
        if P_TOP_COL in merged.columns:
            merged[P_TOP_COL] = merged[P_TOP_COL].round(2)
    return merged


gdf = load_map_data(str(PRIORITY_SCORE_PATH), str(BANDS_PATH))
saved_score_range = (float(gdf["priority_score"].min()), float(gdf["priority_score"].max()))
saved_top = set(gdf[gdf["priority_score"].notna()].nlargest(TOP_N, "priority_score")["subzone_id"])

PILLAR_LABELS = {"exposure": "Heat", "sensitivity": "Vulnerable residents", "adaptive_deficit": "Missing greenery"}


@st.cache_data
def load_pillar_table(lens_key: str):
    """The joined pillar table behind this view's saved score, plus that
    score's weights (PCA for residents, fixed 50/50 for all places)."""
    view = LENSES[lens_key]
    df, _ = load_and_join(include_unranked=view["include_unranked"])
    _, weights = build_score(df, REFERENCE_VARIANT, view["weighting"])
    return df, {k: float(v) for k, v in weights.items()}


def _reset_weights(defaults: dict):
    for key, value in defaults.items():
        st.session_state[f"weight_{lens_key}_{key}"] = value


# Weight sliders, one per pillar the view uses (3 for residents, 2 for all
# places, which has no sensitivity pillar). They re-score from the same
# pillars as the saved score; the band width and chance-of-top-N columns
# belong to the saved weights, so they are hidden while custom weights are on.
# Everything here sits in ONE container, so the number of elements above the
# map never changes as the sliders move; otherwise Streamlit remounts the map
# and the viewer's zoom/pan is lost.
custom_weights = None
weights_panel = st.container()
with weights_panel:
    if st.toggle("Adjust the score weights"):
        try:
            pillar_df, saved_weights = load_pillar_table(lens_key)
        except (FileNotFoundError, ValueError) as err:
            st.warning(f"Can't load the score pillars: {err}")
        else:
            saved_weights = {k: v for k, v in saved_weights.items() if v > 0}
            saved_label = "data-derived (PCA)" if lens["weighting"] == "pca" else "default"
            defaults = {k: round(v, 2) for k, v in saved_weights.items()}
            for key, value in defaults.items():
                st.session_state.setdefault(f"weight_{lens_key}_{key}", value)
            slider_cols = st.columns(len(defaults))
            custom_weights = {
                key: col.slider(PILLAR_LABELS[key], 0.0, 1.0, step=0.01, key=f"weight_{lens_key}_{key}")
                for col, key in zip(slider_cols, defaults)
            }
            st.button(f"Reset to {saved_label} weights", on_click=_reset_weights, args=(defaults,))
            total = sum(custom_weights.values())
            if total == 0:
                st.warning(f"All weights are zero — showing the {saved_label} weights instead.")
                custom_weights = None
            else:
                st.caption(
                    "Weights are rescaled to add up to 100%: "
                    + " · ".join(f"{PILLAR_LABELS[k]} {v / total:.0%}" for k, v in custom_weights.items())
                    + f". {saved_label[0].upper() + saved_label[1:]} weights: "
                    + " · ".join(f"{PILLAR_LABELS[k]} {v:.0%}" for k, v in saved_weights.items()) + "."
                )

    if custom_weights is not None:
        custom_score, _ = build_score(pillar_df, REFERENCE_VARIANT, "custom", weights=custom_weights)
        gdf = gdf.drop(columns=["priority_score"]).merge(
            pd.DataFrame({"subzone_id": pillar_df["subzone_id"], "priority_score": custom_score}), on="subzone_id", how="left",
        )
        gdf = gdf.drop(columns=[c for c in ("band_width", P_TOP_COL) if c in gdf.columns])
        custom_top = set(gdf[gdf["priority_score"].notna()].nlargest(TOP_N, "priority_score")["subzone_id"])
        entered, left = sorted(custom_top - saved_top), sorted(saved_top - custom_top)

        floor = noise_floor(pd.read_csv(BANDS_PATH)) if BANDS_PATH.exists() else None
        metric_cols = st.columns(2)
        metric_cols[0].metric(f"Top-{TOP_N} subzones changed by these weights", len(entered))
        if floor is not None:
            metric_cols[1].metric("Changed by measurement noise alone (average)", f"{floor:.1f}")
            st.caption(
                f"Within measurement noise — these weights move fewer of the top {TOP_N} than the data's own uncertainty does."
                if len(entered) <= floor else
                f"Beyond measurement noise — these weights move more of the top {TOP_N} than the data's own uncertainty does."
            )
        if entered:
            st.caption(f"Enter the top {TOP_N}: {', '.join(entered)}. Leave: {', '.join(left)}.")

has_bands = "band_width" in gdf.columns

# Rank and top-N status under the weights on screen. With custom weights the
# status compares against the saved weights' top N, so the "Top N only" colour
# option shows which subzones stay, enter and drop out as the sliders move.
ranked = gdf["priority_score"].notna()
gdf["rank_label"] = None
gdf.loc[ranked, "rank_label"] = (
    gdf.loc[ranked, "priority_score"].rank(ascending=False, method="first").astype(int).astype(str) + f" of {int(ranked.sum())}"
)
current_top = set(gdf[ranked].nlargest(TOP_N, "priority_score")["subzone_id"])
TOP_STATUS_STYLE = {  # status -> (fill colour, fill opacity, dashed outline)
    f"In top {TOP_N}": ("#b2182b", 0.8, False),
    f"Stays in top {TOP_N}": ("#b2182b", 0.8, False),
    f"New in top {TOP_N}": ("#2166ac", 0.8, False),
    f"Dropped out of top {TOP_N}": ("#969696", 0.6, True),
}
gdf["top_status"] = "Not in top"
if custom_weights is None:
    gdf.loc[gdf["subzone_id"].isin(current_top), "top_status"] = f"In top {TOP_N}"
else:
    gdf.loc[gdf["subzone_id"].isin(current_top & saved_top), "top_status"] = f"Stays in top {TOP_N}"
    gdf.loc[gdf["subzone_id"].isin(current_top - saved_top), "top_status"] = f"New in top {TOP_N}"
    gdf.loc[gdf["subzone_id"].isin(saved_top - current_top), "top_status"] = f"Dropped out of top {TOP_N}"

color_options = {"Priority score": "priority_score", f"Top {TOP_N} only": "top_status"}
if has_bands:
    color_options["Confidence-band width"] = "band_width"
if P_TOP_COL in gdf.columns:
    color_options[f"Chance of top-{TOP_N}"] = P_TOP_COL
# Remembered by label: turning the weight sliders on removes the band options,
# which would otherwise reset the choice (e.g. out of "Top N only").
remembered_color = st.session_state.get("map_color_by")
color_by = st.radio(
    "Color by", list(color_options), horizontal=True,
    index=list(color_options).index(remembered_color) if remembered_color in color_options else 0,
)
st.session_state["map_color_by"] = color_by
value_col = color_options[color_by]
show_top_only = value_col == "top_status"
draw_col = "priority_score" if show_top_only else value_col

plot_gdf = gdf[gdf[draw_col].notna()].copy()
n_missing = len(gdf) - len(plot_gdf)
if n_missing:
    reason = (
        f" Subzones with fewer than {MIN_RESIDENTS_FOR_RANKING} residents (parks, reserves, industrial estates) are not "
        f"ranked in this view, because a resident-based sensitivity can't be estimated for them; switch to "
        f"'{LENSES['all_places']['label']}' to see them."
        if lens_key == "residents" else ""
    )
    st.caption(f"{n_missing} subzone(s) are not drawn: they have no value for '{"Priority score" if show_top_only else color_by}'.{reason}")

# The priority score is always coloured on the SAVED score's range, so moving
# the weight sliders changes a subzone's colour only when its score changes
# (scores outside that range take the end colours).
vmin, vmax = saved_score_range if draw_col == "priority_score" else (
    float(plot_gdf[draw_col].min()), float(plot_gdf[draw_col].max()))
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = color_by

tooltip_fields = [SUBZONE_ID_PROPERTY, "rank_label", "priority_score"]
tooltip_aliases = ["Subzone", "Rank", "Priority score"]
if show_top_only:
    tooltip_fields.append("top_status")
    tooltip_aliases.append(f"Top {TOP_N}")
if has_bands:
    tooltip_fields.append("band_width")
    tooltip_aliases.append("Band width (p95-p05)")
if P_TOP_COL in gdf.columns:
    tooltip_fields.append(P_TOP_COL)
    tooltip_aliases.append(f"Chance of top-{TOP_N}")


def _style(feature):
    if show_top_only:
        fill, opacity, dashed = TOP_STATUS_STYLE.get(feature["properties"].get("top_status"), ("#dddddd", 0.25, False))
        return {
            "fillColor": fill, "fillOpacity": opacity, "color": "#222222" if dashed else "#777777",
            "weight": 1.5 if dashed else 0.5, "dashArray": "5 4" if dashed else None,
        }
    val = feature["properties"].get(value_col)
    return {
        "fillColor": colormap(val) if val is not None else "#cccccc",
        "color": "#555555", "weight": 0.5, "fillOpacity": 0.75,
    }


# The subzones go in a FeatureGroup passed separately, so a slider move swaps
# only that layer: the base map (and its zoom/pan) stays put. The key changes
# with the view and colour option, whose legend lives on the base map.
m = folium.Map(location=SG_CENTER, zoom_start=11, tiles="OpenStreetMap")
if show_top_only:
    shown = [s for s in TOP_STATUS_STYLE if (s == f"In top {TOP_N}") == (custom_weights is None)]
    st.markdown(" &nbsp; ".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:{TOP_STATUS_STYLE[s][0]};'
        f'{"border:1.5px dashed #222;" if TOP_STATUS_STYLE[s][2] else ""}margin-right:4px;vertical-align:middle"></span>{s}'
        for s in shown
    ) + " &nbsp; <span style='color:#888'>(other subzones faded)</span>", unsafe_allow_html=True)
else:
    colormap.add_to(m)
subzone_layer = folium.FeatureGroup(name="Subzones")
folium.GeoJson(
    plot_gdf.__geo_interface__, style_function=_style,
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
).add_to(subzone_layer)

map_state = st_folium(
    m, key=f"island_map_{lens_key}_{value_col}", feature_group_to_add=subzone_layer,
    height=600, use_container_width=True, returned_objects=["last_active_drawing"],
)

clicked = map_state.get("last_active_drawing")
if clicked:
    clicked_id = clicked.get("properties", {}).get(SUBZONE_ID_PROPERTY)
    if clicked_id:
        st.session_state["selected_subzone_id"] = clicked_id

with st.expander(f"Top {TOP_N} subzones in this view"):
    top_table = gdf[gdf["priority_score"].notna()].sort_values("priority_score", ascending=False).head(TOP_N)
    top_cols = ["subzone_id", "priority_score"] + ([P_TOP_COL] if P_TOP_COL in top_table.columns else [])
    if custom_weights is not None:
        top_cols.append("top_status")
    st.dataframe(
        top_table[top_cols].rename(columns={
            "subzone_id": "Subzone", "priority_score": "Priority score", P_TOP_COL: f"Chance of top-{TOP_N}",
            "top_status": "vs saved weights",
        }),
        hide_index=True, use_container_width=True,
    )

selected = st.session_state.get("selected_subzone_id")
if selected:
    st.success(f"Selected subzone: **{selected}**")
    if st.button("View subzone breakdown →", type="primary"):
        st.switch_page("pages/3_Subzone_Breakdown.py")
else:
    st.info("Click a subzone on the map to select it, then view its score breakdown.")
