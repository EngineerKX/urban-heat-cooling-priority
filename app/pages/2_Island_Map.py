"""S7 island-wide interactive map. Loads the URA subzone GeoJSON (already
cached locally, no GEE call needed to view it) + priority_score.csv (and
the confidence-band CSV, if built) and renders a folium choropleth,
matching the map conventions already established in
app/pages/1_Label_Validation_Points.py (Esri-style basemap, folium +
streamlit-folium). Click a subzone, then jump to its breakdown page.

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

from config.settings import MIN_RESIDENTS_FOR_RANKING, PROCESSED_DIR, SG_CENTER, SUBZONE_ID_PROPERTY, TOP_N
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.priority_score.lenses import DEFAULT_LENS, LENSES

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
has_bands = "band_width" in gdf.columns

color_options = {"Priority score": "priority_score"}
if has_bands:
    color_options["Confidence-band width"] = "band_width"
if P_TOP_COL in gdf.columns:
    color_options[f"Chance of top-{TOP_N}"] = P_TOP_COL
color_by = st.radio("Color by", list(color_options), horizontal=True)
value_col = color_options[color_by]

plot_gdf = gdf[gdf[value_col].notna()].copy()
n_missing = len(gdf) - len(plot_gdf)
if n_missing:
    reason = (
        f" Subzones with fewer than {MIN_RESIDENTS_FOR_RANKING} residents (parks, reserves, industrial estates) are not "
        f"ranked in this view, because a resident-based sensitivity can't be estimated for them; switch to "
        f"'{LENSES['all_places']['label']}' to see them."
        if lens_key == "residents" else ""
    )
    st.caption(f"{n_missing} subzone(s) are not drawn: they have no value for '{color_by}'.{reason}")

vmin, vmax = float(plot_gdf[value_col].min()), float(plot_gdf[value_col].max())
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = color_by

tooltip_fields = [SUBZONE_ID_PROPERTY, "priority_score"]
tooltip_aliases = ["Subzone", "Priority score"]
if has_bands:
    tooltip_fields.append("band_width")
    tooltip_aliases.append("Band width (p95-p05)")
if P_TOP_COL in gdf.columns:
    tooltip_fields.append(P_TOP_COL)
    tooltip_aliases.append(f"Chance of top-{TOP_N}")


def _style(feature):
    val = feature["properties"].get(value_col)
    return {
        "fillColor": colormap(val) if val is not None else "#cccccc",
        "color": "#555555", "weight": 0.5, "fillOpacity": 0.75,
    }


m = folium.Map(location=SG_CENTER, zoom_start=11, tiles="OpenStreetMap")
folium.GeoJson(
    plot_gdf.__geo_interface__, style_function=_style,
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
).add_to(m)
colormap.add_to(m)

map_state = st_folium(m, height=600, use_container_width=True, returned_objects=["last_active_drawing"])

clicked = map_state.get("last_active_drawing")
if clicked:
    clicked_id = clicked.get("properties", {}).get(SUBZONE_ID_PROPERTY)
    if clicked_id:
        st.session_state["selected_subzone_id"] = clicked_id

with st.expander(f"Top {TOP_N} subzones in this view"):
    top_table = gdf[gdf["priority_score"].notna()].sort_values("priority_score", ascending=False).head(TOP_N)
    top_cols = ["subzone_id", "priority_score"] + ([P_TOP_COL] if P_TOP_COL in top_table.columns else [])
    st.dataframe(
        top_table[top_cols].rename(columns={"subzone_id": "Subzone", "priority_score": "Priority score", P_TOP_COL: f"Chance of top-{TOP_N}"}),
        hide_index=True, use_container_width=True,
    )

selected = st.session_state.get("selected_subzone_id")
if selected:
    st.success(f"Selected subzone: **{selected}**")
    if st.button("View subzone breakdown →", type="primary"):
        st.switch_page("pages/3_Subzone_Breakdown.py")
else:
    st.info("Click a subzone on the map to select it, then view its score breakdown.")
