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

from config.settings import PROCESSED_DIR, SG_CENTER, SUBZONE_ID_PROPERTY
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson

st.set_page_config(page_title="Island Map — Urban Heat & Cooling Priority", page_icon="🗺️", layout="wide")
st.title("🗺️ Island-wide cooling-priority map")

PRIORITY_SCORE_PATH = PROCESSED_DIR / "priority_score.csv"
BANDS_PATH = PROCESSED_DIR / "priority_score_confidence_bands.csv"

if not PRIORITY_SCORE_PATH.exists():
    st.warning(f"`{PRIORITY_SCORE_PATH}` not found — run `python scripts/build_priority_score.py` first.")
    st.stop()


@st.cache_data
def load_map_data():
    geojson = fetch_subzones_geojson()
    subzones_gdf = as_geodataframe(geojson)
    score_df = pd.read_csv(PRIORITY_SCORE_PATH)
    merged = subzones_gdf.merge(score_df, left_on=SUBZONE_ID_PROPERTY, right_on="subzone_id", how="left")
    if BANDS_PATH.exists():
        bands_df = pd.read_csv(BANDS_PATH)
        merged = merged.merge(bands_df, on="subzone_id", how="left")
    return merged


gdf = load_map_data()
has_bands = "band_width" in gdf.columns

options = ["Priority score"] + (["Confidence-band width"] if has_bands else [])
color_by = st.radio("Color by", options, horizontal=True)
value_col = "priority_score" if color_by == "Priority score" else "band_width"

plot_gdf = gdf[gdf[value_col].notna()].copy()
n_missing = len(gdf) - len(plot_gdf)
if n_missing:
    st.caption(f"{n_missing} subzone(s) have no value for '{color_by}' and are shown uncolored.")

vmin, vmax = float(plot_gdf[value_col].min()), float(plot_gdf[value_col].max())
colormap = cm.linear.YlOrRd_09.scale(vmin, vmax)
colormap.caption = color_by

tooltip_fields = [SUBZONE_ID_PROPERTY, "priority_score"]
tooltip_aliases = ["Subzone", "Priority score"]
if has_bands:
    tooltip_fields.append("band_width")
    tooltip_aliases.append("Band width (p95-p05)")


def _style(feature):
    val = feature["properties"].get(value_col)
    return {
        "fillColor": colormap(val) if val is not None else "#cccccc",
        "color": "#555555", "weight": 0.5, "fillOpacity": 0.75,
    }


m = folium.Map(location=SG_CENTER, zoom_start=11, tiles="cartodbpositron")
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

selected = st.session_state.get("selected_subzone_id")
if selected:
    st.success(f"Selected subzone: **{selected}**")
    if st.button("View subzone breakdown →", type="primary"):
        st.switch_page("pages/3_Subzone_Breakdown.py")
else:
    st.info("Click a subzone on the map to select it, then view its score breakdown.")
