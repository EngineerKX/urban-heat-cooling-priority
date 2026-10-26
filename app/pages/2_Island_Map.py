"""S7 island-wide interactive map. Loads the URA subzone GeoJSON (already
cached locally, no GEE call needed to view it) + priority_score.csv (and
the confidence-band CSV, if built) and renders a folium choropleth on a
light Esri World Gray Canvas basemap (no API key needed; folium +
streamlit-folium, as in app/pages/1_Label_Validation_Points.py).

Layout: the map settings (view, colour, weight sliders) sit in the sidebar,
and a details panel sits beside the map: the top-N list until a subzone is
picked (by clicking it, choosing a row, or searching), then that subzone's
rank, pillars and a link to its breakdown page. In the residents view, three
weight sliders (heat / vulnerable residents / missing greenery) re-score the
map live and count how many of the top-N move compared with the
measurement-noise floor.

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
from app.ui import PILLAR_COLORS, PILLAR_LABELS, subzone_name
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.priority_score.io import load_and_join
from src.priority_score.lenses import DEFAULT_LENS, LENSES
from src.priority_score.score import build_score
from src.utils.geo import normalize
from validation.score_validation.decision_impact import noise_floor

st.set_page_config(page_title="Island Map — Urban Heat & Cooling Priority", page_icon="🗺️", layout="wide")
st.title("🗺️ Island-wide cooling-priority map")
st.sidebar.header("Map settings")

# Which view of the score to map (see src/priority_score/lenses.py). Kept in
# session_state under our own key because a widget's own state is dropped when
# you navigate to another page; the breakdown page reads the same key.
lens_keys = list(LENSES)
remembered = st.session_state.get("priority_lens", DEFAULT_LENS)
lens_key = st.sidebar.radio(
    "View", lens_keys, index=lens_keys.index(remembered if remembered in lens_keys else DEFAULT_LENS),
    format_func=lambda key: LENSES[key]["label"],
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
    merged["subzone_name"] = merged[SUBZONE_ID_PROPERTY].map(subzone_name)
    return merged


gdf = load_map_data(str(PRIORITY_SCORE_PATH), str(BANDS_PATH))
saved_score_range = (float(gdf["priority_score"].min()), float(gdf["priority_score"].max()))
saved_top = set(gdf[gdf["priority_score"].notna()].nlargest(TOP_N, "priority_score")["subzone_id"])

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


try:
    pillar_df, saved_weights = load_pillar_table(lens_key)
    saved_weights = {k: v for k, v in saved_weights.items() if v > 0}
except (FileNotFoundError, ValueError) as err:
    pillar_df, saved_weights, pillar_error = None, None, err

# Weight sliders, one per pillar the view uses (3 for residents, 2 for all
# places, which has no sensitivity pillar). They re-score from the same
# pillars as the saved score; the band width and chance-of-top-N columns
# belong to the saved weights, so they are hidden while custom weights are on.
# They live in the sidebar, so moving them never changes the number of
# elements above the map; otherwise Streamlit remounts the map and the
# viewer's zoom/pan is lost.
custom_weights = None
with st.sidebar:
    if st.toggle("Adjust the score weights"):
        if pillar_df is None:
            st.warning(f"Can't load the score pillars: {pillar_error}")
        else:
            saved_label = "data-derived (PCA)" if lens["weighting"] == "pca" else "default"
            defaults = {k: round(v, 2) for k, v in saved_weights.items()}
            for key, value in defaults.items():
                st.session_state.setdefault(f"weight_{lens_key}_{key}", value)
            custom_weights = {
                key: st.slider(PILLAR_LABELS[key], 0.0, 1.0, step=0.01, key=f"weight_{lens_key}_{key}")
                for key in defaults
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
        metric_cols[0].metric(f"Top-{TOP_N} changed", len(entered), help=f"Top-{TOP_N} subzones changed by these weights.")
        if floor is not None:
            metric_cols[1].metric("Noise alone", f"{floor:.1f}", help="Changed by measurement noise alone (average).")
            st.caption(
                f"Within measurement noise — these weights move fewer of the top {TOP_N} than the data's own uncertainty does."
                if len(entered) <= floor else
                f"Beyond measurement noise — these weights move more of the top {TOP_N} than the data's own uncertainty does."
            )
        if entered:
            st.caption(
                f"Enter the top {TOP_N}: {', '.join(map(subzone_name, entered))}. "
                f"Leave: {', '.join(map(subzone_name, left))}."
            )

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
color_by = st.sidebar.radio(
    "Color by", list(color_options),
    index=list(color_options).index(remembered_color) if remembered_color in color_options else 0,
)
st.session_state["map_color_by"] = color_by
value_col = color_options[color_by]
show_top_only = value_col == "top_status"
draw_col = "priority_score" if show_top_only else value_col

plot_gdf = gdf[gdf[draw_col].notna()].copy()
# Subzones with no value are still drawn, near-white, with a tooltip saying
# why, so the gaps in the map explain themselves. In the residents view these
# are the subzones under MIN_RESIDENTS_FOR_RANKING residents (parks, reserves,
# industrial estates), whose resident-based sensitivity can't be estimated.
no_value_gdf = gdf[gdf[draw_col].isna()].copy()
not_ranked_label = f"Not ranked (fewer than {MIN_RESIDENTS_FOR_RANKING} residents)"
no_value_gdf["no_value_reason"] = (
    not_ranked_label if lens_key == "residents" and draw_col == "priority_score"
    else f"No value for '{color_by}'"
)
if lens_key == "residents":
    no_value_gdf.loc[no_value_gdf["priority_score"].isna(), "no_value_reason"] = not_ranked_label
# Everything printed above the map goes in this ONE container, so the number
# of elements above the map stays fixed whatever the settings (see the
# weight-slider comment above).
notes = st.container()
if len(no_value_gdf):
    notes.caption(
        f"{len(no_value_gdf)} subzone(s) outlined in dashes have no value here — hover one to see why."
        + (f" Switch to '{LENSES['all_places']['label']}' to rank them too." if lens_key == "residents" else "")
    )

# The priority score is always coloured on the SAVED score's range, so moving
# the weight sliders changes a subzone's colour only when its score changes
# (scores outside that range take the end colours).
vmin, vmax = saved_score_range if draw_col == "priority_score" else (
    float(plot_gdf[draw_col].min()), float(plot_gdf[draw_col].max()))
# Uncertainty gets its own palette, so a wide band never reads as "high priority".
palette = cm.linear.Purples_09 if draw_col == "band_width" else cm.linear.YlOrRd_09
colormap = palette.scale(vmin, vmax)
colormap.caption = color_by

# Tooltip copies, formatted for reading; the raw columns stay numeric for colouring.
plot_gdf["score_label"] = plot_gdf["priority_score"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
tooltip_fields = ["subzone_name", "rank_label", "score_label"]
tooltip_aliases = ["Subzone", "Rank", "Priority score"]
if show_top_only:
    tooltip_fields.append("top_status")
    tooltip_aliases.append(f"Top {TOP_N}")
if has_bands:
    plot_gdf["band_label"] = plot_gdf["band_width"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    tooltip_fields.append("band_label")
    tooltip_aliases.append("Band width (p95-p05)")
if P_TOP_COL in gdf.columns:
    plot_gdf["p_top_label"] = plot_gdf[P_TOP_COL].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
    tooltip_fields.append("p_top_label")
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


# Subzones with no value: near-white with a dashed outline, so they read as
# "left out" rather than blending into the grey basemap.
NO_VALUE_STYLE = {"fillColor": "#fbfbfb", "fillOpacity": 0.9, "color": "#6e6e6e", "weight": 1, "dashArray": "3 3"}
no_value_swatch = (
    f' &nbsp; <span style="display:inline-block;width:12px;height:12px;background:{NO_VALUE_STYLE["fillColor"]};'
    f'border:1px dashed {NO_VALUE_STYLE["color"]};margin-right:4px;vertical-align:middle"></span>'
    + ("Not ranked" if lens_key == "residents" and draw_col == "priority_score" else "No value")
    if len(no_value_gdf) else ""
)

# The legend is drawn above the map (in the fixed notes container) rather than
# on it: branca's on-map legend overlaps its tick labels with its caption.
if show_top_only:
    shown = [s for s in TOP_STATUS_STYLE if (s == f"In top {TOP_N}") == (custom_weights is None)]
    notes.markdown(" &nbsp; ".join(
        f'<span style="display:inline-block;width:12px;height:12px;background:{TOP_STATUS_STYLE[s][0]};'
        f'{"border:1.5px dashed #222;" if TOP_STATUS_STYLE[s][2] else ""}margin-right:4px;vertical-align:middle"></span>{s}'
        for s in shown
    ) + " &nbsp; <span style='color:#888'>(other subzones faded)</span>" + no_value_swatch, unsafe_allow_html=True)
else:
    fmt = (lambda v: f"{v:.0%}") if draw_col == P_TOP_COL else (lambda v: f"{v:.2f}")
    stops = ", ".join(colormap(vmin + (vmax - vmin) * i / 8) for i in range(9))
    notes.markdown(
        f'<span style="font-size:0.9em">{color_by} &nbsp; {fmt(vmin)} '
        f'<span style="display:inline-block;width:220px;height:12px;vertical-align:middle;border-radius:2px;'
        f'background:linear-gradient(to right, {stops})"></span> {fmt(vmax)}{no_value_swatch}</span>',
        unsafe_allow_html=True,
    )

# The subzones go in a FeatureGroup passed separately, so a slider move swaps
# only that layer: the base map (and its zoom/pan) stays put. The key changes
# with the view and colour option.
m = folium.Map(location=SG_CENTER, zoom_start=11, tiles="Esri.WorldGrayCanvas")
subzone_layer = folium.FeatureGroup(name="Subzones")
folium.GeoJson(
    plot_gdf.__geo_interface__, style_function=_style,
    tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases),
).add_to(subzone_layer)
if len(no_value_gdf):
    folium.GeoJson(
        no_value_gdf[["subzone_name", "no_value_reason", "geometry"]].__geo_interface__,
        style_function=lambda _feature: NO_VALUE_STYLE,
        tooltip=folium.GeoJsonTooltip(fields=["subzone_name", "no_value_reason"], aliases=["Subzone", "Status"]),
    ).add_to(subzone_layer)

MAP_HEIGHT = 600
map_col, panel_col = st.columns([2.2, 1], gap="medium")
with map_col:
    map_state = st_folium(
        m, key=f"island_map_{lens_key}_{value_col}", feature_group_to_add=subzone_layer,
        height=MAP_HEIGHT, use_container_width=True, returned_objects=["last_active_drawing"],
    )


def _select(subzone_id):
    st.session_state["selected_subzone_id"] = subzone_id


# st_folium keeps returning the LAST clicked feature on every rerun, so a click
# only counts when it differs from the previous one; otherwise it would undo a
# subzone picked from the search box or the list. No-value (dashed) subzones carry
# only a name and reason, no SUBZONE_ID_PROPERTY, so clicking one selects nothing.
clicked = (map_state or {}).get("last_active_drawing") or {}
clicked_id = clicked.get("properties", {}).get(SUBZONE_ID_PROPERTY)
if clicked_id and clicked_id != st.session_state.get("map_last_click"):
    st.session_state["map_last_click"] = clicked_id
    _select(clicked_id)

ranked_gdf = gdf[gdf["priority_score"].notna()].sort_values("priority_score", ascending=False)
ranked_ids = ranked_gdf["subzone_id"].tolist()
search_ids = sorted(ranked_ids, key=subzone_name)
TOP_TABLE_KEY = f"map_top_table_{lens_key}"


def _clear_selection():
    st.session_state["selected_subzone_id"] = None
    st.session_state.pop(TOP_TABLE_KEY, None)  # also forget the list's highlighted row


def _pillar_bars(subzone_id) -> str:
    """One bar per pillar: where this subzone sits between the lowest (0) and
    highest (1) subzone in this view, as in the breakdown page's chart."""
    if pillar_df is None:
        return ""
    match = pillar_df.index[pillar_df["subzone_id"] == subzone_id]
    if not len(match):
        return ""
    norms = {
        "exposure": normalize(pillar_df[REFERENCE_VARIANT]),
        "sensitivity": normalize(pillar_df["sensitivity_raw"]),
        "adaptive_deficit": normalize(1 - pillar_df["greenery_fraction"]),
    }
    rows = []
    for key in saved_weights:
        value = float(norms[key].loc[match[0]])
        rows.append(
            f'<div style="margin:6px 0 2px;font-size:0.85em">{PILLAR_LABELS[key]} '
            f'<span style="float:right;opacity:0.7">{value:.2f}</span></div>'
            f'<div style="background:rgba(128,128,128,0.18);border-radius:4px;height:8px">'
            f'<div style="width:{value * 100:.0f}%;background:{PILLAR_COLORS[key]};height:8px;border-radius:4px"></div></div>'
        )
    return "".join(rows)


with panel_col:
    # The search box mirrors the selection (set before it is drawn), and a pick
    # in it updates the selection through its callback, which runs before the
    # script does; so a map click, a list row or "Show the top N" all reach it.
    selected = st.session_state.get("selected_subzone_id")
    st.session_state["map_search"] = selected if selected in search_ids else None
    st.selectbox(
        "Find a subzone", search_ids, format_func=subzone_name, key="map_search", placeholder="Search by name…",
        on_change=lambda: _select(st.session_state["map_search"]),
    )

    if selected in ranked_ids:
        row = ranked_gdf.loc[ranked_gdf["subzone_id"] == selected].iloc[0]
        with st.container(border=True):
            st.markdown(f"#### {subzone_name(selected)}")
            facts = [f"Rank **{row['rank_label']}**", f"Score **{row['priority_score']:.2f}**"]
            if custom_weights is None and P_TOP_COL in row.index and pd.notna(row[P_TOP_COL]):
                facts.append(f"**{row[P_TOP_COL]:.0%}** chance of top {TOP_N}")
            elif custom_weights is not None:
                facts.append(row["top_status"] if row["top_status"] != "Not in top" else f"Not in top {TOP_N}")
            st.markdown(" · ".join(facts))
            bars = _pillar_bars(selected)
            if bars:
                st.markdown(bars, unsafe_allow_html=True)
                st.caption("0 = lowest in this view, 1 = highest.")
            if st.button("Details", icon=":material/arrow_forward:", type="primary", width="stretch"):
                st.switch_page("pages/3_Subzone_Breakdown.py")
        st.button(f"Show the top {TOP_N}", on_click=_clear_selection, width="stretch")
    else:
        if selected:
            st.caption(f"{subzone_name(selected)} isn't ranked in this view.")
        st.markdown(f"**Top {TOP_N}** · click a row, or a subzone on the map")
        top_table = ranked_gdf.head(TOP_N)
        top_cols = ["subzone_name", "priority_score"] + ([P_TOP_COL] if P_TOP_COL in top_table.columns else [])
        if custom_weights is not None:
            top_cols.append("top_status")
        table_state = st.dataframe(
            top_table[top_cols], hide_index=True, width="stretch", height=MAP_HEIGHT - 130,
            key=TOP_TABLE_KEY, on_select="rerun", selection_mode="single-row",
            column_config={
                "subzone_name": st.column_config.TextColumn("Subzone", width=120),
                "priority_score": st.column_config.NumberColumn("Score", format="%.2f", width=50),
                P_TOP_COL: st.column_config.ProgressColumn(
                    "Chance", min_value=0.0, max_value=1.0, format="percent", width=70,
                    help=f"Chance of staying in the top {TOP_N} across bootstrap draws."),
                "top_status": st.column_config.TextColumn("vs saved weights", width="small"),
            },
        )
        chosen_rows = table_state.selection.rows if table_state else []
        if chosen_rows:
            _select(top_table["subzone_id"].iloc[chosen_rows[0]])
            st.rerun()
