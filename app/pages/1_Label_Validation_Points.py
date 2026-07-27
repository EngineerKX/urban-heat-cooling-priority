"""Interactive joint hand-labeling tool for the 200-point stratified
validation sample. Rebuilds label_points_interactive.ipynb (Track B / SB2)
as a Streamlit page (folium + streamlit-folium) instead of a Colab
ipyleaflet widget, per the migration plan.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Point

from config.settings import INTERIM_DIR, RANDOM_SEED, SG_CENTER

SAMPLE_DIR = INTERIM_DIR / "validation_sample"
SB1_OUTPUT_CSV = SAMPLE_DIR / "validation_sample_200.csv"
WORK_CSV = SAMPLE_DIR / "validation_sample_200_labeling_progress.csv"
FINAL_CSV = SAMPLE_DIR / "validation_sample_200_labeled.csv"
FINAL_GEOJSON = SAMPLE_DIR / "validation_sample_200_labeled.geojson"

TOTAL_POINTS = 200

CLASS_OPTIONS = [
    ("-- select a label --", ""),
    ("🌿 Vegetation — trees, shrubs, grass, crops, mangroves", "vegetation"),
    ("🏢 Built-up — buildings, roads, pavement, construction sites", "built_up"),
    ("🟤 Bare — exposed soil, sand, cleared land", "bare"),
    ("💧 Water — pond/reservoir/drain edge, mixed pixel, or misclassified point", "water"),
    ("❓ Uncertain — flag for team review", "uncertain_flag_for_review"),
]
CONFIDENCE_OPTIONS = [
    ("-- select confidence --", ""),
    ("😕 Low — genuinely unsure, borderline case", "1-low"),
    ("🙂 Medium — fairly confident, minor doubt", "2-medium"),
    ("😄 High — clearly obvious, no doubt", "3-high"),
]
REQUIRED_COLUMNS = ["point_id", "lon", "lat", "agreed_label", "confidence", "notes"]

_MARKER_COLOR = {"": "orange", "uncertain_flag_for_review": "gray"}


def _class_label_for(value: str) -> str:
    for label, v in CLASS_OPTIONS:
        if v == value:
            return label
    return CLASS_OPTIONS[0][0]


def _confidence_label_for(value: str) -> str:
    for label, v in CONFIDENCE_OPTIONS:
        if v == value:
            return label
    return CONFIDENCE_OPTIONS[0][0]


def _generate_fallback_sample() -> pd.DataFrame:
    """Self-contained fallback if generate_validation_sample.py hasn't been
    run yet: plain random points within Singapore's real boundary. NOT
    stratified by WorldCover class (that needs Earth Engine) — flagged
    clearly. Running scripts/generate_validation_sample.py overwrites this
    file with a properly stratified sample; already-saved labeling progress
    is unaffected since it lives in a separate file."""
    import numpy as np

    from src.ingest.subzones import fetch_subzones_geojson

    st.warning(
        "No validation sample found — generating a fallback (plain random, NOT "
        "WorldCover-stratified) sample here instead. Run "
        "`python scripts/generate_validation_sample.py` for a properly stratified sample; "
        "it will overwrite this file without touching your labeling progress."
    )
    geojson = fetch_subzones_geojson()
    boundary = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326").geometry.union_all()

    rng = np.random.default_rng(RANDOM_SEED)
    minx, miny, maxx, maxy = boundary.bounds
    rows = []
    attempts = 0
    while len(rows) < TOTAL_POINTS and attempts < TOTAL_POINTS * 200:
        attempts += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if boundary.contains(Point(x, y)):
            rows.append({
                "point_id": f"P{len(rows) + 1:04d}", "lon": x, "lat": y,
                "worldcover_class_name": "unknown", "worldcover_class_raw_name": "unknown",
                "agreed_label": "", "confidence": "", "notes": "",
            })
    df = pd.DataFrame(rows)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SB1_OUTPUT_CSV, index=False)
    return df


def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Loaded CSV is missing required columns: {missing}")
    for c in ("agreed_label", "confidence", "notes"):
        df[c] = df[c].fillna("").astype(str)
    return df


def load_default() -> pd.DataFrame:
    if WORK_CSV.exists():
        df = pd.read_csv(WORK_CSV)
    elif SB1_OUTPUT_CSV.exists():
        df = pd.read_csv(SB1_OUTPUT_CSV)
    else:
        df = _generate_fallback_sample()
    return _validate_columns(df)


def save_progress(df: pd.DataFrame):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WORK_CSV, index=False)


def export_final(df: pd.DataFrame):
    n_unlabeled = (df["agreed_label"].str.strip() == "").sum()
    df.to_csv(FINAL_CSV, index=False)
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")
    gdf.to_file(FINAL_GEOJSON, driver="GeoJSON")
    if n_unlabeled:
        st.warning(f"{n_unlabeled} point(s) still unlabeled — exported anyway as a checkpoint.")
    else:
        st.success("All points labeled and exported.")
    st.write(f"Wrote:\n- `{FINAL_CSV}`\n- `{FINAL_GEOJSON}`")


st.set_page_config(page_title="Label Validation Points", page_icon="🌿", layout="wide")
st.title("🌿 Label Validation Points")
st.caption(
    "Click a marker, choose the class that matches what's actually there (WorldCover reference "
    "is context only — never the answer), then Save. Progress autosaves after every save."
)

if "points_df" not in st.session_state:
    st.session_state.points_df = load_default()
if "selected_point_id" not in st.session_state:
    st.session_state.selected_point_id = None

points_df = st.session_state.points_df
n_labeled = (points_df["agreed_label"].str.strip() != "").sum()
st.progress(n_labeled / len(points_df) if len(points_df) else 0, text=f"{n_labeled}/{len(points_df)} labeled")

col_map, col_panel = st.columns([2, 1])

with col_map:
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=SG_CENTER, zoom_start=12, tiles="Esri.WorldImagery", attr="Esri")
    for _, row in points_df.iterrows():
        labeled = bool(str(row["agreed_label"]).strip())
        color = _MARKER_COLOR.get(row["agreed_label"], "green" if labeled else "orange")
        folium.CircleMarker(
            location=(row["lat"], row["lon"]), radius=6, color=color, fill_color=color,
            fill_opacity=0.85, weight=2, tooltip=row["point_id"],
        ).add_to(m)

    map_state = st_folium(m, height=560, use_container_width=True, returned_objects=["last_object_clicked_tooltip"])
    clicked_id = map_state.get("last_object_clicked_tooltip")
    if clicked_id and clicked_id != st.session_state.selected_point_id:
        st.session_state.selected_point_id = clicked_id
        st.rerun()

with col_panel:
    selected_id = st.session_state.selected_point_id
    if selected_id is None:
        st.info("Click a marker on the map to select a point.")
    else:
        matches = points_df.index[points_df["point_id"] == selected_id]
        if len(matches) == 0:
            st.warning("Selected point not found — click another marker.")
        else:
            idx = matches[0]
            row = points_df.loc[idx]
            st.subheader(row["point_id"])
            st.caption(f"lon={row['lon']:.5f}, lat={row['lat']:.5f}")

            wc_name = row.get("worldcover_class_name", "unknown")
            wc_raw = row.get("worldcover_class_raw_name", "unknown")
            st.caption(f"WorldCover reference (context only, NOT the answer): {wc_name} ({wc_raw})")

            label_choice = st.selectbox(
                "Label", options=[l for l, _ in CLASS_OPTIONS],
                index=[l for l, _ in CLASS_OPTIONS].index(_class_label_for(row["agreed_label"])),
                key=f"label_{selected_id}",
            )
            confidence_choice = st.selectbox(
                "Confidence", options=[l for l, _ in CONFIDENCE_OPTIONS],
                index=[l for l, _ in CONFIDENCE_OPTIONS].index(_confidence_label_for(row["confidence"])),
                key=f"conf_{selected_id}",
            )
            notes_choice = st.text_area("Notes", value=row["notes"], key=f"notes_{selected_id}")

            if st.button("💾 Save label", type="primary"):
                label_value = dict((l, v) for l, v in CLASS_OPTIONS)[label_choice]
                confidence_value = dict((l, v) for l, v in CONFIDENCE_OPTIONS)[confidence_choice]
                points_df.loc[idx, "agreed_label"] = label_value
                points_df.loc[idx, "confidence"] = confidence_value
                points_df.loc[idx, "notes"] = notes_choice
                st.session_state.points_df = points_df
                save_progress(points_df)
                st.success(f"Saved {row['point_id']} -> '{label_value}'.")
                st.rerun()

    st.divider()
    unlabeled_ids = points_df.loc[points_df["agreed_label"].str.strip() == "", "point_id"].tolist()
    if st.button("⏭️ Jump to next unlabeled point", disabled=not unlabeled_ids):
        st.session_state.selected_point_id = unlabeled_ids[0]
        st.rerun()
    if st.button("💾 Save all progress now"):
        save_progress(points_df)
        st.success(f"Saved to {WORK_CSV}")
    if st.button("📤 Export final CSV + GeoJSON"):
        export_final(points_df)
