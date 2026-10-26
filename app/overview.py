"""Landing page (S7): what the system found, before how it was built. Shows
the three pillars, a few headline numbers and the top 5 subzones from the
production (residents-view) score, each linking to its breakdown. The
pipeline-status checklist that used to be the whole landing page is kept,
collapsed, under "Developer".

Registered as the default page by app/Home.py's st.navigation.
Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app.ui import PILLAR_COLORS, PILLAR_LABELS, subzone_name
from config.settings import (
    CNN_MODEL_SAVE_PATH, INTERIM_DIR, LANDCOVER_PRODUCTION_MODEL, LANDCOVER_RASTER_PATH, MIN_RESIDENTS_FOR_RANKING,
    PROCESSED_DIR, TOP_N,
)
from src.priority_score.lenses import DEFAULT_LENS, LENSES
from validation.score_validation.decision_impact import noise_floor

LENS = LENSES[DEFAULT_LENS]
P_TOP_COL = f"p_top{TOP_N}"
PILLAR_BLURBS = {
    "exposure": "How hot the ground gets: Landsat land surface temperature.",
    "sensitivity": "Who lives there: residents, elderly share and density (SingStat).",
    "adaptive_deficit": "How little shade and greenery there is: share of non-vegetated land.",
}

st.title("Urban Heat & Cooling-Priority Mapping — Singapore")
st.markdown(
    "Which Singapore subzones should be cooled first? This system combines satellite heat, census data and "
    "land-cover maps into one **cooling-priority score** per URA subzone, and says how sure it is of each rank."
)
st.caption("NUS-ISS Practice Module, Graduate Certificate in Pattern Recognition Systems")

# --- The three pillars --------------------------------------------------------
pillar_cols = st.columns(3)
for col, key in zip(pillar_cols, PILLAR_LABELS):
    with col.container(border=True, height="stretch"):  # "stretch": every card matches the tallest in the row
        st.markdown(
            f'<div style="border-left:6px solid {PILLAR_COLORS[key]};padding-left:10px">'
            f'<b>{PILLAR_LABELS[key]}</b><br><span style="font-size:0.9em;opacity:0.8">{PILLAR_BLURBS[key]}</span></div>',
            unsafe_allow_html=True,
        )
st.caption(
    "Each pillar is scaled 0–1 and combined with data-derived (PCA) weights into a priority score: higher means "
    "hotter, more vulnerable residents and less greenery."
)


@st.cache_data
def load_headline(score_path: str, bands_path: str):
    score_df = pd.read_csv(score_path).sort_values("priority_score", ascending=False).reset_index(drop=True)
    bands_df = pd.read_csv(bands_path) if Path(bands_path).exists() else None
    return score_df, bands_df


if not LENS["score_csv"].exists():
    st.warning("The priority score hasn't been built yet, so there are no findings to show.")
else:
    score_df, bands_df = load_headline(str(LENS["score_csv"]), str(LENS["bands_csv"]))
    n_all = len(pd.read_csv(LENSES["all_places"]["score_csv"])) if LENSES["all_places"]["score_csv"].exists() else None

    # --- Headline numbers -------------------------------------------------------
    st.subheader("Key findings")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Subzones ranked" + (f" (of {n_all})" if n_all else ""), f"{len(score_df)}",
        help=f"The residents view ranks subzones with at least {MIN_RESIDENTS_FOR_RANKING} residents. "
             f"'{LENSES['all_places']['label']}' on the Island Map ranks every subzone.",
    )
    k2.metric("Highest priority", subzone_name(score_df.loc[0, "subzone_id"]),
              help=f"Priority score {score_df.loc[0, 'priority_score']:.2f}, the highest in the residents view.")
    if bands_df is not None and P_TOP_COL in bands_df.columns:
        floor = noise_floor(bands_df, TOP_N)
        top_p = bands_df.nlargest(TOP_N, "priority_score_point")[P_TOP_COL]
        k3.metric(
            f"Top-{TOP_N} swappable by noise", f"≈{floor:.0f}",
            help="Expected number of today's top-20 subzones that measurement uncertainty in heat and greenery "
                 "would replace on its own (bootstrap). Read the top 20 as a shortlist, not an exact order.",
        )
        k4.metric(
            f"Top-{TOP_N} likely to stay", f"{int((top_p >= 0.5).sum())}",
            help=f"Subzones in the top {TOP_N} whose chance of staying there across bootstrap draws is at least 50%.",
        )

    # --- Top 5 ----------------------------------------------------------------
    st.subheader("Top 5 subzones to cool first")
    st.caption(f"{LENS['label']}. Open any one to see why it ranks where it does.")
    top5 = score_df.head(5)
    if bands_df is not None and P_TOP_COL in bands_df.columns:
        top5 = top5.merge(bands_df[["subzone_id", P_TOP_COL]], on="subzone_id", how="left")
    for rank, row in enumerate(top5.itertuples(index=False), start=1):
        with st.container(border=True):
            c_rank, c_name, c_score, c_chance, c_open = st.columns([0.4, 2.2, 1.1, 2.6, 1.1], vertical_alignment="center")
            c_rank.markdown(f'<span style="font-size:1.6em;font-weight:700">{rank}</span>', unsafe_allow_html=True)
            c_name.markdown(f"**{subzone_name(row.subzone_id)}**")
            c_score.markdown(f"Score **{row.priority_score:.2f}**")
            p_top = getattr(row, P_TOP_COL, None)
            if p_top is not None and pd.notna(p_top):
                c_chance.progress(float(p_top), text=f"{p_top:.0%} chance to stay in top {TOP_N}")
            if c_open.button("Details", icon=":material/arrow_forward:", key=f"overview_open_{row.subzone_id}", width="stretch"):
                st.session_state["selected_subzone_id"] = row.subzone_id
                st.session_state["priority_lens"] = DEFAULT_LENS
                st.switch_page("pages/3_Subzone_Breakdown.py")

# --- Where to go next -----------------------------------------------------------
st.subheader("Explore")
EXPLORE_CARDS = [
    ("pages/2_Island_Map.py", "Island Map", ":material/map:",
     "See every subzone's priority, re-weight the score, and check how stable the top 20 is."),
    ("pages/3_Subzone_Breakdown.py", "Subzone Breakdown", ":material/bar_chart:",
     "Why one subzone ranks where it does: pillars, uncertainty, land cover, hotspot type."),
    ("pages/4_Counterfactual_Greening.py", "Counterfactual Greening", ":material/park:",
     "What if a subzone were greener? Two heat models predict the cooling."),
    ("pages/5_Validation_Dashboard.py", "Validation Dashboard", ":material/verified:",
     "How each model and choice was checked, and which ones actually move the ranking."),
]
# Two per row, so the page names fit on one line.
for row_start in range(0, len(EXPLORE_CARDS), 2):
    for col, (page, label, icon, blurb) in zip(st.columns(2), EXPLORE_CARDS[row_start:row_start + 2]):
        with col.container(border=True, height="stretch"):
            st.page_link(page, label=label, icon=icon)
            st.caption(blurb)

# --- Developer: pipeline status ---------------------------------------------------
checks = [
    ("Heat-layer variants", INTERIM_DIR / "heat_variants_subzone.csv"),
    ("Sensitivity pillar", INTERIM_DIR / "sensitivity_pillar.csv"),
    ("Adaptive-capacity pillar", INTERIM_DIR / "adaptive_capacity_pillar.csv"),
    ("NEA held-out LST", INTERIM_DIR / "nea_heldout_lst.csv"),
    ("MODIS held-out LST", INTERIM_DIR / "modis_heldout_lst.csv"),
    ("Land-change flags", INTERIM_DIR / "land_change_flags.csv"),
    ("Validation sample (drawn)", INTERIM_DIR / "validation_sample" / "validation_sample_300.csv"),
    ("Validation sample (labeled)", INTERIM_DIR / "validation_sample" / "validation_sample_300_labeled.csv"),
    ("RF land cover raster", PROCESSED_DIR / "landcover" / "rf_landcover.tif"),
    ("U-Net land cover raster", PROCESSED_DIR / "landcover" / "unet_landcover.tif"),
    ("Land-cover hybrid raster", PROCESSED_DIR / "landcover" / "hybrid_landcover.tif"),
    (f"Production land-cover map ({LANDCOVER_PRODUCTION_MODEL})", LANDCOVER_RASTER_PATH),
    ("S4 hotspot clusters", PROCESSED_DIR / "hotspot_clusters.csv"),
    ("S5 XGBoost heat model", PROCESSED_DIR / "heat_model" / "xgb_model.pkl"),
    ("S5 CNN heat model", CNN_MODEL_SAVE_PATH),
    ("Production priority score", PROCESSED_DIR / "priority_score.csv"),
    ("S6 confidence bands", PROCESSED_DIR / "priority_score_confidence_bands.csv"),
    ("All-places priority score", PROCESSED_DIR / "priority_score_all_places.csv"),
    ("All-places confidence bands", PROCESSED_DIR / "priority_score_all_places_confidence_bands.csv"),
]
n_ready = sum(path.exists() for _label, path in checks)
with st.expander(f"Developer: pipeline status ({n_ready} of {len(checks)} outputs built)"):
    for label, path in checks:
        status = "✅ ready" if path.exists() else "⬜ not yet built"
        st.write(f"{status} — **{label}** (`{path}`)")
