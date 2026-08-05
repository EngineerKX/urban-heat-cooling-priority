"""S7 validation dashboard: renders every validation artifact already on
disk (not just a checkmark like Home.py's status list). Each section
checks its own inputs and shows a "not yet built, run <script>" caption
instead of failing if that stage hasn't been run yet — same graceful-
degradation pattern as app/pages/4_Counterfactual_Greening.py.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import spearmanr

from config.settings import CNN_MODEL_SAVE_PATH, DIAGNOSTICS_DIR, INTERIM_DIR, PROCESSED_DIR, VARIANT_COLUMNS
from validation.score_validation.rank_impact import rmse_vs_heldout

st.set_page_config(page_title="Validation Dashboard — Urban Heat & Cooling Priority", page_icon="✅", layout="wide")
st.title("✅ Validation dashboard")
st.caption("Every validated output already on disk, in one place — not just a build/not-built checkmark.")

# --- Week-1 gates -----------------------------------------------------------
with st.expander("Week-1 gates (G1–G5)", expanded=True):
    gates_path = DIAGNOSTICS_DIR / "week1_gate_summary.csv"
    if gates_path.exists():
        gates_df = pd.read_csv(gates_path)
        st.dataframe(gates_df, use_container_width=True, hide_index=True)
        st.caption(
            "G5 (labeling agreement) passed with Cohen's kappa = 0.676 (\"substantial agreement\", 20 shared "
            "points, independently labeled by both team members) — see docs/Progress_Log_2026-08-04_0237.md. "
            "Not recomputed live here; validation/input_validation/labeling_agreement.py has no persisted "
            "per-run output file yet."
        )
    else:
        st.caption(f"`{gates_path}` not found — run `python scripts/run_week1_gates.py` first.")

# --- Land-cover classifiers ---------------------------------------------------
with st.expander("Land-cover classifiers: RF vs. U-Net vs. ensemble", expanded=True):
    eval_dir = PROCESSED_DIR / "landcover" / "evaluation"
    comparison_path = eval_dir / "comparison_table.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        confusion_path = eval_dir / "confusion_matrix_ensemble.csv"
        if confusion_path.exists():
            st.markdown("**Ensemble confusion matrix**")
            confusion_df = pd.read_csv(confusion_path).rename(columns={"Unnamed: 0": "actual \\ predicted"})
            st.dataframe(confusion_df, use_container_width=True, hide_index=True)
    else:
        st.caption(f"`{comparison_path}` not found — run `python scripts/evaluate_landcover_classifiers.py` first.")

# --- Rank-impact / heat-variant ablation -------------------------------------
with st.expander("Rank-impact ablation (C3: heat-variant choice vs. score)", expanded=False):
    for weighting in ("pca", "equal"):
        path = PROCESSED_DIR / f"rank_impact_results_{weighting}.csv"
        if path.exists():
            st.markdown(f"**Weighting: {weighting}**")
            st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
        else:
            st.caption(f"`{path}` not found — run `python scripts/build_priority_score.py` first.")

    heat_variant_plot = DIAGNOSTICS_DIR / "heat_variant_diagnostic.png"
    if heat_variant_plot.exists():
        st.image(str(heat_variant_plot), caption="Heat-variant diagnostic")
    else:
        st.caption(f"`{heat_variant_plot}` not found — run `python scripts/diagnose_heat_variants.py` first.")

# --- Land-change diagnostic ---------------------------------------------------
with st.expander("Land-change diagnostic (multi-year composite risk)", expanded=False):
    land_change_path = INTERIM_DIR / "land_change_flags.csv"
    if land_change_path.exists():
        land_change_df = pd.read_csv(land_change_path)
        st.bar_chart(land_change_df["status"].value_counts())
        st.dataframe(land_change_df, use_container_width=True, hide_index=True)
    else:
        st.caption(f"`{land_change_path}` not found — run `python scripts/diagnose_land_change.py` first.")

# --- S4 hotspot clusters ------------------------------------------------------
with st.expander("S4 — hotspot-typology clusters", expanded=False):
    profile_path = PROCESSED_DIR / "hotspot_cluster_profile.csv"
    cluster_plot = DIAGNOSTICS_DIR / "hotspot_cluster_profile.png"
    if profile_path.exists():
        st.dataframe(pd.read_csv(profile_path), use_container_width=True, hide_index=True)
        st.caption(
            "Land-cover-coherence check: whether the hottest cluster is also the most built-up / least-"
            "vegetated (fractions are NOT clustering inputs — see src/hotspots/cluster.py). Run "
            "`python scripts/diagnose_hotspot_clusters.py` to recompute this printed check and the plot below."
        )
        if cluster_plot.exists():
            st.image(str(cluster_plot), caption="Per-cluster feature profile")
    else:
        st.caption(f"`{profile_path}` not found — run `python scripts/build_hotspot_clusters.py` first.")

# --- S5 heat model -------------------------------------------------------------
with st.expander("S5 — XGBoost + CNN heat model (C2)", expanded=False):
    importance_path = PROCESSED_DIR / "heat_model" / "xgb_feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        fig = px.bar(importance_df, x=importance_df.columns[-1], y=importance_df.columns[0], orientation="h",
                     title="XGBoost feature importance")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"`{importance_path}` not found — run `python scripts/train_heat_model_xgboost.py` first.")

    if CNN_MODEL_SAVE_PATH.exists():
        st.caption("Live XGBoost-vs-CNN counterfactual cross-check (direction agreement) is on the "
                   "**Counterfactual Greening** page.")
    else:
        st.caption(
            "CNN heat model not trained yet — train it via `notebooks/colab_training/train_heat_cnn.ipynb`, then "
            "`python scripts/pull_models.py --model cnn`, to enable the CNN half on the "
            "**Counterfactual Greening** page."
        )

# --- Secondary LST cross-checks (NEA + MODIS) ---------------------------------
with st.expander("Secondary LST cross-checks (NEA air-temp + MODIS)", expanded=False):
    heat_path = INTERIM_DIR / "heat_variants_subzone.csv"
    nea_path = INTERIM_DIR / "nea_heldout_lst.csv"
    modis_path = INTERIM_DIR / "modis_heldout_lst.csv"

    if heat_path.exists() and (nea_path.exists() or modis_path.exists()):
        heat_df = pd.read_csv(heat_path)
        rows = []
        for variant in VARIANT_COLUMNS:
            row = {"variant": variant}
            if nea_path.exists():
                nea_df = pd.read_csv(nea_path)
                row["nea_rmse"] = rmse_vs_heldout(heat_df, variant, nea_df)
            if modis_path.exists():
                modis_df = pd.read_csv(modis_path)
                row["modis_rmse"] = rmse_vs_heldout(heat_df, variant, modis_df)
                merged = heat_df[["subzone_id", variant]].merge(modis_df, on="subzone_id", how="inner")
                row["modis_spearman"] = spearmanr(merged[variant], merged["lst_heldout_c"])[0] if len(merged) > 1 else float("nan")
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "NEA measures air temperature (systematic offset vs. LST); MODIS measures true LST but at 1km "
            "resolution (coarse vs. subzone scale) — different failure modes, which is why both are shown. "
            "Prefer Spearman over RMSE for MODIS; see validation/input_validation/{nea_heldout,modis_heldout}.py."
        )
    else:
        st.caption("Run `python scripts/build_nea_heldout.py` and/or `python scripts/build_modis_heldout.py` first.")

# --- S6 confidence bands -------------------------------------------------------
with st.expander("S6 — calibrated confidence bands", expanded=False):
    bands_path = PROCESSED_DIR / "priority_score_confidence_bands.csv"
    if bands_path.exists():
        bands_df = pd.read_csv(bands_path)
        ranked = bands_df.sort_values("priority_score_point", ascending=False).reset_index(drop=True)
        top20 = ranked.head(20)
        overlaps = sum(
            top20.iloc[i]["priority_score_p05"] <= top20.iloc[i + 1]["priority_score_p95"]
            for i in range(len(top20) - 1)
        )
        st.metric("Mean band width (p95−p05)", f"{bands_df['band_width'].mean():.3f}")
        st.metric("Overlapping adjacent pairs in top 20", f"{overlaps} / {len(top20) - 1}")
        st.dataframe(top20, use_container_width=True, hide_index=True)
        st.caption(
            "Bootstrapped from NEA-heldout RMSE (exposure) + land-cover ensemble recall SE (adaptive capacity) "
            "— NOT the sensitivity pillar (no validation-error estimate exists for it). The exposure RMSE also "
            "mixes real noise with the systematic LST-vs-air-temperature offset, so read these as a pessimistic "
            "upper bound on rank uncertainty. See validation/score_validation/confidence_bands.py's docstring."
        )
    else:
        st.caption(f"`{bands_path}` not found — run `python scripts/build_priority_score_confidence_bands.py` first.")
