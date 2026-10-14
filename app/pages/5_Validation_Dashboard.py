"""S7 validation dashboard: renders every validation artifact already on
disk (not just a checkmark like the overview's status list). Each section
checks its own inputs and shows a "not yet built, run <script>" caption
instead of failing if that stage hasn't been run yet — same graceful-
degradation pattern as app/pages/4_Counterfactual_Greening.py.

Laid out as tabs: a summary (headline numbers + the decision-impact chart,
contribution 3), then one tab per component. Long method notes sit in
"Method notes" expanders under the result they qualify.

Land-cover results come from the final evaluation batch exported from
MLflow (data/processed/landcover/evaluation/final_batch_*.csv, see
final_batch_SOURCE.txt), not the older comparison_table.csv layout.

Run the whole app with: streamlit run app/Home.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui import CLUSTER_NAMES, LANDCOVER_LABELS, subzone_name
from config.settings import (
    CNN_MODEL_SAVE_PATH, DIAGNOSTICS_DIR, EXPOSURE_NOISE_SOURCE, INTERIM_DIR, LANDCOVER_LABEL_SOURCE,
    LANDCOVER_PRODUCTION_MODEL, PROCESSED_DIR, TOP_N, VARIANT_COLUMNS,
)
from validation.score_validation.decision_impact import build_decision_impact_summary, noise_floor
from validation.score_validation.rank_impact import heldout_agreement

# Display names for the LANDCOVER_PRODUCTION_MODEL keys.
_MODEL_NAMES = {"rf": "RF", "unet": "U-Net", "hybrid": "hybrid"}
PRODUCTION_MODEL_NAME = _MODEL_NAMES[LANDCOVER_PRODUCTION_MODEL]
LABEL_SOURCE_NAME = {"dynamicworld": "Dynamic World", "worldcover": "WorldCover"}[LANDCOVER_LABEL_SOURCE]

LANDCOVER_EVAL_DIR = PROCESSED_DIR / "landcover" / "evaluation"
# The headline runs of the final batch, one per (model, label source): the
# same runs the McNemar tests compare (mcnemar_counts.csv).
LANDCOVER_HEADLINE_RUNS = {
    ("RF", "Dynamic World"): "rf_dw_raw", ("U-Net", "Dynamic World"): "unet_dw",
    ("Hybrid", "Dynamic World"): "hybrid_dw_rfraw", ("RF", "WorldCover"): "rf_worldcover",
    ("U-Net", "WorldCover"): "unet_worldcover", ("Hybrid", "WorldCover"): "hybrid_worldcover",
}
NOISE_COLOR, WITHIN_COLOR, EXCEEDS_COLOR = "#6e6e6e", "#2166ac", "#b2182b"

st.set_page_config(page_title="Validation Dashboard — Urban Heat & Cooling Priority", page_icon="✅", layout="wide")
st.title("✅ Validation dashboard")
st.caption("How each component was checked, and which methodological choices actually move the ranking.")


def _missing(path, command):
    st.caption(f"Not built yet — `{Path(path).name}` is missing. Run `{command}` to enable this section.")


def _table(df: pd.DataFrame, height="auto", **column_config):
    """st.dataframe with readable headers (snake_case -> words) and floats at
    3 decimals, unless a column gets its own config."""
    config = {
        col: st.column_config.NumberColumn(col.replace("_", " ").capitalize(), format="%.3f")
        if pd.api.types.is_float_dtype(df[col]) else col.replace("_", " ").capitalize()
        for col in df.columns
    }
    config.update(column_config)
    st.dataframe(df, hide_index=True, width="stretch", height=height, column_config=config)


_ACRONYMS = {"ndvi": "NDVI", "ndbi": "NDBI", "lst": "LST"}
_SEASONS = {"dry": "(dry season)", "wet": "(wet season)"}


def _pretty_feature(name: str) -> str:
    """'ndbi_dry' -> 'NDBI (dry season)', 'fraction_built_up' -> 'Built-up share',
    'population_total' -> 'Residents'."""
    name = str(name)
    if name.startswith("fraction_"):
        return f"{LANDCOVER_LABELS.get(name[len('fraction_'):], name)} share"
    if name == "population_total":
        return "Residents"
    words = name.replace("seasonal_delta", "seasonal change").split("_")
    words = [_ACRONYMS.get(w, _SEASONS.get(w, w)) for w in words]
    text = " ".join(words)
    return text[0].upper() + text[1:]


# --- Load what the summary needs ---------------------------------------------------
gates_path = DIAGNOSTICS_DIR / "week1_gate_summary.csv"
gates_df = pd.read_csv(gates_path) if gates_path.exists() else None

metrics_path = LANDCOVER_EVAL_DIR / "final_batch_metrics.csv"
lc_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
production_run = lc_metrics[lc_metrics["selected"] == "production_landcover"] if lc_metrics is not None else None

decision_inputs = {
    "weighting_df": PROCESSED_DIR / "weighting_comparison.csv",
    "rank_impact_df": PROCESSED_DIR / "rank_impact_results_pca.csv",
    "spec_df": PROCESSED_DIR / "sensitivity_spec_comparison.csv",
    "swap_df": PROCESSED_DIR / "landcover_swap_comparison.csv",
    "bands_df": PROCESSED_DIR / "priority_score_confidence_bands.csv",
}
missing_decision_inputs = [p.name for p in decision_inputs.values() if not p.exists()]
summary_df = None if missing_decision_inputs else build_decision_impact_summary(
    **{name: pd.read_csv(csv_path) for name, csv_path in decision_inputs.items()}, top_n=TOP_N,
)

bands_path = PROCESSED_DIR / "priority_score_confidence_bands.csv"
bands_df = pd.read_csv(bands_path) if bands_path.exists() else None
land_change_path = INTERIM_DIR / "land_change_flags.csv"
land_change_df = pd.read_csv(land_change_path) if land_change_path.exists() else None

(tab_summary, tab_score, tab_landcover, tab_heat, tab_hotspots, tab_inputs) = st.tabs([
    "Summary", "Priority score", "Land cover", "Heat model", "Hotspot types", "Inputs & cross-checks",
])

# =====================================================================================
with tab_summary:
    b1, b2, b3, b4 = st.columns(4)
    if gates_df is not None:
        n_pass = int((gates_df["status"].astype(str).str.upper() == "PASS").sum())
        b1.metric("Week-1 data gates", f"{n_pass} of {len(gates_df)} passed", border=True,
                  help="G1–G5: the data-readiness checks run before modelling (see the Inputs tab).")
    if production_run is not None and not production_run.empty:
        r = production_run.iloc[0]
        b2.metric(f"Land-cover accuracy ({PRODUCTION_MODEL_NAME})", f"{r['accuracy']:.0%}", border=True,
                  help=f"Against {int(r['n_scored'])} hand-labelled points; 95% CI "
                       f"{r['accuracy_ci95_low']:.0%}–{r['accuracy_ci95_high']:.0%}.")
    if summary_df is not None:
        choices = summary_df[summary_df["vs_noise_floor"] != "yardstick"]
        n_within = int((choices["vs_noise_floor"] == "within noise").sum())
        b3.metric("Method choices within noise", f"{n_within} of {len(choices)}", border=True,
                  help=f"Choices that move fewer of the top {TOP_N} than measurement noise does on its own.")
    if bands_df is not None and f"p_top{TOP_N}" in bands_df.columns:
        b4.metric(f"Top-{TOP_N} swappable by noise", f"≈{noise_floor(bands_df, TOP_N):.0f} of {TOP_N}", border=True,
                  help=f"Expected number of today's top {TOP_N} that measurement noise alone would replace (bootstrap).")

    st.subheader(f"Which choices move the top {TOP_N} more than measurement noise does?")
    if summary_df is None:
        st.caption(
            f"Not all inputs exist yet ({', '.join(missing_decision_inputs)}) — run `python scripts/build_priority_score.py`, "
            "`python scripts/build_landcover_swap_comparison.py` and "
            "`python scripts/build_priority_score_confidence_bands.py` first."
        )
    else:
        floor_row = summary_df[summary_df["vs_noise_floor"] == "yardstick"]
        floor = float(floor_row["top_changed_max"].iloc[0]) if not floor_row.empty else None
        choices = summary_df[summary_df["vs_noise_floor"] != "yardstick"].iloc[::-1]  # plotly draws bottom-up
        fig = go.Figure()
        for _, row in choices.iterrows():
            color = EXCEEDS_COLOR if row["vs_noise_floor"] == "exceeds noise" else WITHIN_COLOR
            lo, hi = float(row["top_changed_min"]), float(row["top_changed_max"])
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[row["choice"]] * 2, mode="lines+markers", line=dict(color=color, width=6),
                marker=dict(size=11, color=color),
                hovertemplate=f"{row['choice']}<br>{lo:.0f}–{hi:.0f} of the top {TOP_N} change"
                              f"<br>largest from: {row['largest_change_from']}<extra></extra>",
            ))
        if floor is not None:
            fig.add_vline(x=floor, line_dash="dash", line_color=NOISE_COLOR, line_width=2,
                          annotation_text=f"measurement noise alone ≈ {floor:.1f}", annotation_position="top")
        fig.update_layout(
            height=110 + 60 * len(choices), margin=dict(l=10, r=20, t=40, b=40), showlegend=False,
            xaxis=dict(title=f"Subzones of the top {TOP_N} that change", range=[-0.5, TOP_N + 0.5], dtick=2),
        )
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Each line spans the fewest to the most top-{TOP_N} subzones that move across the alternatives tested for "
            f"one choice, holding everything else fixed. Blue lines left of the dashed line are within noise: for "
            f"planning, that choice is indistinguishable from the data's own uncertainty. A red line would mark a "
            f"choice that needs justifying."
        )
        with st.expander("Method notes and table"):
            _table(summary_df)
            st.caption(
                f"The first row is the yardstick: the expected number of the top {TOP_N} that measurement "
                f"uncertainty in exposure and greenery would replace on its own (bootstrap). Both sides count "
                f"subzones entering or leaving the top {TOP_N}, but the noise figure is an average over draws and "
                f"covers measurement noise only — formula choices (weighting, sensitivity) are what the other rows "
                f"measure."
            )

# =====================================================================================
with tab_score:
    st.subheader("Confidence bands (S6)")
    if bands_df is None:
        _missing(bands_path, "python scripts/build_priority_score_confidence_bands.py")
    else:
        p_col = f"p_top{TOP_N}"
        top = bands_df.sort_values("priority_score_point", ascending=False).reset_index(drop=True).head(TOP_N)
        overlaps = sum(top.iloc[i]["priority_score_p05"] <= top.iloc[i + 1]["priority_score_p95"] for i in range(len(top) - 1))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean band width (p95−p05)", f"{bands_df['band_width'].mean():.3f}")
        c2.metric(f"Overlapping neighbours in top {TOP_N}", f"{overlaps} of {len(top) - 1}")
        if p_col in bands_df.columns:
            c3.metric(f"Top {TOP_N}: ≥90% chance of staying", int((top[p_col] >= 0.9).sum()))
            c4.metric(f"Top {TOP_N}: <50% chance of staying", int((top[p_col] < 0.5).sum()))
        show = top.assign(rank=range(1, len(top) + 1), subzone=top["subzone_id"].map(subzone_name))
        _table(
            show[["rank", "subzone", "priority_score_point", "priority_score_p05", "priority_score_p95", "band_width"]
                 + ([p_col] if p_col in show.columns else [])],
            rank=st.column_config.NumberColumn("Rank", width="small"),
            priority_score_point=st.column_config.NumberColumn("Score", format="%.3f"),
            priority_score_p05=st.column_config.NumberColumn("p05", format="%.3f"),
            priority_score_p95=st.column_config.NumberColumn("p95", format="%.3f"),
            **{p_col: st.column_config.ProgressColumn(f"Chance of top {TOP_N}", min_value=0.0, max_value=1.0, format="percent")},
        )
        with st.expander("Method notes"):
            st.caption(
                f"Bootstrapped from the offset-removed spread of Landsat LST vs the {EXPOSURE_NOISE_SOURCE.upper()} "
                f"held-out source (exposure) + the {PRODUCTION_MODEL_NAME}-map-vs-NDVI-proxy greenery disagreement "
                f"(adaptive capacity) — NOT the sensitivity pillar, whose uncertainty is formula choice, not noise (see "
                f"the specification table below). Exposure noise is an upper bound (it also contains the held-out "
                f"footprint's mismatch with the subzone) and the greenery noise is a rough scale, so read the bands as "
                f"rough, not tight. `{p_col}` is the share of bootstrap draws in which a subzone lands in the top "
                f"{TOP_N}. These are validation-based bands, not statistically calibrated ones. See "
                f"validation/score_validation/confidence_bands.py's docstring."
            )

    st.subheader("Ablations: does each choice change the ranking?")
    for weighting in ("pca", "equal"):
        path = PROCESSED_DIR / f"rank_impact_results_{weighting}.csv"
        st.markdown(f"**Heat-layer variant vs. score — {'PCA' if weighting == 'pca' else 'equal'} weighting (C3)**")
        if path.exists():
            _table(pd.read_csv(path))
        else:
            _missing(path, "python scripts/build_priority_score.py")
    with st.expander("Method notes: heat-layer variants"):
        st.caption(
            f"Each heat layer is compared with the {EXPOSURE_NOISE_SOURCE.upper()} held-out LST table. Offset = "
            f"mean(Landsat − held-out): a systematic gap that can't move any rank. Spread = the std of the residuals "
            f"once that offset is removed. Spearman = rank agreement. RMSE is deliberately not shown — it mostly "
            f"restates the offset. The three layers barely differ because the downscaling (approximately) preserves "
            f"the 30 m mean, so their subzone means are almost identical."
        )

    weighting_path = PROCESSED_DIR / "weighting_comparison.csv"
    membership_path = PROCESSED_DIR / f"weighting_comparison_top{TOP_N}_membership.csv"
    st.markdown("**PCA vs. equal weighting (C1 benchmark)**")
    if weighting_path.exists():
        _table(pd.read_csv(weighting_path))
        if membership_path.exists():
            membership_df = pd.read_csv(membership_path)
            changed_df = membership_df[membership_df["status"] != "both"]
            st.caption(f"Top-{TOP_N} subzones that appear under only one weighting (reference heat layer) — "
                       f"{len(changed_df)} of {len(membership_df)} subzones in either top {TOP_N}.")
            _table(changed_df)
    else:
        _missing(weighting_path, "python scripts/build_priority_score.py")

    spec_path = PROCESSED_DIR / "sensitivity_spec_comparison.csv"
    st.markdown("**Sensitivity specification (formula uncertainty)**")
    if spec_path.exists():
        _table(pd.read_csv(spec_path))
        st.caption(
            f"The sensitivity pillar has no measurement noise to bootstrap (SingStat counts are near-exact); its "
            f"uncertainty is how the counts become a score. Each row changes only that formula and reports how many "
            f"of the top-{TOP_N} subzones move versus the production score."
        )
    else:
        _missing(spec_path, "python scripts/build_priority_score.py")

    swap_path = PROCESSED_DIR / "landcover_swap_comparison.csv"
    st.markdown("**Land-cover classifier swap (does the classifier change the priorities?)**")
    if swap_path.exists():
        swap_df = pd.read_csv(swap_path)
        for col in ("left_top", "entered_top"):
            if col in swap_df.columns:
                swap_df[col] = swap_df[col].fillna("").map(lambda s: "; ".join(subzone_name(z) for z in s.split("; ") if z))
        _table(swap_df, left_top=f"Left the top {TOP_N}", entered_top=f"Entered the top {TOP_N}")
        with st.expander("Method notes: classifier swap"):
            st.caption(
                "The greenery fraction feeding the adaptive-capacity pillar is taken from each classifier's raster in "
                f"turn (exposure, sensitivity and PCA weighting held fixed) and compared with the production "
                f"{PRODUCTION_MODEL_NAME} map. Only the vegetation share reaches the score, so a classifier's errors on "
                "other classes (e.g. bare land) matter only through vegetation. A constant misclassification rate "
                "changes nothing (an affine distortion that the normalisation cancels); only errors that vary between "
                "subzones can move a rank."
            )
    else:
        _missing(swap_path, "python scripts/build_landcover_swap_comparison.py")

    heat_variant_plot = DIAGNOSTICS_DIR / "heat_variant_diagnostic.png"
    if heat_variant_plot.exists():
        st.image(str(heat_variant_plot), caption="Heat-variant diagnostic")

# =====================================================================================
with tab_landcover:
    st.subheader("Land-cover classifiers: RF vs. U-Net vs. hybrid")
    if lc_metrics is None:
        _missing(metrics_path, "python scripts/evaluate_landcover_classifiers.py")
    else:
        rows = []
        for (model, source), run in LANDCOVER_HEADLINE_RUNS.items():
            hit = lc_metrics[lc_metrics["run"] == run]
            if not hit.empty:
                rows.append({"model": model, "labels": source, **hit.iloc[0][
                    ["accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1", "n_scored"]].to_dict()})
        acc = pd.DataFrame(rows)
        fig = go.Figure()
        for source, color in (("Dynamic World", WITHIN_COLOR), ("WorldCover", "#9e9e9e")):
            part = acc[acc["labels"] == source]
            fig.add_trace(go.Bar(
                x=part["model"], y=part["accuracy"], name=f"Trained on {source} labels", marker_color=color,
                error_y=dict(type="data", symmetric=False, array=part["accuracy_ci95_high"] - part["accuracy"],
                             arrayminus=part["accuracy"] - part["accuracy_ci95_low"], color="#333", thickness=1.2),
                text=[f"{v:.0%}" for v in part["accuracy"]], textposition="inside", insidetextanchor="middle",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            ))
        fig.update_layout(
            barmode="group", height=340, margin=dict(l=10, r=10, t=30, b=30), yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        )
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"Accuracy on the hand-labelled validation points, with 95% confidence intervals. The production map is "
            f"**{PRODUCTION_MODEL_NAME}** trained on {LABEL_SOURCE_NAME} labels (config.settings.LANDCOVER_PRODUCTION_MODEL)."
        )

        conf_col, class_col = st.columns(2)
        confusion_path = LANDCOVER_EVAL_DIR / f"{LANDCOVER_PRODUCTION_MODEL}_confusion_matrix.csv"
        with conf_col:
            st.markdown(f"**{PRODUCTION_MODEL_NAME} confusion matrix**")
            if confusion_path.exists():
                cm_df = pd.read_csv(confusion_path, index_col=0)
                classes = [LANDCOVER_LABELS.get(c.split(":")[-1], c) for c in cm_df.columns]
                truths = [LANDCOVER_LABELS.get(c.split(":")[-1], c) for c in cm_df.index]
                cm_fig = go.Figure(go.Heatmap(
                    z=cm_df.values, x=classes, y=truths, colorscale="Blues", showscale=False,
                    text=cm_df.values, texttemplate="%{text}", hovertemplate="true %{y} → predicted %{x}: %{z}<extra></extra>",
                ))
                cm_fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=40),
                                     xaxis_title="Predicted", yaxis=dict(title="Actual", autorange="reversed"))
                cm_fig.update_yaxes(automargin=True)
                st.plotly_chart(cm_fig, width="stretch")
            else:
                _missing(confusion_path, "python scripts/evaluate_landcover_classifiers.py")
        per_class_path = LANDCOVER_EVAL_DIR / f"{LANDCOVER_PRODUCTION_MODEL}_per_class.csv"
        with class_col:
            st.markdown(f"**{PRODUCTION_MODEL_NAME} per-class scores**")
            if per_class_path.exists():
                per_class = pd.read_csv(per_class_path)
                per_class["class"] = per_class["class"].map(lambda c: LANDCOVER_LABELS.get(c, c))
                _table(per_class, support=st.column_config.NumberColumn("Points", format="%d"))
                weakest = per_class.sort_values("f1").iloc[0]
                st.caption(f"Weakest class: {weakest['class']} (F1 {weakest['f1']:.2f}, {int(weakest['support'])} points) — "
                           "scores for small classes rest on few examples.")
            else:
                _missing(per_class_path, "python scripts/evaluate_landcover_classifiers.py")

        mcnemar_path = LANDCOVER_EVAL_DIR / "mcnemar_counts.csv"
        if mcnemar_path.exists():
            st.markdown("**Are the differences real? (McNemar tests on the same points)**")
            mc = pd.read_csv(mcnemar_path)
            mc["significant"] = mc["p_value"].map(lambda p: "yes (p < 0.05)" if p < 0.05 else "no")
            _table(mc[["comparison", "model_a", "model_b", "accuracy_a", "accuracy_b", "only_a_correct", "only_b_correct",
                       "p_value", "significant"]],
                   accuracy_a=st.column_config.NumberColumn("Accuracy A", format="percent"),
                   accuracy_b=st.column_config.NumberColumn("Accuracy B", format="percent"),
                   p_value=st.column_config.NumberColumn("p-value", format="%.4f"))
        source_note = LANDCOVER_EVAL_DIR / "final_batch_SOURCE.txt"
        with st.expander("Method notes: where these numbers come from"):
            if source_note.exists():
                st.caption(source_note.read_text(encoding="utf-8"))
            st.caption("The full sweep (RF smoothing windows, hybrid weights) is in final_batch_metrics.csv and MLflow "
                       "(experiment landcover_classifiers).")
            _table(lc_metrics[["run", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high", "macro_f1", "eval_group",
                               "label_source"]], height=300)

# =====================================================================================
with tab_heat:
    st.subheader("XGBoost + CNN heat model (S5, C2)")
    importance_path = PROCESSED_DIR / "heat_model" / "xgb_feature_importance.csv"
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        name_col, value_col = importance_df.columns[0], importance_df.columns[-1]
        importance_df = importance_df.sort_values(value_col)
        fig = go.Figure(go.Bar(
            x=importance_df[value_col], y=importance_df[name_col].map(_pretty_feature), orientation="h",
            marker_color=WITHIN_COLOR, hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(height=60 + 24 * len(importance_df), margin=dict(l=10, r=10, t=10, b=30),
                          xaxis_title="XGBoost feature importance")
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, width="stretch")
        st.caption("What the subzone heat model leans on most when predicting land surface temperature.")
    else:
        _missing(importance_path, "python scripts/train_heat_model_xgboost.py")

    if CNN_MODEL_SAVE_PATH.exists():
        st.caption("The live XGBoost-vs-CNN counterfactual cross-check (direction agreement) is on the "
                   "**Counterfactual Greening** page.")
    else:
        st.caption(
            "CNN heat model not trained yet — train it via `notebooks/colab_training/train_heat_cnn.ipynb`, then "
            "`python scripts/pull_models.py --model cnn`, to enable the CNN half on the **Counterfactual Greening** page."
        )

# =====================================================================================
with tab_hotspots:
    st.subheader("Hotspot-typology clusters (S4)")
    profile_path = PROCESSED_DIR / "hotspot_cluster_profile.csv"
    cluster_plot = DIAGNOSTICS_DIR / "hotspot_cluster_profile.png"
    if profile_path.exists():
        profile = pd.read_csv(profile_path)
        profile.insert(1, "type", profile["primary_cluster"].map(lambda c: CLUSTER_NAMES.get(int(c), f"Cluster {int(c)}")))
        _table(profile, primary_cluster=st.column_config.NumberColumn("Cluster", width="small"),
               n_subzones=st.column_config.NumberColumn("Subzones", format="%d"),
               **{c: st.column_config.NumberColumn(_pretty_feature(c), format="%.2f")
                  for c in profile.columns if c.startswith(("lst", "ndvi", "ndbi", "fraction"))})
        st.caption(
            "Land-cover-coherence check: whether the hottest cluster is also the most built-up / least-vegetated "
            "(fractions are NOT clustering inputs — see src/hotspots/cluster.py). Run "
            "`python scripts/diagnose_hotspot_clusters.py` to recompute this check and the plot below."
        )
        if cluster_plot.exists():
            st.image(str(cluster_plot), caption="Per-cluster feature profile")
    else:
        _missing(profile_path, "python scripts/build_hotspot_clusters.py")

# =====================================================================================
with tab_inputs:
    st.subheader("Week-1 gates (G1–G5)")
    if gates_df is not None:
        _table(gates_df.assign(status=gates_df["status"].map(lambda s: f"✅ {s}" if str(s).upper() == "PASS" else f"❌ {s}")))
        st.caption(
            "G5 (labeling agreement) passed with Cohen's kappa = 0.676 (\"substantial agreement\", 20 shared "
            "points, independently labeled by both team members) — see docs/archive/Progress_Log_2026-08-04_0237.md. "
            "Not recomputed live here; validation/input_validation/labeling_agreement.py has no persisted "
            "per-run output file yet."
        )
    else:
        _missing(gates_path, "python scripts/run_week1_gates.py")

    st.subheader("Secondary LST cross-checks (NEA air temperature + MODIS)")
    heat_path = INTERIM_DIR / "heat_variants_subzone.csv"
    nea_path = INTERIM_DIR / "nea_heldout_lst.csv"
    modis_path = INTERIM_DIR / "modis_heldout_lst.csv"
    if heat_path.exists() and (nea_path.exists() or modis_path.exists()):
        heat_df = pd.read_csv(heat_path)
        rows = []
        for source_label, source_path in (("NEA (air temperature)", nea_path), ("MODIS (1 km LST)", modis_path)):
            if not source_path.exists():
                continue
            heldout_df = pd.read_csv(source_path)
            for variant in VARIANT_COLUMNS:
                agreement = heldout_agreement(heat_df, variant, heldout_df)
                rows.append({
                    "source": source_label, "variant": variant, "n": agreement["n"],
                    "mean_offset_c": agreement["mean_offset_c"], "spread_c": agreement["spread_c"],
                    "spearman": agreement["spearman"],
                })
        _table(pd.DataFrame(rows), mean_offset_c=st.column_config.NumberColumn("Mean offset (°C)", format="%.2f"),
               spread_c=st.column_config.NumberColumn("Spread (°C)", format="%.2f"),
               spearman=st.column_config.NumberColumn("Spearman", format="%.3f"))
        with st.expander("Method notes: reading the cross-checks"):
            st.caption(
                "Landsat LST reads hotter than both references (see the mean offset): NEA measures air temperature "
                "and MODIS a coarse 1 km surface temperature. That offset is systematic and can't move any rank, so "
                "RMSE — which mostly restates it — is not shown. What matters for a ranking tool is the spread (the "
                "residual std once the offset is removed) and Spearman. MODIS covers ~329 subzones, NEA only ~12, and "
                "the two fail in different ways (footprint mismatch vs. air-temperature-vs-LST), which is why both are "
                "shown. See validation/input_validation/{nea_heldout,modis_heldout}.py."
            )
    else:
        st.caption("Run `python scripts/build_nea_heldout.py` and/or `python scripts/build_modis_heldout.py` first.")

    st.subheader("Land-change diagnostic (multi-year composite risk)")
    if land_change_df is not None:
        counts = land_change_df["status"].value_counts()
        l1, l2 = st.columns(2)
        l1.metric("Subzones flagged for land change", f"{int(counts.get('flagged', 0))} of {len(land_change_df)}")
        l2.metric("Stable", int(counts.get("stable", 0)))
        flagged = land_change_df.sort_values("change_fraction", ascending=False)
        _table(flagged.assign(subzone_id=flagged["subzone_id"].map(subzone_name)), height=300,
               subzone_id="Subzone", change_fraction=st.column_config.ProgressColumn(
                   "Changed share", min_value=0.0, max_value=1.0, format="percent"))
    else:
        _missing(land_change_path, "python scripts/diagnose_land_change.py")
