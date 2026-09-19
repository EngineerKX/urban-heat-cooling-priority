"""Rank-impact / ablation test (from rank_impact.ipynb, Track A / S6): fills
the C3 ablation row by rebuilding the cooling-priority score once per
heat-layer variant (Exposure = that variant's LST, Sensitivity and
Adaptive-capacity deficit frozen) and reporting RMSE-vs-held-out, Spearman
rank correlation, and top-N overlap against a reference variant.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from config.settings import REFERENCE_VARIANT, TOP_N, VARIANT_COLUMNS
from src.priority_score.score import build_score


def rmse_vs_heldout(df: pd.DataFrame, exposure_col: str, heldout: pd.DataFrame):
    if heldout is None:
        return np.nan
    merged = df[["subzone_id", exposure_col]].merge(
        heldout[["subzone_id", "lst_heldout_c"]], on="subzone_id", how="inner",
    )
    if len(merged) == 0:
        return np.nan
    return float(np.sqrt(mean_squared_error(merged["lst_heldout_c"], merged[exposure_col])))


def top_n_overlap(score_a: pd.Series, score_b: pd.Series, n: int = TOP_N) -> float:
    top_a = set(score_a.sort_values(ascending=False).index[:n])
    top_b = set(score_b.sort_values(ascending=False).index[:n])
    return len(top_a & top_b) / n


def run_rank_impact(
    df: pd.DataFrame, heldout: pd.DataFrame, weighting: str,
    variant_columns=VARIANT_COLUMNS, reference_variant: str = REFERENCE_VARIANT, top_n: int = TOP_N,
) -> tuple[pd.DataFrame, dict]:
    """Returns (results_df, scores_by_variant). `results_df` has one row per
    variant: lst_rmse_heldout, spearman_vs_<reference>, top<N>_overlap_vs_<reference>.
    The reference-variant row is self-compared by construction
    (spearman=1.000, overlap=1.000) — it's the frozen baseline, not a result.
    """
    scores, weights_by_variant = {}, {}
    for variant in variant_columns:
        score, weights = build_score(df, variant, weighting)
        scores[variant] = score
        weights_by_variant[variant] = weights

    ref_score = scores[reference_variant]
    rows = []
    for variant in variant_columns:
        rmse = rmse_vs_heldout(df, variant, heldout)
        corr, _ = spearmanr(scores[variant], ref_score)
        overlap = top_n_overlap(scores[variant], ref_score, top_n)
        rows.append({
            "variant": variant,
            "lst_rmse_heldout": rmse,
            f"spearman_vs_{reference_variant}": corr,
            f"top{top_n}_overlap_vs_{reference_variant}": overlap,
        })

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False, float_format=lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"))
    print("\nWeights used per variant (exposure, sensitivity, adaptive_deficit):")
    for variant, w in weights_by_variant.items():
        print(f"  {variant}: {', '.join(f'{k}={v:.3f}' for k, v in w.items())}")
        if any(v < 0 for v in w.values()):
            print("    ⚠️  negative weight present — see src/priority_score/score.py's PCA sign-alignment note.")

    return results_df, scores


def run_weighting_comparison(
    df: pd.DataFrame, variant_columns=VARIANT_COLUMNS, reference_variant: str = REFERENCE_VARIANT, top_n: int = TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C1 benchmark: how far do PCA-derived weights and equal weights
    disagree on the ranking? run_rank_impact varies the heat layer within
    one weighting; this varies the weighting within one heat layer, so the
    two together show which choice moves the top-N more.

    Returns (summary_df, membership_df). `summary_df` has one row per
    variant: Spearman and top-N overlap between the PCA and equal-weight
    scores, how many top-N subzones change, and the fitted PCA weights.
    `membership_df` lists every subzone in EITHER weighting's top-N for
    `reference_variant`, with its rank under each, tagged both / pca_only /
    equal_only -- i.e. which subzones the weighting choice adds or removes.
    """
    rows, membership_df = [], None
    for variant in variant_columns:
        pca_score, pca_weights = build_score(df, variant, "pca")
        equal_score, _ = build_score(df, variant, "equal")

        corr, _ = spearmanr(pca_score, equal_score)
        overlap = top_n_overlap(pca_score, equal_score, top_n)
        rows.append({
            "variant": variant,
            "spearman_pca_vs_equal": corr,
            f"top{top_n}_overlap_pca_vs_equal": overlap,
            f"n_top{top_n}_changed": top_n - round(overlap * top_n),
            **{f"pca_weight_{name}": w for name, w in pca_weights.items()},
        })
        if any(w < 0 for w in pca_weights.values()):
            print(f"⚠️  {variant}: negative PCA weight present — see src/priority_score/score.py's PCA sign-alignment note.")

        if variant == reference_variant:
            top_pca = set(pca_score.sort_values(ascending=False).index[:top_n])
            top_equal = set(equal_score.sort_values(ascending=False).index[:top_n])
            in_either = sorted(top_pca | top_equal)
            rank_pca = pca_score.rank(ascending=False, method="first").astype(int)
            rank_equal = equal_score.rank(ascending=False, method="first").astype(int)
            membership_df = pd.DataFrame({
                "subzone_id": df.loc[in_either, "subzone_id"].values,
                "rank_pca": rank_pca.loc[in_either].values,
                "rank_equal": rank_equal.loc[in_either].values,
                "status": [
                    "both" if i in top_pca and i in top_equal else "pca_only" if i in top_pca else "equal_only"
                    for i in in_either
                ],
            }).sort_values("rank_pca").reset_index(drop=True)

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False, float_format=lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"))
    return summary_df, membership_df
