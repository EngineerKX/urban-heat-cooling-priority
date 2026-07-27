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
