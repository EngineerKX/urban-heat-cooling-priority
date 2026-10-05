"""Cooling-priority score evaluation beyond the existing decision-impact
tables (weighting, heat layer, sensitivity formula, land-cover source): the
structure of the PCA weighting, pillar ablations, the aggregation rule, the
minimum-residents threshold, and an external face-validity check.

Every comparison follows the decision-impact convention: rebuild the score
with ONE change and count how many of the production top-N subzones leave.
"""

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.decomposition import PCA

from config.settings import TOP_N
from src.priority_score.score import build_score
from src.utils.geo import normalize

PILLARS = ("exposure", "sensitivity", "adaptive_deficit")


def pillar_table(df, exposure_col, adaptive_capacity_col="greenery_fraction") -> pd.DataFrame:
    """The three min-max pillars exactly as build_score forms them."""
    return pd.DataFrame({
        "exposure": normalize(df[exposure_col]),
        "sensitivity": normalize(df["sensitivity_raw"]),
        "adaptive_deficit": normalize(1 - df[adaptive_capacity_col]),
    })


def top_changed(ids, score, ref_ids, ref_score, n=TOP_N) -> int:
    """Production top-n subzones (ref) missing from the alternative's top-n."""
    top = set(pd.Series(score.values, index=ids.values).nlargest(n).index)
    ref_top = set(pd.Series(ref_score.values, index=ref_ids.values).nlargest(n).index)
    return len(ref_top - top)


def pca_structure(pillars: pd.DataFrame, score: pd.Series) -> dict:
    """Share of variance the first principal component captures (how well one
    index summarises the three pillars), the pillar correlations, and how
    strongly each pillar drives the final score."""
    z = (pillars - pillars.mean()) / pillars.std(ddof=0)
    pca = PCA().fit(z.values)
    return {
        "explained_variance": pca.explained_variance_ratio_,
        "correlations": pillars.corr(),
        "score_correlation": {p: float(np.corrcoef(pillars[p], score)[0, 1]) for p in pillars},
        "pillar_sd": pillars.std(ddof=0).to_dict(),
    }


def leave_one_out(df, exposure_col, prod_score, prod_weights) -> pd.DataFrame:
    """Drop each pillar in turn, keeping the other two at their production
    weights (rescaled to sum to 1)."""
    rows = []
    for drop in PILLARS:
        w = {p: (0.0 if p == drop else prod_weights[p]) for p in PILLARS}
        score, _ = build_score(df, exposure_col, "custom", weights=w)
        rows.append({"change": f"Without {drop.replace('_', ' ')}",
                     "top_changed": top_changed(df["subzone_id"], score, df["subzone_id"], prod_score),
                     "spearman": spearmanr(score, prod_score)[0]})
    return pd.DataFrame(rows)


def zscore_aggregation(pillars: pd.DataFrame, weights: dict) -> pd.Series:
    """Same weights, but pillars standardised (z-scores) instead of min-max:
    each pillar's influence then matches its weight instead of its spread."""
    z = (pillars - pillars.mean()) / pillars.std(ddof=0)
    return z[list(PILLARS)] @ np.array([weights[p] for p in PILLARS])


def face_validity(score_df, planning_area: pd.Series, hotspot_areas, n=TOP_N) -> dict:
    """How many of the top-n subzones sit in the published hotspot planning
    areas, against the hypergeometric chance expectation."""
    pa = score_df["subzone_id"].map(planning_area)
    in_area = pa.isin(hotspot_areas)
    top = score_df.nlargest(n, "priority_score")
    hits = int(top["subzone_id"].map(planning_area).isin(hotspot_areas).sum())
    total, k = len(score_df), int(in_area.sum())
    return {"n_ranked": total, "n_in_hotspot_areas": k, "top_n": n, "top_in_hotspot_areas": hits,
            "expected_by_chance": n * k / total, "p_value_at_least": float(hypergeom.sf(hits - 1, total, k, n)),
            "subzones": "; ".join(sorted(top.loc[top["subzone_id"].map(planning_area).isin(hotspot_areas), "subzone_id"]))}
