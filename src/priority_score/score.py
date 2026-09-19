"""S6 — cooling-priority score construction (Cooling Singapore UHV-style
structure: Exposure + Sensitivity + Adaptive-capacity deficit), PCA-weighted
(primary) or equal-weighted (mandatory sensitivity check).
"""

import pandas as pd
from sklearn.decomposition import PCA

from config.settings import RANDOM_SEED
from src.utils.geo import normalize


def build_score(
    df: pd.DataFrame, exposure_col: str, weighting: str, seed: int = RANDOM_SEED,
    adaptive_capacity_col: str = "greenery_fraction", reference_df: pd.DataFrame = None,
):
    """Build the cooling-priority score for one exposure (heat-layer) column.
    Sensitivity and adaptive-capacity deficit are held fixed — only the
    exposure input and the weighting scheme vary between calls, which is
    what makes this reusable both for the production score and for the C3
    ablation (see validation/score_validation/rank_impact.py).

    `adaptive_capacity_col` defaults to "greenery_fraction" so every
    existing call site is unaffected; pass a different column name to
    compare the NDVI-threshold vs. land-cover-hybrid adaptive-capacity
    sources (see config.settings.ADAPTIVE_CAPACITY_SOURCE) without
    duplicating this function.

    `reference_df` (default None = normalize `df` against its own range, as
    everywhere else) pins each pillar's min-max scaling to another table's
    range. Only the confidence-band bootstrap uses it, passing the
    unperturbed data so noisy draws are scored on the same scale as the
    point estimate.

    `weighting="pca"` fits a fresh 1-component PCA on every call, on the
    pillars STANDARDISED to z-scores (correlation-matrix PCA) -- but the score
    itself aggregates the 0-1 min-max pillars with those weights, exactly like
    the equal-weight mode, so the two differ only in the weights. Until
    2026-09-19 the PCA ran on the min-max pillars directly, which made the
    "data-derived" weights follow each pillar's SPREAD (a skewed pillar is
    squeezed toward 0 by min-max and got a tiny weight) instead of how the
    pillars co-vary: swapping the sensitivity pillar's population term from
    count to density moved its weight 0.16 -> 0.45 with no change in its
    correlation with the other pillars. Standardising first keeps the weights
    stable (0.38 / 0.23 / 0.39 either way).

    The fitted PCA's sign — the sign
    of the fitted loadings is aligned to the exposure loading only, so a
    pillar whose loading disagrees with exposure can come out negative,
    which INVERTS its contribution rather than just down-weighting it. This
    is a methodological call for whoever owns S6's weighting, not something
    this function resolves silently — inspect the returned `weights` dict;
    a negative entry means this happened.
    """
    ref = df if reference_df is None else reference_df
    exposure_norm = normalize(df[exposure_col], ref[exposure_col])
    sensitivity_norm = normalize(df["sensitivity_raw"], ref["sensitivity_raw"])
    adaptive_deficit_norm = normalize(1 - df[adaptive_capacity_col], 1 - ref[adaptive_capacity_col])

    pillars = pd.DataFrame({
        "exposure": exposure_norm,
        "sensitivity": sensitivity_norm,
        "adaptive_deficit": adaptive_deficit_norm,
    })

    if weighting == "equal":
        score = pillars.mean(axis=1)
        weights = {"exposure": 1 / 3, "sensitivity": 1 / 3, "adaptive_deficit": 1 / 3}

    elif weighting == "pca":
        standardised = (pillars - pillars.mean()) / pillars.std(ddof=0).replace(0, 1)
        pca = PCA(n_components=1, random_state=seed)
        pca.fit(standardised.values)
        raw_weights = pca.components_[0]
        if raw_weights[0] < 0:
            raw_weights = -raw_weights
        weights_arr = raw_weights / raw_weights.sum()
        score = pd.Series(pillars.values @ weights_arr, index=pillars.index)
        weights = dict(zip(pillars.columns, weights_arr))

    else:
        raise ValueError(f"Unknown weighting '{weighting}', expected 'pca' or 'equal'.")

    return score, weights
