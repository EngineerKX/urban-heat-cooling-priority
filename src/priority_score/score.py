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
    weights: dict = None,
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

    `weighting="heat_greenery"` scores exposure and adaptive-capacity deficit
    only (0.5 / 0.5) and never touches `sensitivity_raw` -- the all-places view
    of src/priority_score/lenses.py, which includes subzones with too few
    residents to have a sensitivity value. It says where it is hottest and
    least green, not how many people that affects.

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

    `weighting="custom"` takes the weights from `weights` (a dict keyed
    exposure / sensitivity / adaptive_deficit, rescaled to sum to 1; a missing
    key counts as 0) -- the Island Map's weight sliders, which let a viewer see
    how far the top-N moves when the weights change. With a zero sensitivity
    weight `sensitivity_raw` is never read, as in "heat_greenery", so the
    all-places view's two sliders also cover subzones without one.

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
    adaptive_deficit_norm = normalize(1 - df[adaptive_capacity_col], 1 - ref[adaptive_capacity_col])

    if weighting == "heat_greenery":
        # The "all places" view: heat and missing greenery only, no sensitivity
        # (so it also works for subzones with too few residents to have one).
        # Equal weights: with two positively correlated pillars the PCA weights
        # would be 0.5/0.5 anyway. `weights` keeps the sensitivity key (at 0) so
        # callers that read all three keep working.
        score = 0.5 * exposure_norm + 0.5 * adaptive_deficit_norm
        return score, {"exposure": 0.5, "sensitivity": 0.0, "adaptive_deficit": 0.5}

    if weighting == "custom":
        pillar_names = ("exposure", "sensitivity", "adaptive_deficit")
        if not weights or set(weights) - set(pillar_names):
            raise ValueError(f"weighting='custom' needs `weights` keyed by {pillar_names}, got {weights}.")
        total = sum(weights.values())
        if total <= 0 or any(w < 0 for w in weights.values()):
            raise ValueError(f"Custom weights must be non-negative and not all zero, got {weights}.")
        weights = {k: weights.get(k, 0.0) / total for k in pillar_names}
        score = weights["exposure"] * exposure_norm + weights["adaptive_deficit"] * adaptive_deficit_norm
        if weights["sensitivity"]:
            score = score + weights["sensitivity"] * normalize(df["sensitivity_raw"], ref["sensitivity_raw"])
        return score, weights

    sensitivity_norm = normalize(df["sensitivity_raw"], ref["sensitivity_raw"])

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
        raise ValueError(f"Unknown weighting '{weighting}', expected 'pca', 'equal', 'heat_greenery' or 'custom'.")

    return score, weights
