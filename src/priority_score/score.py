"""S6 — cooling-priority score construction (Cooling Singapore UHV-style
structure: Exposure + Sensitivity + Adaptive-capacity deficit), PCA-weighted
(primary) or equal-weighted (mandatory sensitivity check).
"""

import pandas as pd
from sklearn.decomposition import PCA

from config.settings import RANDOM_SEED
from src.utils.geo import normalize


def build_score(df: pd.DataFrame, exposure_col: str, weighting: str, seed: int = RANDOM_SEED):
    """Build the cooling-priority score for one exposure (heat-layer) column.
    Sensitivity and adaptive-capacity deficit are held fixed — only the
    exposure input and the weighting scheme vary between calls, which is
    what makes this reusable both for the production score and for the C3
    ablation (see validation/score_validation/rank_impact.py).

    `weighting="pca"` fits a fresh 1-component PCA on every call — the sign
    of the fitted loadings is aligned to the exposure loading only, so a
    pillar whose loading disagrees with exposure can come out negative,
    which INVERTS its contribution rather than just down-weighting it. This
    is a methodological call for whoever owns S6's weighting, not something
    this function resolves silently — inspect the returned `weights` dict;
    a negative entry means this happened.
    """
    exposure_norm = normalize(df[exposure_col])
    sensitivity_norm = normalize(df["sensitivity_raw"])
    adaptive_deficit_norm = normalize(1 - df["greenery_fraction"])

    pillars = pd.DataFrame({
        "exposure": exposure_norm,
        "sensitivity": sensitivity_norm,
        "adaptive_deficit": adaptive_deficit_norm,
    })

    if weighting == "equal":
        score = pillars.mean(axis=1)
        weights = {"exposure": 1 / 3, "sensitivity": 1 / 3, "adaptive_deficit": 1 / 3}

    elif weighting == "pca":
        pca = PCA(n_components=1, random_state=seed)
        pca.fit(pillars.values)
        raw_weights = pca.components_[0]
        if raw_weights[0] < 0:
            raw_weights = -raw_weights
        weights_arr = raw_weights / raw_weights.sum()
        score = pd.Series(pillars.values @ weights_arr, index=pillars.index)
        weights = dict(zip(pillars.columns, weights_arr))

    else:
        raise ValueError(f"Unknown weighting '{weighting}', expected 'pca' or 'equal'.")

    return score, weights
