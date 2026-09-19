"""S6 confidence bands: bootstrap the priority score using
already-built validation error estimates as noise models -- NOT an
arbitrary Monte Carlo over unvalidated input ranges (e.g. the 0.35 NDVI
threshold has no natural uncertainty range to invent, so that whole
approach was rejected). Two noise sources:

- Exposure: the spread of (Landsat LST - held-out LST) residuals across
  subzones, AFTER removing their mean offset (`exposure_noise_std` below).
  The score min-max-normalizes exposure, so a constant offset can't move any
  rank; only the spread around it is noise. The held-out source is
  config.settings.EXPOSURE_NOISE_SOURCE (MODIS by default).
- Adaptive capacity: the per-subzone disagreement between the two greenery
  estimates already in the pipeline (land-cover hybrid vs. NDVI-threshold
  proxy), as the residual std after regressing one on the other
  (`adaptive_capacity_noise_std`). A CONSTANT misclassification rate is an
  affine distortion of greenery that the score's min-max normalization
  cancels exactly (checked: max score change 4e-16), so only per-subzone
  HETEROGENEITY of error can move a rank -- which the hybrid's global recall
  standard error (what this module used before 2026-09-19) never measured.

Every noisy draw is scored against the UNperturbed data's normalization
range (build_score's `reference_df`). Re-normalizing each draw against its
own range -- what this module did before 2026-09-19 -- lets the injected
noise stretch the exposure range, which squashes hot subzones' scores: the
point estimate then sat above its own p95 for 31/332 subzones (8 of the top
20). Fixed scaling removes that bias.

Besides per-subzone score quantiles the result carries `p_top<N>`: the
fraction of draws in which each subzone lands in the top N -- the number a
planner actually acts on (which N to fund), which score-unit bands alone
don't give.

STATED LIMITATIONS (not silent omissions):
- The sensitivity pillar is deliberately left UNPERTURBED, and its
  uncertainty is not missing from the picture, it is a different KIND: the
  SingStat counts are near-exact administrative data, so there is no
  measurement noise to bootstrap; what is uncertain is the FORMULA (count vs.
  density, the population/elderly split, which elderly measure). That is
  reported separately as a decision-level table
  (validation/score_validation/sensitivity_specs.py). The bands therefore
  cover measurement noise in exposure and adaptive capacity only, not
  formula choices.
- The exposure noise is an UPPER bound on random error: the residual spread
  also contains the mismatch between the held-out footprint (MODIS is 1km,
  far larger than most subzones) and the subzone itself. With the NEA source
  it is estimated from only ~12 subzones.
- The adaptive-capacity noise is a rough scale, not a validated one: the
  NDVI proxy is itself a crude estimate (it can be wrong where the hybrid is
  right), yet it shares the hybrid's Sentinel-2 imagery, so common-mode
  errors cancel out of the residual. It is ~2.3x the old recall-based value
  but moves the bands by only ~10%, so no conclusion hinges on it.
- These are validation-error-based bands, not statistically CALIBRATED ones:
  their coverage was never tested against ground truth.
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

from config.settings import (
    PRIORITY_SCORE_BAND_QUANTILES, PRIORITY_SCORE_BOOTSTRAP_ITERATIONS, RANDOM_SEED, TOP_N,
)
from src.priority_score.score import build_score
from validation.score_validation.rank_impact import heldout_agreement


def exposure_noise_std(df: pd.DataFrame, exposure_col: str, heldout: pd.DataFrame) -> float:
    """Std of the (exposure_col - held-out) residuals over the subzones both
    tables cover, after removing their mean offset -- the Gaussian noise std
    injected into every subzone's exposure value per bootstrap draw. NaN if
    there is no held-out table or fewer than 3 overlapping subzones (too few
    to estimate a spread from).

    Deliberately NOT the RMSE (what this function returned before
    2026-09-19): RMSE folds in the mean offset -- Landsat LST reads ~6C hotter
    than MODIS and ~11C hotter than NEA air temperature -- which is systematic,
    can't move a rank, and at 11.6C swamped the ~2.5C real spread between
    subzones.
    """
    agreement = heldout_agreement(df, exposure_col, heldout)
    if np.isnan(agreement["spread_c"]):
        return np.nan
    print(f"Exposure noise model: {agreement['n']} subzones, mean offset {agreement['mean_offset_c']:+.2f}C (removed), "
          f"offset-removed std {agreement['spread_c']:.2f}C")
    return agreement["spread_c"]


def adaptive_capacity_noise_std(df: pd.DataFrame, greenery_col: str, reference_col: str) -> float:
    """Per-subzone greenery noise: the std of `greenery_col`'s residuals after
    regressing it on an independent estimate of the same quantity
    (`reference_col`), i.e. how much the two estimators disagree on a subzone
    once any systematic (affine) relationship between them is removed. NaN if
    fewer than 3 subzones have both values.

    Not the hybrid's recall standard error (what this returned before
    2026-09-19): see the module docstring for why only per-subzone
    heterogeneity can move a rank.
    """
    both = df[[greenery_col, reference_col]].dropna()
    if len(both) < 3:
        return np.nan

    fit = linregress(both[reference_col], both[greenery_col])
    residual = both[greenery_col] - (fit.intercept + fit.slope * both[reference_col])
    std = float(residual.std(ddof=2))
    print(f"Adaptive-capacity noise model: {greenery_col} vs {reference_col} over {len(both)} subzones, "
          f"R²={fit.rvalue ** 2:.3f}, residual std={std:.4f}")
    return std


def bootstrap_priority_score(
    df: pd.DataFrame, exposure_col: str, weighting: str,
    exposure_noise_std: float, adaptive_capacity_noise_std: float,
    n_iterations: int = PRIORITY_SCORE_BOOTSTRAP_ITERATIONS, quantiles=PRIORITY_SCORE_BAND_QUANTILES,
    seed: int = RANDOM_SEED, adaptive_capacity_col: str = "greenery_fraction", top_n: int = TOP_N,
) -> pd.DataFrame:
    """Perturbs exposure_col and adaptive_capacity_col with independent
    Gaussian noise (the two noise-model stds above) n_iterations times,
    rebuilds the priority score each time via build_score against the
    UNperturbed data's normalization range (`reference_df=df` -- see the
    module docstring for why), and reports per-subzone quantiles plus
    `p_top<top_n>`, the fraction of draws that land the subzone in the top
    `top_n`. NaN/zero noise stds are treated as "no perturbation for this
    pillar" rather than raising -- lets a caller run with only one noise
    source wired up.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    scores = np.zeros((n_iterations, n))

    has_exposure_noise = exposure_noise_std is not None and not np.isnan(exposure_noise_std) and exposure_noise_std > 0
    has_adaptive_noise = (
        adaptive_capacity_noise_std is not None and not np.isnan(adaptive_capacity_noise_std) and adaptive_capacity_noise_std > 0
    )

    for i in range(n_iterations):
        perturbed = df.copy()
        if has_exposure_noise:
            perturbed[exposure_col] = perturbed[exposure_col] + rng.normal(0, exposure_noise_std, size=n)
        if has_adaptive_noise:
            perturbed[adaptive_capacity_col] = (
                perturbed[adaptive_capacity_col] + rng.normal(0, adaptive_capacity_noise_std, size=n)
            ).clip(0, 1)
        score, _weights = build_score(
            perturbed, exposure_col, weighting, seed=seed, adaptive_capacity_col=adaptive_capacity_col,
            reference_df=df,
        )
        scores[i] = score.values

    point_score, _ = build_score(df, exposure_col, weighting, seed=seed, adaptive_capacity_col=adaptive_capacity_col)

    result = pd.DataFrame({"subzone_id": df["subzone_id"].values, "priority_score_point": point_score.values})
    quantile_cols = []
    for q in sorted(quantiles):
        col = f"priority_score_p{int(round(q * 100)):02d}"
        result[col] = np.quantile(scores, q, axis=0)
        quantile_cols.append(col)
    result["band_width"] = result[quantile_cols[-1]] - result[quantile_cols[0]]

    top_members = np.argsort(-scores, axis=1)[:, :top_n]
    p_top_col = f"p_top{top_n}"
    result[p_top_col] = np.bincount(top_members.ravel(), minlength=n) / n_iterations

    print(f"Bootstrap: {n_iterations} iterations "
          f"(exposure_noise_std={exposure_noise_std if has_exposure_noise else 'none'}, "
          f"adaptive_capacity_noise_std={adaptive_capacity_noise_std if has_adaptive_noise else 'none'})")
    print(f"Mean band width ({quantile_cols[-1]} - {quantile_cols[0]}): {result['band_width'].mean():.4f} "
          f"(priority scores are normalized ~[0,1], so this is directly comparable across subzones)")
    n_outside = int(((result["priority_score_point"] > result[quantile_cols[-1]])
                     | (result["priority_score_point"] < result[quantile_cols[0]])).sum())
    print(f"Point estimate outside its own [{quantile_cols[0]}, {quantile_cols[-1]}] band: {n_outside}/{n} subzones "
          f"(should be near zero -- a large count means the bootstrap is biased).")
    p = result[p_top_col]
    print(f"{p_top_col}: {int((p >= 0.9).sum())} subzones >=0.9, {int(((p > 0.1) & (p < 0.9)).sum())} borderline "
          f"(0.1-0.9), {int((p <= 0.1).sum())} <=0.1.")
    if not has_adaptive_noise:
        print("⚠️  No adaptive-capacity noise model was supplied — bands reflect exposure uncertainty only.")
    print("ℹ️  Sensitivity pillar is not perturbed: its uncertainty is formula choice, not noise — see the "
          "sensitivity-specification table (validation/score_validation/sensitivity_specs.py) and the module docstring.")

    return result
