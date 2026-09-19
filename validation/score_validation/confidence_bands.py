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
- Adaptive capacity: the land-cover hybrid's own vegetation-class recall
  binomial standard error, from its confusion matrix (already produced by
  scripts/evaluate_landcover_classifiers.py).

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
- The sensitivity pillar is deliberately left UNPERTURBED -- no validation-
  error estimate exists anywhere in this repo for SingStat population data,
  and inventing one would be exactly the ungrounded approach already
  rejected for the other two pillars. The bands reflect exposure +
  adaptive-capacity uncertainty only.
- The exposure noise is an UPPER bound on random error: the residual spread
  also contains the mismatch between the held-out footprint (MODIS is 1km,
  far larger than most subzones) and the subzone itself. With the NEA source
  it is estimated from only ~12 subzones.
- The adaptive-capacity noise counts only vegetation RECALL. The hybrid's
  bigger error is built-up labelled as vegetation (low vegetation
  PRECISION), which inflates greenery and is not in this noise model, so
  the bands probably understate adaptive-capacity uncertainty. Not
  quantified (the validation sample is stratified, so its raw counts aren't
  island-representative).
- These are validation-error-based bands, not statistically CALIBRATED ones:
  their coverage was never tested against ground truth.
"""

import numpy as np
import pandas as pd

from config.settings import (
    PRIORITY_SCORE_BAND_QUANTILES, PRIORITY_SCORE_BOOTSTRAP_ITERATIONS, RANDOM_SEED, TOP_N,
)
from src.priority_score.score import build_score


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
    if heldout is None:
        return np.nan
    merged = df[["subzone_id", exposure_col]].merge(
        heldout[["subzone_id", "lst_heldout_c"]], on="subzone_id", how="inner",
    ).dropna()
    if len(merged) < 3:
        return np.nan

    residual = merged[exposure_col] - merged["lst_heldout_c"]
    std = float(residual.std(ddof=1))
    print(f"Exposure noise model: {len(merged)} subzones, mean offset {residual.mean():+.2f}C (removed), "
          f"offset-removed std {std:.2f}C")
    return std


def adaptive_capacity_noise_std(confusion_matrix_df: pd.DataFrame, class_name: str = "vegetation") -> float:
    """Binomial standard error of `class_name`'s recall, computed directly
    from the confusion matrix's own counts (first column holds
    'true:<class>' row labels, other columns 'pred:<class>')."""
    true_col = f"true:{class_name}"
    label_col = confusion_matrix_df.columns[0]
    row = confusion_matrix_df[confusion_matrix_df[label_col] == true_col]
    if row.empty:
        raise ValueError(f"'{true_col}' not found in confusion matrix's '{label_col}' column.")

    pred_cols = [c for c in confusion_matrix_df.columns if c.startswith("pred:")]
    counts = row[pred_cols].iloc[0]
    n = int(counts.sum())
    tp = int(counts[f"pred:{class_name}"])
    recall = tp / n if n else float("nan")
    std = float(np.sqrt(recall * (1 - recall) / n)) if n else float("nan")

    print(f"Adaptive-capacity noise model: {class_name} recall={recall:.3f} (n={n}), binomial SE={std:.4f}")
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
    print("⚠️  Sensitivity pillar is never perturbed — no validation-error estimate exists for it in this repo "
          "(see module docstring). Bands understate true uncertainty to that extent; stated limitation, not a gap.")

    return result
