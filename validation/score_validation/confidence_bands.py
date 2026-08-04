"""S6 calibrated confidence bands: bootstrap the priority score using
already-built validation error estimates as noise models -- NOT an
arbitrary Monte Carlo over unvalidated input ranges (e.g. the 0.35 NDVI
threshold has no natural uncertainty range to invent, so that whole
approach was rejected). Two noise sources:

- Exposure: rmse_vs_heldout's already-computed NEA-heldout RMSE for the
  chosen LST variant (this module's own function below just calls it).
- Adaptive capacity: the land-cover ensemble's own vegetation-class recall
  binomial standard error, from its confusion matrix (already produced by
  scripts/evaluate_landcover_classifiers.py).

The sensitivity pillar is deliberately left UNPERTURBED -- no validation-
error estimate exists anywhere in this repo for SingStat population data,
and inventing one would be exactly the ungrounded approach already
rejected for the other two pillars. This is a STATED LIMITATION of the
resulting bands, not a silent omission: they reflect exposure +
adaptive-capacity uncertainty only, not sensitivity uncertainty.

Second stated limitation, found empirically once this ran on real data:
the NEA-heldout RMSE (~11.6C) is large relative to LST's own ~30-50C
range -- because it mixes genuine random error with the SYSTEMATIC LST-
vs-air-temperature offset (LST reads hotter than air temperature,
especially over built-up surfaces, and NEA measures air temperature, not
LST). Treating that RMSE as an i.i.d. per-subzone noise std is a
conservative (upper-bound), not precise, exposure noise model -- it
likely overstates true random rank uncertainty since a systematic offset
isn't actually independent per subzone. Reported bands should be read as
"how much could ranks plausibly move under a pessimistic noise budget",
not a tight calibrated interval.
"""

import numpy as np
import pandas as pd

from config.settings import PRIORITY_SCORE_BAND_QUANTILES, PRIORITY_SCORE_BOOTSTRAP_ITERATIONS, RANDOM_SEED
from src.priority_score.score import build_score
from validation.score_validation.rank_impact import rmse_vs_heldout


def exposure_noise_std(df: pd.DataFrame, exposure_col: str, heldout: pd.DataFrame) -> float:
    """RMSE-vs-NEA-heldout for this exposure variant (reused unmodified),
    used directly as the Gaussian noise std injected into every subzone's
    exposure value per bootstrap draw."""
    return rmse_vs_heldout(df, exposure_col, heldout)


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
    seed: int = RANDOM_SEED, adaptive_capacity_col: str = "greenery_fraction",
) -> pd.DataFrame:
    """Perturbs exposure_col and adaptive_capacity_col with independent
    Gaussian noise (the two noise-model stds above) n_iterations times,
    rebuilds the priority score each time via build_score (reused
    completely unmodified), and reports per-subzone quantiles. NaN/zero
    noise stds are treated as "no perturbation for this pillar" rather
    than raising -- lets a caller run with only one noise source wired up.
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

    print(f"Bootstrap: {n_iterations} iterations "
          f"(exposure_noise_std={exposure_noise_std if has_exposure_noise else 'none'}, "
          f"adaptive_capacity_noise_std={adaptive_capacity_noise_std if has_adaptive_noise else 'none'})")
    print(f"Mean band width ({quantile_cols[-1]} - {quantile_cols[0]}): {result['band_width'].mean():.4f} "
          f"(priority scores are normalized ~[0,1], so this is directly comparable across subzones)")
    if not has_adaptive_noise:
        print("⚠️  No adaptive-capacity noise model was supplied — bands reflect exposure uncertainty only.")
    print("⚠️  Sensitivity pillar is never perturbed — no validation-error estimate exists for it in this repo "
          "(see module docstring). Bands understate true uncertainty to that extent; stated limitation, not a gap.")

    return result
