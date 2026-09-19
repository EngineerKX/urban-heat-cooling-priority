"""Decision impact summary -- contribution 3 in one table: for every
methodological choice tested, how many of the top-N subzones does it move,
and is that more than measurement noise alone would move?

The yardstick is `noise_floor`: the expected number of the point-estimate
top-N that a bootstrap draw (measurement noise in exposure and greenery)
replaces. It is read straight off the saved `p_top<N>` column, so no new
computation is needed. A choice that moves fewer subzones than that is
indistinguishable, for planning purposes, from data noise; one that moves more
is a real decision that has to be justified rather than defaulted.

Both sides count subzones entering/leaving the top N, so they are comparable in
units -- but the noise figure is an average over draws while each choice is a
single ranking, and the bootstrap covers measurement noise only (exposure and
greenery), NOT formula uncertainty, which is exactly what several rows below
measure.
"""

import pandas as pd

from config.settings import REFERENCE_VARIANT, TOP_N


def noise_floor(bands_df: pd.DataFrame, top_n: int = TOP_N) -> float:
    """Expected number of the point-estimate top-N subzones replaced by
    measurement noise alone: N minus the sum, over those N subzones, of the
    probability each stays in the top N (`p_top<N>` from the bootstrap)."""
    ranked = bands_df.sort_values("priority_score_point", ascending=False).head(top_n)
    return float(top_n - ranked[f"p_top{top_n}"].sum())


def _row(choice: str, changed: pd.Series, labels: pd.Series, floor: float, top_n: int) -> dict:
    """`changed` and `labels` share a 0..k-1 index: top-N subzones moved by
    each alternative, and that alternative's name."""
    # idxmax on an all-zero series just returns the first label, which would read as a finding.
    largest = str(labels.loc[changed.idxmax()]) if changed.max() > 0 else f"none — no alternative changes the top {top_n}"
    return {
        "choice": choice,
        "alternatives_tested": len(changed),
        "top_changed_min": int(changed.min()),
        "top_changed_max": int(changed.max()),
        "largest_change_from": largest,
        "vs_noise_floor": "exceeds noise" if changed.max() > floor else "within noise",
    }


def build_decision_impact_summary(
    weighting_df: pd.DataFrame, rank_impact_df: pd.DataFrame, spec_df: pd.DataFrame,
    swap_df: pd.DataFrame, bands_df: pd.DataFrame,
    top_n: int = TOP_N, reference_variant: str = REFERENCE_VARIANT,
) -> pd.DataFrame:
    """One row per choice family, sorted by the largest change, with the
    measurement-noise floor as the first row. Inputs are the saved CSVs:
    weighting_comparison.csv, rank_impact_results_pca.csv,
    sensitivity_spec_comparison.csv, landcover_swap_comparison.csv and
    priority_score_confidence_bands.csv."""
    floor = noise_floor(bands_df, top_n)

    # Weighting: PCA-derived vs equal weights, on the reference heat layer.
    weighting_changed = weighting_df.loc[weighting_df["variant"] == reference_variant, f"n_top{top_n}_changed"]
    weighting_changed = weighting_changed.reset_index(drop=True)

    # Heat-map resolution: every variant except the reference one.
    resolution = rank_impact_df[rank_impact_df["variant"] != reference_variant].reset_index(drop=True)
    resolution_changed = top_n - (resolution[f"top{top_n}_overlap_vs_{reference_variant}"] * top_n).round().astype(int)

    # Sensitivity formula: every alternative that isn't the production spec (or identical to it).
    alt_specs = spec_df[
        ~spec_df["specification"].str.startswith("PRODUCTION") & (spec_df["spearman_vs_production"] < 1 - 1e-12)
    ].reset_index(drop=True)

    # Land-cover source: every greenery source except the reference.
    alt_swap = swap_df[~swap_df["is_reference"]].reset_index(drop=True)

    rows = []
    if len(weighting_changed):
        rows.append(_row("Weighting (PCA-derived vs equal)", weighting_changed,
                         pd.Series(["equal weights"] * len(weighting_changed)), floor, top_n))
    if len(resolution_changed):
        rows.append(_row("Heat-map resolution (30 m vs downscaled 10 m)", resolution_changed, resolution["variant"], floor, top_n))
    if len(alt_specs):
        rows.append(_row("Sensitivity formula", alt_specs[f"top{top_n}_changed"], alt_specs["specification"], floor, top_n))
    if len(alt_swap):
        rows.append(_row("Land-cover source (RF / U-Net / NDVI proxy vs hybrid)", alt_swap[f"top{top_n}_changed"],
                         alt_swap["greenery_source"], floor, top_n))

    summary = pd.DataFrame(rows).sort_values("top_changed_max", ascending=False, kind="stable").reset_index(drop=True)
    noise_row = pd.DataFrame([{
        "choice": f"MEASUREMENT NOISE ALONE (expected top-{top_n} replaced)", "alternatives_tested": pd.NA,
        "top_changed_min": pd.NA, "top_changed_max": round(floor, 1),
        "largest_change_from": "bootstrap over exposure + greenery noise", "vs_noise_floor": "yardstick",
    }])
    # The verdict (vs_noise_floor) sits next to the numbers it judges: a wide table is clipped on the right in the dashboard.
    column_order = ["choice", "alternatives_tested", "top_changed_min", "top_changed_max", "vs_noise_floor", "largest_change_from"]
    summary = pd.concat([noise_row, summary], ignore_index=True)[column_order]
    summary["alternatives_tested"] = summary["alternatives_tested"].astype("Int64")
    summary["top_changed_min"] = summary["top_changed_min"].astype("Float64")
    summary["top_changed_max"] = summary["top_changed_max"].astype(float)
    return summary
