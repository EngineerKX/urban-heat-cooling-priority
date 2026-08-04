"""MODIS LST secondary cross-check -- in the proposal's data plan table,
never implemented in any notebook or migrated code until now. Mirrors
nea_heldout.py's structure: a second, independent LST-adjacent source to
sanity-check the heat-layer variants against.

⚠️ Read before trusting the output: MOD11A2 IS a genuine LST product (same
physical quantity as Landsat's thermal band, unlike NEA's air-temperature
proxy) -- but at 1km resolution, vs. Landsat's 30m and most subzones being
far smaller than a single MODIS pixel. This is a coarse sanity check on
RELATIVE spatial heat patterns, not a precise per-subzone validation
source. Prefer Spearman / rank correlation (see
validation/score_validation/rank_impact.py) over RMSE magnitude for the
same reason nea_heldout.py prefers rank metrics -- just for a different
underlying cause (resolution mismatch here, vs. air-temp-vs-LST offset
there).
"""

import pandas as pd

from src.utils.geo import zonal_mean


def build_modis_heldout(
    modis_lst_composite, subzones_fc, id_property: str, heat_subzone_ids: pd.Series, scale: int,
) -> pd.DataFrame:
    """Returns [subzone_id, lst_heldout_c] -- same column name as
    nea_heldout.py's output so both slot into rank_impact.rmse_vs_heldout
    unmodified."""
    heldout_df = zonal_mean(modis_lst_composite, "lst_heldout_c", subzones_fc, id_property, scale)

    heat_ids = set(heat_subzone_ids.astype(str))
    heldout_df = heldout_df[heldout_df["subzone_id"].astype(str).isin(heat_ids)].copy()
    n_before_dropna = len(heldout_df)
    # zonal_mean returns one row per requested subzone, NaN where no valid
    # pixel fell in-mask (tiny/sliver subzones) -- drop those so this table
    # matches nea_heldout.py's invariant of never containing a NaN target
    # value (rmse_vs_heldout has no NaN handling of its own).
    heldout_df = heldout_df.dropna(subset=["lst_heldout_c"])
    n_dropped = n_before_dropna - len(heldout_df)
    print(f"MODIS held-out table: {len(heldout_df)} subzones with a valid value "
          f"({n_dropped} dropped for no valid pixel in-mask) — {100 * len(heldout_df) / len(heat_ids):.1f}% "
          f"of {len(heat_ids)} total.")
    print("⚠️  1km MODIS pixels are far coarser than most subzones — treat as a coarse spatial-pattern "
          "sanity check, not precise per-subzone validation. Prefer rank-based comparison (Spearman) over RMSE.")
    return heldout_df
