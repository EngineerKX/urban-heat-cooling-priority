"""Sensitivity and adaptive-capacity pillars for the S6 cooling-priority
score (Cooling Singapore UHV-style structure: Exposure + Sensitivity +
Adaptive capacity).

Both formulas below are carried over EXACTLY as the notebooks that first
built them flagged: placeholders, not locked S6 decisions. Do not "fix"
them without checking with whoever owns S6 — see
`config.settings.NDVI_VEGETATION_THRESHOLD` and
`SENSITIVITY_POPULATION_WEIGHT` / `SENSITIVITY_ELDERLY_WEIGHT`.
"""

import ee
import pandas as pd

from config.settings import (
    ELDERLY_AGE_COLUMNS,
    NDVI_VEGETATION_THRESHOLD,
    SENSITIVITY_ELDERLY_WEIGHT,
    SENSITIVITY_POPULATION_WEIGHT,
    SINGSTAT_NAME_COLUMN,
    TOTAL_POP_COLUMN,
)
from src.ingest.gee import add_spectral_indices, fetch_sentinel2_collection
from src.ingest.singstat import fetch_population_by_subzone
from src.utils.geo import normalize, zonal_mean


# --- Sensitivity pillar (SingStat population + elderly) --------------------

def _clean_population_table(pop_raw: pd.DataFrame) -> pd.DataFrame:
    """Drops the Singapore-wide 'Total' row and every planning-area subtotal
    row ('<Planning Area> - Total'); coerces suppressed cells ('-') to 0."""
    pop = pop_raw.copy()
    is_pa_total = pop[SINGSTAT_NAME_COLUMN].str.contains(" - Total", regex=False, na=False)
    is_grand_total = pop[SINGSTAT_NAME_COLUMN] == "Total"
    n_pa_total, n_grand_total = is_pa_total.sum(), is_grand_total.sum()
    pop = pop[~is_pa_total & ~is_grand_total].copy()
    print(f"Dropped {n_pa_total} planning-area total rows and {n_grand_total} grand-total row.")

    numeric_cols = [TOTAL_POP_COLUMN] + ELDERLY_AGE_COLUMNS
    for col in numeric_cols:
        pop[col] = pd.to_numeric(pop[col].replace("-", "0"), errors="coerce")

    n_nan = pop[numeric_cols].isna().any(axis=1).sum()
    if n_nan:
        print(f"⚠️  {n_nan} rows have non-numeric values outside the expected '-' pattern — inspect before trusting downstream numbers.")
    return pop


def build_sensitivity_pillar(heat_subzone_ids: pd.Series) -> pd.DataFrame:
    """Returns [subzone_id, population_total, elderly_proportion, sensitivity_raw].

    `sensitivity_raw` is PLACEHOLDER: 0.5 * normalize(population_total) +
    0.5 * normalize(elderly_proportion). Whether population should be a raw
    count vs. density, and whether 50/50 is the right split, is an open S6
    call — see the module docstring.
    """
    pop_raw = fetch_population_by_subzone()
    pop = _clean_population_table(pop_raw)

    pop["population_total"] = pop[TOTAL_POP_COLUMN]
    pop["elderly_total"] = pop[ELDERLY_AGE_COLUMNS].sum(axis=1)
    # 0/0 -> NaN (zero-population subzones), not 0 — a 0.0 elderly_proportion
    # would falsely read as "no elderly" instead of "no residents at all".
    pop["elderly_proportion"] = pop["elderly_total"] / pop["population_total"]

    heat_ids = heat_subzone_ids.astype(str)

    def _norm(s: str) -> str:
        return s.strip().upper()

    heat_lookup = {_norm(s): s for s in heat_ids}
    pop["_name_norm"] = pop[SINGSTAT_NAME_COLUMN].apply(_norm)
    pop["subzone_id"] = pop["_name_norm"].map(heat_lookup)

    n_matched = pop["subzone_id"].notna().sum()
    n_unmatched_heat = len(set(heat_ids) - set(pop["subzone_id"].dropna()))
    print(f"Matched: {n_matched} / {len(pop)} SingStat subzone rows")
    if n_unmatched_heat:
        print(f"⚠️  {n_unmatched_heat} heat-CSV subzones have no population match — "
              f"will be dropped from the sensitivity pillar.")

    matched = pop[pop["subzone_id"].notna()].copy()
    matched["elderly_proportion_filled"] = matched["elderly_proportion"].fillna(0)
    matched["sensitivity_raw"] = (
        SENSITIVITY_POPULATION_WEIGHT * normalize(matched["population_total"])
        + SENSITIVITY_ELDERLY_WEIGHT * normalize(matched["elderly_proportion_filled"])
    )
    return matched[["subzone_id", "population_total", "elderly_proportion", "sensitivity_raw"]]


# --- Adaptive-capacity pillar (NDVI-threshold greenery proxy) ---------------

def build_adaptive_capacity_pillar(
    sg_bbox, subzones_fc, id_property, years, months, cloud_prob_max, target_scale,
    heat_subzone_ids: pd.Series, ndvi_threshold: float = NDVI_VEGETATION_THRESHOLD,
) -> pd.DataFrame:
    """Returns [subzone_id, greenery_fraction] — an INTERIM NDVI-threshold
    proxy for vegetation fraction, standing in until the real S3 land-cover
    output (RF/U-Net ensemble) is validated and ready to use instead.
    """
    s2_masked = fetch_sentinel2_collection(sg_bbox, years, months, cloud_prob_max)
    s2_indexed = s2_masked.map(add_spectral_indices)
    ndvi_composite = s2_indexed.select("NDVI").median().clip(sg_bbox)

    vegetation_mask = ndvi_composite.gt(ndvi_threshold).rename("is_vegetation")
    ac_df = zonal_mean(vegetation_mask, "greenery_fraction", subzones_fc, id_property, target_scale)

    heat_ids = set(heat_subzone_ids.astype(str))
    ac_ids = set(ac_df["subzone_id"].astype(str))
    n_matched = len(heat_ids & ac_ids)
    n_unmatched_heat = len(heat_ids - ac_ids)
    print(f"Matched: {n_matched} | Heat-CSV subzones with no greenery match: {n_unmatched_heat}")

    return ac_df[ac_df["subzone_id"].astype(str).isin(heat_ids)].copy()
