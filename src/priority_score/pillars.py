"""Sensitivity and adaptive-capacity pillars for the S6 cooling-priority
score (Cooling Singapore UHV-style structure: Exposure + Sensitivity +
Adaptive capacity).

Both formulas below are carried over EXACTLY as the notebooks that first
built them flagged: placeholders, not locked S6 decisions. Do not "fix"
them without checking with whoever owns S6 — see
`config.settings.NDVI_VEGETATION_THRESHOLD` and
`SENSITIVITY_POPULATION_WEIGHT` / `SENSITIVITY_ELDERLY_WEIGHT`. (One exception,
decided 2026-09-19: the sensitivity pillar's population term is now residents
per km², not a raw count — see `SENSITIVITY_POPULATION_MEASURE`.)
"""

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

from config.settings import (
    ELDERLY_AGE_COLUMNS,
    NDVI_VEGETATION_THRESHOLD,
    S2_UTM_CRS,
    SENSITIVITY_ELDERLY_WEIGHT,
    SENSITIVITY_POPULATION_MEASURE,
    SENSITIVITY_POPULATION_WEIGHT,
    SINGSTAT_NAME_COLUMN,
    SUBZONE_ID_PROPERTY,
    TOTAL_POP_COLUMN,
)
from src.ingest.gee import add_spectral_indices, fetch_sentinel2_collection
from src.ingest.singstat import fetch_population_by_subzone
from src.landcover.hybrid import HYBRID_RASTER_PATH
from src.landcover.zonal import zonal_class_fractions
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


def subzone_areas_km2(subzones_gdf: gpd.GeoDataFrame, id_property: str = SUBZONE_ID_PROPERTY) -> pd.Series:
    """Planar area of every subzone in km², indexed by subzone id (as str).
    Projected to the project's metric CRS first — the GeoJSON is in degrees."""
    projected = subzones_gdf.to_crs(S2_UTM_CRS)
    return pd.Series((projected.geometry.area / 1e6).values, index=projected[id_property].astype(str).values)


def compute_sensitivity_raw(
    population_total: pd.Series, elderly_proportion: pd.Series, area_km2: pd.Series,
    measure: str = SENSITIVITY_POPULATION_MEASURE,
) -> pd.Series:
    """0.5 * normalize(population term) + 0.5 * normalize(elderly_proportion),
    where the population term is residents per km² ("density") or the raw
    resident count ("count"). Zero/missing area gives density 0, not inf.
    The 50/50 split is still a PLACEHOLDER; count-vs-density was decided
    2026-09-19 — see config.settings.SENSITIVITY_POPULATION_MEASURE.
    """
    if measure == "density":
        with np.errstate(divide="ignore", invalid="ignore"):
            population_term = (population_total / area_km2).replace([np.inf, -np.inf], np.nan).fillna(0)
    elif measure == "count":
        population_term = population_total
    else:
        raise ValueError(f"Unknown SENSITIVITY_POPULATION_MEASURE '{measure}', expected 'density' or 'count'.")
    return (
        SENSITIVITY_POPULATION_WEIGHT * normalize(population_term)
        + SENSITIVITY_ELDERLY_WEIGHT * normalize(elderly_proportion)
    )


def build_sensitivity_pillar(heat_subzone_ids: pd.Series, area_km2_by_subzone: pd.Series) -> pd.DataFrame:
    """Returns [subzone_id, population_total, elderly_proportion, area_km2,
    population_density_km2, sensitivity_raw].

    `sensitivity_raw` follows `compute_sensitivity_raw` (population density by
    default). Both the count and the density inputs are kept as columns so the
    alternative specification can be compared without rebuilding
    (validation/score_validation/sensitivity_specs.py). The 50/50 population/
    elderly split remains a placeholder — see the module docstring.
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
    matched["area_km2"] = matched["subzone_id"].astype(str).map(area_km2_by_subzone)
    n_no_area = int(matched["area_km2"].isna().sum())
    if n_no_area:
        print(f"⚠️  {n_no_area} subzones have no polygon area — their population density is treated as 0.")
    with np.errstate(divide="ignore", invalid="ignore"):
        matched["population_density_km2"] = (
            (matched["population_total"] / matched["area_km2"]).replace([np.inf, -np.inf], np.nan)
        )
    matched["sensitivity_raw"] = compute_sensitivity_raw(
        matched["population_total"], matched["elderly_proportion_filled"], matched["area_km2"],
    )
    return matched[[
        "subzone_id", "population_total", "elderly_proportion", "area_km2", "population_density_km2", "sensitivity_raw",
    ]]


# --- Adaptive-capacity pillar (NDVI-threshold greenery proxy) ---------------

def build_adaptive_capacity_pillar(
    sg_bbox, subzones_fc, id_property, years, months, cloud_prob_max, target_scale,
    heat_subzone_ids: pd.Series, ndvi_threshold: float = NDVI_VEGETATION_THRESHOLD,
) -> pd.DataFrame:
    """Returns [subzone_id, greenery_fraction] — an INTERIM NDVI-threshold
    proxy for vegetation fraction, standing in until the real S3 land-cover
    output (RF/U-Net hybrid) is validated and ready to use instead.
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


# --- Adaptive-capacity pillar (real S3 land-cover hybrid) -----------------

def build_adaptive_capacity_pillar_landcover(
    subzones_gdf: gpd.GeoDataFrame,
    id_property: str,
    heat_subzone_ids: pd.Series,
    raster_path=HYBRID_RASTER_PATH,
) -> pd.DataFrame:
    """Returns [subzone_id, greenery_fraction] using the validated RF/U-Net
    hybrid's vegetation fraction per subzone -- the real S3 output that
    build_adaptive_capacity_pillar's NDVI threshold was always described as
    "standing in" for. That NDVI-based function is left untouched so both
    remain callable for comparison (see ADAPTIVE_CAPACITY_SOURCE and
    scripts/build_adaptive_capacity_pillar.py's printed Spearman check).
    """
    frac_df = zonal_class_fractions(raster_path, subzones_gdf, id_property)
    ac_df = frac_df.rename(columns={"fraction_vegetation": "greenery_fraction"})
    ac_df = ac_df[["subzone_id", "greenery_fraction"]]

    heat_ids = set(heat_subzone_ids.astype(str))
    ac_ids = set(ac_df["subzone_id"].astype(str))
    n_matched = len(heat_ids & ac_ids)
    n_unmatched_heat = len(heat_ids - ac_ids)
    print(f"Matched: {n_matched} | Heat-CSV subzones with no land-cover greenery match: {n_unmatched_heat}")

    return ac_df[ac_df["subzone_id"].astype(str).isin(heat_ids)].copy()
