"""S4 feature table: dry/wet-season LST + NDVI/NDBI deltas per subzone --
the genuine temporal signal behind the hotspot-typology clustering, without
duplicating C4's own multi-year composite scope (see
config.settings.WET_SEASON_MONTHS). Reuses the existing variant-builder
machinery (src/downscaling/variants.py) called twice -- once per season --
rather than any new GEE fetch logic.
"""

import pandas as pd

from src.downscaling.variants import build_lst_30m, build_s2_indices_10m
from src.utils.geo import zonal_mean


def build_seasonal_feature_table(
    sg_bbox, subzones_fc, id_property, years, dry_months, wet_months,
    cloud_cover_max, cloud_prob_max, crs, native_scale, target_scale,
) -> pd.DataFrame:
    """One row per subzone: lst_dry, lst_wet, ndvi_dry, ndvi_wet, ndbi_dry,
    ndbi_wet, plus lst/ndvi/ndbi_seasonal_delta = wet - dry. LST reduced at
    native 30m, indices at native 10m -- matches how each is normally
    reduced elsewhere in this pipeline (see zonal_join_variants)."""
    merged = None
    for season_label, months in (("dry", dry_months), ("wet", wet_months)):
        print(f"--- Building {season_label}-season composite (months={months}) ---")
        lst_30m = build_lst_30m(sg_bbox, years, months, cloud_cover_max, crs, native_scale)
        s2_indices_10m = build_s2_indices_10m(sg_bbox, years, months, cloud_prob_max)

        lst_df = zonal_mean(lst_30m, f"lst_{season_label}", subzones_fc, id_property, native_scale)
        ndvi_df = zonal_mean(s2_indices_10m.select("NDVI"), f"ndvi_{season_label}", subzones_fc, id_property, target_scale)
        ndbi_df = zonal_mean(s2_indices_10m.select("NDBI"), f"ndbi_{season_label}", subzones_fc, id_property, target_scale)

        season_df = lst_df.merge(ndvi_df, on="subzone_id").merge(ndbi_df, on="subzone_id")
        merged = season_df if merged is None else merged.merge(season_df, on="subzone_id", how="outer")

    merged["lst_seasonal_delta"] = merged["lst_wet"] - merged["lst_dry"]
    merged["ndvi_seasonal_delta"] = merged["ndvi_wet"] - merged["ndvi_dry"]
    merged["ndbi_seasonal_delta"] = merged["ndbi_wet"] - merged["ndbi_dry"]

    n_total = len(merged)
    n_complete = merged.dropna().shape[0]
    print(f"\nSeasonal feature table: {n_total} subzones, {n_complete} with complete dry+wet features.")
    return merged
