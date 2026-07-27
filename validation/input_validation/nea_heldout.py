"""NEA held-out validation (from nea_heldout_lst.ipynb): the same
data.gov.sg air-temperature API the Week-1 gate (G2) already checked,
aggregated per subzone as an independent LST cross-check.

⚠️ Read before trusting the output: NEA stations measure AIR temperature
(~1.5m, shaded instruments), not LAND SURFACE temperature (the satellite
thermal band the heat-layer variants are built from). LST typically runs
several degrees higher than air temp over built-up surfaces, especially near
satellite overpass time. This is the best independent, zero-cost validation
source available in scope — but it's a proxy with a systematic offset baked
in, not clean ground truth. State that plainly as a limitation; don't
present RMSE against this as validated-against-true-LST. Spearman / top-N
overlap (see validation/score_validation/rank_impact.py) are less sensitive
to this since they're rank-based, not magnitude-based.

Coverage is inherently sparse: Singapore has on the order of a few dozen
weather stations against ~330 subzones, so most subzones have no matched
station and won't appear in the output.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.ingest.nea import fetch_station_metadata, sample_readings


def assign_stations_to_subzones(stations_df: pd.DataFrame, subzones_gdf: gpd.GeoDataFrame, id_property: str) -> dict:
    """Point-in-polygon join. Stations outside every subzone polygon (e.g.
    offshore buoys, reservoir edges) are expected and dropped, not an error."""
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=[Point(xy) for xy in zip(stations_df["longitude"], stations_df["latitude"])],
        crs="EPSG:4326",
    )
    if subzones_gdf.crs is None or subzones_gdf.crs.to_epsg() != 4326:
        subzones_gdf = subzones_gdf.to_crs(epsg=4326)

    joined = gpd.sjoin(stations_gdf, subzones_gdf[[id_property, "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={id_property: "subzone_id"})

    n_matched = joined["subzone_id"].notna().sum()
    n_with_station = joined["subzone_id"].nunique()
    print(f"Stations matched to a subzone: {n_matched} / {len(joined)}")
    print(f"Distinct subzones with >=1 station: {n_with_station} / {len(subzones_gdf)} "
          f"({100 * n_with_station / len(subzones_gdf):.1f}%) — low coverage is expected, not a bug.")

    return joined.dropna(subset=["subzone_id"])[["station_id", "subzone_id"]].set_index("station_id")["subzone_id"].to_dict()


def build_nea_heldout(
    subzones_gdf: gpd.GeoDataFrame, id_property: str, heat_subzone_ids: pd.Series,
    n_sample_days: int = 20, n_months_back: int = 6, seed: int = 42, api_key: str = "",
) -> pd.DataFrame:
    """Returns [subzone_id, lst_heldout_c] — per-subzone mean air temperature
    across `n_sample_days` sampled dates."""
    stations_df = fetch_station_metadata(api_key=api_key)
    station_to_subzone = assign_stations_to_subzones(stations_df, subzones_gdf, id_property)

    readings_df = sample_readings(n_sample_days, n_months_back, seed, api_key=api_key)
    readings_df["subzone_id"] = readings_df["station_id"].map(station_to_subzone)
    n_unmatched = readings_df["subzone_id"].isna().sum()
    readings_df = readings_df.dropna(subset=["subzone_id"])
    print(f"Readings dropped (station not in any subzone): {n_unmatched}")

    heldout_df = (
        readings_df.groupby("subzone_id")
        .agg(lst_heldout_c=("value", "mean"), n_readings=("value", "count"))
        .reset_index()
    )

    heat_ids = set(heat_subzone_ids.astype(str))
    heldout_df = heldout_df[heldout_df["subzone_id"].astype(str).isin(heat_ids)].copy()
    print(f"Final held-out table: {len(heldout_df)} subzones "
          f"({100 * len(heldout_df) / len(heat_ids):.1f}% of {len(heat_ids)} total)")
    print("⚠️  Values are air temperature, a proxy for LST with a systematic offset — "
          "state this as a limitation, don't present as validated-against-true-LST.")

    return heldout_df[["subzone_id", "lst_heldout_c"]]
