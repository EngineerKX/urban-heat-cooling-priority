"""URA Master Plan subzone boundaries (data.gov.sg), cached locally.

This generalizes the ONE caching pattern that already existed in the
original notebooks (`train_unet.ipynb`'s UN1.2, which cached the same fetch
to a Drive folder) — every other notebook re-downloaded this ~330-feature
GeoJSON from data.gov.sg on every single run.
"""

from pathlib import Path

import ee
import geopandas as gpd
import requests

from config.settings import RAW_URA_SUBZONES_DIR, SUBZONE_DATASET_ID, SUBZONE_ID_PROPERTY
from src.utils.caching import load_or_fetch_json
from src.utils.geo import sanitize_dotted_properties

CACHE_PATH = RAW_URA_SUBZONES_DIR / "subzones.geojson"


def _fetch_datagovsg_geojson(dataset_id: str) -> dict:
    poll_url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download"
    r = requests.get(poll_url)
    r.raise_for_status()
    payload = r.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"data.gov.sg API error: {payload.get('errMsg')}")
    download_url = payload["data"]["url"]
    geojson = requests.get(download_url).json()
    return sanitize_dotted_properties(geojson)


def fetch_subzones_geojson(force: bool = False) -> dict:
    """GeoJSON dict, fetched once and cached at data/raw/ura_subzones/subzones.geojson."""

    def _fetch():
        print("No cache found — fetching URA subzones from data.gov.sg (one-time).")
        return _fetch_datagovsg_geojson(SUBZONE_DATASET_ID)

    geojson = load_or_fetch_json(CACHE_PATH, _fetch, force=force)
    n_features = len(geojson.get("features", []))
    if n_features == 0:
        raise ValueError("Subzone GeoJSON has zero features — check the cache/download before proceeding.")
    return geojson


def as_ee_feature_collection(geojson: dict) -> "ee.FeatureCollection":
    fc = ee.FeatureCollection(geojson)
    sample_props = fc.first().propertyNames().getInfo()
    if SUBZONE_ID_PROPERTY not in sample_props:
        candidates = [p for p in sample_props if "name" in p.lower() or "subzone" in p.lower() or "sz" in p.lower()]
        raise ValueError(
            f"'{SUBZONE_ID_PROPERTY}' not found in subzone properties. "
            f"Candidates: {candidates if candidates else sample_props}"
        )
    return fc


def as_geodataframe(geojson: dict) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    if SUBZONE_ID_PROPERTY not in gdf.columns:
        candidates = [c for c in gdf.columns if "name" in c.lower() or "subzone" in c.lower()]
        raise ValueError(f"'{SUBZONE_ID_PROPERTY}' not found. Candidates: {candidates}")
    return gdf


def dissolve_boundary(fc: "ee.FeatureCollection") -> "ee.Geometry":
    """Dissolve all subzone polygons into Singapore's real land boundary —
    used as the actual sampling/training region instead of a loose bbox
    (keeps points off open sea and out of neighboring-country slivers)."""
    boundary = fc.union(1).first().geometry()
    area_km2 = boundary.area(1).divide(1e6).getInfo()
    print(f"Dissolved Singapore boundary built. Approx area: {area_km2:,.1f} km²")
    print("(Sanity check: Singapore's land area is ~730-735 km².)")
    return boundary
