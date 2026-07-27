"""Small geo/tabular helpers reused across several notebooks verbatim:
min-max normalization, the data.gov.sg dotted-property-name fix (Earth
Engine rejects ArcGIS-style `SHAPE.AREA`/`SHAPE.LEN` fields outright), and a
"pull once, count what got dropped" zonal-stats wrapper.
"""

import ee
import pandas as pd


def normalize(s: pd.Series) -> pd.Series:
    """Min-max normalize. Returns all-zero if the series is constant."""
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())


def sanitize_dotted_properties(geojson: dict) -> dict:
    """Earth Engine rejects property names containing '.' (e.g. ArcGIS-style
    SHAPE.AREA / SHAPE.LEN fields common in data.gov.sg exports) with
    "Invalid property name". Rename them in place before uploading."""
    renamed = set()
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        for old_key in list(props.keys()):
            if "." in old_key:
                new_key = old_key.replace(".", "_")
                props[new_key] = props.pop(old_key)
                renamed.add(f"{old_key} -> {new_key}")
    if renamed:
        print(f"Sanitized {len(renamed)} property name(s) containing '.': {sorted(renamed)}")
    return geojson


def zonal_mean(
    image: "ee.Image",
    band_name: str,
    subzones_fc: "ee.FeatureCollection",
    id_property: str,
    scale: int,
    tile_scale: int = 4,
) -> pd.DataFrame:
    """Zonal mean of a SINGLE band over every subzone, pulled down with one
    `.getInfo()` call (repeated `.getInfo()`/`.size()` calls on a lazy EE
    graph re-run the entire computation each time — that redundant
    recomputation was the root cause of at least one "why is this cell so
    slow" bug in the original notebooks).

    Note on Earth Engine's own naming quirk: `reduceRegions` with a
    single-output reducer (`ee.Reducer.mean()`) over a single-band image
    names the output property after the REDUCER ("mean"), not the band —
    confirmed empirically in the source notebooks. Renamed to `band_name`
    here so callers don't have to know that.

    Returns a DataFrame with columns [subzone_id, band_name], plus a printed
    count of subzones with a null value (no valid pixels in-mask — expected
    for tiny/sliver subzones, but worth eyeballing that it's not most of them).
    """
    reduced = image.rename(band_name).reduceRegions(
        collection=subzones_fc, reducer=ee.Reducer.mean(), scale=scale, tileScale=tile_scale,
    )
    records = reduced.getInfo()["features"]
    rows = [
        {"subzone_id": f["properties"].get(id_property), band_name: f["properties"].get("mean")}
        for f in records
    ]
    df = pd.DataFrame(rows)

    n_total = len(df)
    n_null = df[band_name].isna().sum() if n_total else 0
    print(f"Zonal mean '{band_name}': {n_total} subzones, {n_null} null (no valid pixels in-mask).")
    if n_null:
        print("  Expected for tiny/sliver subzones — confirm this isn't most of them.")
    return df
