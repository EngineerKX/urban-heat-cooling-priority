"""Per-subzone class-fraction zonal stats for a LOCAL categorical raster
(the land-cover ensemble). Counterpart to src/utils/geo.py::zonal_mean,
which only handles GEE ee.Image inputs -- the land-cover ensemble is a
rasterio-readable local GeoTIFF instead, so it needs its own zonal tool
rather than a round-trip back through Earth Engine.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask

from config.settings import BUCKET_NAMES, SUBZONE_ID_PROPERTY


def zonal_class_fractions(
    raster_path: Path,
    subzones_gdf: gpd.GeoDataFrame,
    id_property: str = SUBZONE_ID_PROPERTY,
    class_names: dict = BUCKET_NAMES,
) -> pd.DataFrame:
    """Fraction of each land-cover class per subzone polygon, from a single-
    band categorical raster where 0 means nodata/invalid (see
    src/landcover/ensemble.py's convention -- bucket ids are 1-indexed, 0 is
    never a real class).

    Returns [subzone_id, fraction_<class> for each class_names value,
    n_valid_pixels], plus a printed count of subzones with zero valid
    pixels (mirrors src/utils/geo.py::zonal_mean's convention).
    """
    raster_path = Path(raster_path)
    class_ids = sorted(class_names.keys())

    with rasterio.open(raster_path) as src:
        gdf = subzones_gdf if subzones_gdf.crs == src.crs else subzones_gdf.to_crs(src.crs)

        rows = []
        for _, feature in gdf.iterrows():
            row = {"subzone_id": feature[id_property]}
            try:
                out_image, _ = rasterio.mask.mask(src, [feature.geometry.__geo_interface__], crop=True, nodata=0)
                band = out_image[0]
                valid = band != 0
                n_valid = int(valid.sum())
            except ValueError:
                # Geometry doesn't overlap the raster at all.
                n_valid = 0
                band = valid = None

            row["n_valid_pixels"] = n_valid
            for class_id in class_ids:
                frac = float((band[valid] == class_id).sum() / n_valid) if n_valid else np.nan
                row[f"fraction_{class_names[class_id]}"] = frac
            rows.append(row)

    df = pd.DataFrame(rows)
    n_total = len(df)
    n_zero = int((df["n_valid_pixels"] == 0).sum())
    print(f"Zonal class fractions: {n_total} subzones, {n_zero} with zero valid pixels.")
    if n_zero:
        print("  Expected for tiny/sliver subzones outside the raster's coverage — confirm this isn't most of them.")
    return df
