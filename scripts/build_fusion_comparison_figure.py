#!/usr/bin/env python
"""Presentation figure for the sensor-fusion step (C3): one neighbourhood
shown as Landsat heat at 30 m, Sentinel-2 greenness at 10 m, and the fused
(TsHARP-style regress10) heat at 10 m, side by side on one colour scale.

Rebuilds the composites and the regression exactly as
scripts/build_heat_variants.py does (same years, season, cloud limits, seed
and island-wide regression sample), then downloads only a small window, so
the panels match the production heat layers.

Usage: python scripts/build_fusion_comparison_figure.py [--lat 1.3105 --lon 103.8540 --half-size-m 1500]
"""

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import matplotlib.pyplot as plt
import numpy as np
import requests
from pyproj import Transformer

from config.settings import (
    DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX, NATIVE_SCALE_M, RANDOM_SEED, S2_CLOUD_PROB_MAX, S2_UTM_CRS,
    SG_BBOX, SUBZONE_ID_PROPERTY, TARGET_SCALE_M, YEARS,
)
from src.downscaling.variants import build_lst_30m, build_s2_indices_10m, variant_regress10
from src.ingest.gee import init_ee
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"
REG_SAMPLE_N = 4000  # same as scripts/build_heat_variants.py


def _download(image: ee.Image, band: str, region: ee.Geometry) -> np.ndarray:
    """One band of `image` on the 10 m UTM grid over `region`, as a 2-D array.
    The 30 m layer is sampled with nearest neighbour, so its 30 m blocks stay visible."""
    url = image.select(band).reproject(crs=S2_UTM_CRS, scale=TARGET_SCALE_M).getDownloadURL(
        {"region": region, "crs": S2_UTM_CRS, "scale": TARGET_SCALE_M, "format": "NPY"}
    )
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))[band].astype(float)


def main(lat: float, lon: float, half_size_m: float):
    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    lst_30m = build_lst_30m(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX, S2_UTM_CRS, NATIVE_SCALE_M)
    s2_indices_10m = build_s2_indices_10m(sg_bbox, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)
    regress10 = variant_regress10(
        lst_30m, s2_indices_10m, sg_bbox, S2_UTM_CRS, NATIVE_SCALE_M, TARGET_SCALE_M,
        sample_n=REG_SAMPLE_N, sample_seed=RANDOM_SEED,
    )

    to_utm = Transformer.from_crs("EPSG:4326", S2_UTM_CRS, always_xy=True)
    cx, cy = to_utm.transform(lon, lat)
    extent = (cx - half_size_m, cx + half_size_m, cy - half_size_m, cy + half_size_m)
    region = ee.Geometry.Rectangle([extent[0], extent[2], extent[1], extent[3]], proj=S2_UTM_CRS, geodesic=False)

    print("Downloading the window ...")
    native = _download(lst_30m, "LST_C", region)
    ndvi = _download(s2_indices_10m, "NDVI", region)
    fused = _download(regress10, "lst_regress10", region)

    subzones = as_geodataframe(fetch_subzones_geojson()).to_crs(S2_UTM_CRS)
    subzones = subzones.cx[extent[0]:extent[1], extent[2]:extent[3]]

    lo, hi = np.nanpercentile(np.concatenate([native.ravel(), fused.ravel()]), [2, 98])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)
    panels = [
        (native, "Landsat 8/9 thermal — 30 m", "inferno", (lo, hi), "Surface temperature (°C)"),
        (ndvi, "Sentinel-2 greenness (NDVI) — 10 m", "YlGn", (0, 0.8), "NDVI"),
        (fused, "Fused heat map — 10 m", "inferno", (lo, hi), "Surface temperature (°C)"),
    ]
    for ax, (data, title, cmap, (vmin, vmax), label) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest")
        subzones.boundary.plot(ax=ax, color="white" if cmap == "inferno" else "#333333", linewidth=0.8)
        for _, row in subzones.iterrows():
            point = row.geometry.representative_point()
            if extent[0] < point.x < extent[1] and extent[2] < point.y < extent[3]:
                ax.annotate(row[SUBZONE_ID_PROPERTY].title(), (point.x, point.y), ha="center", fontsize=7,
                            color="white" if cmap == "inferno" else "#111111")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    fig.suptitle(
        f"Sensor fusion: Landsat heat + Sentinel-2 detail → 10 m heat map  ({2 * half_size_m / 1000:.0f} × "
        f"{2 * half_size_m / 1000:.0f} km around {lat:.4f}, {lon:.4f}; dry-season composites {YEARS[0]}–{YEARS[-1]})",
        fontsize=12,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "fusion_30m_vs_10m.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=1.3105, help="Window centre latitude (default: Little India).")
    parser.add_argument("--lon", type=float, default=103.8540, help="Window centre longitude.")
    parser.add_argument("--half-size-m", type=float, default=1500, help="Half the window width, in metres.")
    args = parser.parse_args()
    main(args.lat, args.lon, args.half_size_m)
