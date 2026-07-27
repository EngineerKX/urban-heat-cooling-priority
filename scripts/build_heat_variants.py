#!/usr/bin/env python
"""Build the three heat-layer variants (native30 / bicubic10 / regress10)
and their per-subzone zonal means. Replaces gee_heat_variants.ipynb.

Usage: python scripts/build_heat_variants.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee

from config.settings import (
    DRY_SEASON_MONTHS,
    INTERIM_DIR,
    LANDSAT_CLOUD_COVER_MAX,
    NATIVE_SCALE_M,
    RANDOM_SEED,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    YEARS,
)
from src.downscaling.variants import (
    build_lst_30m,
    build_s2_indices_10m,
    check_composite_coverage,
    variant_bicubic10,
    variant_native30,
    variant_regress10,
    zonal_join_variants,
)
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, fetch_subzones_geojson

OUT_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
REG_SAMPLE_N = 4000


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH

    init_ee()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))

    subzones_geojson = fetch_subzones_geojson()
    subzones_fc = as_ee_feature_collection(subzones_geojson)

    print("\n--- Building composites ---")
    lst_30m = build_lst_30m(sg_bbox, YEARS, DRY_SEASON_MONTHS, LANDSAT_CLOUD_COVER_MAX, S2_UTM_CRS, NATIVE_SCALE_M)
    s2_indices_10m = build_s2_indices_10m(sg_bbox, YEARS, DRY_SEASON_MONTHS, S2_CLOUD_PROB_MAX)

    ok, _, _ = check_composite_coverage(lst_30m, s2_indices_10m, sg_bbox, NATIVE_SCALE_M, TARGET_SCALE_M)
    if not ok:
        print("Proceeding despite low coverage — widen the season window in config/settings.py if this looks wrong.")

    print("\n--- Building variants ---")
    variants = {
        "lst_native30": variant_native30(lst_30m),
        "lst_bicubic10": variant_bicubic10(lst_30m, S2_UTM_CRS, TARGET_SCALE_M),
        "lst_regress10": variant_regress10(
            lst_30m, s2_indices_10m, sg_bbox, S2_UTM_CRS, NATIVE_SCALE_M, TARGET_SCALE_M,
            sample_n=REG_SAMPLE_N, sample_seed=RANDOM_SEED,
        ),
    }

    print("\n--- Zonal join ---")
    result_df = zonal_join_variants(variants, subzones_fc, SUBZONE_ID_PROPERTY, NATIVE_SCALE_M, TARGET_SCALE_M)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(result_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if the output CSV already exists.")
    args = parser.parse_args()
    main(force=args.force)
