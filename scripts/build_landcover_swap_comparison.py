#!/usr/bin/env python
"""Land-cover classifier swap: rerun the priority score with the greenery
fraction taken from each classifier's raster (RF, U-Net, hybrid) and from the
NDVI-threshold proxy, everything else held fixed, and count how many of the
top-20 subzones change versus the production (hybrid) score. This is the
decision-level counterpart to the accuracy/F1 evaluation in
scripts/evaluate_landcover_classifiers.py -- see
validation/score_validation/landcover_impact.py for what it can and can't say.

Recomputes on its own when any input (a raster, or a pillar CSV) is newer than
the output -- the plain "skip if the file exists" check most build scripts use
can't notice a rebuilt upstream file. `--force` still overrides.

Usage: python scripts/build_landcover_swap_comparison.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROCESSED_DIR, SUBZONE_ID_PROPERTY, UNET_CLASSIFIED_RASTER_PATH
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.landcover.hybrid import HYBRID_RASTER_PATH
from src.landcover.rf_baseline import RF_RASTER_PATH
from src.landcover.zonal import zonal_class_fractions
from src.priority_score.io import ADAPTIVE_CSV_PATH, HEAT_CSV_PATH, SENSITIVITY_CSV_PATH, load_and_join
from src.utils.caching import is_stale
from validation.score_validation.landcover_impact import run_landcover_swap

OUT_PATH = PROCESSED_DIR / "landcover_swap_comparison.csv"
RASTERS = {"RF": RF_RASTER_PATH, "U-Net": UNET_CLASSIFIED_RASTER_PATH, "hybrid": HYBRID_RASTER_PATH}
REFERENCE_LABEL = "hybrid"


def main(force: bool = False):
    inputs = [*RASTERS.values(), ADAPTIVE_CSV_PATH, SENSITIVITY_CSV_PATH, HEAT_CSV_PATH]
    if not force and not is_stale(OUT_PATH, inputs):
        print(f"{OUT_PATH} is up to date with its inputs — skipping (pass --force to rebuild).")
        return OUT_PATH

    missing = [f"{label} ({path})" for label, path in RASTERS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing land-cover raster(s): {', '.join(missing)} — run scripts/train_landcover_rf.py, "
            f"scripts/run_landcover_unet_inference.py and scripts/build_landcover_hybrid.py first."
        )

    df, _ = load_and_join(toy_mode=False)
    subzones_gdf = as_geodataframe(fetch_subzones_geojson())

    greenery_columns = {}
    for label, path in RASTERS.items():
        print(f"\n--- vegetation fraction from the {label} raster ({path.name}) ---")
        fractions = zonal_class_fractions(path, subzones_gdf, SUBZONE_ID_PROPERTY)[["subzone_id", "fraction_vegetation"]]
        column = f"greenery_{label.lower().replace('-', '')}"
        df = df.merge(fractions.rename(columns={"fraction_vegetation": column}), on="subzone_id", how="left")
        greenery_columns[label] = column

    if "greenery_fraction_ndvi" in df.columns:
        greenery_columns["NDVI-threshold proxy"] = "greenery_fraction_ndvi"

    # The reference row should reproduce the production score. If the adaptive-capacity
    # pillar was built from an older hybrid raster it won't -- say so instead of comparing silently.
    if "greenery_fraction_landcover" in df.columns:
        gap = float((df["greenery_hybrid"] - df["greenery_fraction_landcover"]).abs().max())
        if gap > 1e-6:
            print(f"\n⚠️  The adaptive-capacity pillar's land-cover greenery differs from the current hybrid raster "
                  f"(max gap {gap:.3f}) — rebuild it with scripts/build_adaptive_capacity_pillar.py --force, or the "
                  f"reference row here won't match the production score.")

    print("\n=== Priority score under each greenery source (reference: hybrid) ===")
    result = run_landcover_swap(df, greenery_columns, reference_label=REFERENCE_LABEL)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
