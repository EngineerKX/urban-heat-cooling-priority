#!/usr/bin/env python
"""Read-only diagnostic comparing all 3 LST variants against the MODIS
held-out table, alongside the NEA held-out comparison for context. Same
"build vs. diagnose" split as scripts/diagnose_heat_variants.py. Uses
rank_impact.heldout_agreement, which splits the disagreement into the
systematic offset, the offset-removed spread and the rank correlation -- it
accepts any held-out df with an lst_heldout_c column, and both
nea_heldout.py and modis_heldout.py produce exactly that.

RMSE is printed only in the last column, to make one point visible: it is
almost exactly sqrt(offset^2 + spread^2), i.e. it mostly restates the
systematic offset (which can't move a rank), so don't read it as accuracy.

Usage: python scripts/diagnose_modis_crosscheck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import INTERIM_DIR, VARIANT_COLUMNS
from validation.score_validation.rank_impact import heldout_agreement

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
MODIS_CSV_PATH = INTERIM_DIR / "modis_heldout_lst.csv"
NEA_CSV_PATH = INTERIM_DIR / "nea_heldout_lst.csv"


def main():
    if not MODIS_CSV_PATH.exists():
        raise FileNotFoundError(f"{MODIS_CSV_PATH} not found — run scripts/build_modis_heldout.py first.")
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    heat = pd.read_csv(HEAT_CSV_PATH)
    sources = {"MODIS": pd.read_csv(MODIS_CSV_PATH)}
    if NEA_CSV_PATH.exists():
        sources["NEA"] = pd.read_csv(NEA_CSV_PATH)

    rows = []
    for source, heldout in sources.items():
        for variant in VARIANT_COLUMNS:
            a = heldout_agreement(heat, variant, heldout)
            rows.append({
                "source": source, "variant": variant, "n": a["n"], "mean_offset_c": a["mean_offset_c"],
                "spread_c": a["spread_c"], "spearman": a["spearman"], "rmse_c (mostly offset)": a["rmse_c"],
            })

    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"))
    print("\nRead mean_offset_c as a systematic gap (Landsat reads hotter than both references; it can't move a rank), "
          "spread_c as the disagreement that CAN reorder subzones, and spearman as rank agreement — the numbers that "
          "matter for a ranking tool. rmse_c is shown only to make visible that it mostly restates the offset.")
    print("⚠️  MODIS compares 1km pixels against subzone-scale polygons, so spread_c is an upper bound on random error "
          "— see validation/input_validation/modis_heldout.py's module docstring for the full limitations.")
    if "NEA" in sources:
        print("NEA is shown for context only: 12 subzones, and it measures air temperature, not surface temperature "
              "(see validation/input_validation/nea_heldout.py). The two cross-checks have DIFFERENT failure modes, "
              "which is exactly why using both is more informative than either alone.")


if __name__ == "__main__":
    main()
