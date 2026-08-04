#!/usr/bin/env python
"""Read-only diagnostic comparing all 3 LST variants against the MODIS
held-out table (RMSE + Spearman), alongside the existing NEA held-out
comparison for context. Same "build vs. diagnose" split as
scripts/diagnose_heat_variants.py. Reuses rank_impact.rmse_vs_heldout
unmodified -- it already accepts any heldout df with an lst_heldout_c
column, and both nea_heldout.py and modis_heldout.py produce exactly that.

Usage: python scripts/diagnose_modis_crosscheck.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

from config.settings import INTERIM_DIR, VARIANT_COLUMNS
from validation.score_validation.rank_impact import rmse_vs_heldout

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
MODIS_CSV_PATH = INTERIM_DIR / "modis_heldout_lst.csv"
NEA_CSV_PATH = INTERIM_DIR / "nea_heldout_lst.csv"


def main():
    if not MODIS_CSV_PATH.exists():
        raise FileNotFoundError(f"{MODIS_CSV_PATH} not found — run scripts/build_modis_heldout.py first.")
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    heat = pd.read_csv(HEAT_CSV_PATH)
    modis = pd.read_csv(MODIS_CSV_PATH)
    nea = pd.read_csv(NEA_CSV_PATH) if NEA_CSV_PATH.exists() else None

    rows = []
    for variant in VARIANT_COLUMNS:
        modis_rmse = rmse_vs_heldout(heat, variant, modis)
        merged = heat[["subzone_id", variant]].merge(modis, on="subzone_id", how="inner")
        modis_corr = spearmanr(merged[variant], merged["lst_heldout_c"])[0] if len(merged) > 1 else float("nan")

        row = {"variant": variant, "modis_rmse": modis_rmse, "modis_spearman": modis_corr, "n_modis": len(merged)}
        if nea is not None:
            row["nea_rmse"] = rmse_vs_heldout(heat, variant, nea)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    print(results_df.to_string(index=False, float_format=lambda x: "NaN" if pd.isna(x) else f"{x:.3f}"))
    print("\n⚠️  MODIS RMSE compares 1km pixels against subzone-scale polygons — prefer the Spearman column. "
          "See validation/input_validation/modis_heldout.py's module docstring for the full limitations.")
    if nea is not None:
        print("NEA RMSE is shown for context only — it has its own systematic air-temp-vs-LST offset "
              "(see validation/input_validation/nea_heldout.py). The two cross-checks have DIFFERENT "
              "failure modes, which is exactly why using both is more informative than either alone.")


if __name__ == "__main__":
    main()
