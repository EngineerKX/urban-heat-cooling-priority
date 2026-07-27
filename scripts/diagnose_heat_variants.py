#!/usr/bin/env python
"""Diagnose whether lst_bicubic10's lower held-out RMSE reflects a uniform
offset or heat-dependent smoothing. Replaces diagnostic_heat_variants.ipynb.

Usage: python scripts/diagnose_heat_variants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import DIAGNOSTICS_DIR, INTERIM_DIR
from validation.score_validation.heat_variant_diagnostic import (
    gap_vs_heat_level,
    pairwise_differences,
    plot_diagnostic,
)

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
PLOT_PATH = DIAGNOSTICS_DIR / "heat_variant_diagnostic.png"


def main():
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    df = pd.read_csv(HEAT_CSV_PATH)
    variants = ["lst_native30", "lst_bicubic10", "lst_regress10"]
    missing = set(variants) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Subzones: {len(df)}\n")
    print(df[variants].describe().to_string())

    df = pairwise_differences(df)
    r, p, slope, intercept = gap_vs_heat_level(df)
    plot_diagnostic(df, slope, intercept, r, PLOT_PATH)


if __name__ == "__main__":
    main()
