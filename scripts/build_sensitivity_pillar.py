#!/usr/bin/env python
"""Build the sensitivity pillar (SingStat population + elderly proportion).
Replaces sensitivity_pillar.ipynb.

Usage: python scripts/build_sensitivity_pillar.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import (
    INTERIM_DIR,
    SENSITIVITY_ELDERLY_WEIGHT,
    SENSITIVITY_POPULATION_WEIGHT,
)
from src.priority_score.pillars import build_sensitivity_pillar

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
OUT_PATH = INTERIM_DIR / "sensitivity_pillar.csv"


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return OUT_PATH
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")

    heat_ids = pd.read_csv(HEAT_CSV_PATH)["subzone_id"]
    sp_df = build_sensitivity_pillar(heat_ids)
    print(f"\n⚠️  Reminder: {SENSITIVITY_POPULATION_WEIGHT}/{SENSITIVITY_ELDERLY_WEIGHT} population/elderly split "
          f"is a placeholder, not a locked S6 decision.")

    print("\nTop 10 by sensitivity_raw:")
    print(sp_df.sort_values("sensitivity_raw", ascending=False).head(10).to_string(index=False))

    sp_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(sp_df)} subzones)")
    return OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
