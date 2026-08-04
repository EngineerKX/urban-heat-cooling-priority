"""Loads + joins the 3 pillar CSVs (heat variants, sensitivity, adaptive
capacity) into the single wide table build_score/bootstrap_priority_score
operate on. Extracted from scripts/build_priority_score.py so
scripts/build_priority_score_confidence_bands.py (S6 confidence bands)
doesn't duplicate this join logic.
"""

import numpy as np
import pandas as pd

from config.settings import INTERIM_DIR, RANDOM_SEED, VARIANT_COLUMNS

HEAT_CSV_PATH = INTERIM_DIR / "heat_variants_subzone.csv"
SENSITIVITY_CSV_PATH = INTERIM_DIR / "sensitivity_pillar.csv"
ADAPTIVE_CSV_PATH = INTERIM_DIR / "adaptive_capacity_pillar.csv"
HELDOUT_CSV_PATH = INTERIM_DIR / "nea_heldout_lst.csv"


def make_toy_pillars(subzone_ids: pd.Series, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    sensitivity = pd.DataFrame({"subzone_id": subzone_ids, "sensitivity_raw": rng.uniform(0.3, 1.0, size=len(subzone_ids))})
    adaptive = pd.DataFrame({"subzone_id": subzone_ids, "greenery_fraction": rng.uniform(0.05, 0.6, size=len(subzone_ids))})
    return sensitivity, adaptive


def load_and_join(toy_mode: bool = False):
    if not HEAT_CSV_PATH.exists():
        raise FileNotFoundError(f"{HEAT_CSV_PATH} not found — run scripts/build_heat_variants.py first.")
    heat = pd.read_csv(HEAT_CSV_PATH)
    missing = set(VARIANT_COLUMNS) - set(heat.columns)
    if missing:
        raise ValueError(f"{HEAT_CSV_PATH} is missing columns {missing}.")
    print(f"Loaded {len(heat)} subzones from heat-variants CSV.")

    if toy_mode:
        print("⚠️  TOY_MODE: seeded placeholder sensitivity/adaptive pillars. Not real data — do not report these numbers.")
        sensitivity, adaptive = make_toy_pillars(heat["subzone_id"])
    else:
        for label, path in [("sensitivity", SENSITIVITY_CSV_PATH), ("adaptive capacity", ADAPTIVE_CSV_PATH)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found (needed for the {label} pillar). Run the corresponding build_*_pillar.py "
                    f"script first, or pass --toy to test wiring with placeholder data."
                )
        sensitivity = pd.read_csv(SENSITIVITY_CSV_PATH)
        adaptive = pd.read_csv(ADAPTIVE_CSV_PATH)

    df = heat.merge(sensitivity, on="subzone_id", how="inner").merge(adaptive, on="subzone_id", how="inner")
    n_dropped = len(heat) - len(df)
    print(f"Join summary: {len(heat)} -> {len(df)} subzones retained after pillar join.")
    if n_dropped:
        print(f"⚠️  {n_dropped} subzones dropped — check subzone_id spelling/casing across files.")

    heldout = pd.read_csv(HELDOUT_CSV_PATH) if HELDOUT_CSV_PATH.exists() else None
    if heldout is None:
        print(f"ℹ️  {HELDOUT_CSV_PATH} not found — lst_rmse_heldout will be NaN (run scripts/build_nea_heldout.py to fill it in).")

    return df, heldout
