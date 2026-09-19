"""Sensitivity-pillar SPECIFICATION comparison. The sensitivity pillar's
uncertainty is not measurement noise -- SingStat counts are near-exact
administrative data -- it is how those counts are turned into a score (raw
count vs. density, the population/elderly split, which elderly measure). The
S6 confidence-band bootstrap therefore deliberately leaves the pillar alone
(see confidence_bands.py), and this module reports the formula uncertainty
separately, as a decision-level result: how many of the top-N subzones change
under each defensible alternative, everything else held fixed.

The reference row is the PRODUCTION pillar (the `sensitivity_raw` column the
score is actually built from), so the table tracks
config.settings.SENSITIVITY_POPULATION_MEASURE automatically.
"""

import pandas as pd
from scipy.stats import spearmanr

from config.settings import REFERENCE_VARIANT, SENSITIVITY_POPULATION_MEASURE, TOP_N
from src.priority_score.score import build_score
from src.utils.geo import normalize
from validation.score_validation.rank_impact import top_n_overlap

REQUIRED_COLUMNS = ["population_total", "elderly_proportion", "area_km2", "greenery_fraction", "sensitivity_raw"]


def sensitivity_specifications(df: pd.DataFrame) -> dict:
    """{label: raw sensitivity series} for every specification compared. All
    are 0.5/0.5 blends of two min-max-normalised terms unless the label says
    otherwise; the pillar is re-normalised inside build_score."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} — rebuild the sensitivity pillar (scripts/build_sensitivity_pillar.py --force).")

    population = df["population_total"].astype(float)
    density = (population / df["area_km2"]).where(df["area_km2"] > 0, 0.0).fillna(0.0)
    share = df["elderly_proportion"].fillna(0)
    elderly_density = (population * share / df["area_km2"]).where(df["area_km2"] > 0, 0.0).fillna(0.0)

    return {
        f"PRODUCTION (config: population {SENSITIVITY_POPULATION_MEASURE} + elderly share)": df["sensitivity_raw"],
        "population density + elderly share": 0.5 * normalize(density) + 0.5 * normalize(share),
        "population count + elderly share": 0.5 * normalize(population) + 0.5 * normalize(share),
        "density 0.75 / elderly share 0.25": 0.75 * normalize(density) + 0.25 * normalize(share),
        "density 0.25 / elderly share 0.75": 0.25 * normalize(density) + 0.75 * normalize(share),
        "elderly share only": normalize(share),
        "population density only": normalize(density),
        "elderly density only (elderly residents per km²)": normalize(elderly_density),
        "elderly density + elderly share": 0.5 * normalize(elderly_density) + 0.5 * normalize(share),
    }


def run_sensitivity_spec_comparison(
    df: pd.DataFrame, exposure_col: str = REFERENCE_VARIANT, top_n: int = TOP_N,
) -> pd.DataFrame:
    """One row per specification: Spearman and top-N change versus the
    production score, plus the PCA weights that specification produces. The
    production row is its own reference (Spearman 1.0, 0 changed)."""
    specs = sensitivity_specifications(df)
    reference_score, _ = build_score(df, exposure_col, "pca")

    rows = []
    for label, raw in specs.items():
        score, weights = build_score(df.assign(sensitivity_raw=raw), exposure_col, "pca")
        corr, _ = spearmanr(score, reference_score)
        rows.append({
            "specification": label,
            "spearman_vs_production": corr,
            f"top{top_n}_changed": top_n - round(top_n_overlap(score, reference_score, top_n) * top_n),
            "pca_weight_exposure": weights["exposure"],
            "pca_weight_sensitivity": weights["sensitivity"],
            "pca_weight_adaptive_deficit": weights["adaptive_deficit"],
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return result
