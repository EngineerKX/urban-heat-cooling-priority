"""Season-window diagnostic (from season_window_diagnostic.ipynb): tests
candidate season definitions against actual Landsat scene counts so C4's
"season-controlled" window is picked from real numbers instead of a guess.
Locked result (see config.settings.DRY_SEASON_MONTHS): both inter-monsoon
periods (Apr/May + Oct/Nov), which gave the most usable scenes of the
candidates tested at CLOUD_COVER_MAX=70.
"""

import pandas as pd

from src.ingest.gee import date_filter_for_years_months

MIN_DEFENSIBLE_SCENES = 8  # rough bar: fewer than this is closer to averaging a handful
                            # of arbitrary dates than computing a robust seasonal signal


def count_landsat_scenes(sg_bbox, years, months, cloud_max) -> int:
    import ee

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(sg_bbox).filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(sg_bbox).filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
    merged = date_filter_for_years_months(l8.merge(l9), years, months)
    return merged.size().getInfo()


def run_season_window_candidates(sg_bbox, years, candidates: dict, cloud_thresholds: list) -> pd.DataFrame:
    """`candidates`: {label: [month, ...]}. Runs every candidate x every
    cloud threshold (real, non-cached `.getInfo()` calls — this is a
    one-off diagnostic, not something that needs to run on every pipeline
    invocation)."""
    rows = []
    for name, months in candidates.items():
        for cloud_max in cloud_thresholds:
            n = count_landsat_scenes(sg_bbox, years, months, cloud_max)
            rows.append({"candidate": name, "cloud_cover_max": cloud_max, "n_scenes": n})
            print(f"  {name} | cloud<{cloud_max}: {n} scenes")
    return pd.DataFrame(rows)


def recommend(results_df: pd.DataFrame, cloud_cover_max_col_value: int, min_defensible_scenes: int = MIN_DEFENSIBLE_SCENES):
    pivot = results_df.pivot(index="candidate", columns="cloud_cover_max", values="n_scenes")
    print(pivot.to_string())
    print("\n--- Recommendation guidance (not an automatic decision) ---")
    for name in pivot.index:
        n = pivot.loc[name, cloud_cover_max_col_value]
        flag = "✅ likely defensible" if n >= min_defensible_scenes else "⚠️  thin — treat as a stated limitation if used"
        print(f"  {name}: {n} scenes at cloud<{cloud_cover_max_col_value} -> {flag}")
    return pivot
