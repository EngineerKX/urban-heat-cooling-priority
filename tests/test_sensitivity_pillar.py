#!/usr/bin/env python
"""Standalone verification for the sensitivity pillar's density/count logic
(src/priority_score/pillars.py) and the specification comparison
(validation/score_validation/sensitivity_specs.py) against synthetic data --
no GEE or network required. Same runnable-script + printed pass/fail style as
the rest of tests/.

Usage: python tests/test_sensitivity_pillar.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from config.settings import SENSITIVITY_POPULATION_MEASURE
from src.priority_score.pillars import compute_sensitivity_raw, subzone_areas_km2
from src.utils.geo import normalize
from validation.score_validation.sensitivity_specs import run_sensitivity_spec_comparison


def test_density_and_count_measures():
    pop = pd.Series([1000.0, 4000.0, 0.0, 2000.0])
    elderly = pd.Series([0.1, 0.1, 0.0, 0.1])
    area = pd.Series([1.0, 8.0, 2.0, 0.5])  # densities: 1000, 500, 0, 4000 per km2 -> count ranks #1 first, density #3

    density = compute_sensitivity_raw(pop, elderly, area, "density")
    count = compute_sensitivity_raw(pop, elderly, area, "count")
    assert np.allclose(density, 0.5 * normalize(pop / area) + 0.5 * normalize(elderly))
    assert np.allclose(count, 0.5 * normalize(pop) + 0.5 * normalize(elderly))
    assert density.idxmax() != count.idxmax(), "fixture must make count and density disagree on the top subzone"
    print("PASS: density and count measures follow their formulas and can disagree on the ranking")


def test_zero_or_missing_area_gives_zero_density_not_inf():
    pop = pd.Series([1000.0, 4000.0, 0.0, 2000.0])
    elderly = pd.Series([0.2, 0.1, 0.0, 0.3])
    area = pd.Series([1.0, 0.0, np.nan, 0.5])
    result = compute_sensitivity_raw(pop, elderly, area, "density")
    assert np.isfinite(result).all(), f"zero/missing area must not leak inf/NaN into the pillar: {result.tolist()}"
    print("PASS: zero or missing area is treated as density 0, not inf/NaN")


def test_unknown_measure_raises():
    try:
        compute_sensitivity_raw(pd.Series([1.0, 2.0]), pd.Series([0.1, 0.2]), pd.Series([1.0, 1.0]), "per_capita")
    except ValueError:
        print("PASS: an unknown SENSITIVITY_POPULATION_MEASURE raises")
        return
    raise AssertionError("expected ValueError for an unknown measure")


def test_subzone_areas_are_projected_km2():
    # A 0.009 x 0.009 degree box at Singapore's latitude is ~1 km x ~1 km.
    gdf = gpd.GeoDataFrame(
        {"SUBZONE_N": ["A"]}, geometry=[box(103.80, 1.35, 103.809, 1.359)], crs="EPSG:4326",
    )
    area = subzone_areas_km2(gdf, "SUBZONE_N")
    assert list(area.index) == ["A"]
    assert np.isclose(area["A"], 1.0, rtol=0.03), f"expected ~1 km2, got {area['A']:.3f}"
    print(f"PASS: subzone area is computed in a metric CRS ({area['A']:.3f} km2 for a ~1 km box)")


def _make_pillar_table(n=120, seed=3):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    population = np.exp(rng.normal(8.5, 1.0, n)).round()
    area = np.exp(rng.normal(0.3, 0.6, n))
    elderly = np.clip(0.13 + 0.03 * z + rng.normal(0, 0.03, n), 0, 1)
    df = pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        "lst_native30": 38 + 2.5 * z + rng.normal(0, 1.0, n),
        "greenery_fraction": np.clip(0.35 - 0.12 * z + rng.normal(0, 0.08, n), 0, 1),
        "population_total": population, "elderly_proportion": elderly, "area_km2": area,
    })
    df["sensitivity_raw"] = compute_sensitivity_raw(
        df["population_total"], df["elderly_proportion"], df["area_km2"], SENSITIVITY_POPULATION_MEASURE,
    )
    return df


def test_spec_comparison_shape_and_reference_row():
    result = run_sensitivity_spec_comparison(_make_pillar_table(), top_n=10)
    assert len(result) == 9, f"expected 9 specifications, got {len(result)}"
    reference = result.iloc[0]
    assert reference["specification"].startswith("PRODUCTION")
    assert np.isclose(reference["spearman_vs_production"], 1.0) and reference["top10_changed"] == 0
    assert result["spearman_vs_production"].between(-1, 1).all()
    weight_cols = [c for c in result.columns if c.startswith("pca_weight_")]
    assert np.allclose(result[weight_cols].sum(axis=1), 1.0)
    print("PASS: spec comparison has one row per specification, the production row is its own reference")


def test_spec_comparison_rejects_an_old_pillar_table():
    old = _make_pillar_table().drop(columns=["area_km2"])
    try:
        run_sensitivity_spec_comparison(old, top_n=10)
    except ValueError as e:
        assert "area_km2" in str(e)
        print("PASS: a pillar table without area_km2 is rejected with a rebuild hint")
        return
    raise AssertionError("expected ValueError for a table missing area_km2")


def main():
    test_density_and_count_measures()
    test_zero_or_missing_area_gives_zero_density_not_inf()
    test_unknown_measure_raises()
    test_subzone_areas_are_projected_km2()
    test_spec_comparison_shape_and_reference_row()
    test_spec_comparison_rejects_an_old_pillar_table()
    print("\nAll sensitivity-pillar checks passed.")


if __name__ == "__main__":
    main()
