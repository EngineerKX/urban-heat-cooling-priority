#!/usr/bin/env python
"""Standalone verification for src/heat_model/tabular.py against a small
synthetic subzone table -- no GEE/TensorFlow required. Same runnable-script
+ printed pass/fail style as the rest of tests/.

Usage: python tests/test_heat_model_tabular.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.heat_model.tabular import (
    XGB_FEATURE_COLUMNS,
    XGB_TARGET_COLUMN,
    fit_ndvi_vegetation_slope,
    predict_counterfactual_subzone,
    redistribute_vegetation_fraction,
    train_xgb_model,
)


def _make_synthetic_table(seed=42, n=80):
    """Fractions sum to 1 by construction; LST is a deterministic linear
    function of fraction_vegetation ONLY (with light noise) so 'more
    vegetation -> lower LST' is unambiguous ground truth for the direction
    check, regardless of how well XGBoost fits the noise."""
    rng = np.random.default_rng(seed)
    fraction_water = np.full(n, 0.05)
    fraction_bare = np.full(n, 0.05)
    fraction_vegetation = rng.uniform(0.05, 0.80, size=n)
    fraction_built_up = 1.0 - fraction_vegetation - fraction_bare - fraction_water

    ndvi_dry = 0.1 + 0.6 * fraction_vegetation + rng.normal(0, 0.01, size=n)
    ndvi_wet = 0.1 + 0.6 * fraction_vegetation + rng.normal(0, 0.01, size=n)
    ndbi_dry = -0.1 + 0.3 * fraction_built_up + rng.normal(0, 0.01, size=n)
    ndbi_wet = -0.1 + 0.3 * fraction_built_up + rng.normal(0, 0.01, size=n)

    lst_native30 = 40.0 - 10.0 * fraction_vegetation + rng.normal(0, 0.3, size=n)

    df = pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        XGB_TARGET_COLUMN: lst_native30,
        "fraction_vegetation": fraction_vegetation,
        "fraction_built_up": fraction_built_up,
        "fraction_bare": fraction_bare,
        "fraction_water": fraction_water,
        "ndvi_dry": ndvi_dry, "ndvi_wet": ndvi_wet,
        "ndbi_dry": ndbi_dry, "ndbi_wet": ndbi_wet,
        "population_total": rng.uniform(1000, 20000, size=n),
        "elderly_proportion": rng.uniform(0.05, 0.25, size=n),
        "primary_cluster": rng.integers(0, 3, size=n),
    })
    df["primary_cluster"] = df["primary_cluster"].astype("category")
    return df


def test_redistribute_vegetation_fraction_preserves_total():
    result = redistribute_vegetation_fraction(veg0=0.2, built0=0.6, bare0=0.1, delta=0.15)
    total = result["fraction_vegetation"] + result["fraction_built_up"] + result["fraction_bare"]
    assert np.isclose(total, 0.9), f"expected veg+built+bare to stay at 0.9 (excl. water), got {total}"
    assert np.isclose(result["actual_delta_vegetation"], 0.15)
    print("PASS: redistribute_vegetation_fraction preserves the non-water fraction total")


def test_redistribute_vegetation_fraction_clamps_at_upper_bound():
    result = redistribute_vegetation_fraction(veg0=0.9, built0=0.05, bare0=0.05, delta=0.5)
    assert np.isclose(result["fraction_vegetation"], 1.0), "vegetation fraction must clamp at 1.0"
    assert np.isclose(result["actual_delta_vegetation"], 0.1), "actual delta must reflect the clamp, not the request"
    total = result["fraction_vegetation"] + result["fraction_built_up"] + result["fraction_bare"]
    assert np.isclose(total, 1.0)
    print("PASS: redistribute_vegetation_fraction clamps at the upper bound and reports the true delta")


def test_predict_counterfactual_subzone_direction():
    df = _make_synthetic_table()
    model, metrics = train_xgb_model(df, test_size=0.25)
    assert metrics["test_r2"] > 0.5, f"sanity check on the synthetic fit itself failed, R²={metrics['test_r2']:.3f}"

    ndvi_slope = fit_ndvi_vegetation_slope(df)
    assert ndvi_slope > 0, "NDVI should rise with vegetation fraction by construction of the synthetic data"

    # Pick a mid-range-vegetation row so a +0.2 delta doesn't clamp.
    mid_idx = (df["fraction_vegetation"] - 0.4).abs().idxmin()
    row_df = df.loc[[mid_idx]]

    result = predict_counterfactual_subzone(model, row_df, delta_fraction_vegetation=0.2, ndvi_slope=ndvi_slope)
    assert result["delta_lst"] < 0, (
        f"increasing vegetation should decrease predicted LST given the synthetic ground truth, "
        f"got delta_lst={result['delta_lst']:.3f}"
    )
    print(f"PASS: greening counterfactual predicts a temperature drop (delta_lst={result['delta_lst']:.3f})")


def main():
    test_redistribute_vegetation_fraction_preserves_total()
    test_redistribute_vegetation_fraction_clamps_at_upper_bound()
    test_predict_counterfactual_subzone_direction()
    print("\nAll heat_model/tabular.py checks passed.")


if __name__ == "__main__":
    main()
