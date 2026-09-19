#!/usr/bin/env python
"""Standalone verification for validation/score_validation/landcover_impact.py
and decision_impact.py against synthetic data -- no rasters, GEE or network
required. Same runnable-script + printed pass/fail style as the rest of tests/.

Usage: python tests/test_decision_impact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from validation.score_validation.decision_impact import build_decision_impact_summary, noise_floor
from validation.score_validation.landcover_impact import run_landcover_swap

TOP_N = 10


def _make_table(n=120, seed=5):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    greenery = np.clip(0.35 - 0.12 * z + rng.normal(0, 0.08, n), 0.02, 0.98)
    return pd.DataFrame({
        "subzone_id": [f"SZ_{i}" for i in range(n)],
        "lst_native30": 38 + 2.5 * z + rng.normal(0, 1.0, n),
        "sensitivity_raw": np.clip(0.5 + 0.15 * z + rng.normal(0, 0.1, n), 0, None),
        "g_ref": greenery,
        "g_affine": 0.35 + 0.56 * greenery,  # what a CONSTANT misclassification rate does to greenery
        "g_noisy": np.clip(greenery + rng.normal(0, 0.10, n), 0, 1),
        "g_shuffled": np.random.default_rng(seed + 1).permutation(greenery),  # right values, wrong subzones
    })


COLUMNS = {"ref": "g_ref", "affine": "g_affine", "noisy": "g_noisy", "shuffled": "g_shuffled"}


def test_reference_row_comes_first_and_is_unchanged():
    result = run_landcover_swap(_make_table(), {"noisy": "g_noisy", "ref": "g_ref"}, "ref", top_n=TOP_N)
    first = result.iloc[0]
    assert first["greenery_source"] == "ref" and first["is_reference"]
    assert first[f"top{TOP_N}_changed"] == 0 and np.isclose(first["score_spearman_vs_reference"], 1.0)
    assert first["left_top"] == "" and first["entered_top"] == ""
    assert (result["n_subzones"] == 120).all()
    print("PASS: the reference row is listed first and reproduces itself")


def test_constant_misclassification_rate_changes_nothing():
    """A constant error rate is an affine map of greenery; min-max scaling and
    the standardised PCA are both invariant to it, so the ranking must not move."""
    result = run_landcover_swap(_make_table(), COLUMNS, "ref", top_n=TOP_N).set_index("greenery_source")
    row = result.loc["affine"]
    assert row[f"top{TOP_N}_changed"] == 0, f"an affine distortion moved {row[f'top{TOP_N}_changed']} subzones"
    assert np.isclose(row["score_spearman_vs_reference"], 1.0) and np.isclose(row["greenery_spearman_vs_reference"], 1.0)
    print("PASS: an affine distortion of greenery (a constant misclassification rate) changes no ranking")


def test_heterogeneous_error_moves_subzones():
    result = run_landcover_swap(_make_table(), COLUMNS, "ref", top_n=TOP_N).set_index("greenery_source")
    for label in ("noisy", "shuffled"):
        row = result.loc[label]
        assert row[f"top{TOP_N}_changed"] >= 1, f"per-subzone error ('{label}') should move at least one top-{TOP_N} subzone"
        assert row["greenery_spearman_vs_reference"] < 1.0

        left = set(filter(None, row["left_top"].split("; ")))
        entered = set(filter(None, row["entered_top"].split("; ")))
        assert len(left) == len(entered) == row[f"top{TOP_N}_changed"] and not (left & entered), (
            f"'{label}': left/entered lists must be disjoint and match the changed count"
        )
    print(f"PASS: per-subzone error moves subzones (noisy {result.loc['noisy', f'top{TOP_N}_changed']}, shuffled "
          f"{result.loc['shuffled', f'top{TOP_N}_changed']}), with consistent left/entered lists")


def test_missing_values_are_dropped_from_every_row():
    df = _make_table()
    df.loc[:4, "g_noisy"] = np.nan
    result = run_landcover_swap(df, COLUMNS, "ref", top_n=TOP_N)
    assert (result["n_subzones"] == 115).all(), f"expected every row on the same 115 subzones, got {result['n_subzones'].tolist()}"
    print("PASS: subzones missing in any source are dropped from all rows, so they rank the same set")


def test_unknown_reference_raises():
    try:
        run_landcover_swap(_make_table(), COLUMNS, "nope", top_n=TOP_N)
    except ValueError:
        print("PASS: an unknown reference label raises")
        return
    raise AssertionError("expected ValueError for an unknown reference label")


def test_noise_floor_is_top_n_minus_stay_probability():
    bands = pd.DataFrame({
        "subzone_id": list("abcdef"),
        "priority_score_point": [0.9, 0.8, 0.7, 0.1, 0.05, 0.02],
        "p_top3": [1.0, 0.8, 0.5, 0.4, 0.3, 0.0],  # the point top-3 stay with 1.0 + 0.8 + 0.5 = 2.3
    })
    assert np.isclose(noise_floor(bands, top_n=3), 0.7), noise_floor(bands, top_n=3)
    print("PASS: noise floor = N minus the summed stay-probability of the point top-N")


def test_summary_flags_only_choices_above_the_noise_floor():
    bands = pd.DataFrame({
        "subzone_id": [f"s{i}" for i in range(15)],
        "priority_score_point": np.linspace(1.0, 0.1, 15),
        f"p_top{TOP_N}": [0.5] * TOP_N + [0.1] * 5,  # top-10 stay with 0.5 each -> floor 5.0
    })
    weighting = pd.DataFrame({"variant": ["lst_native30", "lst_bicubic10"], f"n_top{TOP_N}_changed": [4, 4]})
    rank_impact = pd.DataFrame({
        "variant": ["lst_native30", "lst_bicubic10", "lst_regress10"],
        f"top{TOP_N}_overlap_vs_lst_native30": [1.0, 1.0, 0.9],
    })
    spec = pd.DataFrame({
        "specification": ["PRODUCTION (config: density)", "population density + elderly share", "count + share", "0.75 / 0.25"],
        "spearman_vs_production": [1.0, 1.0, 0.95, 0.99], f"top{TOP_N}_changed": [0, 0, 7, 2],
    })
    swap = pd.DataFrame({
        "greenery_source": ["hybrid", "RF", "U-Net", "NDVI proxy"], "is_reference": [True, False, False, False],
        f"top{TOP_N}_changed": [0, 3, 1, 6],
    })

    summary = build_decision_impact_summary(weighting, rank_impact, spec, swap, bands, top_n=TOP_N)
    assert summary.loc[0, "vs_noise_floor"] == "yardstick" and np.isclose(summary.loc[0, "top_changed_max"], 5.0)

    by_choice = summary.iloc[1:].set_index("choice")
    assert list(summary["top_changed_max"].iloc[1:]) == [7, 6, 4, 1], "rows must be sorted by the largest change"
    assert by_choice.loc["Sensitivity formula", "vs_noise_floor"] == "exceeds noise"
    assert by_choice.loc["Sensitivity formula", "largest_change_from"] == "count + share"
    assert by_choice.loc["Sensitivity formula", "alternatives_tested"] == 2, "production and its identical duplicate are excluded"
    assert by_choice.loc["Land-cover source (RF / U-Net / NDVI proxy vs hybrid)", "largest_change_from"] == "NDVI proxy"
    assert by_choice.loc["Weighting (PCA-derived vs equal)", "vs_noise_floor"] == "within noise"
    assert by_choice.loc["Heat-map resolution (30 m vs downscaled 10 m)", "top_changed_max"] == 1
    print("PASS: summary sorts choices by impact and flags exactly those that exceed the noise floor")


def test_summary_says_so_when_no_alternative_changes_anything():
    """idxmax on an all-zero column returns the first label, which would read as
    a finding ('largest change from lst_bicubic10') when nothing changed at all."""
    bands = pd.DataFrame({
        "subzone_id": [f"s{i}" for i in range(15)], "priority_score_point": np.linspace(1.0, 0.1, 15),
        f"p_top{TOP_N}": [0.5] * TOP_N + [0.1] * 5,
    })
    rank_impact = pd.DataFrame({
        "variant": ["lst_native30", "lst_bicubic10", "lst_regress10"], f"top{TOP_N}_overlap_vs_lst_native30": [1.0, 1.0, 1.0],
    })
    weighting = pd.DataFrame({"variant": ["lst_native30"], f"n_top{TOP_N}_changed": [2]})
    spec = pd.DataFrame({
        "specification": ["PRODUCTION x", "alt"], "spearman_vs_production": [1.0, 0.99], f"top{TOP_N}_changed": [0, 1],
    })
    swap = pd.DataFrame({"greenery_source": ["hybrid", "RF"], "is_reference": [True, False], f"top{TOP_N}_changed": [0, 1]})

    summary = build_decision_impact_summary(weighting, rank_impact, spec, swap, bands, top_n=TOP_N).set_index("choice")
    resolution = summary.loc["Heat-map resolution (30 m vs downscaled 10 m)"]
    assert resolution["top_changed_max"] == 0 and resolution["largest_change_from"].startswith("none"), resolution.to_dict()
    assert summary.loc["Sensitivity formula", "largest_change_from"] == "alt"
    print("PASS: a choice that changes nothing is reported as 'none', not attributed to an arbitrary alternative")


def main():
    test_reference_row_comes_first_and_is_unchanged()
    test_constant_misclassification_rate_changes_nothing()
    test_heterogeneous_error_moves_subzones()
    test_missing_values_are_dropped_from_every_row()
    test_unknown_reference_raises()
    test_noise_floor_is_top_n_minus_stay_probability()
    test_summary_flags_only_choices_above_the_noise_floor()
    test_summary_says_so_when_no_alternative_changes_anything()
    print("\nAll decision-impact checks passed.")


if __name__ == "__main__":
    main()
