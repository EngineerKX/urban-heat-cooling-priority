#!/usr/bin/env python
"""Evaluate the production cooling-priority score for the evaluation report.
Reads saved pillar tables and scores only (no Earth Engine). Complements the
existing decision-impact tables (weighting_comparison.csv,
rank_impact_results_*.csv, sensitivity_spec_comparison.csv,
landcover_*swap*.csv, priority_score_confidence_bands.csv). Writes to
data/processed/diagnostics/:

  priority_pca_structure.csv    PCA weights, variance explained, pillar
                                correlation with the score and spread
  priority_pillar_correlations.csv
  priority_ablations.csv        leave-one-pillar-out and z-score aggregation
  priority_min_residents.csv    minimum-residents threshold sweep
  priority_noise_sources.csv    noise floor with each noise source alone
  priority_views.csv            residents view vs all-places view
  priority_face_validity.csv    top 20 vs the Singapore-ETH hotspot planning areas
  priority_bands_summary.csv    band width, coverage check, top-20 membership tiers

Usage: python scripts/evaluate_priority_score.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import DIAGNOSTICS_DIR, INTERIM_DIR, MIN_RESIDENTS_FOR_RANKING, PROCESSED_DIR, REFERENCE_VARIANT, TOP_N
from src.ingest.subzones import as_geodataframe, fetch_subzones_geojson
from src.priority_score.io import load_and_join, load_heldout
from src.priority_score.pillars import compute_sensitivity_raw
from src.priority_score.score import build_score
from validation.score_validation.confidence_bands import (
    adaptive_capacity_noise_std, bootstrap_priority_score, exposure_noise_std,
)
from validation.score_validation.decision_impact import noise_floor
from validation.score_validation.score_evaluation import (
    PILLARS, face_validity, leave_one_out, pca_structure, pillar_table, top_changed, zscore_aggregation,
)

# Singapore-ETH Centre urban heat vulnerability analysis (Philipp & Chow, DOI 10.3929/ethz-b-000419689):
# planning areas named as hotspots at the coolest or hottest hours (docs/Progress_Log_2026-09-19_1703.md §7).
ETH_HOTSPOT_AREAS = ["GEYLANG", "JURONG EAST", "MARINE PARADE", "PUNGGOL", "SENGKANG", "SERANGOON", "WOODLANDS"]
THRESHOLDS = [0, 100, 250, 500, 1000, 2000]


def _save(df, name):
    df.to_csv(DIAGNOSTICS_DIR / name, index=False)
    print(f"\n{name}\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main():
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    df, _ = load_and_join(toy_mode=False)
    prod_score, weights = build_score(df, REFERENCE_VARIANT, "pca")
    ids = df["subzone_id"]

    # 1. PCA structure
    pillars = pillar_table(df, REFERENCE_VARIANT)
    s = pca_structure(pillars, prod_score)
    _save(pd.DataFrame({
        "pillar": list(PILLARS), "pca_weight": [weights[p] for p in PILLARS],
        "pillar_sd": [s["pillar_sd"][p] for p in PILLARS],
        "correlation_with_score": [s["score_correlation"][p] for p in PILLARS],
        "variance_explained_by_component": s["explained_variance"],
    }), "priority_pca_structure.csv")
    _save(s["correlations"].reset_index().rename(columns={"index": "pillar"}), "priority_pillar_correlations.csv")

    # 2. Pillar ablations and aggregation rule
    abl = leave_one_out(df, REFERENCE_VARIANT, prod_score, weights)
    z = zscore_aggregation(pillars, weights)
    abl = pd.concat([abl, pd.DataFrame([{"change": "z-score instead of min-max aggregation",
                                          "top_changed": top_changed(ids, z, ids, prod_score),
                                          "spearman": spearmanr(z, prod_score)[0]}])], ignore_index=True)
    _save(abl, "priority_ablations.csv")

    # 3. Minimum-residents threshold
    heat = pd.read_csv(INTERIM_DIR / "heat_variants_subzone.csv")
    sens = pd.read_csv(INTERIM_DIR / "sensitivity_pillar.csv")
    adapt = pd.read_csv(INTERIM_DIR / "adaptive_capacity_pillar.csv")
    rows = []
    for t in THRESHOLDS:
        alt = sens.copy()
        alt["sensitivity_raw"] = compute_sensitivity_raw(alt["population_total"], alt["elderly_proportion"].fillna(0),
                                                         alt["area_km2"], min_residents=t)
        d = heat.merge(alt, on="subzone_id").merge(adapt, on="subzone_id")
        d = d[d["sensitivity_raw"].notna()].reset_index(drop=True)
        sc, _ = build_score(d, REFERENCE_VARIANT, "pca")
        rows.append({"min_residents": t, "is_production": t == MIN_RESIDENTS_FOR_RANKING, "n_ranked": len(d),
                     "top_changed": top_changed(d["subzone_id"], sc, ids, prod_score)})
    _save(pd.DataFrame(rows), "priority_min_residents.csv")

    # 4. Noise sources
    heldout = load_heldout(required=True)
    exp_std = exposure_noise_std(df, REFERENCE_VARIANT, heldout)
    ac_std = adaptive_capacity_noise_std(df, "greenery_fraction", "greenery_fraction_ndvi")
    rows = []
    for label, e, a in (("Both (production)", exp_std, ac_std), ("Heat exposure only", exp_std, 0.0),
                        ("Greenery only", 0.0, ac_std)):
        bands = bootstrap_priority_score(df, REFERENCE_VARIANT, "pca", e, a)
        rows.append({"noise_sources": label, "exposure_noise_std_c": e, "greenery_noise_std": a,
                     "noise_floor_top20": noise_floor(bands)})
    _save(pd.DataFrame(rows), "priority_noise_sources.csv")

    # 4b. Effect of clipping perturbed greenery to [0, 1] (the production setting)
    clipped = bootstrap_priority_score(df, REFERENCE_VARIANT, "pca", exp_std, ac_std)
    unclipped = bootstrap_priority_score(df, REFERENCE_VARIANT, "pca", exp_std, ac_std, clip_greenery=False)
    no_green = clipped["subzone_id"].map(df.set_index("subzone_id")["greenery_fraction"]) < 0.005
    _save(pd.DataFrame([{
        "noise_floor_clipped": noise_floor(clipped), "noise_floor_unclipped": noise_floor(unclipped),
        "n_subzones_no_greenery": int(no_green.sum()),
        "n_top20_no_greenery": int(no_green[clipped.nlargest(TOP_N, "priority_score_point").index].sum()),
        "mean_p_top20_no_greenery_clipped": clipped.loc[no_green, "p_top20"].mean(),
        "mean_p_top20_no_greenery_unclipped": unclipped.loc[no_green, "p_top20"].mean(),
        "max_abs_p_top20_change": (clipped["p_top20"] - unclipped["p_top20"]).abs().max(),
    }]), "priority_clipping_check.csv")

    # 4c. What the greenery pillar changes beyond top-20 membership
    no_green_score, _ = build_score(df, REFERENCE_VARIANT, "custom",
                                    weights={"exposure": weights["exposure"], "sensitivity": weights["sensitivity"],
                                             "adaptive_deficit": 0.0})
    rank_prod, rank_alt = prod_score.rank(ascending=False), no_green_score.rank(ascending=False)
    shift = (rank_prod - rank_alt).abs()
    top20 = rank_prod <= TOP_N
    _save(pd.DataFrame([{
        **{f"top{n}_changed": top_changed(ids, no_green_score, ids, prod_score, n) for n in (10, 20, 50)},
        "mean_rank_shift_in_top20": shift[top20].mean(), "max_rank_shift_in_top20": shift[top20].max(),
        "mean_rank_shift_all": shift.mean(), "n_moving_10_or_more_places": int((shift >= 10).sum()),
        "n_ranked": len(df),
    }]), "priority_greenery_effect.csv")

    # 5. Residents view vs all-places view
    res = pd.read_csv(PROCESSED_DIR / "priority_score.csv")
    allp = pd.read_csv(PROCESSED_DIR / "priority_score_all_places.csv")
    top_res, top_all = set(res.nlargest(TOP_N, "priority_score")["subzone_id"]), set(allp.nlargest(TOP_N, "priority_score")["subzone_id"])
    both = res.merge(allp, on="subzone_id", suffixes=("_res", "_all"))
    _save(pd.DataFrame([{"n_residents_view": len(res), "n_all_places_view": len(allp),
                         "top20_in_both": len(top_res & top_all),
                         "all_places_top20_not_ranked_in_residents_view": len(top_all - set(res["subzone_id"])),
                         "spearman_on_common_subzones": spearmanr(both["priority_score_res"], both["priority_score_all"])[0]}]),
          "priority_views.csv")

    # 6. Face validity against published hotspot planning areas
    gdf = as_geodataframe(fetch_subzones_geojson())
    planning_area = pd.Series(gdf["PLN_AREA_N"].values, index=gdf["SUBZONE_N"].astype(str).values)
    _save(pd.DataFrame([{"view": "Residents", **face_validity(res, planning_area, ETH_HOTSPOT_AREAS)},
                        {"view": "All places", **face_validity(allp, planning_area, ETH_HOTSPOT_AREAS)}]),
          "priority_face_validity.csv")

    # 7. Band summary
    rows = []
    for view, path in (("Residents", "priority_score_confidence_bands.csv"), ("All places", "priority_score_all_places_confidence_bands.csv")):
        b = pd.read_csv(PROCESSED_DIR / path)
        top = b.nlargest(TOP_N, "priority_score_point")
        outside = ((b["priority_score_point"] < b["priority_score_p05"]) | (b["priority_score_point"] > b["priority_score_p95"])).sum()
        rows.append({"view": view, "n": len(b), "mean_band_width": (b["priority_score_p95"] - b["priority_score_p05"]).mean(),
                     "point_outside_own_band": int(outside), "noise_floor": noise_floor(b),
                     "top20_p_ge_0.9": int((top["p_top20"] >= 0.9).sum()),
                     "top20_p_0.5_to_0.9": int(((top["p_top20"] >= 0.5) & (top["p_top20"] < 0.9)).sum()),
                     "top20_p_lt_0.5": int((top["p_top20"] < 0.5).sum())})
    _save(pd.DataFrame(rows), "priority_bands_summary.csv")
    print(f"\nSaved to {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
