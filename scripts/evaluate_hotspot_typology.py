#!/usr/bin/env python
"""Evaluate the production heat typology (data/processed/hotspot_clusters.csv,
built by scripts/build_hotspot_clusters.py) for the evaluation report. Reads
saved files only (no Earth Engine). Writes to data/processed/diagnostics/:

  typology_quality.csv        production K-means and GMM: silhouette,
                              Davies-Bouldin, Calinski-Harabasz, sizes, and the
                              null-baseline silhouette (features shuffled)
  typology_k_sweep.csv        both methods for k = 2..8 (the selection sweep)
  typology_stability.csv      agreement with production across seeds and
                              bootstrap resamples, and K-means vs GMM
  typology_feature_ablation.csv  K-means at the production k on feature subsets
  typology_coherence.csv      variables that were NOT clustering inputs
                              (land-cover fractions, MODIS temperature) by cluster
  typology_profile.csv        cluster means of the inputs, sizes, GMM certainty
  typology_priority.csv       top-20 subzones of both priority views by cluster

Usage: python scripts/evaluate_hotspot_typology.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from config.settings import DIAGNOSTICS_DIR, INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED, TOP_N
from src.hotspots.cluster import FEATURE_COLUMNS, prepare_features, sweep_gmm, sweep_kmeans
from validation.hotspots_validation.typology_evaluation import (
    bootstrap_stability, coherence, feature_ablation, null_silhouettes, quality, seed_stability,
)

CLUSTERS = PROCESSED_DIR / "hotspot_clusters.csv"
N_NULL, N_SEEDS, N_BOOT = 200, 50, 200
FEATURE_SETS = {
    "All six (production)": FEATURE_COLUMNS,
    "Inter-monsoon only": ["lst_dry", "ndvi_dry", "ndbi_dry"],
    "December-February only": ["lst_wet", "ndvi_wet", "ndbi_wet"],
    "Temperature only": ["lst_dry", "lst_wet"],
    "Indices only (no temperature)": ["ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet"],
    "Without NDBI": ["lst_dry", "lst_wet", "ndvi_dry", "ndvi_wet"],
    "Without NDVI": ["lst_dry", "lst_wet", "ndbi_dry", "ndbi_wet"],
}


def _summary(name, values):
    return {"test": name, "n": len(values), "mean": values.mean(), "min": values.min(),
            "p05": np.percentile(values, 5), "p95": np.percentile(values, 95), "max": values.max()}


def main():
    raw = pd.read_csv(CLUSTERS)
    X, _, df = prepare_features(raw)
    prod = df["primary_cluster"].values
    gmm = df["cluster_gmm"].values
    k = len(np.unique(prod))
    k_gmm = len(np.unique(gmm))
    method = df["primary_cluster_method"].iloc[0]
    print(f"Production typology: {method}, k={k}; GMM k={k_gmm}; {len(df)} subzones")

    # Quality and null baseline
    null = null_silhouettes(X, k, N_NULL, RANDOM_SEED)
    q = pd.DataFrame([{"method": f"{method} (production)", **quality(X, prod)}, {"method": "gmm", **quality(X, gmm)}])
    q["null_silhouette_mean"], q["null_silhouette_p95"] = null.mean(), np.percentile(null, 95)
    q.to_csv(DIAGNOSTICS_DIR / "typology_quality.csv", index=False)
    print("\n" + q.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # The selection sweep
    sweep = pd.concat([sweep_kmeans(X).assign(method="kmeans"), sweep_gmm(X).assign(method="gmm")], ignore_index=True)
    sweep.to_csv(DIAGNOSTICS_DIR / "typology_k_sweep.csv", index=False)

    # Stability
    seeds = list(range(N_SEEDS))
    stab = pd.DataFrame([
        _summary("K-means, other seeds", seed_stability(X, prod, k, seeds, "kmeans")),
        _summary("K-means, bootstrap resamples", bootstrap_stability(X, prod, k, N_BOOT, RANDOM_SEED)),
        _summary("GMM, other seeds", seed_stability(X, gmm, k_gmm, seeds, "gmm")),
        {"test": "K-means vs GMM (production labels)", "n": 1, "mean": adjusted_rand_score(prod, gmm)},
    ])
    stab.to_csv(DIAGNOSTICS_DIR / "typology_stability.csv", index=False)
    print("\n" + stab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Feature ablation
    abl = feature_ablation(df, FEATURE_SETS, prod, k, RANDOM_SEED)
    abl.to_csv(DIAGNOSTICS_DIR / "typology_feature_ablation.csv", index=False)
    print("\n" + abl.drop(columns="features").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Coherence with variables that are not clustering inputs
    modis = pd.read_csv(INTERIM_DIR / "modis_heldout_lst.csv").rename(columns={"lst_heldout_c": "modis_lst"})
    ext = df.merge(modis, on="subzone_id", how="left")
    coh = coherence(ext, "primary_cluster",
                    ["fraction_vegetation", "fraction_built_up", "fraction_water", "fraction_bare", "modis_lst"])
    coh.to_csv(DIAGNOSTICS_DIR / "typology_coherence.csv", index=False)
    print("\n" + coh.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # Profile, with GMM certainty
    prob_cols = [c for c in df.columns if c.startswith("cluster_gmm_prob_")]
    df["gmm_max_prob"] = df[prob_cols].max(axis=1)
    prof = df.groupby("primary_cluster").agg(
        n_subzones=("subzone_id", "size"), **{c: (c, "mean") for c in FEATURE_COLUMNS},
        fraction_vegetation=("fraction_vegetation", "mean"), fraction_built_up=("fraction_built_up", "mean"),
        fraction_water=("fraction_water", "mean"),
    ).reset_index()
    prof.to_csv(DIAGNOSTICS_DIR / "typology_profile.csv", index=False)
    print("\n" + prof.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nGMM membership probability >= 0.9 for {(df['gmm_max_prob'] >= 0.9).mean():.1%} of subzones")

    # Relation to the priority ranking
    rows = []
    for view, path in (("Residents", PROCESSED_DIR / "priority_score.csv"),
                       ("All places", PROCESSED_DIR / "priority_score_all_places.csv")):
        score = pd.read_csv(path).merge(df[["subzone_id", "primary_cluster"]], on="subzone_id", how="left")
        top = score.sort_values("priority_score", ascending=False).head(TOP_N)
        for c, g in score.groupby("primary_cluster"):
            rows.append({"view": view, "cluster": int(c), "n_ranked": len(g),
                         "n_in_top20": int((top["primary_cluster"] == c).sum()),
                         "mean_priority_score": g["priority_score"].mean()})
    pri = pd.DataFrame(rows)
    pri.to_csv(DIAGNOSTICS_DIR / "typology_priority.csv", index=False)
    print("\n" + pri.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    (DIAGNOSTICS_DIR / "typology_gmm_certainty.txt").write_text(f"{(df['gmm_max_prob'] >= 0.9).mean():.6f}\n")
    print(f"\nSaved to {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
