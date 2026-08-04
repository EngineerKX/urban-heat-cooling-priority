#!/usr/bin/env python
"""Read-only diagnostic for the S4 hotspot clusters already built by
scripts/build_hotspot_clusters.py -- same "build vs. diagnose" split as
scripts/diagnose_heat_variants.py / diagnose_land_change.py. Recomputes
cluster-quality metrics and the land-cover coherence check from the saved
CSV, and saves a per-cluster feature-profile bar chart.

Usage: python scripts/diagnose_hotspot_clusters.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from config.settings import DIAGNOSTICS_DIR, PROCESSED_DIR
from src.hotspots.cluster import FEATURE_COLUMNS, prepare_features
from validation.hotspots_validation.cluster_quality import evaluate_cluster_quality, sanity_check_landcover_coherence

CLUSTERS_CSV_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
PROFILE_CSV_PATH = PROCESSED_DIR / "hotspot_cluster_profile.csv"
PLOT_OUT_PATH = DIAGNOSTICS_DIR / "hotspot_cluster_profile.png"


def main():
    if not CLUSTERS_CSV_PATH.exists():
        raise FileNotFoundError(f"{CLUSTERS_CSV_PATH} not found — run scripts/build_hotspot_clusters.py first.")
    df = pd.read_csv(CLUSTERS_CSV_PATH)
    profile_df = pd.read_csv(PROFILE_CSV_PATH)

    X_scaled, _scaler, clean_df = prepare_features(df)
    print("--- Cluster quality (primary_cluster, recomputed from saved CSV) ---")
    evaluate_cluster_quality(X_scaled, clean_df["primary_cluster"].values)

    if "fraction_built_up" in df.columns:
        print("\n--- Land-cover coherence check ---")
        sanity_check_landcover_coherence(df, "primary_cluster")
    else:
        print("\n⚠️  No land-cover fraction columns in the saved CSV — coherence check skipped.")

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    profile_df.set_index("primary_cluster")[FEATURE_COLUMNS].plot(kind="bar", ax=ax)
    ax.set_title("Hotspot cluster profile — mean feature value per cluster (raw units)")
    ax.set_ylabel("°C (lst_*) / index value (ndvi_*, ndbi_*)")
    fig.tight_layout()
    fig.savefig(PLOT_OUT_PATH)
    print(f"\nSaved: {PLOT_OUT_PATH}")


if __name__ == "__main__":
    main()
