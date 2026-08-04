#!/usr/bin/env python
"""Standalone verification for src/hotspots/cluster.py against synthetic
blob data -- no GEE required. Same runnable-script + printed pass/fail
style as the rest of tests/.

Usage: python tests/test_hotspots_cluster.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.hotspots.cluster import FEATURE_COLUMNS, choose_k, fit_kmeans, label_clusters, prepare_features, sweep_kmeans


def _make_synthetic_blobs(seed=42):
    """3 well-separated blobs in 6D feature space (FEATURE_COLUMNS), 40
    points each -- clustering should trivially recover the 3 groups."""
    rng = np.random.default_rng(seed)
    n_per_blob = 40
    centers = {0: -5.0, 1: 0.0, 2: 5.0}
    rows, true_labels = [], []
    for label, center in centers.items():
        block = rng.normal(loc=center, scale=0.5, size=(n_per_blob, len(FEATURE_COLUMNS)))
        rows.append(block)
        true_labels.extend([label] * n_per_blob)
    X = np.vstack(rows)
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df["subzone_id"] = [f"SZ_{i}" for i in range(len(df))]
    return df, np.array(true_labels)


def test_prepare_features_drops_nan_rows():
    df = pd.DataFrame({col: [1.0, 2.0, np.nan] for col in FEATURE_COLUMNS})
    df["subzone_id"] = ["A", "B", "C"]
    X_scaled, _scaler, clean_df = prepare_features(df)
    assert len(clean_df) == 2, "row with a missing feature must be dropped before scaling"
    assert X_scaled.shape == (2, len(FEATURE_COLUMNS))
    print("PASS: prepare_features drops rows with missing features")


def test_choose_k_recovers_known_cluster_count():
    df, true_labels = _make_synthetic_blobs()
    X_scaled, _scaler, _clean_df = prepare_features(df)

    sweep_df = sweep_kmeans(X_scaled, k_range=range(2, 6))
    k = choose_k(sweep_df, min_k=3)
    assert k == 3, f"expected choose_k to recover k=3 on well-separated synthetic blobs, got {k}"
    print("PASS: choose_k recovers k=3 from the silhouette sweep")

    model = fit_kmeans(X_scaled, k)
    ari = adjusted_rand_score(true_labels, model.labels_)
    assert ari > 0.95, f"expected near-perfect cluster recovery on separated blobs, got ARI={ari:.3f}"
    print(f"PASS: K-means recovers the synthetic groups (ARI={ari:.3f})")


def test_label_clusters_adds_expected_column():
    df, _true_labels = _make_synthetic_blobs()
    X_scaled, _scaler, clean_df = prepare_features(df)
    model = fit_kmeans(X_scaled, 3)
    labeled = label_clusters(clean_df, model, X_scaled, "cluster_kmeans")
    assert "cluster_kmeans" in labeled.columns
    assert set(labeled["cluster_kmeans"].unique()) == {0, 1, 2}
    print("PASS: label_clusters adds the cluster column with the expected labels")


def main():
    test_prepare_features_drops_nan_rows()
    test_choose_k_recovers_known_cluster_count()
    test_label_clusters_adds_expected_column()
    print("\nAll hotspots/cluster.py checks passed.")


if __name__ == "__main__":
    main()
