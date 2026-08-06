#!/usr/bin/env python
"""Build S4 -- unsupervised hotspot-typology clustering. New pipeline
stage (src/hotspots/ was empty, no source notebook to migrate). Builds a
dry/wet seasonal feature table (config.settings.WET_SEASON_MONTHS), sweeps
K-means and GMM over k=2..8, picks each algorithm's best k by silhouette,
fits both, and keeps whichever has the higher silhouette as the primary
cluster label -- the other stays as a visible ablation column.

"Day/night typologies" from the original S4 spec is NOT built: Landsat's
overpass is daytime-only, and src/ingest/nea.py::sample_readings doesn't
currently retain per-reading timestamps, so no day/night signal exists
anywhere in this pipeline yet. Documented limitation, not a silent gap.

Usage: python scripts/build_hotspot_clusters.py [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ee
import mlflow

from config.settings import (
    DRY_SEASON_MONTHS,
    INTERIM_DIR,
    LANDSAT_CLOUD_COVER_MAX,
    NATIVE_SCALE_M,
    PROCESSED_DIR,
    S2_CLOUD_PROB_MAX,
    S2_UTM_CRS,
    SG_BBOX,
    SUBZONE_ID_PROPERTY,
    TARGET_SCALE_M,
    WET_SEASON_MONTHS,
    YEARS,
)
from src.hotspots.cluster import (
    FEATURE_COLUMNS,
    choose_k,
    fit_gmm,
    fit_kmeans,
    label_clusters,
    prepare_features,
    profile_clusters,
    sweep_gmm,
    sweep_kmeans,
)
from src.hotspots.features import build_seasonal_feature_table
from src.ingest.gee import init_ee
from src.ingest.subzones import as_ee_feature_collection, as_geodataframe, fetch_subzones_geojson
from src.landcover.ensemble import ENSEMBLE_RASTER_PATH
from src.landcover.zonal import zonal_class_fractions
from src.utils.caching import load_or_fetch_csv
from src.utils.experiment_tracking import HOTSPOTS_EXPERIMENT_NAME, start_run
from validation.hotspots_validation.cluster_quality import evaluate_cluster_quality, sanity_check_landcover_coherence

FEATURES_CSV_PATH = INTERIM_DIR / "hotspot_features_subzone.csv"
CLUSTERS_OUT_PATH = PROCESSED_DIR / "hotspot_clusters.csv"
PROFILE_OUT_PATH = PROCESSED_DIR / "hotspot_cluster_profile.csv"


def _fetch_features(sg_bbox, subzones_fc):
    return build_seasonal_feature_table(
        sg_bbox, subzones_fc, SUBZONE_ID_PROPERTY, YEARS, DRY_SEASON_MONTHS, WET_SEASON_MONTHS,
        LANDSAT_CLOUD_COVER_MAX, S2_CLOUD_PROB_MAX, S2_UTM_CRS, NATIVE_SCALE_M, TARGET_SCALE_M,
    )


def main(force: bool = False):
    if CLUSTERS_OUT_PATH.exists() and not force:
        print(f"{CLUSTERS_OUT_PATH} already exists — skipping recompute (pass --force to rebuild).")
        return CLUSTERS_OUT_PATH

    init_ee()
    geojson = fetch_subzones_geojson()
    sg_bbox = ee.Geometry.Rectangle(list(SG_BBOX))
    subzones_fc = as_ee_feature_collection(geojson)

    features_df = load_or_fetch_csv(FEATURES_CSV_PATH, lambda: _fetch_features(sg_bbox, subzones_fc), force=force)
    print(f"\nFeature table: {len(features_df)} subzones.")

    X_scaled, _scaler, clean_df = prepare_features(features_df)

    print("\n--- K-means sweep (k=2..8) ---")
    kmeans_sweep = sweep_kmeans(X_scaled)
    print(kmeans_sweep.to_string(index=False))
    k_kmeans = choose_k(kmeans_sweep)
    print(f"Chosen k for K-means: {k_kmeans}")

    with start_run("kmeans", experiment_name=HOTSPOTS_EXPERIMENT_NAME, stage="clustering"):
        mlflow.log_param("k_range", f"{kmeans_sweep['k'].min()}-{kmeans_sweep['k'].max()}")
        mlflow.log_param("chosen_k", k_kmeans)
        mlflow.log_param("n_subzones", len(clean_df))
        for _, row in kmeans_sweep.iterrows():
            step = int(row["k"])
            mlflow.log_metric("silhouette", row["silhouette"], step=step)
            mlflow.log_metric("davies_bouldin", row["davies_bouldin"], step=step)
            mlflow.log_metric("inertia", row["inertia"], step=step)
        mlflow.log_metric("chosen_silhouette", float(kmeans_sweep.loc[kmeans_sweep["k"] == k_kmeans, "silhouette"].iloc[0]))

    print("\n--- GMM sweep (k=2..8) ---")
    gmm_sweep = sweep_gmm(X_scaled)
    print(gmm_sweep.to_string(index=False))
    k_gmm = choose_k(gmm_sweep)
    print(f"Chosen k for GMM: {k_gmm}")

    with start_run("gmm", experiment_name=HOTSPOTS_EXPERIMENT_NAME, stage="clustering"):
        mlflow.log_param("k_range", f"{gmm_sweep['k'].min()}-{gmm_sweep['k'].max()}")
        mlflow.log_param("chosen_k", k_gmm)
        mlflow.log_param("n_subzones", len(clean_df))
        for _, row in gmm_sweep.iterrows():
            step = int(row["k"])
            mlflow.log_metric("silhouette", row["silhouette"], step=step)
            mlflow.log_metric("davies_bouldin", row["davies_bouldin"], step=step)
            mlflow.log_metric("bic", row["bic"], step=step)
            mlflow.log_metric("aic", row["aic"], step=step)
        mlflow.log_metric("chosen_silhouette", float(gmm_sweep.loc[gmm_sweep["k"] == k_gmm, "silhouette"].iloc[0]))

    kmeans_model = fit_kmeans(X_scaled, k_kmeans)
    gmm_model = fit_gmm(X_scaled, k_gmm)

    result_df = label_clusters(clean_df, kmeans_model, X_scaled, "cluster_kmeans")
    result_df = label_clusters(result_df, gmm_model, X_scaled, "cluster_gmm")

    kmeans_silhouette = float(kmeans_sweep.loc[kmeans_sweep["k"] == k_kmeans, "silhouette"].iloc[0])
    gmm_silhouette = float(gmm_sweep.loc[gmm_sweep["k"] == k_gmm, "silhouette"].iloc[0])
    if kmeans_silhouette >= gmm_silhouette:
        result_df["primary_cluster_method"] = "kmeans"
        result_df["primary_cluster"] = result_df["cluster_kmeans"]
    else:
        result_df["primary_cluster_method"] = "gmm"
        result_df["primary_cluster"] = result_df["cluster_gmm"]
    print(f"\nPrimary method: {result_df['primary_cluster_method'].iloc[0]} "
          f"(K-means silhouette={kmeans_silhouette:.3f}, GMM silhouette={gmm_silhouette:.3f})")

    print("\n--- Cluster quality (primary_cluster) ---")
    quality_result = evaluate_cluster_quality(X_scaled, result_df["primary_cluster"].values)

    landcover_cols = []
    coherence_result = None
    if ENSEMBLE_RASTER_PATH.exists():
        print("\n--- Land-cover coherence check (fractions are NOT a clustering input) ---")
        subzones_gdf = as_geodataframe(geojson)
        frac_df = zonal_class_fractions(ENSEMBLE_RASTER_PATH, subzones_gdf, SUBZONE_ID_PROPERTY)
        result_df = result_df.merge(frac_df.drop(columns="n_valid_pixels"), on="subzone_id", how="left")
        landcover_cols = [c for c in frac_df.columns if c.startswith("fraction_")]
        coherence_result = sanity_check_landcover_coherence(result_df, "primary_cluster")
    else:
        print(f"\n⚠️  {ENSEMBLE_RASTER_PATH} not found — skipping land-cover coherence check "
              f"(run scripts/build_landcover_ensemble.py first).")

    with start_run("primary_cluster", experiment_name=HOTSPOTS_EXPERIMENT_NAME, stage="selection"):
        mlflow.log_param("primary_cluster_method", result_df["primary_cluster_method"].iloc[0])
        mlflow.log_param("chosen_k", k_kmeans if result_df["primary_cluster_method"].iloc[0] == "kmeans" else k_gmm)
        mlflow.log_metric("kmeans_silhouette", kmeans_silhouette)
        mlflow.log_metric("gmm_silhouette", gmm_silhouette)
        mlflow.log_metric("n_clusters", quality_result["n_clusters"])
        mlflow.log_metric("silhouette", quality_result["silhouette"])
        mlflow.log_metric("davies_bouldin", quality_result["davies_bouldin"])
        mlflow.log_metric("min_cluster_size", quality_result["min_cluster_size"])
        mlflow.log_metric("max_cluster_size", quality_result["max_cluster_size"])
        mlflow.log_param("degenerate", quality_result["degenerate"])
        if coherence_result is not None:
            mlflow.log_param("coherent_built_up", coherence_result["coherent_built_up"])
            mlflow.log_param("coherent_vegetation", coherence_result["coherent_vegetation"])

    profile_df = profile_clusters(result_df, "primary_cluster", FEATURE_COLUMNS, landcover_cols)
    print("\n--- Cluster profile (primary_cluster) ---")
    print(profile_df.to_string(index=False))

    print("\n⚠️  Reminder: 'day/night typologies' from the original S4 spec is NOT implemented — "
          "Landsat's overpass is daytime-only and NEA's sample_readings doesn't retain per-reading "
          "timestamps. Documented limitation, not a silent gap.")

    result_df.to_csv(CLUSTERS_OUT_PATH, index=False)
    profile_df.to_csv(PROFILE_OUT_PATH, index=False)
    print(f"\nSaved: {CLUSTERS_OUT_PATH} ({len(result_df)} subzones)")
    print(f"Saved: {PROFILE_OUT_PATH}")
    return CLUSTERS_OUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
