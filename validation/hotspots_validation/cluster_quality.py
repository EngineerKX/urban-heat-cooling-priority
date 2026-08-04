"""S4 clustering has no ground-truth labels, so this fills the role every
other build in this repo gets from a validation counterpart (heat variants
vs. NEA, land cover vs. hand labels): a check against something real. Two
checks: standard unsupervised-clustering quality metrics, and a land-cover-
fraction coherence check using features that were NEVER a clustering input
-- does the hottest cluster also come out most built-up / least-vegetated?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

MIN_SILHOUETTE_SCORE = 0.15  # placeholder gate threshold -- flagged, not a locked S4 decision


def evaluate_cluster_quality(X_scaled, labels) -> dict:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    result = {
        "n_clusters": len(unique),
        "silhouette": float(silhouette_score(X_scaled, labels)),
        "davies_bouldin": float(davies_bouldin_score(X_scaled, labels)),
        "min_cluster_size": int(counts.min()),
        "max_cluster_size": int(counts.max()),
    }
    result["degenerate"] = result["min_cluster_size"] < 3 or result["silhouette"] < MIN_SILHOUETTE_SCORE

    print(f"Cluster quality: silhouette={result['silhouette']:.3f} (gate: >={MIN_SILHOUETTE_SCORE}), "
          f"davies_bouldin={result['davies_bouldin']:.3f}, sizes={dict(zip(unique.tolist(), counts.tolist()))}")
    if result["degenerate"]:
        print("⚠️  Degenerate solution warning: a cluster is tiny (<3) or silhouette is below the gate threshold.")
    return result


def sanity_check_landcover_coherence(
    df: pd.DataFrame, cluster_col: str, lst_col: str = "lst_dry",
    vegetation_col: str = "fraction_vegetation", built_up_col: str = "fraction_built_up",
) -> dict:
    """Rather than hand-naming specific known-hot/known-cool Singapore
    subzones (risk of guessing real geography wrong), checks internal
    physical coherence against land-cover fractions that were never a
    clustering input (see src/hotspots/cluster.py::FEATURE_COLUMNS): the
    cluster with the highest mean LST should also have the highest mean
    built-up fraction and the lowest mean vegetation fraction."""
    profile = df.groupby(cluster_col)[[lst_col, vegetation_col, built_up_col]].mean()
    hottest = profile[lst_col].idxmax()
    most_built_up = profile[built_up_col].idxmax()
    least_vegetated = profile[vegetation_col].idxmin()

    coherent_built_up = hottest == most_built_up
    coherent_vegetation = hottest == least_vegetated
    print(f"Hottest cluster: {hottest} | Most built-up cluster: {most_built_up} "
          f"({'✅ match' if coherent_built_up else '⚠️  mismatch'})")
    print(f"Hottest cluster: {hottest} | Least-vegetated cluster: {least_vegetated} "
          f"({'✅ match' if coherent_vegetation else '⚠️  mismatch'})")

    return {
        "hottest_cluster": hottest,
        "most_built_up_cluster": most_built_up,
        "least_vegetated_cluster": least_vegetated,
        "coherent_built_up": coherent_built_up,
        "coherent_vegetation": coherent_vegetation,
    }
