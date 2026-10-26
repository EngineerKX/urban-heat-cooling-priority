"""Heat-typology evaluation beyond cluster_quality.py: is the structure real
(null baseline), repeatable (seed and bootstrap stability), what drives it
(feature ablations), does it agree with independent data (land-cover
fractions and MODIS temperature, neither a clustering input), and how it
relates to the priority ranking.

Every function takes the production feature matrix and labels, so the
evaluation scores the typology the pipeline actually produced
(data/processed/hotspot_clusters.csv), not a refit.
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def quality(X, labels) -> dict:
    """Internal quality: silhouette (higher = tighter, better separated),
    Davies-Bouldin (lower = better), Calinski-Harabasz (higher = better)."""
    _, counts = np.unique(labels, return_counts=True)
    return {"silhouette": silhouette_score(X, labels), "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "k": len(counts), "min_size": int(counts.min()), "max_size": int(counts.max())}


def null_silhouettes(X, k, n_reps, seed) -> np.ndarray:
    """Silhouette of K-means (same k) on data with each feature shuffled
    independently: same per-feature distributions, no joint structure. A real
    typology should score well above this."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_reps):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
        labels = KMeans(n_clusters=k, random_state=int(rng.integers(1e9)), n_init=10).fit_predict(Xp)
        out.append(silhouette_score(Xp, labels))
    return np.array(out)


def seed_stability(X, reference, k, seeds, method="kmeans") -> np.ndarray:
    """Agreement (adjusted Rand index) between the production labels and a
    refit with each seed. 1 = identical partition, 0 = chance agreement."""
    out = []
    for s in seeds:
        if method == "kmeans":
            labels = KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X)
        else:
            labels = GaussianMixture(n_components=k, random_state=s, n_init=5).fit(X).predict(X)
        out.append(adjusted_rand_score(reference, labels))
    return np.array(out)


def bootstrap_stability(X, reference, k, n_reps, seed) -> np.ndarray:
    """Refit K-means on a bootstrap resample of subzones, assign every subzone
    to the nearest refitted centre, and compare with the production labels.
    Tests whether the typology depends on which subzones happen to be in it."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_reps):
        idx = rng.integers(0, len(X), len(X))
        model = KMeans(n_clusters=k, random_state=int(rng.integers(1e9)), n_init=10).fit(X[idx])
        out.append(adjusted_rand_score(reference, model.predict(X)))
    return np.array(out)


def feature_ablation(df, feature_sets: dict, reference, k, seed) -> pd.DataFrame:
    """K-means at the production k on each feature subset (z-scored), scored
    by its own silhouette and by agreement with the production typology."""
    rows = []
    for name, cols in feature_sets.items():
        X = StandardScaler().fit_transform(df[cols].values)
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)
        rows.append({"feature_set": name, "features": ", ".join(cols), "n_features": len(cols),
                     "silhouette": silhouette_score(X, labels), "ari_vs_production": adjusted_rand_score(reference, labels)})
    return pd.DataFrame(rows)


def coherence(df, cluster_col, columns) -> pd.DataFrame:
    """Per-column: cluster means, Kruskal-Wallis p-value across clusters, and
    eta-squared (share of the column's variance explained by the clusters).
    Use with columns that were NOT clustering inputs."""
    rows = []
    for col in columns:
        sub = df[[cluster_col, col]].dropna()
        groups = [g[col].values for _, g in sub.groupby(cluster_col)]
        grand = sub[col].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((sub[col] - grand) ** 2).sum()
        row = {"variable": col, "n": len(sub), "kruskal_p": kruskal(*groups).pvalue, "eta_squared": ss_between / ss_total}
        row.update({f"mean_cluster_{c}": g[col].mean() for c, g in sub.groupby(cluster_col)})
        rows.append(row)
    return pd.DataFrame(rows)
