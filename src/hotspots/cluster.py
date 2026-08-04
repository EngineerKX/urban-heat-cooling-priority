"""K-means / GMM hotspot-typology clustering on the S4 seasonal feature
table. Builds BOTH algorithms (the project spec says "K-means/GMM"),
compares them via unsupervised-clustering quality metrics, and keeps
whichever wins as `primary_cluster` while leaving the other visible as an
ablation column -- the same "visible ablation" habit as PCA-vs-equal-weight
scoring and the RF/U-Net/ensemble land-cover comparison.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from config.settings import RANDOM_SEED

FEATURE_COLUMNS = ["lst_dry", "lst_wet", "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet"]


def prepare_features(df: pd.DataFrame, feature_columns=FEATURE_COLUMNS):
    """Z-scores feature_columns after dropping any row with a missing value
    (clustering can't handle NaN). Returns (X_scaled, scaler, clean_df) so
    callers can align cluster labels back onto subzone_id."""
    clean_df = df.dropna(subset=feature_columns).reset_index(drop=True)
    n_dropped = len(df) - len(clean_df)
    if n_dropped:
        print(f"⚠️  Dropped {n_dropped}/{len(df)} subzones with missing seasonal features before clustering.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[feature_columns].values)
    return X_scaled, scaler, clean_df


def sweep_kmeans(X_scaled, k_range=range(2, 9), seed: int = RANDOM_SEED) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)
        labels = model.labels_
        rows.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        })
    return pd.DataFrame(rows)


def sweep_gmm(X_scaled, k_range=range(2, 9), seed: int = RANDOM_SEED) -> pd.DataFrame:
    rows = []
    for k in k_range:
        model = GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(X_scaled)
        labels = model.predict(X_scaled)
        rows.append({
            "k": k,
            "bic": model.bic(X_scaled),
            "aic": model.aic(X_scaled),
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        })
    return pd.DataFrame(rows)


def choose_k(sweep_df: pd.DataFrame, min_k: int = 3) -> int:
    """Highest silhouette score at or above min_k -- the floor rules out a
    trivial/uninformative 2-way split winning purely on separability."""
    candidates = sweep_df[sweep_df["k"] >= min_k]
    if candidates.empty:
        candidates = sweep_df
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"])


def fit_kmeans(X_scaled, k: int, seed: int = RANDOM_SEED) -> KMeans:
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_scaled)


def fit_gmm(X_scaled, k: int, seed: int = RANDOM_SEED) -> GaussianMixture:
    return GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(X_scaled)


def label_clusters(df: pd.DataFrame, model, X_scaled, label_col: str) -> pd.DataFrame:
    """Adds label_col (cluster id) to a copy of df. For a GaussianMixture,
    also adds per-cluster membership-probability columns
    (<label_col>_prob_0..k-1) -- GMM's soft assignment is one of its
    genuine advantages over K-means' hard boundary, worth keeping visible."""
    out = df.copy()
    if isinstance(model, GaussianMixture):
        out[label_col] = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        for i in range(probs.shape[1]):
            out[f"{label_col}_prob_{i}"] = probs[:, i]
    else:
        out[label_col] = model.labels_
    return out


def profile_clusters(df: pd.DataFrame, cluster_col: str, feature_columns=FEATURE_COLUMNS,
                      landcover_fraction_columns=None) -> pd.DataFrame:
    """Mean feature value per cluster -- the interpretability check ("what
    does Cluster 2 actually look like?"), plus, if land-cover fraction
    columns are supplied (from src/landcover/zonal.py, NOT a clustering
    input), a coherence cross-check: the hottest cluster should also be the
    most built-up / least-vegetated one, as a byproduct of real physical
    geography rather than by construction (see
    validation/hotspots_validation/cluster_quality.py::sanity_check_landcover_coherence)."""
    cols = feature_columns + (landcover_fraction_columns or [])
    profile = df.groupby(cluster_col)[cols].mean()
    profile["n_subzones"] = df.groupby(cluster_col).size()
    return profile.reset_index()
