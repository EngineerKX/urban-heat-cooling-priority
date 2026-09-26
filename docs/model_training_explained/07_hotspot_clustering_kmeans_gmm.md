# Hotspot-typology clustering (K-means / GMM) — the unsupervised clustering model

Code: [`src/hotspots/cluster.py`](../../src/hotspots/cluster.py).

## What makes this one different from every other model in the project

Every model in files 02–05 is **supervised** — each has a known correct
answer during training (a hand-labeled land-cover point, a measured
temperature) that the model is explicitly pushed toward. This one has no
such answer. Nobody has hand-labeled "this subzone is Typology A" anywhere
in the pipeline. The task is genuinely exploratory: given each subzone's
seasonal heat/vegetation/built-up profile, find natural groupings in that
data on its own, with no target to be right or wrong against.

## The features: seasonal contrast, not a snapshot

```python
FEATURE_COLUMNS = ["lst_dry", "lst_wet", "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet"]
```

Every feature here comes in a dry-season/wet-season pair. This is a
deliberate choice, not just "more data is better": a subzone that's hot and
built-up *year-round* is a genuinely different kind of place from one
that's hot only in the dry season but cools and greens up notably in the
wet season — the *seasonal swing itself* is informative, and you can only
see it by feeding both seasons in rather than a single averaged snapshot.
This is also the one place `WET_SEASON_MONTHS` (file 01, §1) actually gets
used — everywhere else in the pipeline only ever builds the single
dry-season composite.

**One consequence worth knowing:** `lst_dry` here is the identical number to
XGBoost's target `lst_native30` (file 05) — the two match in all 332
subzones. So the cluster label is partly built from XGBoost's answer, and
must never be fed to a model that predicts it. That is why the label was
removed from XGBoost's features on 2026-09-21. The same feature table does
still supply XGBoost's land-cover fractions and dry/wet NDVI/NDBI — just
not the label.

## Preparing the features: why standardize, why drop NaN

```python
def prepare_features(df, feature_columns=FEATURE_COLUMNS):
    clean_df = df.dropna(subset=feature_columns).reset_index(drop=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_df[feature_columns].values)
```

**Standardization (z-scoring) is not optional here the way it might be for
a tree-based model.** Both K-means and GMM measure *distance* between
points (Euclidean for K-means, Mahalanobis-like for GMM's Gaussian
components) to decide what's "close" to what. LST is measured in °C
(values like 30-45), while NDVI/NDBI are unitless ratios roughly in
[-1, 1]. Without standardizing, the clustering would be dominated almost
entirely by whichever feature happens to have the largest raw numeric
range — not because it's more informative, but purely because of its
units. `StandardScaler` rescales every feature to mean 0, standard
deviation 1, so all six features contribute on equal footing.

**Dropping rows with any missing feature**, rather than imputing a
substitute value: clustering has no concept of "this value is unknown" the
way a supervised loss can just assign zero weight (files 01/03's
`weight` mask trick doesn't have an equivalent here) — a NaN would either
crash the distance computation or silently corrupt it. A subzone missing
one of the six seasonal features is simply excluded from clustering
entirely, with the dropped count printed so a large drop is visible.

## Two algorithms, on purpose, per the project's own stated scope

The project spec explicitly says "K-means/GMM," so both get built and
compared rather than picking one upfront:

- **K-means**: assumes clusters are roughly spherical, equal-sized blobs;
  assigns each point to exactly one cluster (hard boundary).
- **Gaussian Mixture Model (GMM)**: models each cluster as its own Gaussian
  distribution (which can be elongated/oriented, not just spherical) and
  gives **soft** membership — a probability of belonging to each cluster,
  not a single hard label. `label_clusters()` keeps those per-cluster
  probability columns explicitly, since soft assignment is one of GMM's
  genuine advantages over K-means and the code preserves it rather than
  collapsing straight to a hard label.

Both get compared using the same unsupervised-quality metrics
(silhouette score, Davies-Bouldin index — neither needs ground truth, both
just measure "how well-separated are the resulting clusters"), and
whichever one scores better on silhouette becomes `primary_cluster`, while
the other stays visible as an ablation column rather than being discarded —
the same "keep the alternative visible" pattern used elsewhere in this
project (PCA vs. equal-weight scoring, and RF/U-Net/hybrid all being
reported side by side rather than only the winner).

## Choosing k: why silhouette, why a floor of `min_k=3`

```python
def choose_k(sweep_df, min_k=3):
    candidates = sweep_df[sweep_df["k"] >= min_k]
    return int(candidates.loc[candidates["silhouette"].idxmax(), "k"])
```

Both algorithms are swept across `k = 2..8` (`sweep_kmeans`/`sweep_gmm`),
and the k with the best silhouette score wins — silhouette measures how
similar each point is to its own cluster vs. the next-nearest one, so a
higher score means genuinely more separated, more meaningful clusters, not
just "more clusters."

**The `min_k=3` floor exists to rule out a specific failure mode**: with
real-world data, a 2-way split (e.g. "everything hot" vs. "everything
cool") often wins on raw silhouette score simply because *any* 2-way split
of continuous data tends to be more separable than a finer one — that's a
mathematical property of the metric, not evidence of 2 being the most
*useful* number of typologies for this problem. Flooring the search at
`k ≥ 3` prevents the sweep from defaulting to a technically-highest-scoring
but practically uninformative 2-cluster split.

**`n_init=10`** (K-means) / **`n_init=5`** (GMM): both algorithms are
sensitive to their random starting point and can converge to a locally
(not globally) optimal clustering. Running each `k` multiple times from
different random initializations and keeping the best result is the
standard defense against landing on a bad local optimum — not a tuned
value, just the conventional way both algorithms are normally run.

## What it actually found

K-means beat GMM on silhouette and the sweep settled on **k = 4**
(`primary_cluster_method` is `kmeans` for all 332 subzones). The four groups,
from `data/processed/hotspot_cluster_profile.csv`:

| Cluster | Subzones | Dry-season LST | Drop, dry → wet | Vegetation | Built-up | Water | Plain reading |
|---|---|---|---|---|---|---|---|
| 3 | 92 | 42.3 °C | 2.7 °C | 14% | 83% | 2% | hottest, densely built up |
| 0 | 142 | 39.6 °C | 2.6 °C | 29% | 67% | 2% | warm, mostly built up with some green |
| 2 | 31 | 39.9 °C | 6.2 °C | 9% | 75% | 14% | least vegetated, water-heavy, big seasonal swing |
| 1 | 67 | 36.5 °C | 2.1 °C | 69% | 22% | 7% | coolest, green |

The "plain reading" column is a description of the numbers, not a name used
in the code. Cluster 2 is the distinctive one: its dry-to-wet temperature
drop is more than double the other three, which is exactly the kind of
difference the seasonal-pair features were chosen to expose.

## The coherence check — validating an unsupervised result without labels

You can't compute "accuracy" for clustering the way you can for a
classifier — there's no ground-truth cluster id to compare against.
Instead, `profile_clusters()` supports an indirect sanity check: if
land-cover fractions (from the RF+U-Net hybrid, file 06 — **not** a
clustering input) are supplied, the hottest cluster should, as a byproduct
of real physical geography, also turn out to be the most built-up /
least-vegetated one
(`validation/hotspots_validation/cluster_quality.py::sanity_check_landcover_coherence`).
This isn't circular — land-cover fractions were never part of what the
clustering algorithm saw — it's an independent check that the clusters the
algorithm found on its own actually correspond to something physically
real, rather than an artifact of the six input features alone.

**Current result: half a pass.** The hottest cluster (3) is also the most
built-up one, so the built-up half of the check passes. The vegetation half
does not: the least-vegetated cluster is 2 (9%), not the hottest one (14%).
That doesn't make the clustering wrong — cluster 2 is a different kind of
place (much more water, a far larger seasonal swing, cooler in the dry season
than cluster 3) and separating it out is what a typology is for — but it
means the check as written ("hottest = most built-up = least vegetated")
holds for one of its two conditions, not both.
