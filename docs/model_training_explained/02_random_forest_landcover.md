# Random Forest land-cover classifier

Code: [`src/landcover/rf_baseline.py`](../../src/landcover/rf_baseline.py).
Assumes you've read
[01_data_preparation_and_filtering.md](01_data_preparation_and_filtering.md)
— this file only covers what's specific to RF.

## What it predicts

Per-pixel land cover: one of 4 classes (vegetation / built_up / bare /
water), from the 9-band feature composite (6 raw Sentinel-2 bands + 3
spectral indices) described in file 01.

## How training samples are drawn

RF doesn't train on every pixel in Singapore — it trains on a **stratified
sample**: a fixed number of points per class, drawn from the training region
(already excluding the 15m buffer around every validation point).

```python
TRAINING_POINTS_PER_CLASS = 3000
```

`extract_training_samples()` uses Earth Engine's `stratifiedSample`, which
does this sampling **server-side** — important because GEE caps a
synchronous `FeatureCollection` pull at 5000 elements, and 4 classes × 3000
points each (12,000 total) would blow past that if you tried to pull
individual points down first and sample locally. `tileScale=8` in the same
call subdivides the computation into smaller tiles server-side, which
avoids GEE's per-request memory limit on a sampling job covering all of
Singapore at once.

**Why stratified, not random?** Singapore's real land cover is heavily
imbalanced — vastly more built-up area than bare land, for instance. A
plain random sample would hand the classifier a training set dominated by
the majority classes, and it would have very little signal to learn the
rare ones from. Capping *and* stratifying to `TRAINING_POINTS_PER_CLASS`
per class forces a roughly balanced training set regardless of the true
class proportions on the ground.

## The model itself: `smileRandomForest`

```python
RF_NUM_TREES = 200
RF_MIN_LEAF_POPULATION = 1
RF_BAG_FRACTION = 0.5
```

This is Earth Engine's built-in Random Forest implementation (a Java port of
the Weka/SMILE library), trained entirely server-side — there's no local
scikit-learn model here.

- **`numberOfTrees=200`**: a standard "large enough to stabilize, not so
  large it's wasteful" tree count for a 4-class problem — RF's accuracy
  typically plateaus well before 200 trees for a feature set this small (9
  bands), so this isn't a tightly-tuned number so much as a comfortable
  margin past the plateau.
- **`minLeafPopulation=1`**: no minimum — trees are allowed to grow until
  leaves contain a single training point if the data supports it. This is
  the SMILE/Weka default; RF's ensemble averaging (200 independently-bagged
  trees) is what keeps this from overfitting the way a single unconstrained
  tree would, not a leaf-size floor.
- **`bagFraction=0.5`**: each of the 200 trees sees a random 50% of the
  training points (sampled with replacement, standard bagging) — the
  randomness across trees (both in which points and, internally, which
  features each split considers) is what makes an RF an ensemble in the
  first place rather than 200 copies of the same tree.

None of these three were empirically swept in this project — they're
standard, safe RF defaults for a small, low-dimensional (9-feature)
problem. If you wanted to genuinely tune them, `RF_NUM_TREES` and
`RF_MIN_LEAF_POPULATION` are the two most likely to move accuracy, via a
grid search scored against the 300-point validation set.

## Two output modes from one trained classifier

RF is trained once, then queried two different ways depending on what's
needed downstream:

- **`classify()`** → hard labels (`CLASSIFICATION` mode) — one class id per
  pixel, used for `informal_accuracy_check()` and the final classified
  raster.
- **`classify_probability()`** → soft probabilities (`MULTIPROBABILITY`
  mode) — a 4-band probability image (one band per class), used by the
  RF+U-Net hybrid combination (file 06) and the formal evaluation's
  per-class metrics.

One real gotcha documented directly in the code: a classifier that's been
round-tripped through the GEE asset cache (`Export.classifier.toAsset` /
`ee.Classifier.load()`) only supports `CLASSIFICATION` mode — calling
`setOutputMode("MULTIPROBABILITY")` on a *loaded* asset classifier fails
outright. Probability output is only available from a classifier trained
fresh in the current session, which is why `train_rf_classifier()`'s
asset-cache path is skipped whenever `--with-probabilities` is requested.

## Caching: a GEE asset, not a local file

Unlike U-Net/CNN (which save local `.keras` weight files), RF's trained
model isn't something you can serialize to disk the normal way — `ee.Classifier`
objects only exist inside Earth Engine's own execution environment. Instead,
`train_rf_classifier()` persists the trained classifier as a **GEE asset**
(`ee.batch.Export.classifier.toAsset`) and reloads from there on later runs,
skipping retraining entirely if the asset already exists. This is optional
(`use_asset_cache=True` by default) and fails soft: if the asset export
doesn't work (e.g. no asset root configured yet in your GEE project), the
classifier still trains successfully in-session, it just won't be cached
for next time.
