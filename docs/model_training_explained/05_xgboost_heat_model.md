# XGBoost heat model (subzone-level LST regressor)

Code: [`src/heat_model/tabular.py`](../../src/heat_model/tabular.py).

## What it predicts, and why it exists alongside the CNN

Also land surface temperature — but at **subzone granularity** (one number
per URA subzone, ~330 of them), from **tabular** features, not raw imagery.
This is a deliberately different approach from the CNN (file 04), not a
redundant one: the CNN answers "what's the temperature at this specific
10m pixel," XGBoost answers "what's the typical temperature across this
whole planning subzone, given its land-cover mix, seasonal indices, and
population." The two are cross-checked against each other
(`scripts/diagnose_heat_model.py`) rather than one replacing the other —
if a counterfactual "add more greenery" prediction moves in the *same
direction* under both a pixel-level CNN and a subzone-level XGBoost model
built from completely different inputs, that agreement is much more
convincing than either model's answer alone.

## Data: a pure join, no new fetching

```python
XGB_FEATURE_COLUMNS = [
    "fraction_vegetation", "fraction_built_up", "fraction_bare", "fraction_water",
    "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet",
    "population_total", "elderly_proportion", "primary_cluster",
]
```

Unlike RF/U-Net/CNN, this model does **no** GEE fetching of its own — every
feature already exists as a column in three CSVs produced by earlier
pipeline stages, and `build_xgb_training_table()` just inner-joins them on
`subzone_id`:

- Land-cover fractions + seasonal NDVI/NDBI (both dry and wet season) — from
  the S4 hotspot-clustering feature table (file 07), which itself already
  computed these.
- Population + elderly proportion — from the sensitivity pillar (SingStat
  census data).
- `primary_cluster` — the hotspot-typology cluster id each subzone was
  assigned to (file 07's *output* feeding forward as an *input* here).

**Filtering here is an inner join, not a season/cloud filter**: if a subzone
is missing from any of the three source tables, it's silently dropped from
the training set (with a printed count so a large drop is visible, not
silent). Since all three sources are themselves already built from the same
subzone boundary list, drops should be rare — a large one is a genuine
signal something upstream broke, not expected behavior.

**Why `primary_cluster` is cast to pandas `category` dtype**, not left as a
plain integer: cluster ids are labels (Cluster 0, 1, 2...), not an ordered
quantity — cluster "2" isn't twice cluster "1" in any meaningful sense.
Feeding it in as a plain int would let XGBoost's tree splits implicitly
treat it as ordered (e.g. "cluster ≤ 1.5"), which would be a made-up
relationship. `enable_categorical=True` on the model tells XGBoost to use
its native categorical-split handling instead, which doesn't impose that
false ordering.

## Target: same circularity concern as the CNN, same solution

```python
XGB_TARGET_COLUMN = "lst_native30"
```

`lst_native30` — the least-processed LST variant (native 30m Landsat
resolution, no downscaling regression involved) — for the identical reason
the CNN targets `lst_bicubic10` instead of `lst_regress10`: `ndvi_dry` and
`ndbi_dry` are both XGBoost input features here too, and `lst_regress10`
was itself produced by regressing on those same indices. Training against
it would again be circular. `lst_native30` has no such dependency on the
inputs.

## Hyperparameters

```python
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.8
```

XGBoost is a **boosting** ensemble — unlike RF's bagging (200 independent
trees averaged), boosting builds trees *sequentially*, each new tree trying
to correct the previous ensemble's remaining errors. That changes what each
hyperparameter is actually balancing:

- **`n_estimators=300`, `learning_rate=0.05`**: these two work as a pair.
  Boosting adds each new tree's contribution scaled by the learning rate —
  a *small* learning rate (0.05, vs. a more aggressive 0.1–0.3) means each
  individual tree only nudges the prediction a little, which needs *more*
  trees (300) to reach a good fit. This combination — slow steps, more of
  them — is a standard defense against overfitting in boosted trees: a
  high learning rate with few estimators can fit the training set's noise
  too eagerly, while many small steps average out that noise more.
- **`max_depth=4`**: a shallow tree depth on purpose. With only ~330
  subzones and 11 features, deep trees (the XGBoost default is often 6)
  would have enough capacity to memorize individual subzones rather than
  learn generalizable splits — capping depth at 4 keeps each tree simple
  enough that it can only capture broad, more-likely-to-generalize
  patterns.
- **`subsample=0.8`**: each tree trains on a random 80% of the training
  rows, not all of them — the same overfitting defense as RF's
  `bagFraction`, applied here to a boosting model instead of a bagging one.

None of these four were swept via grid search in this project — they're a
coherent, standard "conservative" boosting configuration (shallow trees,
slow learning rate, subsampling) chosen because the dataset is small
(~330 rows) and small tabular datasets are exactly where boosted trees are
most prone to overfitting if left at more aggressive defaults.

## Train/test split

```python
test_size = 0.2  # in train_xgb_model()
```

A plain 80/20 random split (`sklearn.train_test_split`, seeded via
`RANDOM_SEED`), evaluated with RMSE and R² on the held-out 20%. This is a
genuinely separate concept from U-Net/CNN's `TRAIN_VAL_SPLIT` — there's no
early stopping here to justify a validation split; this 20% exists purely
to report an honest, held-out accuracy number
(`scripts/train_heat_model_xgboost.py`'s own printed test RMSE/R² is the
number to trust — `diagnose_heat_model.py`'s "full-table RMSE" explicitly
documents itself as informal/in-sample-mixed, not a substitute for it).

## The counterfactual mechanism (why this model exists beyond just prediction)

Beyond plain prediction, `tabular.py` implements "what if this subzone had
X% more vegetation cover?" — `redistribute_vegetation_fraction()` shifts
`fraction_vegetation` up, pulling the difference proportionally out of
`fraction_built_up`/`fraction_bare` (deliberately never out of
`fraction_water` — greening a carpark doesn't plausibly convert open
water), then nudges `ndvi_dry`/`ndvi_wet` by a fitted OLS slope so those
features stay internally consistent with the hypothetically edited
land-cover mix, rather than leaving them stale while only the fraction
changes underneath them. The edited feature row is then re-predicted with
the *same trained model* — no retraining — and the difference between the
original and edited prediction is the counterfactual temperature delta.
