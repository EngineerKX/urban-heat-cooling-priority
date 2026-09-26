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
convincing than either model's answer alone. (How that has actually played
out is in "How far to trust the what-if" at the bottom — short version: the
direction agrees on the demo cases, but the subzone-level size of the effect
is smaller than the model's own error.)

## Data: a pure join, no new fetching

```python
XGB_FEATURE_COLUMNS = [
    "fraction_vegetation", "fraction_built_up", "fraction_bare", "fraction_water",
    "ndvi_dry", "ndvi_wet", "ndbi_dry", "ndbi_wet",
    "population_total", "elderly_proportion",
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
  census data). Note the priority score's sensitivity pillar now uses population
  *density* (residents per km²), but this model still takes the raw *count*
  (`population_total`); whether density would predict temperature better
  hasn't been tested.
- The hotspot-typology cluster label (file 07) is deliberately **not** a
  feature — see "Why the cluster label is not a feature" below.

**Filtering here is an inner join, not a season/cloud filter**: if a subzone
is missing from any of the three source tables, it's silently dropped from
the training set (with a printed count so a large drop is visible, not
silent). Since all three sources are themselves already built from the same
subzone boundary list, drops should be rare — a large one is a genuine
signal something upstream broke, not expected behavior. The join keeps all
332 subzones.

**It trains on every subzone, including the ones the priority ranking
skips.** The 500-resident rule (`MIN_RESIDENTS_FOR_RANKING`) only decides
which subzones get *ranked* for cooling priority, because the elderly share
of a tiny population is too unstable to rank on. It has no bearing here:
this model predicts temperature, which every subzone has whether or not
anyone lives there. So the 121 subzones under 500 residents (parks,
reserves, industrial estates, water) stay in the training table —
`population_total` is simply small for them.

**Why the cluster label is not a feature** (changed 2026-09-21). It used to
be: `primary_cluster`, file 07's output, fed forward as an input. But file
07's clustering takes `lst_dry` as one of its six inputs, and `lst_dry` is
the very same number as this model's target `lst_native30` (identical in
all 332 subzones). So the cluster label was partly built from the answer and
the model could read some of it back — indirect target leakage. Removing it
cost only a little held-out accuracy (mean over 30 random 80/20 splits: R²
0.812 → 0.785, RMSE 1.05 → 1.13 °C), so the model does not depend on it.
`tests/test_heat_model_tabular.py::test_no_target_derived_features` stops it
(or any LST column) coming back. The pandas `category` / `enable_categorical`
handling stays in the code, unused, for any future categorical feature.

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
  subzones and 10 features, deep trees (the XGBoost default is often 6)
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

**The current number** (after the cluster label was dropped): the seed-42
split scores **RMSE 1.13 °C, R² 0.761** on its 67 held-out subzones (265
train). That is one draw. Over 30 different random 80/20 splits R² averages
0.785 with a standard deviation of 0.043, so with only 67 test subzones a
single split can land a few hundredths either side. For scale, subzone
temperatures vary by about 2.5 °C (standard deviation), so the model
explains roughly three quarters of the differences between subzones and is
typically about 1.1 °C off.

Don't line this 1.13 °C up against the CNN's 6.19 °C validation RMSE
(file 04) as if they were the same test. The CNN predicts each 10 m pixel
and is scored on held-out patches; XGBoost predicts a whole-subzone value
and is scored on held-out subzones. Different targets, different
granularity — the two numbers say how each model does *its own* job.

## Which features it leans on

The training script writes `xgb_feature_importance.csv` (XGBoost's built-in
importance scores, summing to 1). `fraction_built_up` (0.59) and `ndbi_dry`
(0.17) carry most of it, `ndbi_wet` (0.06) is third, and
`fraction_vegetation` is last at 0.011. That is sensible for predicting how
hot a subzone is — how much of it is built up is the strongest single
signal — but it matters for the counterfactual below, which works by
changing the vegetation fraction, the feature the model uses least.

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

## How far to trust the what-if (checked 2026-09-26)

Running the +15-percentage-point vegetation edit on every subzone that has
room for it (326 of 332; the other 6 are already over 85% vegetated, so the
edit is clamped) with the saved model gives a mean change of **−0.17 °C**
and a median of **−0.11 °C**. Only 57% of subzones come out cooler at all,
and the middle 80% range from −1.10 °C to +0.73 °C. That spread is no larger
than the model's ~1.1 °C typical error, and a subzone getting *hotter* when
it is greened is not plausible for surface temperature — a sign the model's
response to this edit is mostly noise rather than physics. So read a single
subzone's what-if number as "indicative, not a forecast".

A likely reason (a hypothesis, not tested): the model barely uses vegetation
fraction, and the edit lowers the built-up fraction and raises NDVI but
leaves NDBI — the model's second most-used feature — exactly where it was.
The edited row then describes a subzone no real place resembles. Making the
counterfactual move NDBI consistently is the obvious next fix; it hasn't
been done.

The CNN cross-check (`scripts/diagnose_heat_model.py`) on its three demo
hotspots (TUAS NORTH, GUL CIRCLE, CHIN BEE — the top three of the 4 Aug
ranking, not of the current top 20) agrees in sign 3 out of 3: XGBoost
−1.71 / −1.35 / −0.46 °C versus CNN −3.67 / −2.78 / −2.99 °C. The sizes
aren't comparable — XGBoost changes the whole subzone's vegetation share by
15 points, while the CNN repaints about 80 pixels (a 50 m-radius circle) and
reports the change inside that circle — so only the sign is compared, and
three cases is a thin basis for confidence.
