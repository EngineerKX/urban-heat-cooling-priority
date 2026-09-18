# CNN heat-model (LST regressor)

Code: [`src/heat_model/cnn_model.py`](../../src/heat_model/cnn_model.py) (architecture),
[`cnn_data.py`](../../src/heat_model/cnn_data.py) (data), [`cnn_train.py`](../../src/heat_model/cnn_train.py) (training loop).

**Read [03_unet_landcover_classifier.md](03_unet_landcover_classifier.md)
first.** This model reuses U-Net's exact backbone — same encoder-decoder,
same layer sizes, same reasoning for all of it. This file only covers what's
genuinely different: the target, the inputs, and the head.

## What it predicts, and why this specific target

Land surface temperature (LST) in °C, per pixel — a continuous value, not a
class. This is a **regression** problem, not classification, even though the
backbone is identical.

The target is `lst_bicubic10` specifically, not the more "natural"-sounding
alternative `lst_regress10`. This matters: `lst_regress10` is itself a
downscaled LST variant that was produced by *regressing on NDVI/NDBI/NDWI*
(the exact same three spectral indices used as CNN input features here). If
the CNN trained against `lst_regress10`, it would partly just be learning to
reproduce the regression that *generated* the label — a circularity problem,
not a genuine "predict LST from spectral input" task. `lst_bicubic10` is
pure bicubic interpolation of the coarser native-resolution Landsat LST, no
spectral regression baked in, so it's a clean, independent target.

## The input is wider than U-Net's — why 13 channels, not 9

```python
n_bands = len(ALL_FEATURE_BANDS)          # 9: same S2 bands + indices as U-Net
n_landcover_classes = N_LANDCOVER_CLASSES  # 4: vegetation/built_up/bare/water
in_channels = n_bands + n_landcover_classes  # 13
```

The CNN reuses U-Net's already-exported inference patches (the same 9
spectral/index bands — no new GEE work needed for those), but **adds 4
one-hot land-cover channels** built from the RF+U-Net hybrid raster (file
06). The reasoning: land surface temperature is heavily influenced by what's
physically on the ground (asphalt retains heat very differently from tree
canopy), and rather than making the CNN re-derive "what kind of surface is
this" purely from raw spectral values, the already-validated hybrid
land-cover classification is handed to it directly as an extra, explicit
input. This is a form of feature engineering: giving the model information
it would otherwise have to learn indirectly and less reliably.

Because `in_channels` changes, `build_unet_backbone(input_shape, base_filters)`
is called with a different `input_shape` — `(128, 128, 13)` instead of
`(128, 128, 9)` — but the *architecture logic itself* (3 encoder levels,
32-filter base, doubling per level, skip connections) is byte-for-byte the
same function as U-Net's. This is precisely why the codebase's earlier
"ensemble vs. hybrid" naming discussion mattered: it's a genuinely shared
backbone between two different models, wrapped by two different heads.

## The head is different: regression, not classification

```python
outputs = layers.Conv2D(1, 1, activation="linear")(d1)   # CNN: 1 channel, linear
outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d1)   # U-Net: 4 channels, softmax
```

Same 1×1-conv mechanism as U-Net's head (a per-pixel linear layer over the
final 32 feature channels — see file 03's explanation of why 1×1 is the
right op here), but:

- **1 output channel, not 4** — one continuous temperature value per pixel,
  not a probability per class.
- **`linear` activation, not `softmax`** — softmax forces its outputs to sum
  to 1 across channels, which only makes sense for "probability of each of
  N classes." A temperature prediction has no such constraint — it's a raw,
  unbounded real number (well, bounded by physics, but not by the model's
  output layer), so `linear` (i.e., no activation function at all) is the
  correct choice: the raw weighted sum passes straight through.

## Loss and metric: MSE/RMSE instead of cross-entropy/accuracy

```python
loss = tf.keras.losses.MeanSquaredError()
weighted_metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")]
```

Cross-entropy measures how wrong a *probability distribution* is; it's
meaningless for a single continuous value. Mean Squared Error is the
standard regression loss — it penalizes larger errors disproportionately
more than smaller ones (squared, not absolute), which is usually the right
behavior for a physical quantity like temperature where a 5°C miss is worse
than twice as bad as a 2.5°C miss, not just twice as bad. RMSE (the square
root of MSE) is reported as the tracked metric purely because it's in the
same units as the target (°C), which makes it directly interpretable — "the
model is off by about 6°C on average" is meaningful in a way "the MSE is
38.4" isn't.

The same `weighted_metrics` mechanism as U-Net applies here too — a
per-pixel `weight` mask (this time combining *two* validity conditions, see
next section) zeroes out the loss/RMSE contribution from invalid pixels.

## The valid_mask here is doing double duty

U-Net's mask only had to answer "is there a satellite observation here?"
The CNN's mask (`cnn_data.py::build_local_feature_target_patches`) has to
answer a second question too, because it has a second data source with its
own gaps:

```python
valid_mask[idx] = (lc_window != 0) & ~np.isnan(lst_window)
```

- `lc_window != 0` — the same land-cover validity condition as before (0 =
  no hybrid land-cover class at this pixel).
- `~np.isnan(lst_window)` — the LST raster has its own nodata pixels
  (wherever the underlying Landsat thermal composite had no clean
  observation), read as `NaN` and explicitly checked for, not silently
  passed through. A pixel is only ever counted as valid if **both** its
  land-cover class and its LST target are genuinely present — either one
  missing invalidates the whole training example at that pixel, for the
  same reason described in file 01: a model must never be trained against
  a target or feature that doesn't actually exist.

## Same hyperparameters as U-Net, on purpose — but as independent constants

```python
CNN_BATCH_SIZE = 8
CNN_EPOCHS = 30
CNN_LEARNING_RATE = 1e-3
CNN_BASE_FILTERS = 32
CNN_EARLY_STOP_PATIENCE = 5
CNN_TRAIN_VAL_SPLIT = 0.85
```

Every value is currently identical to U-Net's (file 03 has the full
reasoning for each). They're deliberately kept as **separate named
constants** in `config/settings.py` rather than one shared `UNET_*` block
reused by both models — not because the values need to differ today, but so
the two models can be tuned independently later without the constant name
misleadingly implying "this is a U-Net setting" on what's actually CNN
training code. If you ever do want to test a different value for one model
without touching the other, this is exactly why that's already possible
with a one-line settings change.

## Real training result, for a concrete reference point

The last real training run (Colab, T4 GPU): stopped early at epoch 20/30
(5 epochs without improvement past epoch 15), best `val_rmse ≈ 6.19°C`. That
number — not the architecture — is the thing actually worth trying to
improve if this model's accuracy matters for your writeup; see the earlier
"why only ~70% land-cover accuracy" discussion in this conversation for the
general pattern (thin labels and a narrow feature set constrain how far any
of these models can go without more/better input data, not just more
tuning).
