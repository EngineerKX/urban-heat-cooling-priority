# U-Net land-cover classifier — architecture and hyperparameters explained

Code: [`src/landcover/unet_model.py`](../../src/landcover/unet_model.py) (architecture),
[`unet_data.py`](../../src/landcover/unet_data.py) (data), [`unet_train.py`](../../src/landcover/unet_train.py) (training loop).
This is the file to read if you want to understand *why the network looks
the way it does*, not just what it does.

## Why U-Net at all, instead of a simpler CNN?

RF (file 02) classifies each pixel using **only that pixel's own 9 feature
values** — it has no idea what's next to it. That's a real limitation: a
single pixel's spectral signature can be ambiguous (e.g. a shadowed patch of
grass can spectrally resemble something else), but a *human* looking at the
same spot would use context — "this pixel is surrounded by clearly
vegetated pixels, so it's probably vegetation too."

U-Net is built specifically to use that spatial context. It's a **semantic
segmentation** architecture: it takes in a whole image patch and outputs a
class prediction for every pixel simultaneously, using the full
neighborhood around each pixel to inform each pixel's answer — not a
sliding-window classifier repeated per-pixel.

## The two-halves shape: encoder → decoder

Every U-Net has the same conceptual shape, and this one is no exception —
look at `build_unet_backbone()`:

```
Input (128×128×9)
  │
  ├─ Encoder: shrink spatially, grow channels  (learn "what")
  │
  ├─ Bottleneck
  │
  └─ Decoder: grow spatially back, shrink channels (learn "where")
       + skip connections from the matching encoder level
```

**Encoder (downsampling path)**: each level applies two 3×3 convolutions
(`_conv_block`) then halves the spatial size with `MaxPooling2D(2)`. Shrinking
the spatial dimensions while growing the channel count is what lets the
network trade "precise pixel location" for "bigger-picture pattern
recognition" as you go deeper — by the bottleneck, each "pixel" in the
feature map represents a much larger patch of the original image, encoding
*what kind of area this is* rather than *exactly which pixel*.

**Decoder (upsampling path)**: the mirror image — `Conv2DTranspose` (a
learned upsampling operation, not simple interpolation) doubles the spatial
size back at each level, followed by another `_conv_block`.

**Skip connections** (the `Concatenate` layers): this is the actual "U" in
U-Net, and it's the single most important design choice in the whole
architecture. Without it, the decoder would only have the bottleneck's
heavily-downsampled, spatially-vague features to work from — it would know
roughly "this region is built-up" but have lost the precise pixel-level
boundaries of *where* built-up starts and stops. Each decoder level
concatenates in the matching encoder level's full-resolution features
(`c3`, `c2`, `c1`) *before* those features got downsampled — literally
handing the decoder back the fine spatial detail it would otherwise have
thrown away, so the final output has both "what" (from the deep path) and
"exactly where" (from the skip path).

## Layer-by-layer, with the actual shapes

For U-Net, `base_filters=32`, `patch_size=128`, `in_channels=9`:

| Stage | Operation | Output shape |
|---|---|---|
| Input | — | 128×128×9 |
| `enc1` | Conv3×3→Conv3×3 (32 filters each) | 128×128×32 |
| `pool1` | MaxPool 2×2 | 64×64×32 |
| `enc2` | Conv3×3→Conv3×3 (64 filters) | 64×64×64 |
| `pool2` | MaxPool 2×2 | 32×32×64 |
| `enc3` | Conv3×3→Conv3×3 (128 filters) | 32×32×128 |
| `pool3` | MaxPool 2×2 | 16×16×128 |
| `bottleneck` | Conv3×3→Conv3×3 (256 filters) | 16×16×256 |
| `up3` + concat(`enc3`) | ConvTranspose→256, concat 128 | 32×32×256 |
| `dec3` | Conv3×3→Conv3×3 (128 filters) | 32×32×128 |
| `up2` + concat(`enc2`) | ConvTranspose→128, concat 64 | 64×64×128 |
| `dec2` | Conv3×3→Conv3×3 (64 filters) | 64×64×64 |
| `up1` + concat(`enc1`) | ConvTranspose→64, concat 32 | 128×128×64 |
| `dec1` | Conv3×3→Conv3×3 (32 filters) | 128×128×32 |
| Head | Conv1×1, softmax, 4 filters | 128×128×4 |

**Why 3 encoder levels, not more or fewer?** Every level halves the spatial
size — 128 → 64 → 32 → 16 at the bottleneck. Going deeper (a 4th level would
reach 8×8) risks shrinking the spatial map so much that fine class
boundaries (e.g. the edge between two adjacent land-cover types) become
impossible to recover even with skip connections. 3 levels is the standard
depth for this size of input image (128px) — deep enough to capture
meaningful neighborhood context, shallow enough that the bottleneck (16×16)
still has enough spatial resolution to be useful.

**Why `base_filters=32`, doubling at each level (32→64→128→256)?** The
doubling pattern is the standard U-Net convention from the original 2015
paper: as spatial resolution halves, channel count doubles, so the total
amount of information each level can represent (spatial × channel) stays
roughly balanced rather than collapsing. `32` specifically as the starting
width is a capacity/compute tradeoff — big enough to represent 9 input
channels' worth of useful combinations, small enough to train in seconds
per epoch on a single GPU with only ~1000 training patches available (this
project's actual training-set size, not a huge dataset). This number **was
not empirically swept** in this project — it's the standard default, not a
tuned value. If you wanted to actually test whether a different capacity
helps, `base_filters=16` (faster, less capacity) or `64` (slower, more
capacity) against the same validation set would be the natural experiment,
and the architecture-preview cell in `train_unet.ipynb` exists specifically
so you can see the resulting parameter count instantly without a full
training run.

**Why two 3×3 convs per block, not one?** Two stacked 3×3 convolutions have
the same effective receptive field as one 5×5 conv, but with fewer
parameters and an extra ReLU non-linearity in between — a standard
efficiency trick, not specific to this project.

**Why a 1×1 conv as the final head, not something larger?** A 1×1
convolution at this point is doing something conceptually different from
every other conv in the network — it's not looking at a neighborhood, it's
just a per-pixel linear layer that maps the 32 feature channels down to 4
class scores at every pixel independently. All the *spatial* reasoning
already happened in the layers before it; this layer's only job is turning
"32 learned features at this pixel" into "4 class probabilities at this
pixel," which is exactly what a 1×1 conv does and nothing more.

## Data: what actually flows into the network

**Patches, not the whole image at once.** Singapore doesn't fit in GPU
memory as a single dense array at 10m resolution, and CNN training needs
many independent training examples anyway. GEE exports the training/
inference regions as `128×128` TFRecord patches (`UNET_PATCH_SIZE = 128`),
gzip-compressed. `128` is a practical middle ground: small enough that many
patches fit in a batch, large enough that each patch still contains
meaningful spatial context (a 128×128 patch at 10m resolution covers
1.28km × 1.28km — several city blocks).

**Channels-last format** (`N, H, W, C`): this is simply Keras/TensorFlow's
native tensor convention (as opposed to PyTorch's channels-first `N, C, H,
W`) — a framework requirement, not a modeling decision.

**The weight mask, revisited from a training perspective.** File 01
explained *why* a per-pixel `weight` (0 or 1) exists. Here's how it's
actually used during training: `weight` is passed as the 3rd element of
every batch tuple `(features, label_onehot, weight)`, and Keras's
`weighted_metrics` mechanism uses it to compute a *weighted* loss/accuracy —
pixels with `weight=0` (no valid label) contribute exactly nothing to the
gradient, without needing to be physically removed from the batch tensor
(which would require variable-shaped patches, a real engineering headache
CNNs don't handle well).

## Training hyperparameters, one at a time

```python
UNET_BATCH_SIZE = 8
UNET_EPOCHS = 30
UNET_LEARNING_RATE = 1e-3
UNET_EARLY_STOP_PATIENCE = 5
UNET_TRAIN_VAL_SPLIT = 0.85
```

- **`batch_size=8`**: how many 128×128×9 patches are processed together per
  gradient step. Small on purpose — larger batches would need more GPU
  memory (each patch already carries 9-13 channels), and with only ~1000
  total patches, a small batch size also means more gradient updates per
  epoch, which tends to help convergence on a small dataset. Not swept —
  it's a memory-comfortable default for a free-tier T4 GPU.
- **`epochs=30`**: an upper *ceiling*, not a target — with early stopping
  active (next point), training stops well before 30 if it's not improving.
  30 is simply high enough that early stopping, not the epoch count, is
  what actually decides when training ends in practice (confirmed by the
  real runs: U-Net stopped at epoch 25, CNN at epoch 20 — both cut short by
  early stopping, never hitting the 30-epoch ceiling).
- **`learning_rate=1e-3`**: Adam's own well-known default starting point
  for a wide range of problems — not tuned here, just the standard first
  choice, and a reasonable one since Adam adapts the *effective* per-
  parameter step size during training anyway, making it fairly forgiving
  of the initial value.
- **`early_stop_patience=5`**: training stops if validation loss hasn't
  improved for 5 consecutive epochs, and the model's weights are rolled
  back to whichever epoch *did* have the best validation loss
  (`restore_best_weights=True`) — so a temporary bad patch late in training
  can't leave you with worse weights than an earlier, better epoch. 5 is a
  reasonable balance: large enough to not stop on ordinary epoch-to-epoch
  noise, small enough to not waste a full 30 epochs chasing a genuinely
  plateaued run.
- **`train_val_split=0.85`**: 85% of the *training* patches are actually
  trained on; the remaining 15% are held out purely to decide when to early
  stop (see file 01 for why this is a different thing from the 300
  hand-labeled validation points). An 85/15 split is a standard ratio for a
  moderate-sized dataset — enough held-out data to get a stable validation
  signal, without giving up so much training data that it hurts the model.

## Loss function and metric — and why they're not the plain Keras defaults

```python
loss = tf.keras.losses.CategoricalCrossentropy()
weighted_metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
optimizer = tf.keras.optimizers.Adam(learning_rate=UNET_LEARNING_RATE)
```

- **`CategoricalCrossentropy`**, not `SparseCategoricalCrossentropy`:
  labels are one-hot encoded (`tf.one_hot(label_idx, depth=n_classes)`)
  before training, because the *weight* mask needs to travel through the
  same per-pixel structure as the label — one-hot labels keep the shapes
  aligned cleanly with the softmax output and the weight tensor. This is a
  standard, correct choice for 4-class segmentation with per-pixel sample
  weights.
- **`weighted_metrics`, not plain `metrics`**: this is the mechanism that
  actually applies the `weight` mask to accuracy, not just the loss — using
  plain `metrics` here would report accuracy computed *including* the
  invalid (`weight=0`) pixels, silently inflating or deflating the reported
  number depending on what those invalid pixels happen to predict.
- **Adam**, not plain SGD: Adam adapts its effective learning rate per
  parameter based on recent gradient history, which generally converges
  faster and more reliably than plain SGD without needing a hand-tuned
  learning-rate schedule — the standard first choice for training a CNN
  from scratch, and not something this project needed to deviate from.

## Caching: skip training entirely if nothing changed

One thing that's easy to miss reading just the architecture: `train_unet()`
doesn't always train. It first checks whether a saved model already exists
*and* was trained on the current data (via a SHA-256 fingerprint of the
validation CSV — see file 01, §6–7). If both match, it loads the cached
`.keras` file and returns immediately, no GPU time spent. This exists
because relabeling the validation sample changes the training region
(different exclusion buffer), so a stale cached model trained against an
*old* labeling pass would be silently wrong to keep reusing — the
fingerprint check makes "did the underlying data change" a real,
automatic check instead of relying on a human to remember to pass
`force_retrain=True`.
