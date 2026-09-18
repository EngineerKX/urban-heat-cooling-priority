# Data preparation and filtering (shared by RF, U-Net, and the CNN heat model)

Every supervised model in this project (RF, U-Net, CNN) is trained on the
**same Sentinel-2 feature stack**, built the same way, filtered the same way.
This file explains that shared pipeline once, so the per-model files don't
repeat it. Code: [`src/ingest/gee.py`](../../src/ingest/gee.py),
[`config/settings.py`](../../config/settings.py).

## 1. Why a "season window" filter exists at all

Singapore has no real winter/summer — but it does have a monsoon cycle, and
cloud cover, wind, and insolation genuinely differ month to month. If you
build a "one composite" image from raw satellite passes across all 12
months, you're silently mixing wet-season and dry-season conditions into one
average, which blurs real signal (e.g. NDVI shifts, thermal patterns).

**What's actually filtered**: only images captured in specific months are
used at all — everything else is discarded before it ever reaches the model.

```python
DRY_SEASON_MONTHS = [4, 5, 10, 11]   # both inter-monsoon periods
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
```

`DRY_SEASON_MONTHS` = the two inter-monsoon periods (weak winds, low cloud,
high sun) — locked after testing candidate windows and picking the one that
produced the most *usable* (non-cloud-blocked) Landsat scenes (see
`validation/input_validation/season_window.py`). This isn't an arbitrary
choice — "peak heat, minimum cloud interference" is exactly the condition
you want for a heat-mapping model, and it's also the condition that survives
cloud filtering with enough scenes left to composite from.

`date_filter_for_years_months()` builds a filter that keeps an image if its
(year, month) falls in *any* of the requested combinations — a union filter,
not a single date range — so images from six different years can all
contribute to one "typical dry season" composite without pulling in
wet-season months in between.

`WET_SEASON_MONTHS = [12, 1, 2]` is a **separate** thing — it's only used as
an extra *feature* for hotspot clustering (S4), not part of the main heat
composite. The idea: a subzone's dry-vs-wet seasonal *contrast* (how much
hotter/greener it gets between seasons) is itself informative for typing
hotspots, even though the main pipeline only ever composites the dry season.

## 2. Why cloud filtering happens at two different levels

Clouds don't just block a satellite's optical view — for Sentinel-2 and
Landsat, an unmasked cloudy pixel would corrupt the reflectance/temperature
values a model trains on with values that describe *the cloud*, not the
ground. Two independent filters exist because they operate at different
granularities:

**Scene-level prefilter** (cheap, coarse): drop whole images whose metadata
already says "too cloudy overall," before doing anything expensive with them.

```python
LANDSAT_CLOUD_COVER_MAX = 70   # % cloud cover in the scene's own metadata
S2_CLOUD_PROB_MAX = 70         # per-pixel cloud probability threshold
```

**Pixel-level mask** (expensive, precise): even a scene that passes the
scene-level check usually still has some cloudy pixels in it. Each
data source gets its own per-pixel masking method, because each satellite
program encodes cloud information differently:

- **Landsat** (`mask_landsat_c2l2`): reads the `QA_PIXEL` band's bits —
  dedicated bit flags for dilated cloud, cirrus, cloud, shadow, and snow —
  and masks a pixel out if *any* of those bits are set. Also checks
  `QA_RADSAT` (radiometric saturation — a pixel that clipped the sensor,
  e.g. glint off water, isn't trustworthy either).
- **Sentinel-2** (`mask_s2_cloud_prob`): uses the separate `s2cloudless`
  product's per-pixel cloud *probability* band, masking anything above
  `S2_CLOUD_PROB_MAX` (70%). This runs on a probability score, not a hard
  bitmask, because Sentinel-2's own QA bands are less reliable than a
  dedicated ML-based cloud classifier.

The historical note in `config/settings.py`'s module docstring is worth
knowing: `S2_CLOUD_PROB_MAX` used to silently drift to `40` in one notebook
while staying `70` everywhere else — an easy mistake when the same constant
is copy-pasted across files instead of imported from one place. That drift
is exactly why this constant now lives in exactly one file.

## 3. How the composite itself is built

After season + cloud filtering, what's left is a collection of individual
cloud-masked images — not yet one picture. `build_feature_image()`
(`src/landcover/rf_baseline.py`) reduces that whole filtered collection down
to a single image with `.median()` per pixel: for every pixel location, take
the median value across every clean (non-cloud-masked) observation. Median,
not mean, because it's robust to the occasional missed cloud or sensor
artifact that a mean would let skew the result.

The composite is then reprojected onto a fixed 10m grid in UTM Zone 48N
(`S2_UTM_CRS = "EPSG:32648"`, `TARGET_SCALE_M = 10`) — Singapore's own local
projected CRS, not raw lat/lon degrees, so that a "meter" in the buffer/area
math elsewhere in the pipeline is an actual meter.

**Feature bands used** (`ALL_FEATURE_BANDS`, 9 total):

```python
S2_FEATURE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
INDEX_BANDS = ["NDVI", "NDBI", "NDWI"]  # derived, not raw satellite bands
```

The three index bands are computed, not measured — each is a normalized
difference between two raw bands, chosen because each isolates a specific
physical signal a single raw band can't:

- **NDVI** = (NIR − Red)/(NIR + Red) — vegetation "greenness" (healthy
  vegetation reflects NIR strongly, absorbs red).
- **NDBI** = (SWIR1 − NIR)/(SWIR1 + NIR) — built-up surfaces (asphalt,
  concrete reflect SWIR1 more than NIR; vegetation is the opposite).
- **NDWI** = (Green − NIR)/(Green + NIR) — open water (water absorbs NIR
  almost completely, unlike land).

These three map almost directly onto the project's 4-class scheme
(vegetation / built_up / bare / water), which is exactly why they're
included as explicit features rather than leaving the model to rediscover
the same ratios from the 6 raw bands on its own.

## 4. Why these 4 classes, not WorldCover's original 11

The training labels come from ESA WorldCover, which natively ships **11**
classes: tree cover, shrubland, grassland, cropland, built-up, bare/sparse
vegetation, snow/ice, water, herbaceous wetland, mangroves, moss/lichen
(`ORIGINAL_WC_NAMES`, `src/ingest/worldcover.py`). Every model in this
project is trained on a collapsed **4-class** scheme instead — vegetation,
built_up, bare, water — via an explicit remap table:

```python
WC_TO_BUCKET_FROM = [WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP, WC_BUILTUP,
                      WC_BARE, WC_SNOWICE, WC_WATER, WC_WETLAND, WC_MANGROVE, WC_MOSSLICHEN]
WC_TO_BUCKET_TO   = [VEG,     VEG,       VEG,      VEG,     BUILTUP,
                      BARE,    0,        WATER,    VEG,      VEG,        VEG]
```

Look at what actually happens in that table: **7 of WorldCover's 11
original classes** — tree cover, shrubland, grassland, cropland,
herbaceous wetland, mangroves, and moss/lichen — all collapse into the
single `vegetation` bucket. Only `built_up`, `bare`, and `water` stay as
their own distinct classes, and `snow/ice` is dropped entirely (remapped to
the sentinel `0`, i.e. masked out — Singapore has none).

Three separate reasons drive this, not one:

**1. Thermal relevance, not ecological accuracy, is what this project
needs.** This is a project about urban *heat*, not a land-cover survey for
its own sake. The four buckets map onto four genuinely different
thermal behaviors that matter for a cooling-priority score:
- **Vegetation** cools via shading + evapotranspiration (a tree canopy and
  a grass verge do this differently in ecological terms, but similarly
  enough in *thermal* terms for this project's purposes).
- **Built-up** surfaces absorb heat during the day and re-radiate it
  (asphalt/concrete's high thermal mass is the core driver of the urban
  heat island effect this whole project exists to address).
- **Bare** surfaces (exposed soil, cleared/construction land) heat up
  strongly in daytime sun but have neither vegetation's cooling mechanism
  nor built-up's thermal mass — a genuinely distinct thermal category, not
  a leftover "other."
- **Water** cools via evaporation and has very high thermal inertia
  (slow to heat, slow to cool) — mechanically different from vegetation's
  cooling even though both reduce surface temperature.

  Whether a patch of vegetation is specifically tree canopy, shrubland,
  grassland, or cropland doesn't change which of these four thermal
  behaviors it exhibits — so collapsing them loses ecological detail the
  project was never trying to capture in the first place.

**2. Several of WorldCover's classes barely exist in Singapore at all.**
Snow/ice obviously never occurs here (hence dropped entirely, not just
merged). Cropland, wetland, mangroves, and moss/lichen are all minor,
patchy land covers in a small, dense, tropical city-state — keeping them as
separate classes would mean training a model on classes with only a
handful of real examples each, which tends to produce unstable,
low-confidence predictions for exactly those classes (a smaller-scale
version of the same class-imbalance problem this project's own U-Net
evaluation ran into with just 4 classes — see file 03/06's discussion of
`bare_f1 = 0.000`). Folding them into `vegetation` (where they belong
thermally, per point 1) avoids manufacturing several near-empty classes.

**3. A coarser scheme is what makes hand-labeling *reliable* between two
independent people.** The 300-point validation sample (and the smaller
Week-1 gate check before it) is labeled by hand, by two different project
members independently, and then checked for agreement (`G5`,
`validation/input_validation/labeling_agreement.py`, scored via Cohen's
kappa — the real run scored κ=0.676, "substantial agreement"). Asking two
people to consistently distinguish shrubland from grassland from cropland
from a satellite image, by eye, is a much harder and much less reliable
task than asking them to distinguish "vegetation" from "built-up" from
"bare" from "water" — visually and spectrally, those four are far more
separable than WorldCover's finer subdivisions of "green stuff." A
labeling scheme that two humans can't agree on isn't a usable ground truth
regardless of how ecologically precise it is, so labeling reliability was a
real constraint on how many classes this project could responsibly use, not
just a modeling-convenience shortcut. This is also why the 4-class scheme
is described in the code as the **"gate-review scheme"** — it's the exact
scheme the project's own dropdown-based labeling tool
(`app/pages/1_Label_Validation_Points.py`) uses, so what a human labels and
what a model is trained/scored against are always the identical set of
categories.

## 5. NA / nodata handling — the actual mechanics, not just "drop missing values"

This is where "filtering" gets more subtle than the season/cloud filters
above. Three distinct kinds of "invalid" exist in this pipeline, each
handled differently, and mixing them up is a real bug class the code
explicitly guards against:

**(a) No valid satellite observation at a pixel** — after cloud masking, if
*every* image in the filtered collection was cloudy at a given pixel, that
pixel has no data to composite from at all. GEE represents this as a masked
(not zero, not NaN) pixel natively. `build_feature_image()` captures this as
an explicit `valid_mask` band (`feature_image.select("B4").mask()`) rather
than letting it slip through as a numeric 0 — a real reflectance value of 0
and "no data here" must never be confused, because 0 is what a very dark
surface would legitimately read.

**(b) No land-cover label at a pixel** — WorldCover (the training-label
source) has its own classes that don't map onto this project's 4-bucket
scheme (e.g. snow/ice, which doesn't occur in Singapore). Unmapped and
outside-Singapore pixels get remapped to sentinel value `0`
(`get_worldcover_bucket_image`, `src/ingest/worldcover.py`), explicitly
masked out (`wc_bucket.updateMask(wc_bucket.neq(0))`) — and separately,
*also* masked wherever the satellite composite itself has no valid pixel
(passing `valid_mask` through from step (a)) — "a label never survives
where there's no feature data to pair it with." Without that second mask, a
patch could carry a real land-cover label sitting on top of a
satellite pixel with literally no spectral data, and a model would be
trained to predict a class from a feature vector that never actually
existed.

**(c) GEE's export-time float encoding gotcha** — this is a narrower,
format-level issue worth knowing about if you ever touch the patch export
code: GEE's TFRecord patch export encodes each band independently based on
its own pixel type, and if you mix a non-float band in among float bands,
that one band gets written as an opaque bytes blob instead of a flat
per-pixel float array — silently breaking the fixed-shape parsing the
training code expects. This is why `build_training_stack()`
(`src/landcover/unet_data.py`) explicitly `.toFloat()`s the label band
before export, even though it's conceptually an integer class id — a
dtype-consistency requirement of the export format, not a modeling choice.

Downstream (inside `unet_data.py::parse_training_patches`), this whole story
collapses into one number per pixel: `weight = (label_raw > 0)`. A pixel
with `weight = 0` still occupies space in the training array, but
contributes exactly zero to the loss — the model is never penalized or
rewarded for whatever it predicts there. This is the standard way to handle
a partially-valid grid of pixels without physically deleting pixels
(which would break the fixed patch-tensor shape the CNN needs).

## 6. Non-circularity: why validation points are physically cut out of the training region

This is the part of the pipeline most specific to *this* project, not a
generic ML step. The training labels ultimately come from WorldCover, a
global product — but WorldCover is explicitly **never used as the
validation answer key** (that's the 300 hand-labeled points, labeled
independently by project members). Still, if a model happened to train on
pixels sitting right on top of validation points, a good score on those
points would partly just be "the model saw this exact location during
training," not genuine generalization.

`build_training_region()` (`src/landcover/rf_baseline.py`) enforces this
directly rather than just asserting it: it buffers every validation point by
15m (`VALIDATION_EXCLUSION_BUFFER_M`, ~1.5x a 10m pixel, chosen to err
generous) and subtracts the union of those buffers from Singapore's
boundary before any training sample is drawn from it. The function prints
the excluded area and raises a warning flag if it looks suspiciously small
(a real bug that would otherwise fail silently — e.g. wrong CRS, an empty
validation set). This is what makes the project's "no circularity" claim
checkable by re-running the code, not just a comment asserting it's true.

## 7. Train/validation split — two different meanings in this pipeline

It's worth being precise here because "validation" means two different
things depending on which split you're looking at:

- **The 300 hand-labeled points** = the held-out *accuracy* ground truth,
  never trained on (enforced by §6 above). This is what
  `evaluate_landcover_classifiers.py` scores RF/U-Net/hybrid against.
- **U-Net's internal 85/15 train/val split** (`UNET_TRAIN_VAL_SPLIT = 0.85`)
  = a completely separate split of the *training* patches themselves, used
  only to decide when to early-stop and which epoch's weights to keep. It
  has nothing to do with the 300 labeled points — U-Net never sees the 300
  points' pixels during training at all (§6), so this internal split is
  purely about not overfitting to the training patches it *does* see.

## Summary table

| Filter | What it removes | Why |
|---|---|---|
| 4-class collapse (`WC_TO_BUCKET_*`) | WorldCover's 11 native classes → 4 (7 of them folded into `vegetation`, snow/ice dropped) | Thermal relevance over ecological detail, avoids near-empty classes, keeps hand-labeling reliable between 2 people |
| Season window (`DRY_SEASON_MONTHS`) | Images outside Apr/May/Oct/Nov | Avoid mixing wet/dry-season signal into one composite |
| Scene-level cloud cover (`LANDSAT_CLOUD_COVER_MAX`, scene metadata) | Whole images too cloudy to bother with | Cheap prefilter before per-pixel work |
| Pixel-level cloud mask (`QA_PIXEL` bits / `s2cloudless` probability) | Individual cloudy/shadowed/saturated pixels | Precise, per-source cloud handling |
| `valid_mask` (no satellite observation) | Pixels with zero clean observations after masking | Never confuse "no data" with a real 0 |
| WorldCover label masking (`wc_class != 0`, + `valid_mask`) | Pixels with no mappable class, or no paired feature data | A label must never outlive its features |
| Validation exclusion buffer (15m) | Training pixels near any of the 300 hand-labeled points | Non-circularity — the accuracy check must be genuine |
