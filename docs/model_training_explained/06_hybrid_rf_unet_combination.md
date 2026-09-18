# The RF + U-Net hybrid combination

Code: [`src/landcover/hybrid.py`](../../src/landcover/hybrid.py).

## This isn't a third trained model

Worth saying explicitly, since it's easy to assume otherwise: there is no
separate training step here. RF and U-Net are each trained completely
independently (files 02 and 03) — the hybrid is a **post-hoc, inference-time
combination** of their two already-trained outputs. Nothing in this file
involves gradient descent, GEE's classifier training, or any learned
parameters at all.

## Why "hybrid," not "ensemble" — and why that distinction is functionally real here

RF is a tree-based classifier; U-Net is a convolutional neural network —
two structurally different model types, combined together. Per the
definition used for this project's rubric (ensemble = multiple instances of
the *same* model type; hybrid = different model types working together),
that makes this a hybrid, and the project's own original proposal already
called it "Hybrid ML" before this codebase's internal naming caught up to
that (see the earlier renaming work in this project).

## How the combination actually works: soft voting

1. Both RF and U-Net are queried in probability mode — a 4-band image where
   each band is "probability this pixel is class X," not a single hard
   label (see file 02's `classify_probability()` and file 03's softmax
   head).
2. Both probability rasters are reprojected onto a **common grid** —
   specifically, U-Net's grid is reprojected onto RF's (`load_prob_raster`),
   because RF's raster is exported directly from Singapore's real boundary
   geometry, while U-Net's grid is an artifact of GEE's rectangular
   patch-export mechanism. Using RF's grid as the reference keeps the
   hybrid raster on the same footprint convention as the RF raster that
   already existed in the repo.
3. The two probability rasters are **averaged** per class, per pixel
   (`average_probabilities`) — a pixel is only counted if it's valid in
   **both** sources (`combined_valid = rf_valid AND unet_valid`); the mean
   is taken over exactly those two numbers, nothing more sophisticated
   (not a learned weighting, not a majority vote of hard labels).
4. The averaged probabilities are argmaxed back into a single hard label
   per pixel (`probabilities_to_hard_labels`) — bucket id = whichever class
   has the highest averaged probability, `+1` because bucket ids are
   1-indexed.

## Why averaging, and why it's worth the (small) accuracy cost

RF and U-Net don't fail on the same pixels. RF, judging each pixel in
isolation, tends to do reasonably on every class including rare ones (its
own `bare_f1 = 0.368` in the last real evaluation). U-Net, using spatial
context, was noticeably better on the majority classes but — in the last
real training run — never predicted "bare" at all (`bare_f1 = 0.000`, a
complete class collapse; see the "why is ensemble weaker" discussion earlier
in this project's history for the concrete confusion-matrix evidence).
Averaging lets U-Net's stronger majority-class calls dominate where RF is
weaker, while still letting RF's genuine (if imperfect) "bare" signal pull
through where U-Net has literally nothing to contribute — at the real,
measured cost of occasionally overriding a handful of points U-Net alone
would've gotten right (a −0.7 percentage-point raw-accuracy difference in
the last evaluation, within statistical noise for a 296-point validation
set), in exchange for a much larger macro-F1 gain (0.547 → 0.652) from no
longer being structurally blind to an entire class.

## A subtlety worth knowing: NaN vs. 0.0 as "invalid," inconsistently, by source

`_to_valid_bool()` exists because RF's and U-Net's exported rasters don't
agree on how they represent "outside the mask": GEE's own export pipeline
writes masked/clipped pixels as `NaN` in RF's float32 probability bands,
while U-Net's locally-written raster (produced by this repo's own rasterio
code, not GEE's export path) uses a clean `0.0` instead. Treating `NaN` as
"valid" would be a real, silent bug — `.astype(bool)` on a raw NaN bit
pattern evaluates as truthy in numpy, which (confirmed empirically during
development) inflated RF's apparent valid area by roughly 58% before this
was caught and fixed by routing every valid-mask read through
`np.nan_to_num(..., nan=0.0)` first, so both sources' "invalid" pixels are
treated identically regardless of which convention they were written with.
