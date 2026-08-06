# Tasks for you

Five things, roughly in the order that makes sense to tackle them (1 feeds
2, 2 feeds 3; 4 and 5 are more independent and can happen anytime). Each
section is self-contained — you shouldn't need to jump between docs to get
through any one of these.

## Before you start

### Play with the app first

Before touching any code or labels: `streamlit run app/Home.py`, click
through all 6 pages, get a real feel for what this pipeline already
produces end-to-end. Task 4 below has a page-by-page tour if you want a
guide, but do this first regardless of when you get to Task 4 properly —
it'll make everything else in this doc make more sense.

### See what's already been trained — MLflow

Every training run (RF, XGBoost, and every U-Net/CNN run, including all 3
U-Net retrains from this session) is logged to a local MLflow store —
worth a look before you start, so you're not training blind into what's
already been tried:

```
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Opens a browser UI (`http://127.0.0.1:5000`) — click into any run for its
logged hyperparameters and per-epoch loss/accuracy curves, or select
multiple runs to compare them side by side. This is also exactly the tool
for the "is this hyperparameter change real or just noise" question in
Task 3 — compare curves across runs rather than eyeballing a single final
number.

### Create a branch

Everything in this project so far has gone straight to `main` — for your
own work, branch off first (`git checkout -b <something-descriptive>`,
e.g. `300-point-relabel` or `teammate-fine-tuning`) rather than committing
directly to `main`. Push the branch and open a PR when you've got
something worth reviewing, rather than pushing straight to `main` as you
go — easier for both of us to review/discuss before it lands.

### Worth a literature check — full disclosure on how we got here

Be aware going in: the model choices in this pipeline (U-Net for
land-cover segmentation, the XGBoost+CNN split for the heat model, RF+U-Net
soft-voted for the ensemble) came out of AI-assisted sessions (Claude)
picking reasonable, standard defaults — not a from-scratch literature
review comparing alternatives. That's not necessarily wrong, but it means
nobody's actually checked whether these are the *best* choices for this
specific problem, just that they're defensible, commonly-used ones.

Worth spending some real time on before/alongside Task 3 (fine-tuning):
- **Land-cover segmentation from Sentinel-2**: is U-Net still what's
  typically recommended for a small dataset like this (~1000 training
  patches)? Are there simpler or more modern architectures worth knowing
  about, and how do they tend to compare on small remote-sensing datasets
  specifically (not just on huge benchmark datasets, which may not
  transfer to this data regime)?
- **Predicting land surface temperature / urban heat from remote sensing**:
  how do others typically frame this — patch-based CNN regression (what
  we're doing), a different downscaling approach, something else entirely?
- **Combining two classifiers' outputs (RF + U-Net)**: is a flat 50/50
  probability average (what `build_landcover_ensemble.py` does) standard
  practice, or is there a better-justified way to combine two models
  besides an unweighted average?

You don't need to land on a different architecture — even confirming
"yes, this is a defensible standard choice, here's why" is a legitimate
and useful outcome, and gives the report something more solid to point to
than "an AI suggested it." If you *do* find something worth trying
differently, that's exactly the kind of thing Task 3/5 are for.

---

## Task 1: Regenerate the validation sample at 300 points, then label it

### Step 0 — code change needed first (the sample size is currently hardcoded)

`TOTAL_POINTS = 200` is hardcoded in two places, and the number "200" is
baked directly into filenames and a GCS path across the codebase. Before
you can actually draw a 300-point sample, these need updating together
(the two `TOTAL_POINTS` constants, and every `..._200...` filename/prefix
renamed to `..._300...`):

- `scripts/generate_validation_sample.py`:
  - line `TOTAL_POINTS = 200` → `300`
  - line `csv_path = OUT_DIR / "validation_sample_200.csv"` → `"validation_sample_300.csv"`
  - line `export_labeling_table(labeling_df, OUT_DIR, prefix="validation_sample_200")` → `prefix="validation_sample_300"`
- `app/pages/1_Label_Validation_Points.py`:
  - line `TOTAL_POINTS = 200` → `300` (this file has its **own** separate copy of the constant, not shared with the script above)
  - `SB1_OUTPUT_CSV`, `WORK_CSV`, `FINAL_CSV`, `FINAL_GEOJSON` — each has `"validation_sample_200..."` in it, rename all four to `"validation_sample_300..."`
  - **Leave this one alone**: `attempts < TOTAL_POINTS * 200` — that `200` is an unrelated retry-budget multiplier (try up to 200× the target count of random draws before giving up), not the sample size. It'll naturally become `300 * 200` and that's correct.
- `validation/input_validation/labeling_sample.py`: `def export_labeling_table(..., prefix: str = "validation_sample_200")` → default `"validation_sample_300"`
- `config/settings.py`: `VALIDATION_SAMPLE_GCS_PREFIX = "training_inputs/validation_sample_200_labeled"` → `"training_inputs/validation_sample_300_labeled"`
- `scripts/evaluate_landcover_classifiers.py`, `scripts/train_landcover_rf.py`, `scripts/run_landcover_unet_inference.py`: each has `VALIDATION_CSV = INTERIM_DIR / "validation_sample" / "validation_sample_200_labeled.csv"` → change `200` to `300`
- `notebooks/colab_training/train_unet.ipynb`: the cell with `VALIDATION_CSV = Path("data/interim/validation_sample/validation_sample_200_labeled.csv")` and the GCS-push command comment above it — both say `200`, change to `300`. (It's a notebook JSON file — if your editor won't let you edit cell source directly, ask, there's a scripted way to do it.)
- `app/Home.py`: two status-list lines reference `validation_sample_200.csv` / `validation_sample_200_labeled.csv` — update both.
- `SETUP.md`: a few prose mentions of `validation_sample_200_labeled.csv` (§8a, §9c) — update for consistency, not functionally required.

Grep for `validation_sample_200\|TOTAL_POINTS` from the repo root before you start to confirm you've got everything — that's the exact search that produced this list.

### Step 1 — draw the new 300-point sample

```
.venv\Scripts\python.exe scripts\generate_validation_sample.py
```

This pulls a stratified sample from Earth Engine (stratified by WorldCover
class, so rare classes like "bare" get proportionally enough points to be
scoreable) and writes `data/interim/validation_sample/validation_sample_300.csv`
— unlabeled, just candidate points with lon/lat + a WorldCover hint. Watch
the printed "sampling-frame valid-pixel coverage" — should be ≥90%; if
it's lower something's off with the season window and it'll print a
warning.

### Step 2 — label all 300 points via the web page

```
streamlit run app/Home.py
```

Navigate to **Label Validation Points** in the sidebar. What you'll see:

- A satellite-imagery map on the left, colored markers for each point
  (orange = unlabeled, green = labeled, gray = flagged uncertain).
- Click a marker → it opens in a detail panel on the right, zoomed in
  close enough to actually judge what's at that single ~10m pixel.
- Pick one of 5 options: **Vegetation** (trees/shrubs/grass/crops/mangroves),
  **Built-up** (buildings/roads/pavement/construction), **Bare** (exposed
  soil/sand/cleared land), **Water** (pond/reservoir/drain edge — also use
  this for a mixed pixel or a point that looks misclassified), or
  **Uncertain — flag for review** if you genuinely can't tell.
- Also pick a **Confidence** level (Low/Medium/High) and optionally leave
  a **Notes** comment — both get saved, useful later if a label ever needs
  revisiting.
- Click **💾 Save label** — it autosaves your progress to a working CSV
  immediately and auto-advances to the next unlabeled point, so you don't
  need to click a marker again for every single point; just keep going.
- If you close the browser/lose your place: reopening this page reloads
  wherever you left off (the autosave file), nothing is lost.
- Use **⏭️ Jump to next unlabeled point** any time you want to skip around
  rather than go in strict order.
- **When all 300 are done**: click **📤 Export final CSV + GeoJSON** — this
  is a required final step, not automatic. It writes the actual file
  (`validation_sample_300_labeled.csv`) that training/evaluation scripts
  read; the autosave working file isn't the same thing and won't be picked
  up by anything downstream until you export.

**One real thing worth doing carefully**: the original 200-point labeling
was done fast for speed, and it's plausibly why some runs on the small
"bare" class swung around a lot in accuracy — take your time on genuinely
ambiguous pixels rather than guessing, and use "Uncertain" honestly when
you mean it rather than picking a class you're not sure of. This directly
feeds the ground truth every accuracy number in this project is measured
against.

### Step 3 — push the labeled sample to GCS

Training happens on Colab (Task 2), which can't see your local files —
push the finished labeled CSV once:

```
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, VALIDATION_SAMPLE_GCS_PREFIX; gcs.upload_file('data/interim/validation_sample/validation_sample_300_labeled.csv', GCS_MODEL_BUCKET, f'{VALIDATION_SAMPLE_GCS_PREFIX}.csv')"
```

---

## Task 2: Run the full training pipeline

Nothing here needs a GPU on your own machine — training happens on Google
Colab's free GPU, everything local is CPU-only.

### 2a. One-time: install the Colab extension

Install **"Google Colab"** from the VS Code Extensions panel. To use it:
open a notebook under `notebooks/colab_training/` → **Select Kernel**
(top-right) → **Colab** → sign in with your Google account → **New Colab
Server** → pick **GPU** (the free T4 tier is enough) → connect. The
notebook file itself stays local and git-tracked as normal — only the
*execution* happens on Colab's VM.

### 2b. Auth: paste your service-account key when prompted

Each training notebook's first real cell prompts a masked paste-in box
(via `getpass` — plain Jupyter stdin, not a browser login). When you run
that cell: open your `credentials/*.json` key in a text editor, copy the
whole contents, paste into the box that appears, press Enter. You'll do
this fresh every Colab session (the file lives on Colab's ephemeral VM
disk, not saved between sessions).

### 2c. Train RF first (local, no Colab needed, do this before U-Net/CNN)

```
.venv\Scripts\python.exe scripts\train_landcover_rf.py --with-probabilities
```

`--with-probabilities` is required — the later ensemble step needs RF's
per-class probability raster, not just its hard classified labels.
Trains server-side on Earth Engine (a couple of minutes), no GPU needed.
Verify: `data/processed/landcover/rf_landcover.tif` and
`rf_landcover_prob.tif` should exist afterward.

### 2d. Train U-Net (Colab)

Open `notebooks/colab_training/train_unet.ipynb`, connect to a Colab GPU
kernel (2a), run every cell top to bottom. It builds the GEE feature
stack, exports/caches training patches (first run for a given labeled
sample pays for a real multi-minute GEE export; later runs against the
same data just download the cache), trains, and pushes the model + a
training-run summary to GCS.

**Gotcha to know about**: the training cell is set to `force_retrain=True`
on purpose — if it were `False` and you reconnect to/reuse the same Colab
session from an earlier run, it would silently reload the cached model
instead of actually retraining, and the push afterward would silently
no-op too (identical weights → identical hash → skipped). Leave it as
`True` for a real retrain; only reason to ever flip it to `False` is if
you're just re-running cells to debug something downstream without
wanting to pay for a full retrain each time.

Then, back on your own machine:

```
python scripts/pull_models.py --model unet
python scripts/run_landcover_unet_inference.py
python scripts/build_landcover_ensemble.py
```

Downloads the trained weights (hash-verified), runs full-Singapore CPU
inference, rebuilds the ensemble raster. Push that ensemble raster to GCS
(the CNN notebook needs it, can't make it itself):

```
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, ENSEMBLE_RASTER_GCS_PREFIX; gcs.upload_file('data/processed/landcover/ensemble_landcover.tif', GCS_MODEL_BUCKET, f'{ENSEMBLE_RASTER_GCS_PREFIX}.tif')"
```

### 2e. Train the CNN heat model (Colab)

Same pattern: open `notebooks/colab_training/train_heat_cnn.ipynb`,
connect to Colab GPU, run top to bottom (needs 2d's ensemble raster to
already be pushed). Then locally:

```
python scripts/pull_models.py --model cnn
```

### 2f. Train XGBoost too (separate from everything above, needed for verification)

The heat model (S5) is XGBoost (subzone-level) + CNN (patch-level) working
together — 2c-2e above only covers the land-cover classifiers (RF/U-Net/
ensemble) that feed *into* the heat model, not the heat model itself.
XGBoost trains natively on Windows, no GPU/Colab needed:

```
.venv\Scripts\python.exe scripts\train_heat_model_xgboost.py
```

### 2g. Check your work — land-cover classifiers

```
python scripts/evaluate_landcover_classifiers.py
```

Prints formal accuracy/macro-F1/per-class F1 for RF, U-Net, and ensemble,
scored against whatever validation CSV currently exists (the new 300-point
one, once Task 1 is done). Expect numbers in the 75-80% accuracy range —
if something's wildly off (e.g. near-random ~25%), something upstream
broke; ask before assuming it's fine.

### 2h. Check your work — XGBoost and CNN specifically

This is the part most worth actually looking at closely, not just running
and glancing at the last line — the whole point of this model pair is
that they're supposed to **cross-check each other**, and we already found
one real case where they don't (see Task 4's ALJUNIED note).

```
python scripts/diagnose_heat_model.py
```

What to actually check in the output:
- **XGBoost's full-table RMSE** (printed near the top) — should be in the
  same ballpark as its own training run's held-out test RMSE (~1°C is
  what we've seen; this script's number is informal/in-sample-mixed, not
  the honest held-out figure, so don't over-read small differences).
- **The 3 demo subzones' cooling-direction agreement** (TUAS NORTH, GUL
  CIRCLE, CHIN BEE) — each should print "✅ same direction as XGBoost".
  If any flip to "⚠️ OPPOSITE direction," that's a real disagreement worth
  flagging, not something to shrug off.
- **CNN's magnitude vs. XGBoost's** — CNN's local ΔLST should generally be
  *larger* in magnitude than XGBoost's subzone-averaged ΔLST (a concentrated
  local edit vs. a diffuse subzone-wide change) — this has held in every
  run so far; if it ever inverts, worth a closer look.

If you're curious beyond the 3 hardcoded subzones, try a live query on
others via `scripts/run_counterfactual.py --subzone-id "<name>"
--delta-vegetation 0.2 --lon <x> --lat <y> --radius-m 50` (or just use the
Counterfactual Greening page in the app, Task 4) — that's actually how the
ALJUNIED disagreement got found in the first place, by poking at subzones
outside the 3 that `diagnose_heat_model.py` hardcodes.

**Browsing tracked runs**: every `pull_models.py` call imports Colab run
summaries into a local MLflow store alongside RF/XGBoost's own runs:
```
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
opens a browser UI (`http://127.0.0.1:5000`) for comparing runs' loss/
accuracy curves without re-reading printed epoch logs.

---

## Task 3: Fine-tuning

### Where the knobs are

`config/settings.py`, under "RF / U-Net hyperparameters" — these are
**shared** between U-Net and the CNN heat model (the CNN's training code
imports the same `UNET_*` constants, doesn't have its own copies):

```python
UNET_PATCH_SIZE = 128           # don't change this one -- tied to how patches were exported
UNET_BATCH_SIZE = 8
UNET_EPOCHS = 30
UNET_LEARNING_RATE = 1e-3
UNET_BASE_FILTERS = 32          # controls model capacity/size
UNET_EARLY_STOP_PATIENCE = 5
UNET_TRAIN_VAL_SPLIT = 0.85
```

To try a change: edit the value here, commit + push to GitHub (the Colab
notebook `git clone`/`git pull`s the repo fresh each session, so it only
sees what's actually pushed), then run the training notebook as in Task 2.

### The one thing you need to know before tuning anything

**Retraining with *identical* settings doesn't give identical results.**
We measured this directly: 3 back-to-back U-Net retrains, same
hyperparameters, same data, came back at 79.3%, 77.8%, and 76.3%
accuracy — a ~3-point spread from GPU non-determinism alone (weight init,
exact early-stopping epoch). That means **a single before/after comparison
can't tell you if a hyperparameter change actually helped** — the
difference could easily just be noise of the same size. If you want to
trust a "this made it better" conclusion, retrain 2-3 times at each
setting and compare the ranges, not single numbers.

### What's actually worth trying, and what to optimize for

Don't just chase overall accuracy — we found RF is consistently the
weakest on accuracy across every run, but **U-Net is consistently the
weakest on macro-F1** (it does fine on the big classes, vegetation/
built-up, but struggles on "bare," the smallest/most ambiguous class —
its `bare_f1` swung between 0.13 and 0.21 across 3 runs). If you're
tuning U-Net specifically, macro-F1 (or `bare_f1` directly, printed by
`evaluate_landcover_classifiers.py`) is the more informative number to
watch than accuracy alone.

Reasonable things to try, roughly in order of how likely they are to
matter:
- **`UNET_LEARNING_RATE`**: currently `1e-3`. Try `5e-4` (slower, maybe
  more stable convergence) or `2e-3` (faster, riskier).
- **`UNET_BASE_FILTERS`**: currently `32` (controls model width/capacity).
  Try `64` for more capacity — will roughly 4x memory/compute per layer
  and slow training down, GPU permitting.
- **`UNET_EARLY_STOP_PATIENCE`**: currently `5`. If training logs show
  val_loss still improving right up to when it stops, try `8-10` to give
  it more room.
- **`UNET_EPOCHS`**: currently `30` (a ceiling, not a target — early
  stopping usually triggers well before this). Only worth raising if
  patience increases and it's still not stopping.

Whatever you try, save the model/rasters/evaluation CSVs somewhere before
overwriting them with the next attempt (there's already a precedent for
this: `models/_backup_unet_run1/`, `_run2/`, `_run3/`, each with a
`SUMMARY.txt` of that run's numbers) so you can actually compare attempts
side by side instead of only ever seeing the latest one.

---

## Task 4: UI

```
streamlit run app/Home.py
```

Six pages, in sidebar order:
- **Home** — pipeline status checklist (what's built vs. not, on disk).
- **Label Validation Points** — the labeling tool from Task 1.
- **Island Map** — folium choropleth of the cooling-priority score,
  click a subzone to select it.
- **Subzone Breakdown** — per-subzone pillar detail, confidence bands,
  land-cover fractions, for whichever subzone is selected.
- **Counterfactual Greening** — "what if this area were greener?" — a
  live XGBoost subzone-level slider plus a live CNN patch-level slider
  (both call real trained models directly, no precomputed examples).
- **Validation Dashboard** — every validated output already on disk in
  one place (Week-1 gates, classifier evaluation, hotspot cluster
  quality, S5/S6 diagnostics, LST cross-checks).

Open-ended: poke around, fix/polish whatever looks rough. One concrete
starting point if you want one — on the Counterfactual Greening page, try
subzone **ALJUNIED**: XGBoost predicts *warming* (+0.42°C) for the same
"+0.15 vegetation fraction" scenario where CNN predicts *cooling*
(−2.344°C) — opposite signs, a real disagreement between the two models
that hasn't been investigated yet. Worth a look either as a UI thing (is
something displayed misleadingly?) or a modeling thing (why do they
actually disagree here?) — genuinely don't know which yet.

---

## Task 5: What else needs improving — especially evaluation

Open question, genuinely want your independent take rather than just
mine — spend some time with the pipeline/results (Tasks 1-4 will already
have gotten you deep into most of it) and think about what's weakest,
particularly on the **evaluation** side specifically, since that's where
the project is most exposed if something's wrong: a model choice we can't
actually distinguish from noise is a much worse problem to find out about
late than a slow script.

Known, real, currently-unresolved gaps to prime your thinking (not an
exhaustive list, don't feel limited to these):

- **The land-cover ensemble consistently underperforms the original
  TensorFlow version by ~2-3.5 accuracy points**, reproducibly across 3
  retrains — individual U-Net accuracy is fine, so it looks like a
  probability-calibration difference specifically affecting the RF+U-Net
  soft-vote average. Nobody's confirmed the actual cause yet.
- **`validation/input_validation/labeling_agreement.py`** (the script
  behind the G5 labeling-agreement kappa number) has no persisted per-run
  output — the Validation Dashboard shows a static, hand-copied number
  rather than something it actually recomputes live. With 300 fresh
  points to work with (Task 1), this might be worth properly wiring up.
- **`diagnose_heat_model.py` only ever checks 3 hardcoded subzones**
  (TUAS NORTH, GUL CIRCLE, CHIN BEE) for the XGBoost/CNN cross-check —
  ALJUNIED disagreeing was found by accident, not by systematic checking.
  Is there a case for scoring the cross-check across *all* subzones
  instead of 3 cherry-picked ones, to know how often they actually agree?
- More generally: are there other places in the pipeline where a single
  number gets reported without anyone checking whether it's stable
  across reruns, the way we found U-Net's accuracy swings ~3 points just
  from retraining with identical settings?

Write up whatever you find/think — doesn't need to be fixed immediately,
a clear-eyed list of "here's what I don't fully trust yet and why" is
valuable on its own.
