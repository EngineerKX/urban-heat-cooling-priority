# CoolPriority SG

An AI-driven cooling-priority ranking for Singapore's URA subzones
(ISY5002 Pattern Recognition Systems practice module, Group 28). It combines
Landsat surface temperature, Sentinel-2 imagery, a U-Net land-cover map and
SingStat census data into one priority score per subzone, gives each rank a
90% confidence band, and lets planners test what added greenery might do to
surface temperature. A Streamlit app presents the map, per-subzone
breakdowns, the greening what-if tool and the validation results.

## Download the full project (code + datasets)

The datasets and trained models are too large for git (about 7 GB), so the
complete project, including every dataset the app needs, is shared as a
single zip:

**[Download the full project (zip)](PASTE_GOOGLE_DRIVE_LINK_HERE)**

The zip contains this repository plus the folders git doesn't track:
`data/` (raw inputs, processed layers, scores, confidence bands, clusters),
`models/` (trained U-Net, CNN and XGBoost) and `mlruns/` + `mlflow.db`
(experiment history). It does **not** contain any credentials.

## Quick start: run the app from the zip

You don't need a Google Earth Engine or Google Cloud account for this —
the app reads only the files in the zip.

1. **Install Python 3.11 or newer** (we used 3.13).
2. **Unzip to a short path**, e.g. `C:\coolpriority`. Windows can fail on
   very long file paths, and some files sit several folders deep.
3. **Create a fresh virtual environment and install the packages.** Open a
   terminal in the unzipped folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS/Linux
   pip install -r requirements.txt
   ```
   (PowerShell blocking activation? See the gotcha in §2b below.)
4. **Run the app:**
   ```
   python -m streamlit run app/Home.py
   ```
   It opens at `http://localhost:8501`. Stop it with **Ctrl+C** in the
   terminal.

Where to start: the **Overview** page summarises the findings; **Island
Map**, **Subzone Breakdown** and **Counterfactual Greening** are the
planner tools; **Validation Dashboard** holds the evaluation results. The
first visit to Counterfactual Greening takes about a minute while the CNN
loads its image patches.

To *re-run parts of the pipeline* from the zip (not needed just to use the
app), you'll need your own Earth Engine credentials — see
`TEAMMATE_SETUP.md`, which covers working from a full folder copy.

---

# Full setup: rebuilding the pipeline from scratch

Everything below is only needed if you start from a plain `git clone`
(without the zip) or want to regenerate the datasets yourself. Follow it
top to bottom.

> **Reproducibility note.** Following this guide rebuilds the full pipeline
> and gives results that are *consistent with*, but not *identical to*, the
> numbers in our final report. Seeds are fixed (`RANDOM_SEED = 42`), the
> satellite date window is fixed (to `2026-07-01`) and the hand-labelled
> validation sample is in git, but:
> - the U-Net and CNN train on Colab GPUs, which are not bit-for-bit
>   deterministic, so retrained weights — and the land-cover map, greenery
>   pillar and ranks built from them — differ slightly between runs;
> - a few packages in `requirements.txt` (`tensorflow`, `mlflow`, `folium`,
>   `matplotlib`, `streamlit-folium`) are unpinned, so a later install may
>   pick up newer versions;
> - Earth Engine collections (Landsat, Sentinel-2, Dynamic World) are
>   occasionally reprocessed by their providers.
>
> Expect metrics such as land-cover accuracy to move by a point or so and a
> few top-20 subzones to swap. Our evaluation shows measurement noise alone
> swaps about 11 of the top 20, so differences of this size do not change
> the project's conclusions.

## 1. Prerequisites

- Python 3.11+
- Git

## 2. Clone & Python environment

First, clone the repo and move into it:
```
git clone <repo-url>
cd urban-heat-cooling-priority
```

### 2a. Create a virtual environment

A virtual environment ("venv") is a private, isolated copy of Python just
for this project — packages installed into it don't affect any other
project or your system-wide Python. **Note**: `.venv/` is gitignored, so
cloning the repo does *not* give you one — everyone creates their own
locally, once, on their own machine.

```
python -m venv .venv
```

This creates a `.venv/` folder in the repo root. Nothing runs yet — this
just creates the environment, it doesn't activate it.

### 2b. Activate it

You need to activate the venv in **every new terminal window/session**
before running any script in this repo — otherwise Python falls back to
your system-wide install, which won't have the packages this project
needs.

```
.venv\Scripts\activate         # Windows, PowerShell or cmd
source .venv/bin/activate      # macOS/Linux, or Git Bash on Windows
```

**How to tell it worked**: your terminal prompt should now start with
`(.venv)`, e.g. `(.venv) PS C:\...\urban-heat-cooling-priority>`.

**Windows PowerShell gotcha**: if activation fails with something like
*"cannot be loaded because running scripts is disabled on this system"*,
PowerShell's execution policy is blocking it. Fix (one-time, per user, run
as your normal user — not admin):
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Then retry activation. Alternatively, just use Git Bash or `cmd.exe`
instead of PowerShell, which don't have this restriction.

**To deactivate later** (switch back to system Python), just run:
```
deactivate
```

### 2c. Install dependencies

`requirements.txt` lists every third-party Python package this project
needs, pinned to specific versions so everyone's environment matches.
With the venv active (from 2b — check for that `(.venv)` prefix first):

```
pip install -r requirements.txt
```

This reads the file and installs everything listed into `.venv/` — takes
a few minutes the first time (some packages, like `pandas`/`scikit-learn`,
are large). You only need to re-run this when `requirements.txt` itself
changes (e.g. after pulling a teammate's commit that added a package).

**Verify it worked**:
```
python -c "import ee, pandas, streamlit; print('OK')"
```
Should print `OK` with no errors. If you get `ModuleNotFoundError`, double
check the `(.venv)` prefix is showing in your prompt — the install
probably went to the wrong Python.

## 3. Google Earth Engine credentials

This project authenticates to Earth Engine via a **GCP service account**
(not an interactive browser login), so scripts can run unattended. Every
script needs this working before it can do anything.

### If you're joining an existing team project (most common case)

Ask a teammate who already has access to add you first:
- They go to **IAM & Admin → IAM → + GRANT ACCESS**, enter your Google
  account email, role **Editor**, Save.

Once you have access:
1. Go to `https://console.cloud.google.com/iam-admin/serviceaccounts?project=<PROJECT_ID>`
   (ask your teammate for the project ID, e.g. `nus-iss-urban-heat-sg`)
2. **+ CREATE SERVICE ACCOUNT** → give it a name (e.g. `<yourname>-urban-heat`)
3. **Create and continue** → in the role picker, add **both** of these
   roles (click **+ ADD ANOTHER ROLE** for the second one) — both are
   required, Earth Engine will reject requests missing either:
   - **Earth Engine Resource Writer**
   - **Service Usage Consumer**
4. **Continue** → **Done**
5. Click into the service account you just created → **Keys** tab →
   **Add Key → Create new key → JSON → Create**. This downloads a
   `.json` file — treat it like a password, never commit it or share it
   outside a private channel.
6. Move that file into a `credentials/` folder at the repo root (create
   the folder if it doesn't exist — it's already gitignored, so anything
   placed there stays local to your machine).

### If you're starting a brand new GCP project from scratch

1. [console.cloud.google.com](https://console.cloud.google.com) → **New
   Project** → note the Project ID (lowercase-with-hyphens)
2. [code.earthengine.google.com/register](https://code.earthengine.google.com/register)
   → register that project (noncommercial/academic option for a student
   project)
3. Then follow the "create a service account" steps above (step 3
   onward) under your own new project

## 4. Configure `.env`

Copy the template and fill it in:
```
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

```
GEE_SERVICE_ACCOUNT=<service-account-email>@<project-id>.iam.gserviceaccount.com
GEE_PRIVATE_KEY_PATH=<full local path to the .json key you downloaded>
GEE_PROJECT_ID=<project-id>
GEE_EXPORT_BUCKET=
GCS_MODEL_BUCKET=
```

`GEE_EXPORT_BUCKET` can stay blank for now — it's only needed later for
U-Net training / exporting the Random Forest land-cover raster (see
§7 below). Everything else requires the first three.

`GCS_MODEL_BUCKET` can also stay blank — it's for trained-model sync
(`scripts/pull_models.py`/`push_models.py`, §9 below) and Colab
training-run summaries. Leaving it unset makes it default to reusing
whatever you set `GEE_EXPORT_BUCKET` to, which is fine at this project's
scale — only set it separately if you specifically want model artifacts
on a different bucket/lifecycle policy than raw GEE exports.

**Never commit `.env`** — it's already gitignored, but always worth a
second glance before pushing.

## 5. Verify Earth Engine auth works

```
.venv\Scripts\python.exe -c "from src.ingest.gee import init_ee; init_ee()"
```

Expect: `EE initialized OK (service account), project: ...`

**Common error**: `USER_PROJECT_DENIED` / "Caller does not have required
permission to use project ...". This means the service account is
missing the **Service Usage Consumer** role (step 3 above). Fix: IAM &
Admin → IAM → edit your service account's row → **+ Add another role** →
**Service Usage Consumer** → Save. Wait 1-2 minutes for the permission to
propagate, then retry.

## 6. Run the Week-1 gates (full pipeline smoke test)

```
.venv\Scripts\python.exe scripts\run_week1_gates.py
```

This pulls a real Earth Engine composite, checks the NEA weather-station
API, runs an end-to-end smoke test (tile download → downscaling → toy
land cover → subzone join → dummy score), and a downscaling sanity check.
All five gates should PASS if steps 1-5 above are correct. G5 (labeling
agreement) scores the real labels from the team's joint-labeling session
(kappa 0.676).

## 7. Optional: Cloud Storage bucket

Needed once you get to the Random Forest raster export in
`scripts/train_landcover_rf.py` (§8 below), or anything U-Net/CNN-related
(patch export, training, model sync — §9 below). Create a bucket in the same
GCP project (Cloud Storage → Buckets → Create), then set `GEE_EXPORT_BUCKET`
in `.env` to its name. The same bucket doubles as the model-artifact sync
target (`GCS_MODEL_BUCKET`, defaults to reusing this one — see `.env.example`).

## 8. Land-cover comparison model: Random Forest

The production land-cover model is the **U-Net** (§9,
`LANDCOVER_PRODUCTION_MODEL = "unet"` in `config/settings.py`), trained on
**Dynamic World** labels (`LANDCOVER_LABEL_SOURCE`). The Random Forest and
the RF + U-Net hybrid are kept as comparison models: the Validation
Dashboard, `evaluate_landcover_classifiers.py` and
`build_landcover_swap_comparison.py` (§10) all need them. RF is the
simplest place to start — no GPU, no Colab, and it trains server-side on
Earth Engine in a couple of minutes.

### 8a. Prerequisite: the hand-labeled validation sample

Both RF and U-Net need `data/interim/validation_sample/validation_sample_300_labeled.csv`
— it defines both the accuracy scoring ground truth *and* the spatial
exclusion zone (so training never sees the points it'll later be scored
against). **This file is tracked in git, so you already have it after
cloning** — no labeling needed.

Only if you deliberately want to redo the labeling:

1. Regenerate the (unlabeled) stratified sample (`--force` overwrites the
   existing one):
   ```
   .venv\Scripts\python.exe scripts\generate_validation_sample.py --force
   ```
2. Label it by hand via the Streamlit labeling page:
   ```
   streamlit run app/Home.py
   ```
   then open **Team tools → Label Validation Points** in the sidebar and
   work through all 300 points, picking one of the 4 bucket classes
   (vegetation / built_up / bare / water) for each.

Relabeling changes the scoring ground truth, so every land-cover accuracy
figure will then differ from the report's.

### 8b. Train RF and classify all of Singapore

```
.venv\Scripts\python.exe scripts\train_landcover_rf.py --with-probabilities
```

`--with-probabilities` is required here specifically because the
hybrid comparison model (§9d) needs RF's per-class probability raster,
not just its hard classified labels — without this flag you'd only get
the classified raster and the hybrid build would fail with a
missing-file error later. This step needs `GEE_EXPORT_BUCKET` set (§7) since it
exports the classified/probability rasters via Cloud Storage. Takes a
few minutes — training happens server-side on Earth Engine (`ee.Classifier`),
not on your machine, so no GPU or heavy local compute either.

Other useful flags: `--no-asset-cache` forces retraining instead of
reusing a cached GEE classifier asset from a previous run.

**Verify it worked**: you should see
`data/processed/landcover/rf_landcover.tif` (classified) and
`rf_landcover_prob.tif` (per-class probabilities) afterward.

## 9. Training deep-learning models (Colab)

U-Net (land-cover) and the CNN heat model both train exclusively on Google
Colab's free GPU — **never locally**, no GPU or WSL2 needed on your machine
at all. Local machines only ever run inference on the trained weights
(CPU is plenty).

### 9a. Install the Google Colab extension in VS Code

Google publishes an official extension that connects a local `.ipynb` file
directly to a real Colab-hosted GPU kernel. Install "Google Colab" from the
VS Code Extensions panel, then to use it: open a notebook under
`notebooks/colab_training/` → **Select Kernel** (top-right) → **Colab** →
sign in with your Google account → **New Colab Server** → pick **GPU**
(the free T4 tier is enough) → connect. The notebook file itself stays
local and git-tracked as normal — only the *execution* happens on Colab's
VM.

### 9b. Auth: paste your service-account key when prompted

Each training notebook's auth cell prompts a masked paste-in box (via
`getpass`, standard Jupyter stdin — not an interactive Google login,
matching this project's service-account-only convention everywhere else).
When you run that cell: open your `credentials/*.json` key from §3 in a
text editor, copy the whole contents, paste into the box that appears,
press Enter. No setup needed in advance.

Two other approaches were tried and don't work through VS Code's
remote-kernel connection to a Colab runtime — `google.colab.files.upload()`
hangs indefinitely (it depends on a JS/widget bridge specific to Colab's
own web frontend, which VS Code's notebook renderer doesn't implement),
and the Colab extension's Secrets UI wasn't reliably locatable. `getpass`
only depends on the standard Jupyter stdin protocol, so it works reliably
regardless of frontend — confirmed via a real successful training run,
2026-08-05.

Note you'll paste this fresh **every Colab session** (the file lives on
the ephemeral VM disk, not saved between sessions) — a small repeated
step, but a reliable one.

### 9c. Push the training inputs Colab can't regenerate itself

A couple of local build outputs need pushing to GCS once (and again after
you relabel/rebuild them) — Colab can't produce these on its own:

```
# the hand-labeled validation sample (needed by train_unet.ipynb)
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, VALIDATION_SAMPLE_GCS_PREFIX; gcs.upload_file('data/interim/validation_sample/validation_sample_300_labeled.csv', GCS_MODEL_BUCKET, f'{VALIDATION_SAMPLE_GCS_PREFIX}.csv')"

# the production land-cover map (needed by train_heat_cnn.ipynb) -- run
# this after U-Net inference, see §9d
python scripts/push_landcover_raster.py
```

`push_landcover_raster.py` uploads whichever map `LANDCOVER_PRODUCTION_MODEL`
points at (the U-Net's) with a hash and a note of the model and label
source that produced it. The CNN notebook checks both and refuses to train
on a stale or mismatched map. (The older hybrid-raster upload, under
`HYBRID_RASTER_GCS_PREFIX`, is no longer used by the notebook.)

### 9d. Run the notebooks, then pull the results locally

Open and run `notebooks/colab_training/train_unet.ipynb` top to bottom
(GPU-connected, per 9a). It exports/caches training patches, trains, and
pushes the model + a training-run summary to GCS. Then, on your own
machine:

```
python scripts/pull_models.py --model unet
python scripts/run_landcover_unet_inference.py --with-probabilities
python scripts/push_landcover_raster.py
python scripts/build_landcover_hybrid.py
```

This downloads the trained weights (hash-verified), imports the run into
your local MLflow store, runs full-Singapore CPU inference to produce the
production land-cover map, pushes that map to GCS for the CNN notebook
(§9c), and builds the hybrid comparison map. (`--with-probabilities` keeps
U-Net's per-class probabilities, which the hybrid needs.) Then run
`notebooks/colab_training/train_heat_cnn.ipynb` the same way, and locally:

```
python scripts/pull_models.py --model cnn
```

Re-running any notebook later (a hyperparameter change, a relabel) is
cheap — the GCS-cached exports mean only the first run per fingerprint pays
for a real GEE export; everything after that is a fast download.

**Gotcha: retraining in the same Colab session silently no-ops.** Both
`train_unet()` and `train_cnn_regressor()` default to `force_retrain=False`
— if a cached model already exists (matching training-data fingerprint)
at `models/unet_landcover.keras` / `models/heat_cnn.keras` *on that Colab
VM's own disk*, the training cell just reloads it instead of actually
retraining. This is harmless on a genuinely fresh VM (nothing cached
yet), but if you **reconnect to or stay on the same Colab session** you
used for an earlier run and want to force a real new training pass
(e.g. to compare against a previous result, or because you changed a
hyperparameter in `config/settings.py` that isn't part of the
fingerprint), you must explicitly edit the training cell to pass
`force_retrain=True` before re-running it — otherwise `push_models.py`
will also silently no-op afterward (same weights → same hash → "already
matches, skipping"), and you'll be looking at last run's numbers without
realizing nothing new actually happened. Safest sign something's wrong:
if GCS's `models/unet_landcover.keras` timestamp doesn't change after a
"finished" training run, this is almost certainly why — check with:
```
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET; c = gcs.get_client(); print([b.updated for b in c.list_blobs(GCS_MODEL_BUCKET, prefix='models/unet_landcover.keras')])"
```

**Browsing tracked training runs**: every `pull_models.py` call imports
any new Colab run summaries into your local MLflow store alongside
RF/XGBoost's own tracked runs. To browse them:
```
.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
then open the URL it prints (defaults to `http://127.0.0.1:5000`) in a
browser. Useful for comparing loss/accuracy curves across multiple
retrains without manually re-reading printed epoch logs.

## 10. Build the analysis outputs

With the production land-cover map in place (§9d), build everything the
app reads. Run in this order — later steps read earlier steps' outputs:

```
.venv\Scripts\python.exe scripts\build_heat_variants.py
.venv\Scripts\python.exe scripts\build_sensitivity_pillar.py
.venv\Scripts\python.exe scripts\build_priority_score.py
.venv\Scripts\python.exe scripts\build_modis_heldout.py
.venv\Scripts\python.exe scripts\build_priority_score_confidence_bands.py
.venv\Scripts\python.exe scripts\build_hotspot_clusters.py
.venv\Scripts\python.exe scripts\train_heat_model_xgboost.py
```

| Script | Builds | Used by |
|---|---|---|
| `build_heat_variants.py` | 30 m, bicubic 10 m and regression 10 m heat layers, averaged per subzone | heat pillar |
| `build_sensitivity_pillar.py` | resident density and share aged 65+ per subzone (SingStat) | sensitivity pillar |
| `build_priority_score.py` | PCA-weighted priority score (residents and all-places views) | Island Map, Subzone Breakdown |
| `build_modis_heldout.py` | Landsat-vs-MODIS heat error, the noise model for the bands | confidence bands |
| `build_priority_score_confidence_bands.py` | 90% bands and chance of staying in the top 20 | Subzone Breakdown, Validation Dashboard |
| `build_hotspot_clusters.py` | K-means heat typology | Subzone Breakdown, Validation Dashboard |
| `train_heat_model_xgboost.py` | subzone heat model (CPU only) | Counterfactual Greening |

Most build scripts skip work if their output already exists; pass
`--force` to rebuild (`--force-retrain` for the XGBoost script).

For the Validation Dashboard's land-cover tabs, also run (needs the RF,
U-Net and hybrid maps from §8–9):

```
.venv\Scripts\python.exe scripts\evaluate_landcover_classifiers.py
.venv\Scripts\python.exe scripts\build_landcover_swap_comparison.py
```

## 11. Run the app

```
.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Opens in your browser (defaults to `http://localhost:8501`); stop it with
**Ctrl+C** in that terminal. Any page whose inputs are missing shows a
message naming the script to run, rather than crashing.

The sidebar is grouped by audience:
- **Overview** (landing page) — key findings and the top 5 subzones.
- **Explore** — **Island Map** (ranked choropleth with adjustable pillar
  weights), **Subzone Breakdown** (one subzone's pillars, score
  contributions and uncertainty band), **Counterfactual Greening**
  (XGBoost subzone what-if plus the CNN street-block before/after map;
  the CNN half needs `pull_models.py --model cnn`, §9d).
- **Evidence** — **Validation Dashboard**, six tabs: summary, priority
  score, land cover, heat model, hotspot types, inputs.
- **Team tools** — **Label Validation Points** (the labeling tool, §8a).

The first visit to Counterfactual Greening takes about a minute while the
CNN loads its image patches; later visits are fast.

## 12. Optional: regenerate the evaluation results

The evaluation and ablation results in the final report come from:

```
.venv\Scripts\python.exe scripts\evaluate_downscaling_skill.py
.venv\Scripts\python.exe scripts\evaluate_landcover_supplementary.py
.venv\Scripts\python.exe scripts\evaluate_hotspot_typology.py
.venv\Scripts\python.exe scripts\evaluate_priority_score.py
.venv\Scripts\python.exe scripts\evaluate_scenario_models.py
```

Run them after §10. `evaluate_scenario_models.py --skip-cnn` runs only the
XGBoost part if you don't have the CNN patch data. As the note at the top
explains, expect values close to, but not identical to, the report's.

## Notes

- `data/`, `models/`, `credentials/`, and `.env` are gitignored. The one
  exception is the hand-labeled validation sample under
  `data/interim/validation_sample/`, which is tracked so everyone scores
  against the same points. Everything else under `data/` and `models/` is
  rebuilt locally by the steps above, and each teammate needs their own
  credentials.
- The original Colab notebooks are kept under
  `notebooks/Colab Notebooks/` for reference; the maintained pipeline now
  lives in `src/`, `validation/`, `scripts/`, and `app/`.
- `config/settings.py` is the single source of truth for pipeline
  constants (AOI, season window, seeds, etc.) — check there before
  assuming a value is hardcoded somewhere else.
