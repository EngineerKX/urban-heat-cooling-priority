# Setup guide

One-time setup for a new machine/teammate after cloning this repo. Follow
top to bottom.

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
```

`GEE_EXPORT_BUCKET` can stay blank for now — it's only needed later for
U-Net training / exporting the Random Forest land-cover raster (see
§7 below). Everything else requires the first three.

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
All five gates should PASS if steps 1-5 above are correct — G5 (labeling
agreement) uses placeholder example labels until you and your teammate do
the real joint-labeling exercise, so ignore that particular result until
then.

## 7. Optional: Cloud Storage bucket

Needed once you get to the Random Forest raster export in
`scripts/train_landcover_rf.py`, or anything U-Net/CNN-related (patch
export, training, model sync — see §8 below). Create a bucket in the same
GCP project (Cloud Storage → Buckets → Create), then set `GEE_EXPORT_BUCKET`
in `.env` to its name. The same bucket doubles as the model-artifact sync
target (`GCS_MODEL_BUCKET`, defaults to reusing this one — see `.env.example`).

## 8. Training deep-learning models (Colab)

U-Net (land-cover) and the CNN heat model both train exclusively on Google
Colab's free GPU — **never locally**, no GPU or WSL2 needed on your machine
at all. Local machines only ever run inference on the trained weights
(CPU is plenty).

### 8a. Install the Google Colab extension in VS Code

Google publishes an official extension that connects a local `.ipynb` file
directly to a real Colab-hosted GPU kernel. Install "Google Colab" from the
VS Code Extensions panel, then to use it: open a notebook under
`notebooks/colab_training/` → **Select Kernel** (top-right) → **Colab** →
sign in with your Google account → **New Colab Server** → pick **GPU**
(the free T4 tier is enough) → connect. The notebook file itself stays
local and git-tracked as normal — only the *execution* happens on Colab's
VM.

### 8b. Auth: paste your service-account key when prompted

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

### 8c. Push the training inputs Colab can't regenerate itself

A couple of local build outputs need pushing to GCS once (and again after
you relabel/rebuild them) — Colab can't produce these on its own:

```
# the hand-labeled validation sample (needed by train_unet.ipynb)
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, VALIDATION_SAMPLE_GCS_PREFIX; gcs.upload_file('data/interim/validation_sample/validation_sample_200_labeled.csv', GCS_MODEL_BUCKET, f'{VALIDATION_SAMPLE_GCS_PREFIX}.csv')"

# the land-cover ensemble raster (needed by train_heat_cnn.ipynb, after
# you've trained U-Net and run scripts/build_landcover_ensemble.py locally)
python -c "from src.utils import gcs; from config.settings import GCS_MODEL_BUCKET, ENSEMBLE_RASTER_GCS_PREFIX; gcs.upload_file('data/processed/landcover/ensemble_landcover.tif', GCS_MODEL_BUCKET, f'{ENSEMBLE_RASTER_GCS_PREFIX}.tif')"
```

### 8d. Run the notebooks, then pull the results locally

Open and run `notebooks/colab_training/train_unet.ipynb` top to bottom
(GPU-connected, per 8a). It exports/caches training patches, trains, and
pushes the model + a training-run summary to GCS. Then, on your own
machine:

```
python scripts/pull_models.py --model unet
python scripts/run_landcover_unet_inference.py
python scripts/build_landcover_ensemble.py
```

This downloads the trained weights (hash-verified), imports the run into
your local MLflow store, runs full-Singapore CPU inference, and rebuilds
the ensemble raster — which `train_heat_cnn.ipynb` needs (push it per §8c
first). Then run that notebook the same way, and locally:

```
python scripts/pull_models.py --model cnn
```

Re-running any notebook later (a hyperparameter change, a relabel) is
cheap — the GCS-cached exports mean only the first run per fingerprint pays
for a real GEE export; everything after that is a fast download.

## Notes

- `data/`, `models/`, `credentials/`, and `.env` are all gitignored —
  nothing under them is shared via git. Each teammate needs their own
  local copy of everything and their own credentials.
- The original Colab notebooks are kept under
  `notebooks/Colab Notebooks/` for reference; the maintained pipeline now
  lives in `src/`, `validation/`, `scripts/`, and `app/`.
- `config/settings.py` is the single source of truth for pipeline
  constants (AOI, season window, seeds, etc.) — check there before
  assuming a value is hardcoded somewhere else.
