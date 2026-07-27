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

Only needed once you get to `scripts/train_landcover_unet.py` (TFRecord
patch export) or the Random Forest raster export in
`scripts/train_landcover_rf.py`. Create a bucket in the same GCP project
(Cloud Storage → Buckets → Create), then set `GEE_EXPORT_BUCKET` in `.env`
to its name.

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
